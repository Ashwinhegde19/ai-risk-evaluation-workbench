"""Configuration system for the AI Risk Evaluation Workbench.

Configuration is loaded from a YAML file (``config.yaml`` by default). API keys
are *never* stored in configuration files: only the name of the environment
variable that holds the key is recorded, and the key is resolved at runtime via
:func:`os.getenv`. Secrets therefore never touch disk or source control.

The configuration file may reference environment variables using
``${VAR_NAME}`` or ``${VAR_NAME:-default}`` syntax; these are substituted
before parsing.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ModelConfig(BaseModel):
    """Configuration for a single model backend."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str = Field(..., description="Unique model identifier.")
    provider: str = Field(
        ..., description="Backend provider: 'openai', 'anthropic', or 'local'."
    )
    api_key_env: Optional[str] = Field(
        default=None,
        description="Name of the environment variable holding the API key.",
    )
    base_url: Optional[str] = Field(
        default=None, description="Optional override for the provider API base URL."
    )
    default_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Default sampling temperature."
    )
    max_tokens: int = Field(
        default=2048, ge=1, description="Default maximum tokens to generate."
    )
    model_path: Optional[str] = Field(
        default=None, description="Local model path / HF repo id (for 'local')."
    )

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        """Normalize and validate the provider name."""
        value = value.strip().lower()
        if value not in {"openai", "anthropic", "local"}:
            raise ValueError(
                f"Unsupported provider '{value}'. Use openai, anthropic, or local."
            )
        return value

    @field_validator("default_temperature", "max_tokens", mode="before")
    @classmethod
    def _coerce_numeric(cls, value: object) -> object:
        """Coerce env-substituted string values into the correct numeric type."""
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return value
            try:
                return float(value) if "." in value else int(value)
            except ValueError:
                return value
        return value


class EvalSuiteConfig(BaseModel):
    """Configuration for an evaluation suite."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str = Field(..., description="Suite name.")
    description: str = Field(default="", description="Human-readable description.")
    dimensions: List[str] = Field(
        default_factory=list, description="Risk dimensions covered by the suite."
    )
    judge_models: List[str] = Field(
        default_factory=list, description="Judge models used to score the suite."
    )


class GuardrailPolicyConfig(BaseModel):
    """Configuration for a guardrail policy tied to a deployment tier."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str = Field(..., description="Policy / deployment tier name.")
    block_pii: bool = Field(default=False, description="Block outputs containing PII.")
    max_toxicity: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Toxicity threshold for blocking."
    )
    block_injection: bool = Field(
        default=False, description="Block detected prompt injections."
    )
    log_only: bool = Field(
        default=False, description="Log violations instead of blocking."
    )


class JudgeConfig(BaseModel):
    """Configuration for the LLM-as-Judge ensemble's token budget.

    The judge models emit a small JSON object plus a short reasoning. A too-low
    ``max_tokens`` truncates the JSON mid-response (no closing brace), which used
    to crash the whole run; the budget here is deliberately generous. When a first
    parse fails, the ensemble retries once at ``retry_max_tokens`` before dropping
    the vote.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    max_tokens: int = Field(
        default=2048,
        ge=1,
        description="Max tokens for the initial judge generation call.",
    )
    retry_max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Max tokens for the single retry after an unparseable response.",
    )


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(strict=True, extra="forbid")

    models: Dict[str, ModelConfig] = Field(
        default_factory=dict, description="Models keyed by name."
    )
    eval_suites: Dict[str, EvalSuiteConfig] = Field(
        default_factory=dict, description="Eval suites keyed by name."
    )
    guardrail_policies: Dict[str, GuardrailPolicyConfig] = Field(
        default_factory=dict, description="Guardrail policies keyed by tier name."
    )
    judge: JudgeConfig = Field(
        default_factory=JudgeConfig,
        description="Token budget for the LLM-as-Judge ensemble.",
    )
    default_model: Optional[str] = Field(
        default=None, description="Name of the default model."
    )
    target_models: List[str] = Field(
        default_factory=list,
        description=(
            "Ordered slugs evaluated in a frontier-vs-open comparison run. "
            "Namespaced slugs (e.g. 'openai/gpt-5') route to the Kilo gateway; "
            "'qwen3-8b' routes to the self-deployed Modal L4 endpoint."
        ),
    )

    def get_model(self, model_name: str) -> Optional[ModelConfig]:
        """Return the configuration for a named model, if present.

        Args:
            model_name: The model identifier to look up.

        Returns:
            The matching :class:`ModelConfig`, or ``None`` if unknown.
        """
        return self.models.get(model_name)

    def get_api_key(self, model_name: str) -> Optional[str]:
        """Resolve the API key for a model from the environment.

        The key itself is never stored in config; only the environment
        variable *name* is recorded and resolved here at runtime.

        Args:
            model_name: The model whose key should be resolved.

        Returns:
            The API key string, or ``None`` if unset / not applicable.
        """
        model = self.get_model(model_name)
        if model is None or model.api_key_env is None:
            return None
        return os.getenv(model.api_key_env)

    def get_policy(self, name: str) -> Optional[GuardrailPolicyConfig]:
        """Return a guardrail policy by name.

        Args:
            name: Policy / tier name.

        Returns:
            The matching :class:`GuardrailPolicyConfig` or ``None``.
        """
        return self.guardrail_policies.get(name)


def _substitute_env(raw: object) -> object:
    """Recursively substitute ``${VAR}`` / ``${VAR:-default}`` in config values.

    Args:
        raw: A parsed YAML scalar, list, or dict.

    Returns:
        The same structure with environment variables substituted in strings.
    """
    if isinstance(raw, dict):
        return {key: _substitute_env(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return [_substitute_env(item) for item in raw]
    if isinstance(raw, str):
        def _replace(match: "re.Match[str]") -> str:
            var_name, default = match.group(1), match.group(2)
            return os.environ.get(var_name, default if default is not None else "")

        return _ENV_PATTERN.sub(_replace, raw)
    return raw


def _default_config_path() -> Path:
    """Resolve the default configuration file path.

    Returns:
        Path to ``config.yaml`` resolved via ``AI_WORKBENCH_CONFIG`` env var or
        the file shipped alongside this module.
    """
    env_path = os.environ.get("AI_WORKBENCH_CONFIG")
    if env_path:
        return Path(env_path)
    return Path(__file__).with_name("config.yaml")


def default_config() -> AppConfig:
    """Build a sensible built-in default configuration.

    Returns:
        An :class:`AppConfig` with commonly used models pre-registered.
    """
    models = {
        "gpt-4o": ModelConfig(
            name="gpt-4o", provider="openai", api_key_env="OPENAI_API_KEY"
        ),
        "claude-sonnet": ModelConfig(
            name="claude-sonnet",
            provider="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
        ),
        "local": ModelConfig(
            name="local", provider="local", model_path="Qwen/Qwen2.5-0.5B-Instruct"
        ),
    }
    suites = {
        "full": EvalSuiteConfig(
            name="full",
            description="Full safety evaluation suite.",
            dimensions=[
                "hallucination",
                "bias",
                "toxicity",
                "jailbreak_resistance",
                "privacy",
                "ip_theft",
                "harmful_content",
            ],
            judge_models=["gpt-4o", "claude-sonnet"],
        )
    }
    policies = {
        "production": GuardrailPolicyConfig(
            name="production", block_pii=True, max_toxicity=0.7, block_injection=True
        ),
        "testing": GuardrailPolicyConfig(
            name="testing", block_pii=False, max_toxicity=0.9, log_only=True
        ),
    }
    return AppConfig(
        models=models,
        eval_suites=suites,
        guardrail_policies=policies,
        default_model="gpt-4o",
    )


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load application configuration from a YAML file.

    Environment variables of the form ``${VAR}`` / ``${VAR:-default}`` are
    substituted before parsing. If the file does not exist, a built-in
    :func:`default_config` is returned so the system works out of the box.

    Args:
        path: Optional path to a ``config.yaml``. Falls back to the default
            location when omitted.

    Returns:
        A validated :class:`AppConfig`.
    """
    config_path = Path(path) if path else _default_config_path()
    if not config_path.exists():
        return default_config()
    raw_text = config_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw_text) or {}
    substituted = _substitute_env(parsed)
    return AppConfig.model_validate(substituted)


__all__ = [
    "ModelConfig",
    "EvalSuiteConfig",
    "GuardrailPolicyConfig",
    "JudgeConfig",
    "AppConfig",
    "load_config",
    "default_config",
]
