"""Unified model backend interface for the AI Risk Evaluation Workbench.

This module defines a provider-agnostic :class:`ModelBackend` abstraction with
concrete implementations for OpenAI, Anthropic, and local models, plus a
:func:`get_backend` factory that wires a configured backend to its secrets
(resolved from the environment, never hardcoded).

Concrete SDK clients are imported lazily inside the methods that use them, so
importing this module never requires the provider SDKs to be installed, and
unit tests can inject mock clients without performing any network I/O.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from src.core.config import AppConfig, load_config


class ModelBackend(ABC):
    """Abstract base class for all model backends.

    Subclasses must implement :meth:`generate`, which turns a prompt into a
    string completion. Temperature and system prompts are optional so the
    interface can be used uniformly across providers.
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the backend for a given model.

        Args:
            model_name: Identifier of the model this backend will call.
        """
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate a completion for ``prompt``.

        Args:
            prompt: The user prompt to send to the model.
            system_prompt: Optional system instructions.
            temperature: Sampling temperature for this call.

        Returns:
            The model's response as a string.
        """
        raise NotImplementedError


class OpenAIBackend(ModelBackend):
    """Backend for OpenAI-compatible chat completions APIs."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize the OpenAI backend.

        Args:
            model_name: Model identifier (e.g. ``gpt-4o``).
            api_key: API key resolved from the environment (never hardcoded).
            base_url: Optional API base URL override.
            default_temperature: Default sampling temperature.
            max_tokens: Default maximum tokens to generate.
        """
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.default_temperature = default_temperature
        self.max_tokens = max_tokens
        self._client: Any = None

    def _create_client(self) -> Any:
        """Lazily construct the OpenAI client.

        Returns:
            An ``openai.OpenAI`` client instance.

        Raises:
            ImportError: If the ``openai`` SDK is not installed.
            ValueError: If no API key is available.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on installed SDK
            raise ImportError(
                "The 'openai' package is required for OpenAIBackend. "
                "Install it with: pip install openai"
            ) from exc
        if not self.api_key:
            raise ValueError(
                f"No API key resolved for model '{self.model_name}'. "
                "Set the appropriate environment variable."
            )
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    @property
    def client(self) -> Any:
        """Return the underlying OpenAI client, creating it on first use.

        Returns:
            The ``openai.OpenAI`` client.
        """
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @client.setter
    def client(self, value: Any) -> None:
        """Allow injecting a mock client for testing.

        Args:
            value: The client object to use for requests.
        """
        self._client = value

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate a chat completion via the OpenAI API.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system instructions.
            temperature: Sampling temperature for this call.

        Returns:
            The assistant message content as a string.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


class AnthropicBackend(ModelBackend):
    """Backend for Anthropic's Messages API."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize the Anthropic backend.

        Args:
            model_name: Model identifier (e.g. ``claude-sonnet``).
            api_key: API key resolved from the environment (never hardcoded).
            base_url: Optional API base URL override.
            default_temperature: Default sampling temperature.
            max_tokens: Default maximum tokens to generate.
        """
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.default_temperature = default_temperature
        self.max_tokens = max_tokens
        self._client: Any = None

    def _create_client(self) -> Any:
        """Lazily construct the Anthropic client.

        Returns:
            An ``anthropic.Anthropic`` client instance.

        Raises:
            ImportError: If the ``anthropic`` SDK is not installed.
            ValueError: If no API key is available.
        """
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on installed SDK
            raise ImportError(
                "The 'anthropic' package is required for AnthropicBackend. "
                "Install it with: pip install anthropic"
            ) from exc
        if not self.api_key:
            raise ValueError(
                f"No API key resolved for model '{self.model_name}'. "
                "Set the appropriate environment variable."
            )
        return anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)

    @property
    def client(self) -> Any:
        """Return the underlying Anthropic client, creating it on first use.

        Returns:
            The ``anthropic.Anthropic`` client.
        """
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @client.setter
    def client(self, value: Any) -> None:
        """Allow injecting a mock client for testing.

        Args:
            value: The client object to use for requests.
        """
        self._client = value

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate a message via the Anthropic API.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system instructions.
            temperature: Sampling temperature for this call.

        Returns:
            The assistant message content as a string.
        """
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        text_parts = [
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ]
        return "".join(text_parts)


class LocalBackend(ModelBackend):
    """Backend for locally hosted models (e.g. Hugging Face transformers).

    A transformer pipeline is loaded lazily from ``model_path``. For testing or
    deterministic behavior, a ``pipeline`` callable can be injected directly.
    """

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        pipeline: Optional[Callable[[str], str]] = None,
        default_temperature: float = 0.7,
    ) -> None:
        """Initialize the local backend.

        Args:
            model_name: Model identifier.
            model_path: Hugging Face repo id or local path. Required unless a
                ``pipeline`` callable is supplied.
            pipeline: Optional callable mapping a prompt to a response, used for
                testing or custom local inference.
            default_temperature: Default sampling temperature (informational).
        """
        super().__init__(model_name)
        self.model_path = model_path
        self._pipeline = pipeline
        self.default_temperature = default_temperature

    def _load_pipeline(self) -> Callable[[str], str]:
        """Lazily load a transformers text-generation pipeline.

        Returns:
            A callable that accepts a prompt and returns generated text.

        Raises:
            ImportError: If ``transformers`` is not installed.
            ValueError: If no ``model_path`` is configured.
        """
        if not self.model_path:
            raise ValueError(
                f"LocalBackend '{self.model_name}' requires a model_path or an "
                "injected pipeline callable."
            )
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:  # pragma: no cover - depends on installed SDK
            raise ImportError(
                "The 'transformers' package is required for local inference. "
                "Install it with: pip install transformers"
            ) from exc
        return hf_pipeline("text-generation", model=self.model_path)

    @property
    def pipeline(self) -> Callable[[str], str]:
        """Return the inference callable, loading it lazily if needed.

        Returns:
            The pipeline callable.
        """
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value: Callable[[str], str]) -> None:
        """Inject a pipeline callable (e.g. for tests).

        Args:
            value: The callable to use for inference.
        """
        self._pipeline = value

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate a completion using the local pipeline.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system instructions (prefixed to prompt).
            temperature: Sampling temperature (informational for HF pipeline).

        Returns:
            The generated text as a string.
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        result = self.pipeline(full_prompt)
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                return str(first.get("generated_text", ""))
            return str(first)
        return str(result)


def _infer_provider(model_name: str) -> str:
    """Infer a provider from a model name when none is configured.

    Args:
        model_name: The model identifier.

    Returns:
        One of ``"openai"``, ``"anthropic"``, or ``"local"``.
    """
    lowered = model_name.lower()
    if "claude" in lowered or "anthropic" in lowered:
        return "anthropic"
    if "gpt" in lowered or "openai" in lowered or "o1" in lowered or "o3" in lowered:
        return "openai"
    return "local"


def get_backend(
    model_name: str, config: Optional[AppConfig] = None
) -> ModelBackend:
    """Factory that returns a configured :class:`ModelBackend` for a model.

    The model's API key is resolved from the environment via the configured
    environment variable name; it is never passed as a literal value.

    Args:
        model_name: The model identifier to instantiate a backend for.
        config: Optional pre-loaded :class:`AppConfig`. Loaded from disk when
            omitted.

    Returns:
        A ready-to-use :class:`ModelBackend` subclass instance.

    Raises:
        ValueError: If the provider is unknown or misconfigured.
    """
    cfg = config or load_config()
    model_cfg = cfg.get_model(model_name)

    if model_cfg is not None:
        provider = model_cfg.provider
        api_key = cfg.get_api_key(model_name)
        if provider == "openai":
            return OpenAIBackend(
                model_name=model_name,
                api_key=api_key,
                base_url=model_cfg.base_url,
                default_temperature=model_cfg.default_temperature,
                max_tokens=model_cfg.max_tokens,
            )
        if provider == "anthropic":
            return AnthropicBackend(
                model_name=model_name,
                api_key=api_key,
                base_url=model_cfg.base_url,
                default_temperature=model_cfg.default_temperature,
                max_tokens=model_cfg.max_tokens,
            )
        if provider == "local":
            return LocalBackend(
                model_name=model_name,
                model_path=model_cfg.model_path,
                default_temperature=model_cfg.default_temperature,
            )
        raise ValueError(f"Unsupported provider '{provider}' for model '{model_name}'.")

    # Unknown model: infer provider and build a backend with no API key wired,
    # deferring key resolution to runtime (e.g. via env var set later).
    inferred = _infer_provider(model_name)
    if inferred == "openai":
        return OpenAIBackend(model_name=model_name)
    if inferred == "anthropic":
        return AnthropicBackend(model_name=model_name)
    return LocalBackend(model_name=model_name, model_path=model_name)


__all__ = [
    "ModelBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "LocalBackend",
    "get_backend",
]
