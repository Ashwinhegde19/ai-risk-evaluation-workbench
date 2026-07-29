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

import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from src.core.config import AppConfig, load_config

# Prefix identifying the self-deployed open-source target. Slugs starting with
# this (e.g. ``qwen3-8b``) are routed to the Modal L4 endpoint via
# ``OPEN_MODEL_BASE_URL`` rather than the Kilo gateway.
OPEN_MODEL_PREFIX = "qwen3-8b"

# Prefix identifying Cline-routed models (e.g. ``cline-free/glm-5.2``).
CLINE_MODEL_PREFIXES = ("cline-free/", "cline/")


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
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate a completion for ``prompt``.

        Args:
            prompt: The user prompt to send to the model.
            system_prompt: Optional system instructions.
            temperature: Sampling temperature for this call.
            max_tokens: Optional per-call override for the maximum tokens to
                generate. When ``None`` the backend's configured default is used.
            seed: Optional integer seed for reproducible sampling. Backends that
                support a native seed (e.g. OpenAI's ``seed``) forward it; others
                treat it as best-effort (the caller should also set
                ``temperature=0.0`` for determinism on those backends).

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
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate a chat completion via the OpenAI API.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system instructions.
            temperature: Sampling temperature for this call.
            max_tokens: Optional per-call override for the max tokens to
                generate; defaults to the backend's configured ``max_tokens``.
            seed: Optional integer seed forwarded to the OpenAI ``seed`` request
                parameter. Combined with ``temperature=0.0`` this makes repeated
                calls deterministic (best-effort, per the OpenAI API contract).

        Returns:
            The assistant message content as a string.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed

        response = self.client.chat.completions.create(**kwargs)
        if not response.choices:
            raise ValueError(f"API returned no choices for model {self.model_name}")
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
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate a message via the Anthropic API.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system instructions.
            temperature: Sampling temperature for this call.
            max_tokens: Optional per-call override for the max tokens to
                generate; defaults to the backend's configured ``max_tokens``.
            seed: Accepted for interface parity. The Anthropic Messages API has
                no native seed parameter, so reproducibility here is best-effort:
                callers should pair this with ``temperature=0.0`` for determinism.

        Returns:
            The assistant message content as a string.
        """
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
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
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate a completion using the local pipeline.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system instructions (prefixed to prompt).
            temperature: Sampling temperature (informational for HF pipeline).
            max_tokens: Optional max tokens to generate (informational for the
                HF pipeline, which is not token-budgeted per call here).
            seed: Accepted for interface parity. The Hugging Face pipeline used
                here does not thread a per-call seed, so reproducibility is
                best-effort; callers should pair this with ``temperature=0.0``.

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


class ClineBackend(ModelBackend):
    """Backend for Cline-routed models (e.g. ``cline-free/glm-5.2``).

    The Cline API requires HTTP/2 and wraps the standard OpenAI chat-completion
    response inside ``{"data": {...}, "success": true}``.  This backend uses
    ``httpx`` with ``http2=True`` and unwraps the envelope.

    Token resolution order (fail loud, never mock):
      1. ``CLINE_API_KEY`` env var — the full ``workos:...`` string, or a bare
         token (the ``workos:`` prefix is prepended automatically if missing).
      2. ``~/.cline/data/settings/providers.json`` — the ``accessToken`` field
         inside ``providers.cline.settings.auth``.
      3. Raise ``ValueError`` with a clear message.

    Retry: up to 3 attempts with exponential backoff on HTTP 429 / 5xx and on
    transient ``httpx.TransportError``.
    """

    DEFAULT_BASE_URL = "https://api.cline.bot/api/v1"

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        default_temperature: float = 0.7,
        max_tokens: int = 4096,
        _transport: Any = None,
    ) -> None:
        super().__init__(model_name)
        self.base_url = (base_url or os.getenv("CLINE_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.default_temperature = default_temperature
        self.max_tokens = max_tokens
        # Private test seam: when set, the httpx client uses this transport and
        # skips http2 (mock transports do not negotiate HTTP/2). Production calls
        # leave this None and get a real http2=True client.
        self._transport = _transport

    # ── token resolution ──────────────────────────────────────────────

    @staticmethod
    def _read_token() -> str:
        """Resolve the Cline OAuth token (env → providers.json → raise)."""
        # 1. Environment variable
        env_key = os.getenv("CLINE_API_KEY", "").strip()
        if env_key:
            return env_key if env_key.startswith("workos:") else f"workos:{env_key}"

        # 2. Cline CLI providers file
        import json
        from pathlib import Path

        path = Path.home() / ".cline" / "data" / "settings" / "providers.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                token = (
                    data.get("providers", {})
                    .get("cline", {})
                    .get("settings", {})
                    .get("auth", {})
                    .get("accessToken", "")
                )
                if token:
                    return token
            except (json.JSONDecodeError, OSError):
                pass  # fall through to raise

        # 3. Fail loud
        raise ValueError(
            "Cline token not found. Set CLINE_API_KEY or run the `cline` CLI "
            "once to refresh ~/.cline/data/settings/providers.json "
            "(the token expires)."
        )

    # ── generate ──────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate via the Cline API using ``httpx`` with HTTP/2 + retry.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system instructions.
            temperature: Sampling temperature.
            max_tokens: Optional per-call max tokens override.
            seed: Accepted for interface parity (not forwarded).

        Returns:
            The model's response text (``<think>`` blocks are stripped
            downstream by the model-agnostic stripper in
            ``src/redteam/strategies/base.py``).
        """
        import time

        import httpx

        token = self._read_token()
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                # Production: real HTTP/2 client. Tests inject a MockTransport
                # via _transport, which must NOT request http2 (mock transports
                # do not negotiate it).
                if self._transport is not None:
                    client_ctx = httpx.Client(transport=self._transport)
                else:
                    client_ctx = httpx.Client(http2=True, timeout=120.0)
                with client_ctx as client:
                    resp = client.post(url, json=payload, headers=headers)
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_exc = RuntimeError(
                        f"Cline API HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                    time.sleep(min(2 ** attempt, 8))
                    continue
                resp.raise_for_status()
                body = resp.json()
                break
            except httpx.TransportError as exc:
                last_exc = exc
                time.sleep(min(2 ** attempt, 8))
                continue
        else:
            raise RuntimeError(
                f"Cline API failed after 3 attempts: {last_exc}"
            ) from last_exc

        # Unwrap the Cline envelope: {"data": {...}, "success": true}
        if body.get("success") is not True:
            raise RuntimeError(
                f"Cline API returned success=false: {str(body)[:300]}"
            )

        data = body.get("data", body)
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"Cline API returned no choices: {str(body)[:300]}")

        return choices[0].get("message", {}).get("content", "")


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


def _is_mock_enabled() -> bool:
    """Return whether mock mode is explicitly enabled via ``MOCK=1``.

    Mock mode is opt-in: only the literal value ``"1"`` enables it, so an unset
    or empty ``MOCK`` variable means real mode.

    Returns:
        ``True`` when ``MOCK`` is set to ``"1"``.
    """
    return os.getenv("MOCK", "").strip() == "1"


def _is_open_model(model_name: str) -> bool:
    """Return whether a slug refers to the self-deployed open-source model.

    Args:
        model_name: The model slug (e.g. ``qwen3-8b``).

    Returns:
        ``True`` when the slug starts with the open-model prefix.
    """
    return model_name.lower().startswith(OPEN_MODEL_PREFIX)


def _is_frontier_slug(model_name: str) -> bool:
    """Return whether a slug is a namespaced frontier model.

    Frontier models are addressed as ``provider/model`` (e.g. ``openai/gpt-5``,
    ``anthropic/claude-opus-4.1``, ``google/gemini-2.5-pro``) and are reached
    through the Kilo gateway.

    Args:
        model_name: The model slug.

    Returns:
        ``True`` when the slug contains a ``provider/`` namespace prefix.
    """
    return "/" in model_name


def _is_cline_model(model_name: str) -> bool:
    """Return whether a slug refers to a Cline-routed model.

    Args:
        model_name: The model slug (e.g. ``cline-free/glm-5.2``).

    Returns:
        ``True`` when the slug starts with ``cline-free/`` or ``cline/``.
    """
    lowered = model_name.lower()
    return lowered.startswith(CLINE_MODEL_PREFIXES)


def _resolve_cline_model(model_name: str) -> ModelBackend:
    """Build a backend for a Cline-routed model.

    The slug is passed through VERBATIM as the ``model`` field (Cline expects
    the exact string ``cline-free/glm-5.2``, not a stripped id).

    Args:
        model_name: The Cline slug (e.g. ``cline-free/glm-5.2``).

    Returns:
        A :class:`ClineBackend` instance.
    """
    base_url = os.getenv("CLINE_BASE_URL", ClineBackend.DEFAULT_BASE_URL)
    print(
        f"[backend] target={model_name} base_url={base_url} "
        f"mock={'on' if _is_mock_enabled() else 'off'}"
    )
    if _is_mock_enabled():
        return OpenAIBackend(model_name=model_name, api_key="mock", base_url="http://mock.local/v1")
    return ClineBackend(model_name=model_name, base_url=base_url)


def _resolve_open_model(model_name: str) -> ModelBackend:
    """Build a backend for the open-source model via the Modal L4 endpoint.

    Args:
        model_name: The open-model slug (e.g. ``qwen3-8b``).

    Returns:
        An :class:`OpenAIBackend` pointed at ``OPEN_MODEL_BASE_URL``.

    Raises:
        ValueError: If ``OPEN_MODEL_BASE_URL`` is unset and mock mode is off.
    """
    base_url = os.getenv("OPEN_MODEL_BASE_URL")
    api_key = os.getenv("OPEN_MODEL_API_KEY", "none")
    if not base_url:
        if _is_mock_enabled():
            base_url = "http://mock.local/v1"
        else:
            raise ValueError(
                f"Cannot route open-source model '{model_name}': "
                "OPEN_MODEL_BASE_URL is not set. Deploy the Modal endpoint "
                "(modal deploy modal_deploy.py) and set OPEN_MODEL_BASE_URL to "
                "its /v1 URL, or set MOCK=1 for offline runs."
            )
    print(f"[backend] target={model_name} base_url={base_url} mock={'on' if _is_mock_enabled() else 'off'}")
    return OpenAIBackend(model_name=model_name, api_key=api_key, base_url=base_url)


def _resolve_frontier_model(model_name: str) -> ModelBackend:
    """Build a backend for a namespaced frontier model via the Kilo gateway.

    Args:
        model_name: The frontier slug (e.g. ``openai/gpt-5``).

    Returns:
        An :class:`OpenAIBackend` pointed at the Kilo (or OpenAI) gateway.

    Raises:
        ValueError: If no gateway base URL is configured and mock mode is off.
    """
    base_url = os.getenv("KILO_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("KILO_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not base_url:
        if _is_mock_enabled():
            base_url = "http://mock.local/v1"
        else:
            raise ValueError(
                f"Cannot route frontier model '{model_name}': neither "
                "KILO_BASE_URL nor OPENAI_BASE_URL is set. Configure the Kilo "
                "gateway (KILO_BASE_URL + KILO_API_KEY), or set MOCK=1 for "
                "offline runs."
            )
    print(f"[backend] target={model_name} base_url={base_url} mock={'on' if _is_mock_enabled() else 'off'}")
    return OpenAIBackend(model_name=model_name, api_key=api_key, base_url=base_url)


def get_backend(
    model_name: str, config: Optional[AppConfig] = None
) -> ModelBackend:
    """Factory that returns a configured :class:`ModelBackend` for a model.

    Routing is decided by the model slug:

    * ``qwen3-8b*`` (open-source) -> Modal L4 endpoint via ``OPEN_MODEL_BASE_URL``.
    * ``provider/model`` (frontier, e.g. ``openai/gpt-5``) -> Kilo gateway via
      ``KILO_BASE_URL`` (falling back to ``OPENAI_BASE_URL``).
    * Otherwise -> the configured provider, or a provider inferred from the name.

    The model's API key is resolved from the environment via the configured
    environment variable name; it is never passed as a literal value. When a
    routed slug has no resolvable base URL and mock mode is off, this raises
    rather than silently falling back to mock.

    Args:
        model_name: The model identifier to instantiate a backend for.
        config: Optional pre-loaded :class:`AppConfig`. Loaded from disk when
            omitted.

    Returns:
        A ready-to-use :class:`ModelBackend` subclass instance.

    Raises:
        ValueError: If the provider is unknown or misconfigured, or a routed
            slug has no base URL in real (non-mock) mode.
    """
    # Slug-based routing takes precedence so the open-source and frontier lanes
    # are selected purely by name, independent of config.yaml contents.
    if _is_open_model(model_name):
        return _resolve_open_model(model_name)
    if _is_cline_model(model_name):
        return _resolve_cline_model(model_name)
    if _is_frontier_slug(model_name):
        return _resolve_frontier_model(model_name)

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
    "ClineBackend",
    "get_backend",
    "OPEN_MODEL_PREFIX",
    "CLINE_MODEL_PREFIXES",
]
