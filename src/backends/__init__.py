"""Model backend implementations and factory for the workbench.

The :func:`get_backend` factory routes a model slug to the correct lane:

* ``qwen3-8b*`` (self-deployed open-source) -> Modal L4 endpoint via
  ``OPEN_MODEL_BASE_URL``.
* ``opencode/*`` (free-tier red-team targets) -> OpenCode Zen gateway via
  ``OPENCODE_BASE_URL``, with the namespace stripped before the call.
* ``cline/*`` -> the Cline gateway.
* ``provider/model`` (frontier, e.g. ``openai/gpt-5``) -> Kilo gateway via
  ``KILO_BASE_URL`` (falling back to ``OPENAI_BASE_URL``).
* Otherwise -> the configured provider, or a provider inferred from the name.

See :mod:`src.backends.base` for the full implementation.
"""

from src.backends.base import (
    OPENCODE_DEFAULT_BASE_URL,
    OPENCODE_MODEL_PREFIX,
    OPEN_MODEL_PREFIX,
    AnthropicBackend,
    ClineBackend,
    LocalBackend,
    MistralBackend,
    MistralShieldstralBackend,
    ModelBackend,
    OpenAIBackend,
    get_backend,
)

__all__ = [
    "ModelBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "LocalBackend",
    "MistralBackend",
    "MistralShieldstralBackend",
    "ClineBackend",
    "get_backend",
    "OPEN_MODEL_PREFIX",
    "OPENCODE_MODEL_PREFIX",
    "OPENCODE_DEFAULT_BASE_URL",
]
