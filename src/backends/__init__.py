"""Model backend implementations and factory for the workbench.

The :func:`get_backend` factory routes a model slug to the correct lane:

* ``qwen3-8b*`` (self-deployed open-source) -> Modal L4 endpoint via
  ``OPEN_MODEL_BASE_URL``.
* ``provider/model`` (frontier, e.g. ``openai/gpt-5``) -> Kilo gateway via
  ``KILO_BASE_URL`` (falling back to ``OPENAI_BASE_URL``).
* Otherwise -> the configured provider, or a provider inferred from the name.

See :mod:`src.backends.base` for the full implementation.
"""

from src.backends.base import (
    OPEN_MODEL_PREFIX,
    AnthropicBackend,
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
    "MistralBackend",
    "MistralShieldstralBackend",
    "LocalBackend",
    "get_backend",
    "OPEN_MODEL_PREFIX",
]
