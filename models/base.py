"""Shared interface for model backends."""

from __future__ import annotations

from typing import Protocol

ChatMessage = dict[str, str]


class ModelClient(Protocol):
    """Small interface used by the assistant orchestration layer."""

    name: str

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """Generate a response from chat-formatted messages."""
