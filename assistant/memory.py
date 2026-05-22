"""Short-term conversational memory."""

from __future__ import annotations

from dataclasses import dataclass, field

from models.base import ChatMessage


@dataclass
class SlidingWindowMemory:
    """Keeps the most recent chat messages for short-term context."""

    max_messages: int = 8
    messages: list[ChatMessage] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported memory role: {role}")
        self.messages.append({"role": role, "content": content})
        self.messages = self.messages[-self.max_messages :]

    def context(self) -> list[ChatMessage]:
        return list(self.messages)

    def reset(self) -> None:
        self.messages.clear()
