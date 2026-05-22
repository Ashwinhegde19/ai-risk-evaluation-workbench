"""OpenAI-compatible frontier model client."""

from __future__ import annotations

import os

from models.base import ChatMessage


class OpenAIModelClient:
    """Thin wrapper around the OpenAI chat completions API."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("FRONTIER_MODEL", "gpt-4.1-mini")
        self.name = self.model
        self._client = None

    def _load_client(self):
        if self._client is not None:
            return self._client

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the frontier assistant.")

        from openai import OpenAI

        base_url = os.getenv("OPENAI_BASE_URL")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        client = self._load_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content or ""
