"""Gateway-backed frontier model client."""

from __future__ import annotations

import os

from models.base import ChatMessage


class FrontierGatewayClient:
    """Thin wrapper around Kilo and other OpenAI-compatible chat APIs."""

    def __init__(self, model: str | None = None) -> None:
        self.provider = os.getenv("FRONTIER_PROVIDER", "kilo").lower()
        self.model = model or os.getenv("FRONTIER_MODEL", self._default_model())
        self.name = self.model
        self._client = None

    def _load_client(self):
        if self._client is not None:
            return self._client

        api_key = self._api_key()
        if not api_key:
            raise RuntimeError(
                "A frontier API key is required. Set KILO_API_KEY for Kilo Gateway "
                "or OPENAI_API_KEY for direct OpenAI-compatible access."
            )

        from openai import OpenAI

        kwargs = {"api_key": api_key}
        base_url = self._base_url()
        if base_url:
            kwargs["base_url"] = base_url
        headers = self._default_headers()
        if headers:
            kwargs["default_headers"] = headers
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

    def _default_model(self) -> str:
        if self.provider in {"kilo", "kilocode", "kilo_gateway"}:
            return "deepseek/deepseek-v3.2"
        return "gpt-4.1-mini"

    def _api_key(self) -> str | None:
        if self.provider in {"kilo", "kilocode", "kilo_gateway"}:
            return os.getenv("KILO_API_KEY") or os.getenv("OPENAI_API_KEY")
        return os.getenv("OPENAI_API_KEY") or os.getenv("KILO_API_KEY")

    def _base_url(self) -> str | None:
        if os.getenv("OPENAI_BASE_URL"):
            return os.getenv("OPENAI_BASE_URL")
        if self.provider in {"kilo", "kilocode", "kilo_gateway"}:
            return os.getenv("KILO_BASE_URL", "https://api.kilo.ai/api/gateway")
        return None

    def _default_headers(self) -> dict[str, str]:
        mode = os.getenv("KILOCODE_MODE")
        if self.provider in {"kilo", "kilocode", "kilo_gateway"} and mode:
            return {"x-kilocode-mode": mode}
        return {}
