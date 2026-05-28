"""HTTP client for optional Modal-hosted OSS model endpoints."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from models.base import ChatMessage


class ModalEndpointClient:
    """Call a Modal-hosted chat completion endpoint with the shared model interface."""

    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.endpoint_url = endpoint_url or os.getenv("MODAL_OSS_ENDPOINT", "")
        self.api_key = api_key or os.getenv("MODAL_API_KEY", "")
        self.name = model_name or os.getenv("MODAL_OSS_MODEL_NAME", "modal-oss-assistant")
        self.timeout_seconds = timeout_seconds or int(os.getenv("MODAL_TIMEOUT_SECONDS", "120"))

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        if not self.endpoint_url:
            raise RuntimeError("MODAL_OSS_ENDPOINT is required when OSS_BACKEND=modal.")

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            self.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Modal endpoint failed with HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Modal endpoint request failed: {exc}") from exc

        return parse_modal_response(data)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def parse_modal_response(data: dict[str, Any]) -> str:
    """Accept a few common response shapes from custom Modal endpoints."""

    if isinstance(data.get("response"), str):
        return data["response"]
    if isinstance(data.get("text"), str):
        return data["text"]
    if isinstance(data.get("content"), str):
        return data["content"]

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]

    raise RuntimeError("Modal endpoint response did not include response, text, content, or choices.")
