"""Optional web-search evidence provider for current factual questions."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass(frozen=True)
class WebEvidence:
    title: str
    url: str
    snippet: str
    source_type: str
    retrieved_at: str
    score: float = 0.0

    @property
    def source(self) -> str:
        return self.url

    @property
    def text(self) -> str:
        return self.snippet

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WebSearchClient(Protocol):
    def search(
        self,
        query: str,
        *,
        allowed_domains: list[str] | None = None,
        max_results: int = 5,
    ) -> list[WebEvidence]:
        """Return web evidence for a query."""


class DisabledWebSearchClient:
    """No-op client used when web search is not configured."""

    def search(
        self,
        query: str,
        *,
        allowed_domains: list[str] | None = None,
        max_results: int = 5,
    ) -> list[WebEvidence]:
        return []


class TavilySearchClient:
    """Minimal Tavily-compatible search client.

    The interface is deliberately tiny so we can swap providers later without
    changing retrieval or eval code.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 20,
    ) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self.base_url = base_url or os.getenv("TAVILY_BASE_URL", "https://api.tavily.com/search")
        self.timeout_seconds = timeout_seconds

    def search(
        self,
        query: str,
        *,
        allowed_domains: list[str] | None = None,
        max_results: int = 5,
    ) -> list[WebEvidence]:
        if not self.api_key:
            return []

        payload: dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        if allowed_domains:
            payload["include_domains"] = allowed_domains

        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return []

        return parse_tavily_results(data)


def build_web_search_client() -> WebSearchClient:
    if not env_flag("ENABLE_WEB_SEARCH"):
        return DisabledWebSearchClient()
    return TavilySearchClient()


def parse_tavily_results(data: dict[str, Any]) -> list[WebEvidence]:
    retrieved_at = datetime.now(timezone.utc).isoformat()
    results = data.get("results") or []
    evidence: list[WebEvidence] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "")
        snippet = str(item.get("content") or item.get("snippet") or "")
        if not url or not snippet:
            continue
        evidence.append(
            WebEvidence(
                title=str(item.get("title") or "Untitled source"),
                url=url,
                snippet=snippet,
                source_type=classify_source(url),
                retrieved_at=retrieved_at,
                score=float(item.get("score") or 0.0),
            )
        )
    return evidence


def classify_source(url: str) -> str:
    lowered = url.lower()
    if ".gov" in lowered:
        return "government"
    if any(domain in lowered for domain in ("docs.", "developer.", "platform.")):
        return "primary_docs"
    if any(domain in lowered for domain in ("ollive.ai", "openai.com", "anthropic.com")):
        return "official_site"
    return "web"


def env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
