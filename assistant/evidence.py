"""Evidence routing across local knowledge base and optional web search."""

from __future__ import annotations

from typing import Any

from assistant.retrieval import SimpleRetriever
from assistant.web_search import WebSearchClient, build_web_search_client


class EvidenceRetriever:
    """Retrieve trusted evidence using local KB first and web search second."""

    def __init__(
        self,
        *,
        local_retriever: SimpleRetriever | None = None,
        web_client: WebSearchClient | None = None,
    ) -> None:
        self.local_retriever = local_retriever or SimpleRetriever()
        self.web_client = web_client or build_web_search_client()

    def search(
        self,
        query: str,
        *,
        min_score: int = 2,
        limit: int = 3,
        allowed_domains: list[str] | None = None,
    ) -> list[Any]:
        local_results = self.local_retriever.search(query, min_score=min_score, limit=limit)
        if local_results:
            return local_results
        return self.web_client.search(query, allowed_domains=allowed_domains, max_results=limit)

    def format_context(self, contexts: list[Any]) -> str:
        sections = []
        for index, context in enumerate(contexts, start=1):
            matched_terms = getattr(context, "matched_terms", [])
            metadata = [
                f"[Source {index}] {getattr(context, 'title', 'Untitled source')}",
                f"Source: {getattr(context, 'source', '')}",
            ]
            if matched_terms:
                metadata.append(f"Matched terms: {', '.join(matched_terms)}")
            source_type = getattr(context, "source_type", "")
            if source_type:
                metadata.append(f"Source type: {source_type}")
            metadata.append(getattr(context, "text", ""))
            sections.append("\n".join(metadata))
        return "\n\n".join(sections)
