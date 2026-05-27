"""Local trusted-document retrieval for grounded assistant answers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KB_DIR = ROOT / "knowledge_base"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "does",
    "for",
    "from",
    "has",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "say",
    "should",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
}


@dataclass(frozen=True)
class RetrievedContext:
    source: str
    title: str
    text: str
    score: int
    matched_terms: list[str] = field(default_factory=list)
    source_type: str = "local_kb"


class SimpleRetriever:
    """Keyword/BM25-style retrieval over local trusted text documents."""

    def __init__(self, kb_dir: str | Path = DEFAULT_KB_DIR, *, chunk_size: int = 1800) -> None:
        self.kb_dir = Path(kb_dir)
        self.chunk_size = chunk_size

    def search(
        self,
        query: str,
        *,
        min_score: int = 2,
        limit: int = 3,
    ) -> list[RetrievedContext]:
        query_terms = tokenize(query)
        if not query_terms or not self.kb_dir.exists():
            return []

        results: list[RetrievedContext] = []
        for path in sorted(self.kb_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            for chunk in self._chunks(text):
                chunk_terms = set(tokenize(chunk))
                matched_terms = sorted(set(query_terms) & chunk_terms)
                score = len(matched_terms)
                if score >= min_score:
                    results.append(
                        RetrievedContext(
                            source=str(path),
                            title=path.stem.replace("_", " ").title(),
                            text=chunk.strip(),
                            score=score,
                            matched_terms=matched_terms,
                        )
                    )

        return sorted(results, key=lambda item: item.score, reverse=True)[:limit]

    def best_context(self, query: str, *, min_score: int = 2) -> RetrievedContext | None:
        results = self.search(query, min_score=min_score, limit=1)
        return results[0] if results else None

    def format_context(self, contexts: list[RetrievedContext]) -> str:
        if not contexts:
            return ""
        sections = []
        for index, context in enumerate(contexts, start=1):
            sections.append(
                "\n".join(
                    [
                        f"[Source {index}] {context.title}",
                        f"Path: {context.source}",
                        f"Matched terms: {', '.join(context.matched_terms)}",
                        context.text,
                    ]
                )
            )
        return "\n\n".join(sections)

    def _chunks(self, text: str) -> list[str]:
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if not current:
                current = paragraph
            elif len(current) + len(paragraph) + 2 <= self.chunk_size:
                current = f"{current}\n\n{paragraph}"
            else:
                chunks.append(current)
                current = paragraph
        if current:
            chunks.append(current)
        return chunks


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9.-]*", text.lower())
    normalized = [normalize_token(token) for token in tokens]
    return [token for token in normalized if len(token) > 2 and token not in STOPWORDS]


def normalize_token(token: str) -> str:
    token = token.removesuffix(".ai")
    token = token.strip(".-")
    if token.endswith("ies") and len(token) > 5:
        return f"{token[:-3]}y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token
