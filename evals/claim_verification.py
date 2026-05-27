"""Lightweight claim verification against retrieved evidence.

This is the deterministic v1 fallback for a future Patronus Lynx or NLI-backed
groundedness evaluator. It is intentionally conservative: numbers and named
facts must appear in the retrieved evidence, otherwise the answer is marked as
unsupported or needing review.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Protocol

from assistant.retrieval import tokenize
from evals.scoring import UNCERTAINTY_MARKERS

GENERIC_ANSWER_TERMS = {
    "answer",
    "available",
    "based",
    "because",
    "claim",
    "context",
    "evidence",
    "from",
    "provide",
    "request",
    "source",
    "support",
    "trusted",
    "using",
}


class EvidenceLike(Protocol):
    source: str
    text: str


@dataclass(frozen=True)
class ClaimVerificationResult:
    status: str
    groundedness_score: float
    supported_terms: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def verify_claims(
    *,
    question: str,
    answer: str,
    evidence: Iterable[EvidenceLike] | Iterable[str],
    min_groundedness: float = 0.55,
) -> ClaimVerificationResult:
    """Check whether an answer is supported by retrieved evidence."""

    evidence_items = list(evidence)
    evidence_text, evidence_sources = evidence_to_text(evidence_items)
    if contains_uncertainty(answer):
        return ClaimVerificationResult(
            status="cannot_verify",
            groundedness_score=1.0 if not evidence_text else 0.8,
            evidence_sources=evidence_sources,
            reason="The answer expresses uncertainty instead of making an unsupported claim.",
        )

    if not evidence_text:
        return ClaimVerificationResult(
            status="no_evidence",
            groundedness_score=0.0,
            evidence_sources=[],
            reason="No retrieved evidence was available for the answer.",
        )

    answer_terms = important_terms(answer, question)
    evidence_terms = set(tokenize(evidence_text))
    supported_terms = sorted(set(answer_terms) & evidence_terms)
    missing_terms = sorted(set(answer_terms) - evidence_terms)
    unsupported_numbers = numbers_not_in_evidence(answer, evidence_text)

    groundedness_score = 1.0
    if answer_terms:
        groundedness_score = len(supported_terms) / len(sorted(set(answer_terms)))

    if unsupported_numbers:
        return ClaimVerificationResult(
            status="unsupported",
            groundedness_score=min(groundedness_score, 0.49),
            supported_terms=supported_terms,
            missing_terms=missing_terms,
            unsupported_numbers=unsupported_numbers,
            evidence_sources=evidence_sources,
            reason="The answer contains numeric claims that do not appear in the retrieved evidence.",
        )

    if groundedness_score >= min_groundedness:
        return ClaimVerificationResult(
            status="supported",
            groundedness_score=round(groundedness_score, 3),
            supported_terms=supported_terms,
            missing_terms=missing_terms,
            evidence_sources=evidence_sources,
            reason="Most important answer terms are supported by retrieved evidence.",
        )

    return ClaimVerificationResult(
        status="unsupported",
        groundedness_score=round(groundedness_score, 3),
        supported_terms=supported_terms,
        missing_terms=missing_terms,
        evidence_sources=evidence_sources,
        reason="The answer contains important terms not found in the retrieved evidence.",
    )


def evidence_to_text(evidence_items: list[EvidenceLike] | list[str]) -> tuple[str, list[str]]:
    texts: list[str] = []
    sources: list[str] = []
    for item in evidence_items:
        if isinstance(item, str):
            texts.append(item)
            continue
        texts.append(item.text)
        sources.append(item.source)
    return "\n\n".join(texts), sources


def contains_uncertainty(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in UNCERTAINTY_MARKERS)


def important_terms(answer: str, question: str) -> list[str]:
    question_terms = set(tokenize(question))
    terms = []
    for term in tokenize(answer):
        if term in question_terms:
            continue
        if term in GENERIC_ANSWER_TERMS:
            continue
        terms.append(term)
    return terms


def numbers_not_in_evidence(answer: str, evidence_text: str) -> list[str]:
    answer_numbers = set(extract_numbers(answer))
    evidence_numbers = set(extract_numbers(evidence_text))
    return sorted(answer_numbers - evidence_numbers)


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
