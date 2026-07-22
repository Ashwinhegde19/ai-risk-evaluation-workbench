"""Shared helpers for the compliance mappers.

Internal module -- not part of the public API. Provides severity ordering,
threshold checks, evidence formatting, and risk-tier aggregation used by the
EU AI Act, NIST AI RMF, and ISO/IEC 42001 mappers and by the report
generator.
"""

from __future__ import annotations

from typing import List

from src.core.models import EvalResult, RiskTier, Severity

# Higher number == more severe. Used to compare an eval result's severity
# against a configurable threshold before raising a compliance finding.
SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}

# Higher number == more regulated. Used to pick the overall risk tier for a
# report (the single most severe tier across all findings).
RISK_TIER_ORDER: dict[RiskTier, int] = {
    RiskTier.MINIMAL: 0,
    RiskTier.LIMITED: 1,
    RiskTier.HIGH: 2,
    RiskTier.UNACCEPTABLE: 3,
}


def severity_meets(result: EvalResult, threshold: Severity) -> bool:
    """Return True when a result's severity meets or exceeds a threshold.

    Args:
        result: The evaluation result being classified.
        threshold: Minimum severity required to count as a compliance gap.

    Returns:
        True if ``result.severity`` ranks at or above ``threshold``.
    """
    return SEVERITY_ORDER[result.severity] >= SEVERITY_ORDER[threshold]


def evidence_for(result: EvalResult, max_len: int = 200) -> str:
    """Build a concise, human-readable evidence string from an eval result.

    Args:
        result: The evaluation result to summarize.
        max_len: Maximum length of the quoted response snippet.

    Returns:
        A single-line evidence string suitable for a compliance finding.
    """
    snippet = result.raw_response
    if len(snippet) > max_len:
        snippet = snippet[:max_len] + "..."
    return (
        f"Dimension '{result.dimension}' scored {result.score:.2f} "
        f"(severity={result.severity.value}); sample: {snippet}"
    )


def max_risk_tier(tiers: List[RiskTier]) -> RiskTier:
    """Return the highest risk tier in a list, or MINIMAL if empty.

    Args:
        tiers: Risk tiers to aggregate.

    Returns:
        The most severe tier present (MINIMAL when the list is empty).
    """
    if not tiers:
        return RiskTier.MINIMAL
    return max(tiers, key=lambda tier: RISK_TIER_ORDER[tier])
