"""NIST AI Risk Management Framework (AI RMF 1.0) mapping.

Maps evaluation results onto the four core NIST AI RMF functions:

    * GOVERN  -- policies, accountability
    * MAP     -- context, risk identification
    * MEASURE -- assessment, testing  (where model evals primarily map)
    * MANAGE  -- mitigation, monitoring

Each finding receives a control identifier of the form ``FUNCTION-SUBSECTION``
(e.g. ``MEASURE-2.6``). The assigned risk tier is taken from the canonical EU
AI Act mapping (see :mod:`src.compliance.eu_ai_act`) so that the same
underlying issue carries a consistent tier across all frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.core.models import (
    ComplianceFinding,
    ComplianceFramework,
    EvalResult,
    Severity,
)
from src.compliance._common import evidence_for, severity_meets
from src.compliance.eu_ai_act import classify_dimension


@dataclass(frozen=True)
class NISTControl:
    """A NIST AI RMF function/subsection an eval dimension can map to."""

    function: str
    subsection: str
    title: str
    description: str


# Baseline mapping of eval dimensions -> NIST AI RMF functions.
DIMENSION_TO_NIST: Dict[str, NISTControl] = {
    "bias": NISTControl(
        "GOVERN", "2.1", "Bias governance",
        "Establish policy and accountability for bias mitigation.",
    ),
    "privacy": NISTControl(
        "GOVERN", "1.3", "Privacy governance",
        "Assign accountability for privacy risks.",
    ),
    "social_scoring": NISTControl(
        "GOVERN", "2.2", "Prohibited-use governance",
        "Govern unacceptable-use scenarios such as social scoring.",
    ),
    "employment": NISTControl(
        "MAP", "3.1", "Context -- employment",
        "Identify risks in the employment context.",
    ),
    "credit": NISTControl(
        "MAP", "3.2", "Context -- financial services",
        "Identify risks in the credit / financial-services context.",
    ),
    "law_enforcement": NISTControl(
        "MAP", "3.3", "Context -- law enforcement",
        "Identify risks in the law-enforcement context.",
    ),
    "education": NISTControl(
        "MAP", "3.4", "Context -- education",
        "Identify risks in the education context.",
    ),
    "hallucination": NISTControl(
        "MEASURE", "2.6", "Accuracy measurement",
        "Measure factual accuracy / hallucination rate.",
    ),
    "toxicity": NISTControl(
        "MEASURE", "2.4", "Toxicity measurement",
        "Measure toxic-output rate.",
    ),
    "jailbreak_resistance": NISTControl(
        "MEASURE", "2.5", "Robustness measurement",
        "Measure resistance to adversarial jailbreaks.",
    ),
    "harmful_content": NISTControl(
        "MEASURE", "2.3", "Harmful-content measurement",
        "Measure harmful-content generation rate.",
    ),
    "manipulation": NISTControl(
        "MEASURE", "2.2", "Manipulation measurement",
        "Measure susceptibility to manipulation.",
    ),
    "biometric": NISTControl(
        "MEASURE", "2.1", "Biometric measurement",
        "Measure biometric-handling risks.",
    ),
    "ip_theft": NISTControl(
        "MANAGE", "4.1", "IP risk mitigation",
        "Mitigate intellectual-property risks.",
    ),
    "deepfake": NISTControl(
        "MANAGE", "4.2", "Synthetic-content controls",
        "Manage disclosure of synthetic content.",
    ),
    "emotion": NISTControl(
        "MANAGE", "4.3", "Emotion-recognition controls",
        "Manage emotion-recognition disclosures.",
    ),
}

# Fallback: treat an unmapped dimension as a general MEASURE control.
DEFAULT_NIST = NISTControl(
    "MEASURE", "2.0", "General measurement",
    "General assessment of an unmapped dimension.",
)


def classify_dimension_nist(dimension: str) -> NISTControl:
    """Resolve the NIST AI RMF control an eval dimension maps to.

    Args:
        dimension: The risk dimension name (case-insensitive, trimmed).

    Returns:
        The matching :class:`NISTControl`, or the general MEASURE default.
    """
    return DIMENSION_TO_NIST.get(dimension.strip().lower(), DEFAULT_NIST)


def map_to_nist_rmf(
    eval_results: List[EvalResult],
    severity_threshold: Severity = Severity.MEDIUM,
) -> List[ComplianceFinding]:
    """Map evaluation results to NIST AI RMF compliance findings.

    A finding is emitted for every eval result whose severity meets or exceeds
    ``severity_threshold``. Each finding cites the relevant RMF function and a
    ``FUNCTION-SUBSECTION`` control identifier (e.g. ``MEASURE-2.6``).

    Args:
        eval_results: Evaluation outcomes to map.
        severity_threshold: Minimum severity required to raise a finding.

    Returns:
        NIST AI RMF compliance findings, one per qualifying eval result.
    """
    findings: List[ComplianceFinding] = []
    for result in eval_results:
        if not severity_meets(result, severity_threshold):
            continue
        control = classify_dimension_nist(result.dimension)
        # Risk tier is sourced from the canonical EU AI Act mapping so that
        # a given issue carries a consistent tier across frameworks.
        risk_tier = classify_dimension(result.dimension).risk_tier
        control_id = f"{control.function}-{control.subsection}"
        findings.append(
            ComplianceFinding(
                framework=ComplianceFramework.NIST_RMF,
                control_id=control_id,
                risk_tier=risk_tier,
                description=control.description,
                evidence=evidence_for(result),
                severity=result.severity,
            )
        )
    return findings
