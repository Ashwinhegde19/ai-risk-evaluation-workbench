"""EU AI Act compliance mapping.

Maps evaluation results (one per risk dimension) onto the EU AI Act risk
classification:

    * Unacceptable Risk  -- Art. 5  (prohibited practices)
    * High Risk          -- Art. 6  (Annex III high-risk systems)
    * Limited Risk       -- Art. 50 (transparency obligations)
    * Minimal Risk       -- everything else (general transparency, Art. 13)

Public API:
    * :func:`classify_dimension` -- resolve the provision a dimension maps to.
    * :func:`classify_risk_tier` -- turn eval results into EU AI Act findings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.core.models import (
    ComplianceFinding,
    ComplianceFramework,
    EvalResult,
    RiskTier,
    Severity,
)
from src.compliance._common import evidence_for, severity_meets


@dataclass(frozen=True)
class EUAIActControl:
    """A single EU AI Act provision that an eval dimension can map to."""

    article: str
    risk_tier: RiskTier
    title: str
    description: str


# Baseline mapping of eval dimensions -> EU AI Act provisions. Dimensions are
# matched case-insensitively (see :func:`classify_dimension`).
DIMENSION_TO_EU_AI_ACT: Dict[str, EUAIActControl] = {
    "social_scoring": EUAIActControl(
        "Art. 5(1)(c)",
        RiskTier.UNACCEPTABLE,
        "Prohibited: social scoring",
        "General social scoring of natural persons is prohibited under Art. 5.",
    ),
    "manipulation": EUAIActControl(
        "Art. 5(1)(a)",
        RiskTier.UNACCEPTABLE,
        "Prohibited: manipulative techniques",
        "Subliminal or manipulative techniques that distort behavior are prohibited.",
    ),
    "biometric": EUAIActControl(
        "Art. 5(1)(h)",
        RiskTier.UNACCEPTABLE,
        "Prohibited: real-time biometric identification",
        "Real-time remote biometric identification in public spaces is prohibited.",
    ),
    "bias": EUAIActControl(
        "Art. 6 / Annex III",
        RiskTier.HIGH,
        "High-risk: bias & discrimination",
        "Biased outputs in high-risk contexts (employment, credit, ...) trigger Art. 6 duties.",
    ),
    "privacy": EUAIActControl(
        "Art. 6 / Annex III",
        RiskTier.HIGH,
        "High-risk: fundamental rights (privacy)",
        "Privacy infringements in high-risk systems trigger Art. 6 duties.",
    ),
    "employment": EUAIActControl(
        "Art. 6(1) Annex III(1)",
        RiskTier.HIGH,
        "High-risk: employment",
        "Employment and worker-management systems are high-risk.",
    ),
    "credit": EUAIActControl(
        "Art. 6(1) Annex III(3)",
        RiskTier.HIGH,
        "High-risk: access to services (credit)",
        "Creditworthiness and access-to-services systems are high-risk.",
    ),
    "law_enforcement": EUAIActControl(
        "Art. 6(1) Annex III(2)",
        RiskTier.HIGH,
        "High-risk: law enforcement",
        "Law-enforcement systems are high-risk.",
    ),
    "education": EUAIActControl(
        "Art. 6(1) Annex III(1)",
        RiskTier.HIGH,
        "High-risk: education",
        "Education and vocational-training systems are high-risk.",
    ),
    "jailbreak_resistance": EUAIActControl(
        "Art. 50(1)",
        RiskTier.LIMITED,
        "Limited risk: AI system interaction",
        "Conversational systems must disclose they are AI (Art. 50).",
    ),
    "chatbot": EUAIActControl(
        "Art. 50(1)",
        RiskTier.LIMITED,
        "Limited risk: AI system interaction",
        "Conversational AI systems must disclose they are AI (Art. 50).",
    ),
    "deepfake": EUAIActControl(
        "Art. 50(2)",
        RiskTier.LIMITED,
        "Limited risk: deepfakes / synthetic content",
        "Synthetic content must be disclosed (Art. 50).",
    ),
    "emotion": EUAIActControl(
        "Art. 50(3)",
        RiskTier.LIMITED,
        "Limited risk: emotion recognition",
        "Emotion-recognition systems must disclose their operation (Art. 50).",
    ),
    "harmful_content": EUAIActControl(
        "Art. 50(2)",
        RiskTier.LIMITED,
        "Limited risk: synthetic / harmful content",
        "AI-generated harmful content should be clearly disclosed (Art. 50).",
    ),
    "hallucination": EUAIActControl(
        "Art. 13",
        RiskTier.MINIMAL,
        "Minimal risk: accuracy",
        "Hallucinations are an accuracy/transparency concern (Art. 13) -- minimal risk.",
    ),
    "toxicity": EUAIActControl(
        "Art. 13",
        RiskTier.MINIMAL,
        "Minimal risk: output quality",
        "Toxic outputs are a quality concern -- minimal regulatory risk under the Act.",
    ),
    "ip_theft": EUAIActControl(
        "Art. 53",
        RiskTier.MINIMAL,
        "Minimal risk: IP / copyright",
        "Copyright/training obligations (Art. 53) -- minimal direct regulatory risk.",
    ),
}

# Fallback for dimensions not present in the mapping above.
DEFAULT_EU_AI_ACT = EUAIActControl(
    "Art. 13",
    RiskTier.MINIMAL,
    "Minimal risk: general obligation",
    "Unmapped dimension -- treated as minimal risk (general transparency obligation).",
)


def classify_dimension(dimension: str) -> EUAIActControl:
    """Resolve the EU AI Act control an eval dimension maps to.

    Args:
        dimension: The risk dimension name (case-insensitive, trimmed).

    Returns:
        The matching :class:`EUAIActControl`, or the minimal-risk default.
    """
    return DIMENSION_TO_EU_AI_ACT.get(dimension.strip().lower(), DEFAULT_EU_AI_ACT)


def classify_risk_tier(
    eval_results: List[EvalResult],
    severity_threshold: Severity = Severity.MEDIUM,
) -> List[ComplianceFinding]:
    """Map evaluation results to EU AI Act compliance findings.

    A finding is emitted for every eval result whose severity meets or exceeds
    ``severity_threshold`` -- i.e. a real compliance gap was observed. Each
    finding cites the relevant article, risk tier, and supporting evidence.

    Args:
        eval_results: Evaluation outcomes to classify.
        severity_threshold: Minimum severity required to raise a finding.

    Returns:
        EU AI Act compliance findings, one per qualifying eval result.
    """
    findings: List[ComplianceFinding] = []
    for result in eval_results:
        if not severity_meets(result, severity_threshold):
            continue
        control = classify_dimension(result.dimension)
        findings.append(
            ComplianceFinding(
                framework=ComplianceFramework.EU_AI_ACT,
                control_id=control.article,
                risk_tier=control.risk_tier,
                description=control.description,
                evidence=evidence_for(result),
                severity=result.severity,
            )
        )
    return findings
