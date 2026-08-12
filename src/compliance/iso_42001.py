"""ISO/IEC 42001 (AI Management System) control mapping.

Maps evaluation results onto ISO/IEC 42001 Annex A controls, focusing on:

    * A.7 -- AI system impact assessment
    * A.8 -- AI system lifecycle

Each finding receives a control identifier of the form ``A.<section>.<n>``
(e.g. ``A.7.2``). The assigned risk tier is the *declared system class*
(see :mod:`src.compliance.system_class`), never a tier invented from the
eval dimension name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.compliance._common import evidence_for, severity_meets
from src.compliance.system_class import (
    DEFAULT_USE_CASE,
    SystemClassification,
    classify_system,
)
from src.core.models import (
    ComplianceFinding,
    ComplianceFramework,
    EvalResult,
    Severity,
)


@dataclass(frozen=True)
class ISOControl:
    """An ISO/IEC 42001 Annex A control an eval dimension can map to."""

    control_id: str
    clause: str
    title: str
    description: str


# Baseline mapping of eval dimensions -> ISO/IEC 42001 Annex A controls,
# emphasizing A.7 (impact assessment) and A.8 (lifecycle).
DIMENSION_TO_ISO: Dict[str, ISOControl] = {
    "social_scoring": ISOControl(
        "A.7.1", "A.7 AI system impact assessment", "Human-rights impact",
        "Assess fundamental-rights impacts (e.g. social scoring).",
    ),
    "bias": ISOControl(
        "A.7.2", "A.7 AI system impact assessment", "Impact on affected persons",
        "Assess discriminatory / biased impacts on affected persons.",
    ),
    "privacy": ISOControl(
        "A.7.3", "A.7 AI system impact assessment", "Impact on privacy",
        "Assess privacy impacts on data subjects.",
    ),
    "employment": ISOControl(
        "A.7.2", "A.7 AI system impact assessment", "Impact in employment",
        "Assess employment-context impacts.",
    ),
    "credit": ISOControl(
        "A.7.2", "A.7 AI system impact assessment", "Impact in financial services",
        "Assess credit / financial-services-context impacts.",
    ),
    "law_enforcement": ISOControl(
        "A.7.2", "A.7 AI system impact assessment", "Impact in law enforcement",
        "Assess law-enforcement-context impacts.",
    ),
    "education": ISOControl(
        "A.7.2", "A.7 AI system impact assessment", "Impact in education",
        "Assess education-context impacts.",
    ),
    "ip_theft": ISOControl(
        "A.8.2", "A.8 AI system lifecycle", "Intellectual property",
        "Assess IP / compliance controls across the lifecycle.",
    ),
    "hallucination": ISOControl(
        "A.8.3", "A.8 AI system lifecycle", "Data & model quality",
        "Assess accuracy / quality controls in the lifecycle.",
    ),
    "toxicity": ISOControl(
        "A.8.4", "A.8 AI system lifecycle", "Content safety controls",
        "Assess output-safety controls in the lifecycle.",
    ),
    "harmful_content": ISOControl(
        "A.8.4", "A.8 AI system lifecycle", "Content safety controls",
        "Assess harmful-content controls in the lifecycle.",
    ),
    "deepfake": ISOControl(
        "A.8.4", "A.8 AI system lifecycle", "Content safety / disclosure",
        "Assess synthetic-content disclosure controls.",
    ),
    "emotion": ISOControl(
        "A.8.4", "A.8 AI system lifecycle", "Content safety controls",
        "Assess emotion-recognition disclosures.",
    ),
    "jailbreak_resistance": ISOControl(
        "A.8.5", "A.8 AI system lifecycle", "Robustness & security",
        "Assess adversarial robustness in the lifecycle.",
    ),
    "manipulation": ISOControl(
        "A.8.5", "A.8 AI system lifecycle", "Robustness & security",
        "Assess resistance to manipulation.",
    ),
    "biometric": ISOControl(
        "A.8.5", "A.8 AI system lifecycle", "Robustness & security",
        "Assess biometric data handling in the lifecycle.",
    ),
}

# Fallback: treat an unmapped dimension as a general A.8 lifecycle control.
DEFAULT_ISO = ISOControl(
    "A.8.1", "A.8 AI system lifecycle", "General lifecycle control",
    "Assess lifecycle controls for an unmapped dimension.",
)


def classify_dimension_iso(dimension: str) -> ISOControl:
    """Resolve the ISO/IEC 42001 control an eval dimension maps to.

    Args:
        dimension: The risk dimension name (case-insensitive, trimmed).

    Returns:
        The matching :class:`ISOControl`, or the general A.8 default.
    """
    return DIMENSION_TO_ISO.get(dimension.strip().lower(), DEFAULT_ISO)


def map_to_iso_42001(
    eval_results: List[EvalResult],
    severity_threshold: Severity = Severity.MEDIUM,
    system_class: SystemClassification | None = None,
) -> List[ComplianceFinding]:
    """Map evaluation results to ISO/IEC 42001 compliance findings.

    A finding is emitted for every eval result whose severity meets or exceeds
    ``severity_threshold``. Each finding cites the relevant Annex A control
    identifier (e.g. ``A.7.2``). ``risk_tier`` is the declared system class.

    Args:
        eval_results: Evaluation outcomes to map.
        severity_threshold: Minimum severity required to raise a finding.
        system_class: Declared use-case classification.

    Returns:
        ISO/IEC 42001 compliance findings, one per qualifying eval result.
    """
    classification = system_class or classify_system(DEFAULT_USE_CASE)
    findings: List[ComplianceFinding] = []
    for result in eval_results:
        if not severity_meets(result, severity_threshold):
            continue
        control = classify_dimension_iso(result.dimension)
        findings.append(
            ComplianceFinding(
                framework=ComplianceFramework.ISO_42001,
                control_id=control.control_id,
                risk_tier=classification.risk_tier,
                description=control.description,
                evidence=evidence_for(result),
                severity=result.severity,
            )
        )
    return findings
