"""EU AI Act mapping: use-case class first, evals as residual evidence.

Public API:

    * :func:`classify_system` -- legal class from declared use case
      (re-exported from :mod:`src.compliance.system_class`).
    * :func:`classify_dimension` -- which *duty* an eval dimension can
      speak to; does **not** assign a risk tier.
    * :func:`classify_risk_tier` -- residual findings under a declared
      :class:`~src.compliance.system_class.SystemClassification`.

A poor bias or jailbreak score never upgrades a chatbot to Annex III
high-risk. A clean score never downgrades a declared employment or credit
system. Social-scoring *probes* are misuse residual, not an Art. 5 finding
unless the declared use case is itself social scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.compliance._common import evidence_for, severity_meets
from src.compliance.system_class import (
    DEFAULT_USE_CASE,
    LEGAL_DISCLAIMER,
    SystemClassification,
    SystemUseCase,
    classify_from_deployment_context,
    classify_system,
    parse_use_case,
)
from src.core.models import (
    ComplianceFinding,
    ComplianceFramework,
    EvalResult,
    RiskTier,
    Severity,
)


@dataclass(frozen=True)
class EUAIActControl:
    """An eval dimension's residual-duty mapping.

    ``risk_tier`` is intentionally absent. The legal class lives on
    :class:`SystemClassification`, not on the dimension name.
    """

    article: str
    title: str
    description: str
    high_risk_duty: str
    other_duty: str


# Eval dimensions -> duties they can evidence. These are not use cases.
DIMENSION_TO_EU_AI_ACT: dict[str, EUAIActControl] = {
    "bias": EUAIActControl(
        "Art. 10 / Art. 15",
        "Fairness / discrimination residual",
        "Biased outputs are residual evidence, not a risk-class upgrade.",
        "Art. 10 data governance and Art. 15 accuracy for this Annex III system.",
        "Residual fairness / discrimination risk; does not make the system Annex III.",
    ),
    "privacy": EUAIActControl(
        "Art. 10 / GDPR",
        "Privacy residual",
        "Privacy leakage is residual evidence (and usually a GDPR issue).",
        "Art. 10 data-governance gap for this Annex III system; GDPR still applies.",
        "Residual privacy / GDPR concern; not an Art. 6 classification.",
    ),
    "jailbreak_resistance": EUAIActControl(
        "Art. 15",
        "Adversarial robustness residual",
        "Jailbreak success is a robustness/cybersecurity residual.",
        "Art. 15 robustness and cybersecurity gap for this Annex III system.",
        "Residual robustness / product-safety concern (GPAI or Art. 50 system).",
    ),
    "harmful_content": EUAIActControl(
        "Art. 15 / residual safety",
        "Harmful-content residual",
        "Harmful-content leakage is residual safety evidence.",
        "Art. 15 / Art. 9 residual for this Annex III system.",
        "Residual content-safety concern; not an Art. 50 reclassification.",
    ),
    "toxicity": EUAIActControl(
        "residual quality",
        "Output-quality residual",
        "Toxic outputs are a quality residual, not a legal class.",
        "Art. 15 output-quality residual for this Annex III system.",
        "Residual quality concern; not a regulatory risk-class change.",
    ),
    "hallucination": EUAIActControl(
        "Art. 15",
        "Accuracy residual",
        "Hallucinations are an accuracy residual (Art. 15 if high-risk).",
        "Art. 15 accuracy gap for this Annex III system.",
        "Residual accuracy / grounding concern; not a risk-class upgrade.",
    ),
    "ip_theft": EUAIActControl(
        "Art. 53",
        "IP / copyright residual",
        "IP leakage is mainly a Chapter V / copyright residual.",
        "Art. 53 GPAI copyright / IP residual (plus any sector duty).",
        "Art. 53 GPAI copyright / training-data residual.",
    ),
    "chatbot": EUAIActControl(
        "Art. 50(1)",
        "Chatbot transparency",
        "Conversational systems must disclose they are AI when Art. 50 applies.",
        "Art. 50 still applies on top of Annex III duties.",
        "Art. 50(1) transparency duty for AI-system interaction.",
    ),
    "deepfake": EUAIActControl(
        "Art. 50(2)",
        "Synthetic-content disclosure",
        "Synthetic content must be disclosed under Art. 50(2).",
        "Art. 50(2) disclosure still applies on top of Annex III duties.",
        "Art. 50(2) synthetic-content disclosure.",
    ),
    "emotion": EUAIActControl(
        "Art. 50(3)",
        "Emotion-recognition disclosure",
        "Where not prohibited, emotion recognition has an Art. 50 disclosure duty.",
        "Confirm the use is not an Art. 5 workplace/education prohibition.",
        "Art. 50(3) disclosure if the use is lawful.",
    ),
    # The next four names look like use cases but appear only as *eval probes*
    # in this repo. They never classify the system.
    "social_scoring": EUAIActControl(
        "Art. 5 awareness",
        "Misuse probe: social scoring",
        "The model produced social-scoring-like content. That is misuse residual, "
        "not a finding that this system *is* a social-scoring system.",
        "Misuse residual plus Annex III duties; does not by itself trigger Art. 5.",
        "Misuse residual. Art. 5 applies only if the declared use *is* social scoring.",
    ),
    "manipulation": EUAIActControl(
        "Art. 5 awareness",
        "Misuse probe: manipulation",
        "Manipulative-content probe. Not an Art. 5 classification of the system.",
        "Misuse residual plus Annex III duties.",
        "Misuse residual. Art. 5 applies only if the declared use is manipulative.",
    ),
    "biometric": EUAIActControl(
        "Art. 5 awareness",
        "Misuse probe: biometric identification",
        "Biometric-identification probe. Not an Art. 5 classification.",
        "Misuse residual plus Annex III duties.",
        "Misuse residual. Art. 5 applies only if the declared use is prohibited biometric ID.",
    ),
    "employment": EUAIActControl(
        "Annex III §4 probe",
        "Context probe: employment",
        "Employment-themed eval item. Classify the *system* as employment to apply Art. 6.",
        "Consistent with this Annex III employment system.",
        "Context probe only; does not turn a chatbot into an employment system.",
    ),
    "credit": EUAIActControl(
        "Annex III §5 probe",
        "Context probe: credit",
        "Credit-themed eval item. Classify the *system* as credit to apply Art. 6.",
        "Consistent with this Annex III credit system.",
        "Context probe only; does not turn a chatbot into a credit system.",
    ),
    "law_enforcement": EUAIActControl(
        "Annex III §6 probe",
        "Context probe: law enforcement",
        "Law-enforcement-themed eval item. Classify the system to apply Art. 6.",
        "Consistent with this Annex III law-enforcement system.",
        "Context probe only; does not turn a chatbot into a law-enforcement system.",
    ),
    "education": EUAIActControl(
        "Annex III §3 probe",
        "Context probe: education",
        "Education-themed eval item. Classify the system to apply Art. 6.",
        "Consistent with this Annex III education system.",
        "Context probe only; does not turn a chatbot into an education system.",
    ),
}

DEFAULT_EU_AI_ACT = EUAIActControl(
    "residual",
    "Unmapped residual",
    "Unmapped eval dimension -- treated as residual technical evidence.",
    "Residual evidence for Arts. 9-15 on this Annex III system.",
    "Residual technical evidence; not a risk-class change.",
)


def classify_dimension(dimension: str) -> EUAIActControl:
    """Resolve the residual duty an eval dimension can speak to.

    This does not assign a legal risk tier. Use :func:`classify_system`
    for classification.

    Args:
        dimension: The risk dimension name (case-insensitive, trimmed).

    Returns:
        The matching :class:`EUAIActControl`, or the residual default.
    """
    return DIMENSION_TO_EU_AI_ACT.get(dimension.strip().lower(), DEFAULT_EU_AI_ACT)


def _duty_text(control: EUAIActControl, classification: SystemClassification) -> str:
    """Pick the high-risk or residual duty wording."""
    if classification.is_prohibited:
        return (
            f"{control.title}: the declared use is prohibited ({classification.articles}). "
            "Eval evidence is secondary to the Art. 5 ban."
        )
    duty = control.high_risk_duty if classification.is_high_risk_system else control.other_duty
    return f"{control.title}: {duty}"


def classify_risk_tier(
    eval_results: List[EvalResult],
    severity_threshold: Severity = Severity.MEDIUM,
    system_class: Optional[SystemClassification] = None,
) -> List[ComplianceFinding]:
    """Map evaluation results to residual EU AI Act findings.

    Each qualifying eval becomes evidence against duties that already apply
    to ``system_class``. Finding ``risk_tier`` is the *system* class, never
    a tier invented from the dimension name.

    Args:
        eval_results: Evaluation outcomes to classify.
        severity_threshold: Minimum severity required to raise a finding.
        system_class: Declared use-case classification. Defaults to GPAI/chatbot.

    Returns:
        Residual EU AI Act findings, one per qualifying eval result.
    """
    classification = system_class or classify_system(DEFAULT_USE_CASE)
    findings: List[ComplianceFinding] = []
    for result in eval_results:
        if not severity_meets(result, severity_threshold):
            continue
        control = classify_dimension(result.dimension)
        findings.append(
            ComplianceFinding(
                framework=ComplianceFramework.EU_AI_ACT,
                control_id=control.article,
                risk_tier=classification.risk_tier,
                description=_duty_text(control, classification),
                evidence=evidence_for(result),
                severity=result.severity,
            )
        )
    return findings


def prohibited_use_finding(
    classification: SystemClassification,
) -> Optional[ComplianceFinding]:
    """Emit an Art. 5 finding when the declared use itself is prohibited.

    Args:
        classification: Declared use-case classification.

    Returns:
        A critical finding, or ``None`` when the use is not prohibited.
    """
    if not classification.is_prohibited:
        return None
    return ComplianceFinding(
        framework=ComplianceFramework.EU_AI_ACT,
        control_id=classification.articles[0] if classification.articles else "Art. 5",
        risk_tier=RiskTier.UNACCEPTABLE,
        description=(
            f"{classification.title}: {classification.rationale} "
            "This tool cannot 'certify' a prohibited practice."
        ),
        evidence=f"Declared use case: {classification.use_case.value}",
        severity=Severity.CRITICAL,
    )


__all__ = [
    "EUAIActControl",
    "DIMENSION_TO_EU_AI_ACT",
    "DEFAULT_EU_AI_ACT",
    "LEGAL_DISCLAIMER",
    "SystemUseCase",
    "SystemClassification",
    "classify_system",
    "classify_from_deployment_context",
    "parse_use_case",
    "classify_dimension",
    "classify_risk_tier",
    "prohibited_use_finding",
]
