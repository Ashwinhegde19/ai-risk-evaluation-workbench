"""EU AI Act system classification from declared use case.

The Act classifies *AI systems and practices*, not model eval scores.

    * Art. 5   -- prohibited practices (social scoring, manipulative techniques,
                  untargeted facial scraping, real-time remote biometric ID in
                  publicly accessible spaces with listed exceptions, emotion
                  recognition in workplace/education, biometric categorisation
                  of sensitive attributes).
    * Art. 6 + Annex III -- high-risk *systems* by intended purpose
                  (employment, credit/essential services, education, law
                  enforcement, ...). A general chatbot or a foundation model
                  is not high-risk because a bias rubric scored poorly.
    * Art. 50  -- transparency duties for chatbots, synthetic content, emotion
                  recognition (where not already prohibited).
    * Arts. 51-56 -- GPAI / foundation-model provider duties, independent of
                  whether a downstream deployer later puts the model in an
                  Annex III system.
    * Residual -- eval scores (bias, jailbreak, hallucination) are *evidence*
                  for duties that already apply (Art. 9/10/14/15 if high-risk;
                  GPAI/systemic-risk and product safety if not). They do not
                  reclassify the system.

This module is the only place that assigns :class:`~src.core.models.RiskTier`
for the EU AI Act. Eval mappers must take a :class:`SystemClassification` and
must not invent a tier from a dimension name.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import Field

from src.core.models import BaseWorkbenchModel, RiskTier


LEGAL_DISCLAIMER = (
    "This is a research evaluation record, not an EU AI Act conformity "
    "assessment, CE marking, notified-body certificate, or legal opinion. "
    "Risk class comes from the declared use case (Art. 5 / Art. 6 + Annex III "
    "/ Art. 50 / Chapter V), not from eval scores. Residual findings are "
    "technical evidence only."
)


class SystemUseCase(str, Enum):
    """Declared intended purpose of the system under assessment.

    Values are use cases, not eval dimensions. Evaluating a foundation model
    in isolation is :attr:`GPAI_OR_CHATBOT`, even if the prompt set includes
    bias or jailbreak items.
    """

    GPAI_OR_CHATBOT = "gpai_or_chatbot"
    CUSTOMER_SUPPORT = "customer_support"
    SYNTHETIC_MEDIA = "synthetic_media"
    EDUCATION = "education"
    EMPLOYMENT = "employment"
    CREDIT = "credit"
    LAW_ENFORCEMENT = "law_enforcement"
    ANNEX_III_OTHER = "annex_iii_other"
    SOCIAL_SCORING = "social_scoring"
    MANIPULATION = "manipulation"
    REALTIME_REMOTE_BIOMETRIC = "realtime_remote_biometric"
    EMOTION_WORKPLACE_OR_EDUCATION = "emotion_workplace_or_education"
    MINIMAL = "minimal"


class SystemClassification(BaseWorkbenchModel):
    """Resolved EU AI Act class for one declared use case."""

    use_case: SystemUseCase = Field(..., description="Declared intended purpose.")
    risk_tier: RiskTier = Field(
        ..., description="Legal class from the use case, never from eval scores."
    )
    articles: List[str] = Field(
        default_factory=list, description="Primary articles that apply."
    )
    title: str = Field(..., description="Short label for reports.")
    rationale: str = Field(..., description="Why this class was assigned.")
    applicable_duties: List[str] = Field(
        default_factory=list,
        description="Duties that residual eval evidence can speak to.",
    )
    disclaimer: str = Field(
        default=LEGAL_DISCLAIMER,
        description="Required non-certification notice.",
    )

    @property
    def is_prohibited(self) -> bool:
        """Return True when the declared use is an Art. 5 practice."""
        return self.risk_tier == RiskTier.UNACCEPTABLE

    @property
    def is_high_risk_system(self) -> bool:
        """Return True when Annex III / Art. 6 duties apply."""
        return self.risk_tier == RiskTier.HIGH


_CLASSIFICATIONS: dict[SystemUseCase, SystemClassification] = {
    SystemUseCase.GPAI_OR_CHATBOT: SystemClassification(
        use_case=SystemUseCase.GPAI_OR_CHATBOT,
        risk_tier=RiskTier.LIMITED,
        articles=["Art. 50", "Arts. 51-53"],
        title="GPAI model or general-purpose chatbot",
        rationale=(
            "A general-purpose model or chatbot is not an Annex III high-risk "
            "system. Art. 50 transparency (disclose that the user is interacting "
            "with AI) and Chapter V GPAI provider duties apply. Putting the same "
            "model later into employment, credit, or law-enforcement software is "
            "a different system that must be classified separately."
        ),
        applicable_duties=[
            "Art. 50 transparency / chatbot disclosure",
            "Art. 53 GPAI documentation and copyright policy",
            "Residual product-safety and acceptable-use controls",
        ],
    ),
    SystemUseCase.CUSTOMER_SUPPORT: SystemClassification(
        use_case=SystemUseCase.CUSTOMER_SUPPORT,
        risk_tier=RiskTier.LIMITED,
        articles=["Art. 50"],
        title="Customer-support assistant",
        rationale=(
            "A support chatbot is typically limited-risk under Art. 50 unless "
            "it actually decides access to essential private or public services "
            "(Annex III point 5), in which case it must be declared as credit "
            "or essential-services instead."
        ),
        applicable_duties=[
            "Art. 50 transparency",
            "Residual quality, privacy, and robustness controls",
        ],
    ),
    SystemUseCase.SYNTHETIC_MEDIA: SystemClassification(
        use_case=SystemUseCase.SYNTHETIC_MEDIA,
        risk_tier=RiskTier.LIMITED,
        articles=["Art. 50(2)"],
        title="Synthetic / deepfake media",
        rationale=(
            "Art. 50 requires disclosure of AI-generated or manipulated content. "
            "That is a transparency duty, not an Annex III high-risk class."
        ),
        applicable_duties=["Art. 50(2) synthetic-content disclosure"],
    ),
    SystemUseCase.EDUCATION: SystemClassification(
        use_case=SystemUseCase.EDUCATION,
        risk_tier=RiskTier.HIGH,
        articles=["Art. 6", "Annex III §3"],
        title="Education / vocational assessment (Annex III)",
        rationale=(
            "AI used to determine access to education or to evaluate students "
            "is high-risk under Annex III. Eval scores do not create this class; "
            "the intended purpose does. Arts. 9-15 then apply."
        ),
        applicable_duties=[
            "Art. 9 risk management",
            "Art. 10 data governance",
            "Art. 13 transparency to deployers",
            "Art. 14 human oversight",
            "Art. 15 accuracy, robustness, cybersecurity",
        ],
    ),
    SystemUseCase.EMPLOYMENT: SystemClassification(
        use_case=SystemUseCase.EMPLOYMENT,
        risk_tier=RiskTier.HIGH,
        articles=["Art. 6", "Annex III §4"],
        title="Employment / worker management (Annex III)",
        rationale=(
            "Recruitment, promotion, or worker-management AI is high-risk "
            "because of its purpose, even if every eval score is clean."
        ),
        applicable_duties=[
            "Art. 9 risk management",
            "Art. 10 data governance",
            "Art. 14 human oversight",
            "Art. 15 accuracy, robustness, cybersecurity",
        ],
    ),
    SystemUseCase.CREDIT: SystemClassification(
        use_case=SystemUseCase.CREDIT,
        risk_tier=RiskTier.HIGH,
        articles=["Art. 6", "Annex III §5"],
        title="Credit / essential services (Annex III)",
        rationale=(
            "Creditworthiness and access to essential public or private "
            "services are Annex III high-risk uses."
        ),
        applicable_duties=[
            "Art. 9 risk management",
            "Art. 10 data governance",
            "Art. 14 human oversight",
            "Art. 15 accuracy, robustness, cybersecurity",
        ],
    ),
    SystemUseCase.LAW_ENFORCEMENT: SystemClassification(
        use_case=SystemUseCase.LAW_ENFORCEMENT,
        risk_tier=RiskTier.HIGH,
        articles=["Art. 6", "Annex III §6"],
        title="Law enforcement (Annex III)",
        rationale=(
            "Law-enforcement AI listed in Annex III is high-risk by purpose. "
            "Some biometric uses are instead prohibited under Art. 5."
        ),
        applicable_duties=[
            "Art. 9 risk management",
            "Art. 10 data governance",
            "Art. 14 human oversight",
            "Art. 15 accuracy, robustness, cybersecurity",
        ],
    ),
    SystemUseCase.ANNEX_III_OTHER: SystemClassification(
        use_case=SystemUseCase.ANNEX_III_OTHER,
        risk_tier=RiskTier.HIGH,
        articles=["Art. 6", "Annex III"],
        title="Other Annex III high-risk system",
        rationale=(
            "The deployer declared an Annex III purpose that is not one of "
            "the named shortcuts in this tool. Arts. 9-15 apply. This is not "
            "a legal determination that Annex III actually applies."
        ),
        applicable_duties=[
            "Art. 9 risk management",
            "Art. 10 data governance",
            "Art. 14 human oversight",
            "Art. 15 accuracy, robustness, cybersecurity",
        ],
    ),
    SystemUseCase.SOCIAL_SCORING: SystemClassification(
        use_case=SystemUseCase.SOCIAL_SCORING,
        risk_tier=RiskTier.UNACCEPTABLE,
        articles=["Art. 5(1)(c)"],
        title="Prohibited: social scoring",
        rationale=(
            "General-purpose social scoring of natural persons is a prohibited "
            "practice. The system should not be placed on the market. Eval "
            "scores cannot 'pass' this use case."
        ),
        applicable_duties=["Art. 5 prohibition -- do not deploy"],
    ),
    SystemUseCase.MANIPULATION: SystemClassification(
        use_case=SystemUseCase.MANIPULATION,
        risk_tier=RiskTier.UNACCEPTABLE,
        articles=["Art. 5(1)(a)"],
        title="Prohibited: manipulative techniques",
        rationale=(
            "AI that deploys subliminal or purposefully manipulative techniques "
            "to distort behaviour and cause significant harm is prohibited."
        ),
        applicable_duties=["Art. 5 prohibition -- do not deploy"],
    ),
    SystemUseCase.REALTIME_REMOTE_BIOMETRIC: SystemClassification(
        use_case=SystemUseCase.REALTIME_REMOTE_BIOMETRIC,
        risk_tier=RiskTier.UNACCEPTABLE,
        articles=["Art. 5(1)(h)"],
        title="Prohibited: real-time remote biometric identification",
        rationale=(
            "Real-time remote biometric identification in publicly accessible "
            "spaces is prohibited, subject to narrow law-enforcement exceptions "
            "that this tool does not evaluate."
        ),
        applicable_duties=["Art. 5 prohibition -- do not deploy without a listed exception"],
    ),
    SystemUseCase.EMOTION_WORKPLACE_OR_EDUCATION: SystemClassification(
        use_case=SystemUseCase.EMOTION_WORKPLACE_OR_EDUCATION,
        risk_tier=RiskTier.UNACCEPTABLE,
        articles=["Art. 5(1)(f)"],
        title="Prohibited: emotion recognition at work or school",
        rationale=(
            "Emotion recognition in the workplace or education institutions is "
            "prohibited, with limited medical/safety exceptions."
        ),
        applicable_duties=["Art. 5 prohibition -- do not deploy"],
    ),
    SystemUseCase.MINIMAL: SystemClassification(
        use_case=SystemUseCase.MINIMAL,
        risk_tier=RiskTier.MINIMAL,
        articles=["Art. 13 (general)"],
        title="Minimal-risk system",
        rationale=(
            "No Annex III purpose and no Art. 50 interaction/synthetic-content "
            "duty was declared. Residual eval findings remain technical quality "
            "evidence, not a legal upgrade to high-risk."
        ),
        applicable_duties=["Voluntary codes of practice; residual product safety"],
    ),
}


# Default when the workbench is evaluating a model, not a deployed Annex III app.
DEFAULT_USE_CASE = SystemUseCase.GPAI_OR_CHATBOT


def classify_system(use_case: SystemUseCase | str) -> SystemClassification:
    """Classify a declared use case under the EU AI Act.

    Args:
        use_case: A :class:`SystemUseCase` or its string value.

    Returns:
        The matching :class:`SystemClassification`.

    Raises:
        ValueError: If ``use_case`` is not a known value.
    """
    if isinstance(use_case, str):
        try:
            use_case = SystemUseCase(use_case.strip().lower())
        except ValueError as exc:
            raise ValueError(
                f"Unknown system use case '{use_case}'. "
                f"Expected one of: {[item.value for item in SystemUseCase]}"
            ) from exc
    return _CLASSIFICATIONS[use_case]


def use_case_from_deployment_context(context: object) -> SystemUseCase:
    """Map the legacy deployment-context flag onto a use case.

    ``high_risk`` means 'treat this as an unspecified Annex III system'.
    ``limited`` is the default GPAI/chatbot class. ``minimal`` is residual-only.

    Args:
        context: A :class:`~src.compliance.redteam_mapping.DeploymentContext`
            or a string with the same value.

    Returns:
        The corresponding :class:`SystemUseCase`.
    """
    value = getattr(context, "value", context)
    if value in {"high_risk", "high"}:
        return SystemUseCase.ANNEX_III_OTHER
    if value in {"minimal", "low"}:
        return SystemUseCase.MINIMAL
    return SystemUseCase.GPAI_OR_CHATBOT


def classify_from_deployment_context(context: object) -> SystemClassification:
    """Classify using the legacy high/limited/minimal deployment flag.

    Args:
        context: Deployment context enum or string.

    Returns:
        The resolved :class:`SystemClassification`.
    """
    return classify_system(use_case_from_deployment_context(context))


def parse_use_case(
    use_case: Optional[str],
    deployment_context: object = None,
) -> SystemClassification:
    """Resolve a classification from an optional explicit use case.

    Args:
        use_case: Explicit :class:`SystemUseCase` value, or ``None``.
        deployment_context: Fallback legacy context when ``use_case`` is omitted.

    Returns:
        The resolved :class:`SystemClassification`.
    """
    if use_case:
        return classify_system(use_case)
    if deployment_context is not None:
        return classify_from_deployment_context(deployment_context)
    return classify_system(DEFAULT_USE_CASE)


__all__ = [
    "LEGAL_DISCLAIMER",
    "SystemUseCase",
    "SystemClassification",
    "DEFAULT_USE_CASE",
    "classify_system",
    "use_case_from_deployment_context",
    "classify_from_deployment_context",
    "parse_use_case",
]
