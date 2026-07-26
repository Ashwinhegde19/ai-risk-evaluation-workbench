"""Pydantic v2 data models for the AI Risk Evaluation Workbench.

These models are the single source of truth for evaluation results,
red-team attack artifacts, compliance findings, and guardrail outcomes.
Every model uses ``strict=True`` validation so that, for example, an ``int``
is never silently coerced into a ``float`` score, and out-of-range or
unknown categorical values are rejected.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _extract_enum(annotation: object) -> Optional[Type[Enum]]:
    """Return the ``Enum`` class for an annotation, unwrapping ``Optional``.

    Args:
        annotation: A type annotation (e.g. ``RiskTier`` or
            ``Optional[RiskTier]``).

    Returns:
        The ``Enum`` subclass if the annotation is an enum or ``Optional[enum]``,
        otherwise ``None``.
    """
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    if get_origin(annotation) is Union:
        for arg in get_args(annotation):
            if isinstance(arg, type) and issubclass(arg, Enum):
                return arg
    return None


class Severity(str, Enum):
    """Severity levels used across findings, results, and guardrails."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskTier(str, Enum):
    """Regulatory risk tiers (aligned with EU AI Act classification)."""

    UNACCEPTABLE = "unacceptable"
    HIGH = "high"
    LIMITED = "limited"
    MINIMAL = "minimal"


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""

    EU_AI_ACT = "eu_ai_act"
    NIST_RMF = "nist_rmf"
    ISO_42001 = "iso_42001"


class BaseWorkbenchModel(BaseModel):
    """Base class providing strict validation and string->enum coercion.

    In strict mode Pydantic rejects plain strings for ``Enum`` fields, so a
    ``model_validator(mode="before")`` coerces any string value whose field is
    annotated as an ``Enum`` into the corresponding enum member before the
    strict field validators run.
    """

    model_config = ConfigDict(strict=True)

    @model_validator(mode="before")
    @classmethod
    def _coerce_enum_strings(cls, data: object) -> object:
        """Coerce string values into enum members for enum-annotated fields.

        Handles both direct ``Enum`` annotations and ``Optional[Enum]``
        (i.e. ``Union[Enum, None]``) so that JSON round-trips of optional enum
        fields (e.g. ``adversarial_risk_tier``) deserialize correctly.

        Args:
            data: The raw input, expected to be a mapping of field -> value.

        Returns:
            The (possibly mutated) input mapping.
        """
        if not isinstance(data, dict):
            return data
        for name, field in cls.model_fields.items():
            if name not in data:
                continue
            value = data[name]
            if not isinstance(value, str):
                continue
            enum_cls = _extract_enum(field.annotation)
            if enum_cls is not None:
                data[name] = enum_cls(value)
        return data


def _coerce_datetime(value: object) -> object:
    """Coerce an ISO-8601 string (or datetime) into a timezone-aware datetime.

    Args:
        value: Either a ``datetime`` instance or an ISO-8601 string.

    Returns:
        A ``datetime`` instance (UTC if the input was naive).
    """
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return value


class JudgeScore(BaseWorkbenchModel):
    """A single judge model's score for one risk dimension."""

    judge_model: str = Field(..., description="Identifier of the judging model.")
    dimension: str = Field(..., description="Risk dimension that was scored.")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated score in the range [0, 1]."
    )
    reasoning: str = Field(..., description="Free-text justification for the score.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Judge's self-reported confidence [0, 1]."
    )


class EvalResult(BaseWorkbenchModel):
    """The outcome of evaluating one model on one risk dimension."""

    model_name: str = Field(..., description="Model under evaluation.")
    dimension: str = Field(..., description="Risk dimension evaluated.")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Aggregate score in the range [0, 1]."
    )
    severity: Severity = Field(..., description="Severity of the observed behavior.")
    raw_response: str = Field(..., description="Raw response produced by the model.")
    judge_scores: List[JudgeScore] = Field(
        default_factory=list, description="Per-judge scores contributing to ``score``."
    )


class AttackTurn(BaseWorkbenchModel):
    """A single turn within a multi-turn red-team attack."""

    turn_number: int = Field(..., ge=1, description="1-based turn index.")
    attacker_prompt: str = Field(..., description="Prompt issued by the attacker.")
    model_response: str = Field(..., description="Target model's response.")
    strategy_used: str = Field(..., description="Attack strategy applied this turn.")
    escalation_level: int = Field(
        ..., ge=0, description="Escalation level reached at this turn."
    )


class AttackTree(BaseWorkbenchModel):
    """A full multi-turn attack captured as a tree of turns."""

    root_prompt: str = Field(..., description="Initial seed prompt for the attack.")
    turns: List[AttackTurn] = Field(
        default_factory=list, description="Ordered attack turns."
    )
    final_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Final success score in the range [0, 1].",
    )
    strategy_chain: List[str] = Field(
        default_factory=list, description="Ordered strategy names used across turns."
    )
    success: bool = Field(
        default=False, description="Whether the attack ultimately succeeded."
    )
    adjudication_needs_review: bool = Field(
        default=False,
        exclude=True,
        description=(
            "Whether an unparseable or failed break adjudication was treated as "
            "uncertain compliance and requires human review."
        ),
    )


class ComplianceFinding(BaseWorkbenchModel):
    """A single mapping of an evaluation outcome to a regulatory control."""

    framework: ComplianceFramework = Field(
        ..., description="Compliance framework this finding belongs to."
    )
    control_id: str = Field(..., description="Control identifier (e.g. 'MEASURE-2.6').")
    risk_tier: RiskTier = Field(..., description="Assigned regulatory risk tier.")
    description: str = Field(..., description="Human-readable description of the gap.")
    evidence: str = Field(..., description="Supporting evidence from eval results.")
    severity: Severity = Field(..., description="Severity of the finding.")


class ComplianceReport(BaseWorkbenchModel):
    """An audit-ready compliance report for one model."""

    model_name: str = Field(..., description="Model the report was generated for.")
    timestamp: datetime = Field(
        ..., description="Report generation time (ISO-8601, UTC)."
    )
    findings: List[ComplianceFinding] = Field(
        default_factory=list, description="All compliance findings."
    )
    redteam_findings: List[ComplianceFinding] = Field(
        default_factory=list,
        description=(
            "Compliance findings derived from red-team breaks, kept separate "
            "from passive-eval findings so the report distinguishes the two."
        ),
    )
    overall_risk_tier: RiskTier = Field(
        ..., description="Highest risk tier across all findings."
    )
    adversarial_risk_tier: Optional[RiskTier] = Field(
        default=None,
        description=(
            "Risk tier derived from red-team break rate and deployment context; "
            "None when no red-team assessment was run."
        ),
    )
    gaps: List[str] = Field(
        default_factory=list, description="Identified compliance gaps / recommendations."
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def _validate_timestamp(cls, value: object) -> object:
        """Coerce ISO-8601 strings into timezone-aware datetimes."""
        return _coerce_datetime(value)


class GuardrailResult(BaseWorkbenchModel):
    """The outcome of a single guardrail check on a prompt or response."""

    check_type: str = Field(..., description="Guardrail check type (pii/toxicity/...).")
    triggered: bool = Field(
        ..., description="Whether the guardrail condition was triggered."
    )
    details: str = Field(
        default="", description="Human-readable details about the trigger."
    )
    severity: Severity = Field(
        default=Severity.INFO, description="Severity if the check triggered."
    )


__all__ = [
    "Severity",
    "RiskTier",
    "ComplianceFramework",
    "BaseWorkbenchModel",
    "JudgeScore",
    "EvalResult",
    "AttackTurn",
    "AttackTree",
    "ComplianceFinding",
    "ComplianceReport",
    "GuardrailResult",
]
