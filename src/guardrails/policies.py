"""Guardrail policy enforcement and the chained guardrail pipeline.

A :class:`GuardrailPipeline` combines the PII, toxicity and injection
detectors and applies a deployment-tier :class:`~src.core.config.GuardrailPolicyConfig`
to decide whether the content should be blocked, allowed, or merely logged.

Example policies (defined in ``src.core.config``):

* ``production`` — block any PII, block toxicity ``>= 0.7``, block injections.
* ``testing`` — never block; log violations for later review.
"""

from __future__ import annotations

from typing import List, Optional

from src.core.config import AppConfig, GuardrailPolicyConfig, default_config, load_config
from src.core.models import BaseWorkbenchModel, GuardrailResult, Severity

from src.guardrails.injection import InjectionDetector
from src.guardrails.pii import PiiDetector
from src.guardrails.toxicity import ToxicityScorer


class GuardrailPipelineResult(BaseWorkbenchModel):
    """Aggregate outcome of running the guardrail pipeline on a text.

    Attributes:
        blocking: Whether the policy decided to block the content.
        action: One of ``"block"``, ``"allow"`` or ``"log"``.
        results: The individual :class:`GuardrailResult` from each detector.
        summary: A human-readable summary of the decision.
    """

    blocking: bool
    action: str
    results: List[GuardrailResult]
    summary: str


class GuardrailPipeline:
    """Chain multiple guardrail checks and enforce a deployment policy.

    Args:
        policy: The :class:`GuardrailPolicyConfig` describing the tier's rules.
        pii_detector: Optional detector; a default one is constructed if omitted.
        toxicity_scorer: Optional scorer; a default one is constructed if omitted.
        injection_detector: Optional detector; a default one is constructed.
    """

    def __init__(
        self,
        policy: GuardrailPolicyConfig,
        pii_detector: Optional[PiiDetector] = None,
        toxicity_scorer: Optional[ToxicityScorer] = None,
        injection_detector: Optional[InjectionDetector] = None,
    ) -> None:
        """Store the policy and ensure each detector is initialised."""
        self.policy: GuardrailPolicyConfig = policy
        self.pii: PiiDetector = pii_detector or PiiDetector()
        self.toxicity: ToxicityScorer = toxicity_scorer or ToxicityScorer()
        self.injection: InjectionDetector = injection_detector or InjectionDetector()

    def run(self, text: str) -> GuardrailPipelineResult:
        """Run all checks against ``text`` and apply the policy.

        Args:
            text: The text (prompt or model output) to evaluate.

        Returns:
            A :class:`GuardrailPipelineResult` describing the decision.
        """
        results: List[GuardrailResult] = []

        pii_result = self.pii.scan(text)
        results.append(pii_result)
        tox_result = self.toxicity.scan(text, threshold=self.policy.max_toxicity)
        results.append(tox_result)
        inj_result = self.injection.scan(text)
        results.append(inj_result)

        block_reasons: List[str] = []

        if self.policy.block_pii and pii_result.triggered:
            block_reasons.append("PII detected")
        if tox_result.triggered:
            block_reasons.append(
                f"toxicity >= {self.policy.max_toxicity}"
            )
        if self.policy.block_injection and inj_result.triggered:
            block_reasons.append("prompt injection detected")

        if self.policy.log_only:
            action = "log"
            blocking = False
            summary = (
                "Policy is log-only; "
                + (
                    "violations logged: " + "; ".join(block_reasons)
                    if block_reasons
                    else "no violations detected"
                )
            )
        elif block_reasons:
            action = "block"
            blocking = True
            summary = "Content blocked: " + "; ".join(block_reasons)
        else:
            action = "allow"
            blocking = False
            summary = "Content passed all guardrails."

        return GuardrailPipelineResult(
            blocking=blocking,
            action=action,
            results=results,
            summary=summary,
        )


def build_default_pipeline(
    policy_name: str = "production",
    config: Optional[AppConfig] = None,
) -> GuardrailPipeline:
    """Build a :class:`GuardrailPipeline` from a named policy in configuration.

    Args:
        policy_name: Name of a policy defined under ``guardrail_policies``.
        config: Optional pre-loaded :class:`AppConfig`; loaded from disk /
            defaults when omitted.

    Returns:
        A ready-to-use :class:`GuardrailPipeline`.

    Raises:
        ValueError: If the named policy does not exist in the configuration.
    """
    app_config = config or load_config()
    policy = app_config.get_policy(policy_name)
    if policy is None:
        raise ValueError(f"Unknown guardrail policy: {policy_name!r}")
    return GuardrailPipeline(policy=policy)


def build_production_pipeline() -> GuardrailPipeline:
    """Build a pipeline using the built-in ``production`` policy.

    Returns:
        A :class:`GuardrailPipeline` configured to block PII, high toxicity and
        injections.
    """
    return GuardrailPipeline(policy=default_config().get_policy("production"))


def build_testing_pipeline() -> GuardrailPipeline:
    """Build a pipeline using the built-in ``testing`` (log-only) policy.

    Returns:
        A :class:`GuardrailPipeline` that logs but never blocks.
    """
    return GuardrailPipeline(policy=default_config().get_policy("testing"))


__all__ = [
    "GuardrailPipelineResult",
    "GuardrailPipeline",
    "build_default_pipeline",
    "build_production_pipeline",
    "build_testing_pipeline",
]
