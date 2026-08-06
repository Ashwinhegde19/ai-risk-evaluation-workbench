"""Shieldstral policy-adaptive safety classifier guardrail.

This module wraps the self-deployed Mistral Shieldstral model as a guardrail
backend. Shieldstral frames content moderation as a binary question-answering
task: you supply a plain-language policy question at inference time and the
model returns a calibrated safety score.

This is a drop-in replacement for the regex/lexicon-based guardrails in
:mod:`src.guardrails` — it provides policy-adaptive classification without
retraining, and returns calibrated yes/no probabilities.

Usage::

    from src.guardrails.shieldstral import ShieldstralGuardrail

    guardrail = ShieldstralGuardrail(
        policy="Does this content promote physical violence?",
        strictness="high",
    )
    result = guardrail.scan("Some content to check.")
    print(result.triggered, result.details)
"""

from __future__ import annotations

from typing import Optional

from src.backends.base import MistralShieldstralBackend, get_backend
from src.core.models import GuardrailResult, Severity


class ShieldstralGuardrail:
    """Policy-adaptive safety classifier backed by Shieldstral.

    Shieldstral accepts plain-language policies at inference time, so the
    same checkpoint can enforce different safety rules per deployment tier
    without retraining.

    Args:
        policy: The safety question to evaluate (e.g. "Does this content
            promote physical violence?").
        strictness: Evaluation strictness — "low", "medium", or "high".
        model_slug: The backend slug (default: ``mistral-shieldstral``).
        backend: Optional pre-built backend instance. When ``None``, the
            backend is resolved from the slug via :func:`get_backend`.
    """

    def __init__(
        self,
        policy: str,
        strictness: str = "medium",
        model_slug: str = "mistral-shieldstral",
        backend: Optional[MistralShieldstralBackend] = None,
    ) -> None:
        """Initialize the guardrail with a policy question."""
        self.policy = policy
        self.strictness = strictness
        self.model_slug = model_slug
        self.backend: MistralShieldstralBackend = (
            backend or get_backend(model_slug)  # type: ignore[assignment]
        )

    def _build_instruct(self) -> str:
        """Build the evaluation context instruction.

        Uses the Shieldstral-specific <Instruct> tag format with
        strictness level and policy context.
        """
        return (
            f"Evaluate for safety. "
            f"Strictness: {self.strictness}."
        )

    def scan(self, text: str) -> GuardrailResult:
        """Classify ``text`` against the configured policy.

        Args:
            text: The content to evaluate.

        Returns:
            A :class:`GuardrailResult` with the classification outcome.
        """
        try:
            result = self.backend.classify_safety(
                query=self.policy,
                document=text,
                instruct=self._build_instruct(),
            )
        except Exception as exc:
            return GuardrailResult(
                check_type="shieldstral",
                triggered=False,
                details=f"Shieldrail unavailable: {exc}",
                severity=Severity.INFO,
            )

        unsafe_score = result.get("unsafe_score", 0.5)
        verdict = result.get("verdict", "uncertain")

        # Map to severity based on the policy strictness and unsafe score
        if verdict == "unsafe":
            if self.strictness == "high":
                severity = Severity.CRITICAL
            elif self.strictness == "medium":
                severity = Severity.HIGH
            else:
                severity = Severity.MEDIUM
        elif verdict == "uncertain":
            severity = Severity.LOW
        else:
            severity = Severity.INFO

        return GuardrailResult(
            check_type="shieldstral",
            triggered=verdict == "unsafe",
            details=(
                f"Policy: '{self.policy}' | "
                f"unsafe_score={unsafe_score:.2f} | "
                f"verdict={verdict} | "
                f"strictness={self.strictness}"
            ),
            severity=severity,
        )

    def scan_with_threshold(
        self,
        text: str,
        threshold: float = 0.5,
    ) -> GuardrailResult:
        """Classify ``text`` with a custom unsafe-score threshold.

        Args:
            text: The content to evaluate.
            threshold: The unsafe score above which content is flagged.

        Returns:
            A :class:`GuardrailResult` with the classification outcome.
        """
        try:
            result = self.backend.classify_safety(
                query=self.policy,
                document=text,
                instruct=self._build_instruct(),
            )
        except Exception as exc:
            return GuardrailResult(
                check_type="shieldstral",
                triggered=False,
                details=f"Shieldrail unavailable: {exc}",
                severity=Severity.INFO,
            )

        unsafe_score = result.get("unsafe_score", 0.5)
        triggered = unsafe_score >= threshold

        if triggered:
            severity = (
                Severity.CRITICAL if unsafe_score > 0.8
                else Severity.HIGH if unsafe_score > 0.6
                else Severity.MEDIUM
            )
        else:
            severity = Severity.INFO

        return GuardrailResult(
            check_type="shieldstral",
            triggered=triggered,
            details=(
                f"Policy: '{self.policy}' | "
                f"unsafe_score={unsafe_score:.2f} | "
                f"threshold={threshold:.2f} | "
                f"triggered={triggered}"
            ),
            severity=severity,
        )


__all__ = ["ShieldstralGuardrail"]
