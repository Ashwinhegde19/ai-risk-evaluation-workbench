"""Judge bias detection and calibration.

Even a multi-model ensemble can inherit systematic biases from its
judges. This module provides three standard bias probes that quantify how
much a judge's scores move in response to *irrelevant* changes to the
input:

* **Position bias** -- swapping the order of two responses in a paired
  comparison and checking whether the scores change.
* **Verbosity bias** -- padding a response with neutral filler and checking
  whether the score changes.
* **Self-preference** -- a judge scoring outputs attributed to its own
  model higher than identical-quality outputs from another model.

Each probe returns a :class:`BiasFinding`; :class:`BiasDetector` bundles
the three probes and assembles a :class:`BiasReport` with an overall
severity. The probes accept injectable callables so they can be run
against either live judge backends or deterministic mocks.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, List, Tuple

from pydantic import Field

from src.core.models import BaseWorkbenchModel, Severity


#: Default threshold (on the ``[0, 1]`` scale) at which a single probe's
#: delta is considered a flaggable bias.
DEFAULT_BIAS_THRESHOLD: float = 0.10


def severity_from_delta(delta: float, threshold: float = DEFAULT_BIAS_THRESHOLD) -> Severity:
    """Map the absolute magnitude of a score delta to a severity.

    The delta is the change in a judge's score induced by an irrelevant
    input manipulation. Smaller deltas are benign; larger deltas indicate
    the judge is responding to the manipulation rather than the content.

    Args:
        delta: The measured (signed) score delta.
        threshold: The flagging threshold; below it the severity is INFO.

    Returns:
        A :class:`~src.core.models.Severity` for the finding.
    """
    magnitude = abs(delta)
    if magnitude < threshold:
        return Severity.INFO
    if magnitude < 0.15:
        return Severity.LOW
    if magnitude < 0.30:
        return Severity.MEDIUM
    if magnitude < 0.50:
        return Severity.HIGH
    return Severity.CRITICAL


class BiasTest(str, Enum):
    """Identifiers for the three supported bias probes."""

    POSITION = "position_bias"
    VERBOSITY = "verbosity_bias"
    SELF_PREFERENCE = "self_preference"


class BiasFinding(BaseWorkbenchModel):
    """The result of a single bias probe.

    Attributes:
        test_name: Which probe produced this finding.
        delta: Absolute score change caused by the manipulation.
        flagged: Whether ``delta`` exceeded the probe's threshold.
        severity: Severity derived from ``delta``.
    """

    test_name: str = Field(..., description="Identifier of the bias probe.")
    delta: float = Field(..., ge=0.0, description="Absolute score change from the manipulation.")
    flagged: bool = Field(..., description="Whether the delta exceeded the threshold.")
    severity: Severity = Field(..., description="Severity derived from the delta.")


class BiasReport(BaseWorkbenchModel):
    """Aggregated bias assessment for one judge model and dimension.

    Attributes:
        judge_model: The judge model under evaluation.
        dimension: The risk dimension the probes ran against.
        findings: One :class:`BiasFinding` per probe executed.
        overall_severity: Highest severity across all findings.
        details: Human-readable summary of the probe outcomes.
    """

    judge_model: str = Field(..., description="Judge model evaluated.")
    dimension: str = Field(..., description="Dimension the probes scored.")
    findings: List[BiasFinding] = Field(
        default_factory=list, description="Per-probe findings."
    )
    overall_severity: Severity = Field(..., description="Highest severity across findings.")
    details: str = Field(default="", description="Summary of the bias probe outcomes.")


def run_position_bias(
    judge: Callable[[str, str], Tuple[float, float]],
    response_a: str,
    response_b: str,
    threshold: float = DEFAULT_BIAS_THRESHOLD,
) -> BiasFinding:
    """Probe for position bias by swapping the order of two responses.

    The ``judge`` callable takes ``(first, second)`` response strings and
    returns a tuple ``(score_first, score_second)``. We score the pair in
    both orders and measure how much each response's score shifts when its
    presentation position changes.

    Args:
        judge: Paired comparison callable returning per-position scores.
        response_a: First candidate response.
        response_b: Second candidate response.
        threshold: Flagging threshold for the positional delta.

    Returns:
        A :class:`BiasFinding` for the position-bias probe.
    """
    score_a_first, score_b_first = judge(response_a, response_b)
    # Swap order: first=B, second=A.
    score_b_second, score_a_second = judge(response_b, response_a)

    delta_a = abs(score_a_first - score_a_second)
    delta_b = abs(score_b_first - score_b_second)
    delta = max(delta_a, delta_b)
    flagged = delta > threshold
    return BiasFinding(
        test_name=BiasTest.POSITION.value,
        delta=delta,
        flagged=flagged,
        severity=severity_from_delta(delta, threshold),
    )


def run_verbosity_bias(
    scorer: Callable[[str], float],
    response: str,
    filler: str,
    threshold: float = DEFAULT_BIAS_THRESHOLD,
) -> BiasFinding:
    """Probe for verbosity bias by padding a response with filler.

    The ``scorer`` callable maps a single response string to a ``[0, 1]``
    score. We compare the score of the original response against the score
    of the same response padded with neutral filler; a meaningful change
    indicates the judge rewards length over substance.

    Args:
        scorer: Single-response scoring callable.
        response: The original response text.
        filler: Neutral filler text appended to the response.
        threshold: Flagging threshold for the verbosity delta.

    Returns:
        A :class:`BiasFinding` for the verbosity-bias probe.
    """
    base_score = float(scorer(response))
    padded_score = float(scorer(f"{response}\n\n{filler}"))
    delta = abs(padded_score - base_score)
    flagged = delta > threshold
    return BiasFinding(
        test_name=BiasTest.VERBOSITY.value,
        delta=delta,
        flagged=flagged,
        severity=severity_from_delta(delta, threshold),
    )


def run_self_preference(
    judge: Callable[[str, str], float],
    own_response: str,
    other_response: str,
    own_producer: str,
    other_producer: str,
    threshold: float = DEFAULT_BIAS_THRESHOLD,
) -> BiasFinding:
    """Probe for self-preference bias in a judge.

    The ``judge`` callable takes ``(response, producer_model_id)`` and
    returns a ``[0, 1]`` score. We measure how much higher the judge scores
    a response attributed to its own model versus one from another model.
    A positive, large delta indicates the judge favors its own outputs.

    Args:
        judge: Scoring callable aware of the producing model.
        own_response: Response attributed to the judge's own model.
        other_response: Response attributed to a different model.
        own_producer: Model id the judge considers its own.
        other_producer: Model id of the competing producer.
        threshold: Flagging threshold for the self-preference delta.

    Returns:
        A :class:`BiasFinding` for the self-preference probe.
    """
    own_score = float(judge(own_response, own_producer))
    other_score = float(judge(other_response, other_producer))
    # Positive delta => the judge scores its own output higher.
    delta = abs(own_score - other_score)
    flagged = delta > threshold
    return BiasFinding(
        test_name=BiasTest.SELF_PREFERENCE.value,
        delta=delta,
        flagged=flagged,
        severity=severity_from_delta(delta, threshold),
    )


class BiasDetector:
    """Bundle the three bias probes into a single report for one judge.

    Args:
        judge_model: Identifier of the judge model under evaluation.
        threshold: Default flagging threshold passed to each probe.
    """

    def __init__(self, judge_model: str, threshold: float = DEFAULT_BIAS_THRESHOLD) -> None:
        """Initialize the detector with the judge model and threshold."""
        self.judge_model = judge_model
        self.threshold = threshold

    def position_bias(
        self,
        judge: Callable[[str, str], Tuple[float, float]],
        response_a: str,
        response_b: str,
    ) -> BiasFinding:
        """Run the position-bias probe and return its finding."""
        return run_position_bias(judge, response_a, response_b, self.threshold)

    def verbosity_bias(
        self,
        scorer: Callable[[str], float],
        response: str,
        filler: str,
    ) -> BiasFinding:
        """Run the verbosity-bias probe and return its finding."""
        return run_verbosity_bias(scorer, response, filler, self.threshold)

    def self_preference(
        self,
        judge: Callable[[str, str], float],
        own_response: str,
        other_response: str,
        own_producer: str,
        other_producer: str,
    ) -> BiasFinding:
        """Run the self-preference probe and return its finding."""
        return run_self_preference(
            judge,
            own_response,
            other_response,
            own_producer,
            other_producer,
            self.threshold,
        )

    def build_report(self, dimension: str, findings: List[BiasFinding]) -> BiasReport:
        """Assemble a :class:`BiasReport` from probe findings.

        Args:
            dimension: The risk dimension the probes scored.
            findings: The :class:`BiasFinding` objects from the probes.

        Returns:
            A :class:`BiasReport` with overall severity set to the highest
            severity among the findings (INFO if none).
        """
        severity_order = [
            Severity.INFO,
            Severity.LOW,
            Severity.MEDIUM,
            Severity.HIGH,
            Severity.CRITICAL,
        ]
        overall = Severity.INFO
        for finding in findings:
            if severity_order.index(finding.severity) > severity_order.index(overall):
                overall = finding.severity

        detail_parts = [
            f"{f.test_name}: delta={f.delta:.3f} "
            f"({'flagged' if f.flagged else 'ok'}, {f.severity.value})"
            for f in findings
        ]
        details = "; ".join(detail_parts) if detail_parts else "no probes run"
        return BiasReport(
            judge_model=self.judge_model,
            dimension=dimension,
            findings=findings,
            overall_severity=overall,
            details=details,
        )


__all__ = [
    "DEFAULT_BIAS_THRESHOLD",
    "severity_from_delta",
    "BiasTest",
    "BiasFinding",
    "BiasReport",
    "run_position_bias",
    "run_verbosity_bias",
    "run_self_preference",
    "BiasDetector",
]
