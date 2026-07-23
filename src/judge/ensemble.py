"""Multi-model LLM-as-Judge ensemble.

This module implements :class:`JudgeEnsemble`, which coordinates three (or
more, configurable) judge models that each score a target response
independently using the rubrics in :mod:`src.judge.rubrics`. Their scores
are aggregated with the **median** (robust to a single outlier judge) and
any large inter-judge spread is flagged as a disagreement that warrants
human review.

The ensemble is provider-agnostic: it resolves each judge model to a
:class:`~src.backends.base.ModelBackend` through a pluggable factory (by
default :func:`src.backends.base.get_backend`, which reads API keys from
the environment and never hardcodes them). For deterministic unit testing,
a ``judge_function`` callable can be injected instead of live backends.

Both synchronous (:meth:`JudgeEnsemble.score`, :meth:`JudgeEnsemble.score_all`)
and asynchronous (:meth:`JudgeEnsemble.score_async`,
:meth:`JudgeEnsemble.score_all_async`) entry points are provided. The async
variants dispatch all judge calls concurrently via :func:`asyncio.gather`;
sync-only backends are wrapped with :func:`asyncio.to_thread` so they do
not block the event loop. Both paths share the same aggregation logic and
produce identical results.
"""

from __future__ import annotations

import asyncio
import statistics
from typing import Callable, List, Optional, Union

from pydantic import Field

from src.core.models import BaseWorkbenchModel, JudgeScore
from src.judge.rubrics import (
    RiskDimension,
    build_judge_prompt,
    parse_judge_response,
    rubric_to_judge_score,
)


#: Default judge models used when none are supplied. These are free-tier models
#: on the Kilo gateway (provider/model format). API keys are resolved from the
#: environment by the backend factory -- never hardcoded.
DEFAULT_JUDGE_MODELS: List[str] = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]

#: A disagreement is flagged when the spread (max - min) of judge scores
#: exceeds this threshold on the normalized ``[0, 1]`` scale.
DEFAULT_DISAGREEMENT_THRESHOLD: float = 0.20

#: Signature of an injectable judge: given a judge model id, a dimension, the
#: response (and optional prompt), return a validated ``JudgeScore``.
JudgeFunction = Callable[[str, str, Optional[str]], JudgeScore]


class EnsembleResult(BaseWorkbenchModel):
    """Aggregated outcome of scoring one dimension across all judges.

    Attributes:
        dimension: The risk dimension that was scored.
        judge_scores: The per-judge :class:`JudgeScore` results.
        aggregate_score: Median of the per-judge normalized scores.
        score_spread: ``max - min`` of the per-judge scores.
        disagreement_flag: ``True`` when ``score_spread`` exceeds the
            ensemble's disagreement threshold.
        confidence: Median of the per-judge confidence values.
    """

    dimension: str = Field(..., description="Risk dimension that was scored.")
    judge_scores: List[JudgeScore] = Field(
        default_factory=list, description="Per-judge scores for this dimension."
    )
    aggregate_score: float = Field(
        ..., ge=0.0, le=1.0, description="Median of the per-judge scores."
    )
    score_spread: float = Field(
        ..., ge=0.0, le=1.0, description="max(judge scores) - min(judge scores)."
    )
    disagreement_flag: bool = Field(
        ..., description="Whether the inter-judge spread exceeded the threshold."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Median judge confidence for this dimension."
    )


class JudgeEnsemble:
    """Coordinate multiple independent judge models into one score.

    Each judge scores a response on a given dimension using the matching
    rubric; the ensemble then takes the median of those scores as the
    aggregate and flags disagreements for review.

    Args:
        judge_models: Identifiers of the judge models. Defaults to
            :data:`DEFAULT_JUDGE_MODELS`.
        backend_factory: Callable mapping a model id to a
            :class:`~src.backends.base.ModelBackend`. Defaults to
            ``get_backend``. Only used when no ``judge_function`` is given.
        disagreement_threshold: Spread above which ``disagreement_flag`` is
            set ``True``.
        judge_function: Optional callable replacing live backend calls
            entirely (used for deterministic tests). Must match
            :data:`JudgeFunction`.
    """

    def __init__(
        self,
        judge_models: Optional[List[str]] = None,
        backend_factory: Optional[Callable[[str], object]] = None,
        disagreement_threshold: float = DEFAULT_DISAGREEMENT_THRESHOLD,
        judge_function: Optional[JudgeFunction] = None,
    ) -> None:
        """Initialize the ensemble with its judge models and wiring."""
        self.judge_models: List[str] = (
            list(judge_models) if judge_models else list(DEFAULT_JUDGE_MODELS)
        )
        self.disagreement_threshold: float = disagreement_threshold
        self._backend_factory = backend_factory
        self._judge_function = judge_function
        self._backends: dict[str, object] = {}

    def _get_backend(self, model_name: str) -> object:
        """Return (and lazily cache) the backend for a judge model.

        Args:
            model_name: The judge model identifier.

        Returns:
            A :class:`~src.backends.base.ModelBackend` instance.
        """
        if model_name not in self._backends:
            if self._backend_factory is None:
                from src.backends.base import get_backend

                self._backend_factory = get_backend
            self._backends[model_name] = self._backend_factory(model_name)
        return self._backends[model_name]

    def _default_judge_function(
        self, judge_model: str, dimension: str, response: Optional[str], prompt: Optional[str]
    ) -> JudgeScore:
        """Score one response using a live judge backend.

        Builds the rubric prompt, calls the judge model at temperature 0 for
        determinism, and parses the structured response.

        Args:
            judge_model: The judging model identifier.
            dimension: The risk dimension to score.
            response: The target response text.
            prompt: Optional original prompt/context.

        Returns:
            A validated :class:`JudgeScore` from this judge.
        """
        backend = self._get_backend(judge_model)
        judge_prompt = build_judge_prompt(dimension, str(response or ""), prompt)
        raw = backend.generate(prompt=judge_prompt, temperature=0.0)
        parsed = parse_judge_response(raw)
        return rubric_to_judge_score(judge_model, dimension, parsed)

    def _invoke_judge(
        self, judge_model: str, dimension: str, response: Optional[str], prompt: Optional[str]
    ) -> JudgeScore:
        """Invoke a single judge (injected function or live backend).

        Args:
            judge_model: The judging model identifier.
            dimension: The risk dimension to score.
            response: The target response text.
            prompt: Optional original prompt/context.

        Returns:
            A validated :class:`JudgeScore` from this judge.
        """
        if self._judge_function is not None:
            return self._judge_function(judge_model, dimension, response, prompt)
        return self._default_judge_function(judge_model, dimension, response, prompt)

    def _aggregate(self, dim_value: str, judge_scores: List[JudgeScore]) -> EnsembleResult:
        """Aggregate per-judge scores into an :class:`EnsembleResult`.

        Shared by the sync and async scoring paths so both produce identical
        results: median aggregate, max-min spread, disagreement flag (with an
        epsilon guard against float error), and median confidence.

        Args:
            dim_value: The normalized risk-dimension identifier.
            judge_scores: Per-judge scores, in judge-model order.

        Returns:
            The aggregated :class:`EnsembleResult`.
        """
        scores = [js.score for js in judge_scores]
        confidences = [js.confidence for js in judge_scores]

        aggregate_score = float(statistics.median(scores))
        score_spread = float(max(scores) - min(scores))
        # A small epsilon keeps an exactly-equal spread (subject to float
        # error, e.g. 0.8 - 0.6 == 0.20000000000000007) from being flagged.
        disagreement_flag = score_spread > (self.disagreement_threshold + 1e-9)
        confidence = float(statistics.median(confidences))

        return EnsembleResult(
            dimension=dim_value,
            judge_scores=judge_scores,
            aggregate_score=aggregate_score,
            score_spread=score_spread,
            disagreement_flag=disagreement_flag,
            confidence=confidence,
        )

    def score(
        self,
        dimension: Union[RiskDimension, str],
        response: str,
        prompt: Optional[str] = None,
    ) -> EnsembleResult:
        """Score ``response`` on ``dimension`` across all judges.

        Judges are invoked sequentially. For concurrent execution use
        :meth:`score_async`, which produces identical results.

        Args:
            dimension: The risk dimension to evaluate.
            response: The target model's response text.
            prompt: Optional original prompt/context for context-dependent
                dimensions (e.g. jailbreak resistance).

        Returns:
            An :class:`EnsembleResult` with per-judge scores, the median
            aggregate, spread, and disagreement flag.

        Raises:
            ValueError: If no judge models are configured.
        """
        if not self.judge_models:
            raise ValueError("JudgeEnsemble requires at least one judge model.")
        dim_value = dimension.value if isinstance(dimension, RiskDimension) else str(dimension)

        judge_scores: List[JudgeScore] = [
            self._invoke_judge(model, dim_value, response, prompt)
            for model in self.judge_models
        ]
        return self._aggregate(dim_value, judge_scores)

    async def score_async(
        self,
        dimension: Union[RiskDimension, str],
        response: str,
        prompt: Optional[str] = None,
    ) -> EnsembleResult:
        """Score ``response`` on ``dimension`` with all judges running concurrently.

        Every judge call is dispatched via :func:`asyncio.gather`. Sync-only
        judges (the default backend path and plain-callable ``judge_function``
        injections) are wrapped with :func:`asyncio.to_thread` so they do not
        block the event loop; coroutine-function judges are awaited directly.
        Aggregation is identical to :meth:`score`.

        Args:
            dimension: The risk dimension to evaluate.
            response: The target model's response text.
            prompt: Optional original prompt/context.

        Returns:
            An :class:`EnsembleResult` identical to what :meth:`score` returns.

        Raises:
            ValueError: If no judge models are configured.
        """
        if not self.judge_models:
            raise ValueError("JudgeEnsemble requires at least one judge model.")
        dim_value = dimension.value if isinstance(dimension, RiskDimension) else str(dimension)

        judge_scores = await asyncio.gather(
            *(self._invoke_judge_async(model, dim_value, response, prompt)
              for model in self.judge_models)
        )
        return self._aggregate(dim_value, list(judge_scores))

    async def _invoke_judge_async(
        self, judge_model: str, dimension: str, response: Optional[str], prompt: Optional[str]
    ) -> JudgeScore:
        """Invoke a single judge without blocking the event loop.

        Coroutine-function judges are awaited directly; sync judges (injected
        callables or live backends) run in a worker thread via
        :func:`asyncio.to_thread`.

        Args:
            judge_model: The judging model identifier.
            dimension: The risk dimension to score.
            response: The target response text.
            prompt: Optional original prompt/context.

        Returns:
            A validated :class:`JudgeScore` from this judge.
        """
        if self._judge_function is not None and asyncio.iscoroutinefunction(self._judge_function):
            return await self._judge_function(judge_model, dimension, response, prompt)
        return await asyncio.to_thread(self._invoke_judge, judge_model, dimension, response, prompt)

    def score_all(
        self,
        dimensions: List[Union[RiskDimension, str]],
        response: str,
        prompt: Optional[str] = None,
    ) -> dict[str, EnsembleResult]:
        """Score ``response`` on every dimension in ``dimensions``.

        Dimensions are processed sequentially; within each dimension judges
        are also sequential. For full concurrency (all judges across all
        dimensions dispatched at once) use :meth:`score_all_async`.

        Args:
            dimensions: The risk dimensions to evaluate.
            response: The target model's response text.
            prompt: Optional original prompt/context.

        Returns:
            A mapping of dimension identifier to its :class:`EnsembleResult`.
        """
        return {
            (d.value if isinstance(d, RiskDimension) else str(d)): self.score(d, response, prompt)
            for d in dimensions
        }

    async def score_all_async(
        self,
        dimensions: List[Union[RiskDimension, str]],
        response: str,
        prompt: Optional[str] = None,
    ) -> dict[str, EnsembleResult]:
        """Score ``response`` on every dimension with all judge calls concurrent.

        All ``len(dimensions) * len(judge_models)`` judge calls are dispatched
        together via :func:`asyncio.gather` -- e.g. 3 judges x 7 dimensions
        run as 21 concurrent calls rather than 21 serial ones. Sync-only
        judges run in worker threads via :func:`asyncio.to_thread`.
        Aggregation is identical to :meth:`score_all`.

        Args:
            dimensions: The risk dimensions to evaluate.
            response: The target model's response text.
            prompt: Optional original prompt/context.

        Returns:
            A mapping of dimension identifier to its :class:`EnsembleResult`,
            identical to what :meth:`score_all` returns.

        Raises:
            ValueError: If no judge models are configured.
        """
        if not self.judge_models:
            raise ValueError("JudgeEnsemble requires at least one judge model.")
        dim_values = [
            d.value if isinstance(d, RiskDimension) else str(d) for d in dimensions
        ]

        # Dispatch every (dimension, judge) pair concurrently, preserving
        # per-dimension judge order for deterministic aggregation.
        all_scores = await asyncio.gather(
            *(
                self._invoke_judge_async(model, dim_value, response, prompt)
                for dim_value in dim_values
                for model in self.judge_models
            )
        )
        return {
            dim_value: self._aggregate(
                dim_value,
                list(all_scores[i * len(self.judge_models):(i + 1) * len(self.judge_models)]),
            )
            for i, dim_value in enumerate(dim_values)
        }


__all__ = [
    "DEFAULT_JUDGE_MODELS",
    "DEFAULT_DISAGREEMENT_THRESHOLD",
    "JudgeFunction",
    "EnsembleResult",
    "JudgeEnsemble",
]
