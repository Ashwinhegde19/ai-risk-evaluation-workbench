"""Tests for the judge ensemble (Phase 3, Task 3.2)."""

import asyncio
import time
import unittest

from src.judge.ensemble import (
    DEFAULT_DISAGREEMENT_THRESHOLD,
    DEFAULT_JUDGE_MODELS,
    EnsembleResult,
    JudgeEnsemble,
)
from src.core.models import JudgeScore


def _make_judge(mapping):
    """Build an injected judge function from a {model: score} mapping."""

    def judge(judge_model, dimension, response, prompt):
        return JudgeScore(
            judge_model=judge_model,
            dimension=dimension,
            score=mapping[judge_model],
            reasoning="deterministic mock",
            confidence=0.9,
        )

    return judge


class EnsembleDefaultsTests(unittest.TestCase):
    def test_default_judge_models(self):
        ensemble = JudgeEnsemble()
        self.assertEqual(ensemble.judge_models, DEFAULT_JUDGE_MODELS)
        self.assertEqual(len(ensemble.judge_models), 3)

    def test_default_disagreement_threshold(self):
        self.assertAlmostEqual(DEFAULT_DISAGREEMENT_THRESHOLD, 0.20)
        self.assertAlmostEqual(JudgeEnsemble().disagreement_threshold, 0.20)

    def test_custom_judge_models(self):
        ensemble = JudgeEnsemble(judge_models=["a", "b"])
        self.assertEqual(ensemble.judge_models, ["a", "b"])


class AggregationTests(unittest.TestCase):
    def test_median_aggregation_odd_count(self):
        judge = _make_judge({"gpt-4o": 0.8, "claude-sonnet": 0.6, "gemini-pro": 0.7})
        result = JudgeEnsemble(judge_function=judge).score("bias", "resp")
        self.assertIsInstance(result, EnsembleResult)
        self.assertAlmostEqual(result.aggregate_score, 0.7)

    def test_median_aggregation_even_count(self):
        judge = _make_judge({"a": 0.2, "b": 0.8})
        result = JudgeEnsemble(judge_models=["a", "b"], judge_function=judge).score(
            "toxicity", "resp"
        )
        # median of [0.2, 0.8] is their mean
        self.assertAlmostEqual(result.aggregate_score, 0.5)

    def test_score_spread_computed(self):
        judge = _make_judge({"gpt-4o": 0.9, "claude-sonnet": 0.1, "gemini-pro": 0.5})
        result = JudgeEnsemble(judge_function=judge).score("privacy", "resp")
        self.assertAlmostEqual(result.score_spread, 0.8)

    def test_no_disagreement_when_spread_at_threshold(self):
        # spread == threshold (0.2) must NOT flag (strictly greater).
        judge = _make_judge({"gpt-4o": 0.8, "claude-sonnet": 0.6, "gemini-pro": 0.7})
        result = JudgeEnsemble(judge_function=judge).score("bias", "resp")
        self.assertAlmostEqual(result.score_spread, 0.2)
        self.assertFalse(result.disagreement_flag)

    def test_disagreement_flagged_when_spread_exceeds_threshold(self):
        judge = _make_judge({"gpt-4o": 0.95, "claude-sonnet": 0.1, "gemini-pro": 0.5})
        result = JudgeEnsemble(judge_function=judge).score("bias", "resp")
        self.assertTrue(result.disagreement_flag)

    def test_custom_threshold(self):
        judge = _make_judge({"gpt-4o": 0.8, "claude-sonnet": 0.6, "gemini-pro": 0.7})
        ensemble = JudgeEnsemble(
            disagreement_threshold=0.1, judge_function=judge
        )
        result = ensemble.score("bias", "resp")
        # spread 0.2 > 0.1 -> flagged
        self.assertTrue(result.disagreement_flag)

    def test_per_judge_scores_returned(self):
        judge = _make_judge({"gpt-4o": 0.8, "claude-sonnet": 0.6, "gemini-pro": 0.7})
        result = JudgeEnsemble(judge_function=judge).score("bias", "resp")
        self.assertEqual(len(result.judge_scores), 3)
        models = {js.judge_model for js in result.judge_scores}
        self.assertEqual(models, {"gpt-4o", "claude-sonnet", "gemini-pro"})

    def test_confidence_is_median_of_judge_confidences(self):
        def judge(judge_model, dimension, response, prompt):
            conf = {"gpt-4o": 0.9, "claude-sonnet": 0.3, "gemini-pro": 0.5}[
                judge_model
            ]
            return JudgeScore(
                judge_model=judge_model,
                dimension=dimension,
                score=0.5,
                reasoning="r",
                confidence=conf,
            )

        result = JudgeEnsemble(judge_function=judge).score("bias", "resp")
        self.assertAlmostEqual(result.confidence, 0.5)

    def test_score_all_runs_every_dimension(self):
        judge = _make_judge({"gpt-4o": 0.5, "claude-sonnet": 0.5, "gemini-pro": 0.5})
        results = JudgeEnsemble(judge_function=judge).score_all(
            ["bias", "toxicity", "privacy"], "resp"
        )
        self.assertEqual(set(results.keys()), {"bias", "toxicity", "privacy"})
        for res in results.values():
            self.assertAlmostEqual(res.aggregate_score, 0.5)

    def test_empty_judge_models_falls_back_to_defaults(self):
        # An empty (falsy) list means "use defaults", so the ensemble always
        # has at least the three default judges configured.
        ensemble = JudgeEnsemble(judge_models=[], judge_function=_make_judge({}))
        self.assertEqual(ensemble.judge_models, DEFAULT_JUDGE_MODELS)


class DefaultJudgeIntegrationTests(unittest.TestCase):
    """Exercise the live-backend code path with a fake backend factory."""

    def test_default_judge_function_builds_and_parses(self):
        class FakeBackend:
            def generate(self, prompt, system_prompt=None, temperature=0.7):
                # Echo the dimension into the reasoning to prove the prompt
                # was built, and return a valid rubric JSON payload.
                return (
                    '{"rating": 4, "confidence": 0.7, '
                    '"reasoning": "evaluated via rubric"}'
                )

        def factory(model_name):
            return FakeBackend()

        ensemble = JudgeEnsemble(
            judge_models=["m1", "m2", "m3"], backend_factory=factory
        )
        result = ensemble.score("harmful_content", "a safe response")
        self.assertAlmostEqual(result.aggregate_score, 0.75)  # rating 4 -> 0.75
        self.assertFalse(result.disagreement_flag)
        for js in result.judge_scores:
            self.assertEqual(js.reasoning, "evaluated via rubric")
            self.assertEqual(js.dimension, "harmful_content")


class AsyncEnsembleTests(unittest.TestCase):
    """Verify the async scoring paths match the sync paths exactly."""

    def test_score_async_matches_sync(self):
        judge = _make_judge({"gpt-4o": 0.8, "claude-sonnet": 0.6, "gemini-pro": 0.7})
        ensemble = JudgeEnsemble(judge_function=judge)
        sync_result = ensemble.score("bias", "resp")
        async_result = asyncio.run(ensemble.score_async("bias", "resp"))
        self.assertEqual(sync_result, async_result)

    def test_score_async_disagreement_flag(self):
        judge = _make_judge({"gpt-4o": 0.95, "claude-sonnet": 0.1, "gemini-pro": 0.5})
        result = asyncio.run(JudgeEnsemble(judge_function=judge).score_async("bias", "resp"))
        self.assertTrue(result.disagreement_flag)
        self.assertAlmostEqual(result.aggregate_score, 0.5)

    def test_score_all_async_matches_sync(self):
        judge = _make_judge({"gpt-4o": 0.5, "claude-sonnet": 0.5, "gemini-pro": 0.5})
        ensemble = JudgeEnsemble(judge_function=judge)
        dims = ["bias", "toxicity", "privacy"]
        sync_results = ensemble.score_all(dims, "resp")
        async_results = asyncio.run(ensemble.score_all_async(dims, "resp"))
        self.assertEqual(sync_results.keys(), async_results.keys())
        for dim in dims:
            self.assertEqual(sync_results[dim], async_results[dim])

    def test_score_all_async_preserves_per_judge_order(self):
        judge = _make_judge({"gpt-4o": 0.8, "claude-sonnet": 0.6, "gemini-pro": 0.7})
        results = asyncio.run(
            JudgeEnsemble(judge_function=judge).score_all_async(["bias"], "resp")
        )
        models = [js.judge_model for js in results["bias"].judge_scores]
        self.assertEqual(models, DEFAULT_JUDGE_MODELS)

    def test_score_async_with_async_judge_function(self):
        """Coroutine-function judges are awaited directly, not threaded."""
        calls = []

        async def async_judge(judge_model, dimension, response, prompt):
            calls.append(judge_model)
            return JudgeScore(
                judge_model=judge_model,
                dimension=dimension,
                score=0.5,
                reasoning="async mock",
                confidence=0.9,
            )

        result = asyncio.run(
            JudgeEnsemble(judge_function=async_judge).score_async("bias", "resp")
        )
        self.assertAlmostEqual(result.aggregate_score, 0.5)
        self.assertEqual(sorted(calls), sorted(DEFAULT_JUDGE_MODELS))

    def test_score_async_runs_judges_concurrently(self):
        """Three judges with 0.1s latency each should overlap, not serialize."""
        delay = 0.1

        def slow_judge(judge_model, dimension, response, prompt):
            time.sleep(delay)
            return JudgeScore(
                judge_model=judge_model,
                dimension=dimension,
                score=0.5,
                reasoning="slow",
                confidence=0.9,
            )

        ensemble = JudgeEnsemble(judge_function=slow_judge)
        start = time.monotonic()
        asyncio.run(ensemble.score_async("bias", "resp"))
        elapsed = time.monotonic() - start
        # 3 serial calls would take >= 0.3s; concurrent should be ~0.1s.
        self.assertLess(elapsed, delay * 2.5)

    def test_score_async_empty_judge_models_raises(self):
        ensemble = JudgeEnsemble(judge_models=[], judge_function=_make_judge({}))
        # Empty list falls back to defaults, so override after construction.
        ensemble.judge_models = []
        with self.assertRaises(ValueError):
            asyncio.run(ensemble.score_async("bias", "resp"))


if __name__ == "__main__":
    unittest.main()
