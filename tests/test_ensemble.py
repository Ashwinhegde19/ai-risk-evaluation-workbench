"""Tests for the judge ensemble (Phase 3, Task 3.2)."""

import asyncio
import io
import math
import time
import unittest
from contextlib import redirect_stdout

from src.judge.ensemble import (
    DEFAULT_DISAGREEMENT_THRESHOLD,
    DEFAULT_JUDGE_MODELS,
    DEFAULT_MAX_CONCURRENCY,
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
        judge = _make_judge({"openai/gpt-5": 0.8, "anthropic/claude-opus-4.1": 0.6, "google/gemini-2.5-pro": 0.7})
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
        judge = _make_judge({"openai/gpt-5": 0.9, "anthropic/claude-opus-4.1": 0.1, "google/gemini-2.5-pro": 0.5})
        result = JudgeEnsemble(judge_function=judge).score("privacy", "resp")
        self.assertAlmostEqual(result.score_spread, 0.8)

    def test_no_disagreement_when_spread_at_threshold(self):
        # spread == threshold (0.2) must NOT flag (strictly greater).
        judge = _make_judge({"openai/gpt-5": 0.8, "anthropic/claude-opus-4.1": 0.6, "google/gemini-2.5-pro": 0.7})
        result = JudgeEnsemble(judge_function=judge).score("bias", "resp")
        self.assertAlmostEqual(result.score_spread, 0.2)
        self.assertFalse(result.disagreement_flag)

    def test_disagreement_flagged_when_spread_exceeds_threshold(self):
        judge = _make_judge({"openai/gpt-5": 0.95, "anthropic/claude-opus-4.1": 0.1, "google/gemini-2.5-pro": 0.5})
        result = JudgeEnsemble(judge_function=judge).score("bias", "resp")
        self.assertTrue(result.disagreement_flag)

    def test_custom_threshold(self):
        judge = _make_judge({"openai/gpt-5": 0.8, "anthropic/claude-opus-4.1": 0.6, "google/gemini-2.5-pro": 0.7})
        ensemble = JudgeEnsemble(
            disagreement_threshold=0.1, judge_function=judge
        )
        result = ensemble.score("bias", "resp")
        # spread 0.2 > 0.1 -> flagged
        self.assertTrue(result.disagreement_flag)

    def test_per_judge_scores_returned(self):
        judge = _make_judge({"openai/gpt-5": 0.8, "anthropic/claude-opus-4.1": 0.6, "google/gemini-2.5-pro": 0.7})
        result = JudgeEnsemble(judge_function=judge).score("bias", "resp")
        self.assertEqual(len(result.judge_scores), 3)
        models = {js.judge_model for js in result.judge_scores}
        self.assertEqual(models, {"openai/gpt-5", "anthropic/claude-opus-4.1", "google/gemini-2.5-pro"})

    def test_confidence_is_median_of_judge_confidences(self):
        def judge(judge_model, dimension, response, prompt):
            conf = {"openai/gpt-5": 0.9, "anthropic/claude-opus-4.1": 0.3, "google/gemini-2.5-pro": 0.5}[
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
        judge = _make_judge({"openai/gpt-5": 0.5, "anthropic/claude-opus-4.1": 0.5, "google/gemini-2.5-pro": 0.5})
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
            def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=None):
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

    def test_default_judge_function_retries_then_drops_on_garbage(self):
        """A backend returning garbage twice: retry once, then drop the vote."""
        import sys
        from io import StringIO

        class GarbageBackend:
            def __init__(self):
                self.calls = []

            def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=None):
                self.calls.append(max_tokens)
                return "this is not json at all, no braces anywhere"

        backend = GarbageBackend()

        def factory(model_name):
            return backend

        ensemble = JudgeEnsemble(judge_models=["m1"], backend_factory=factory)

        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            result = ensemble.score("bias", "resp")
            stderr_output = sys.stderr.getvalue()
        finally:
            sys.stderr = old_stderr

        # Two generate calls: initial + one retry (never more).
        self.assertEqual(len(backend.calls), 2)
        # The retry used the higher retry_max_tokens budget.
        self.assertEqual(backend.calls[0], ensemble.judge_config.max_tokens)
        self.assertEqual(backend.calls[1], ensemble.judge_config.retry_max_tokens)
        # The single vote was dropped -> dimension unscored, no exception.
        self.assertTrue(result.unscored)
        self.assertEqual(len(result.judge_scores), 0)
        # A loud warning was logged naming the judge and dimension.
        self.assertIn("WARNING", stderr_output)
        self.assertIn("m1", stderr_output)
        self.assertIn("bias", stderr_output)

    def test_default_judge_function_retry_succeeds(self):
        """First response is garbage, retry returns valid JSON -> vote kept."""

        class FlakyBackend:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=None):
                self.calls += 1
                if self.calls == 1:
                    return "garbage, no json"
                return '{"rating": 5, "confidence": 0.95, "reasoning": "recovered"}'

        def factory(model_name):
            return FlakyBackend()

        ensemble = JudgeEnsemble(judge_models=["m1"], backend_factory=factory)
        result = ensemble.score("bias", "resp")

        # Retry recovered the vote: rating 5 -> score 1.0.
        self.assertFalse(result.unscored)
        self.assertEqual(len(result.judge_scores), 1)
        self.assertAlmostEqual(result.aggregate_score, 1.0)
        self.assertEqual(result.judge_scores[0].reasoning, "recovered")


class AsyncEnsembleTests(unittest.TestCase):
    """Verify the async scoring paths match the sync paths exactly."""

    def test_score_async_matches_sync(self):
        judge = _make_judge({"openai/gpt-5": 0.8, "anthropic/claude-opus-4.1": 0.6, "google/gemini-2.5-pro": 0.7})
        ensemble = JudgeEnsemble(judge_function=judge)
        sync_result = ensemble.score("bias", "resp")
        async_result = asyncio.run(ensemble.score_async("bias", "resp"))
        self.assertEqual(sync_result, async_result)

    def test_score_async_disagreement_flag(self):
        judge = _make_judge({"openai/gpt-5": 0.95, "anthropic/claude-opus-4.1": 0.1, "google/gemini-2.5-pro": 0.5})
        result = asyncio.run(JudgeEnsemble(judge_function=judge).score_async("bias", "resp"))
        self.assertTrue(result.disagreement_flag)
        self.assertAlmostEqual(result.aggregate_score, 0.5)

    def test_score_all_async_matches_sync(self):
        judge = _make_judge({"openai/gpt-5": 0.5, "anthropic/claude-opus-4.1": 0.5, "google/gemini-2.5-pro": 0.5})
        ensemble = JudgeEnsemble(judge_function=judge)
        dims = ["bias", "toxicity", "privacy"]
        sync_results = ensemble.score_all(dims, "resp")
        async_results = asyncio.run(ensemble.score_all_async(dims, "resp"))
        self.assertEqual(sync_results.keys(), async_results.keys())
        for dim in dims:
            self.assertEqual(sync_results[dim], async_results[dim])

    def test_score_all_async_preserves_per_judge_order(self):
        judge = _make_judge({"openai/gpt-5": 0.8, "anthropic/claude-opus-4.1": 0.6, "google/gemini-2.5-pro": 0.7})
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


class ParallelBatchTests(unittest.TestCase):
    """Prove score_responses fans out all (dimension x judge) calls concurrently.

    These tests are what make the parallelism provable: a mock judge that sleeps
    0.2s per call must finish in ~ceil(21/concurrency) *0.2s, NOT 21 * 0.2s.
    """

    DELAY = 0.2

    @staticmethod
    def _sleeping_judge(delay):
        def judge(judge_model, dimension, response, prompt):
            time.sleep(delay)
            return JudgeScore(
                judge_model=judge_model,
                dimension=dimension,
                score=0.5,
                reasoning="sleeping mock",
                confidence=0.9,
            )

        return judge

    def test_score_responses_is_parallel_not_serial(self):
        """21 calls (7 dims x 3 judges) at 0.2s each must overlap, not serialize."""
        dims = ["d1", "d2", "d3", "d4", "d5", "d6", "d7"]
        responses = {d: f"response for {d}" for d in dims}
        ensemble = JudgeEnsemble(judge_function=self._sleeping_judge(self.DELAY))

        buf = io.StringIO()
        start = time.monotonic()
        with redirect_stdout(buf):
            results = asyncio.run(ensemble.score_responses(responses))
        wall = time.monotonic() - start

        # Correctness: every dimension scored with all judges.
        self.assertEqual(set(results.keys()), set(dims))
        for res in results.values():
            self.assertEqual(len(res.judge_scores), len(DEFAULT_JUDGE_MODELS))
            self.assertAlmostEqual(res.aggregate_score, 0.5)

        n_calls = len(dims) * len(DEFAULT_JUDGE_MODELS)  # 7 * 3 = 21
        sum_of_calls = n_calls * self.DELAY              # 21 * 0.2 = 4.2s serial
        # THE PROOF: wall clock must be far less than the serial total.
        self.assertLess(wall, sum_of_calls / 2)
        # And it should land near ceil(21/concurrency) * delay, not 21 * delay.
        expected_batches = math.ceil(n_calls / DEFAULT_MAX_CONCURRENCY)
        self.assertLess(wall, expected_batches * self.DELAY * 3)

        # The timing log line is emitted and reports a real speedup.
        log_line = next(l for l in buf.getvalue().splitlines() if l.startswith("[judge]"))
        self.assertIn("dims=7", log_line)
        self.assertIn("judges=3", log_line)
        self.assertIn("wall_clock=", log_line)
        self.assertIn("sum_of_calls=", log_line)
        self.assertIn("speedup=", log_line)
        print(f"\n[proof] {log_line}")

    def test_score_responses_matches_score_all(self):
        """The batch path produces identical results to the per-dimension path."""
        judge = _make_judge({"openai/gpt-5": 0.8, "anthropic/claude-opus-4.1": 0.6, "google/gemini-2.5-pro": 0.7})
        dims = ["bias", "toxicity", "privacy"]
        responses = {d: f"resp-{d}" for d in dims}

        ensemble = JudgeEnsemble(judge_function=judge)
        batch = asyncio.run(ensemble.score_responses(responses))
        # score_all uses one shared response; give it the same map values by
        # scoring each dim individually and comparing aggregation per dim.
        for d in dims:
            single = ensemble.score(d, responses[d])
            self.assertEqual(batch[d].aggregate_score, single.aggregate_score)
            self.assertEqual(batch[d].score_spread, single.score_spread)
            self.assertEqual(batch[d].disagreement_flag, single.disagreement_flag)

    def test_semaphore_caps_concurrency(self):
        """With max_concurrency=1, calls serialize (wall ~ sum_of_calls)."""
        dims = ["d1", "d2", "d3"]
        responses = {d: "resp" for d in dims}
        ensemble = JudgeEnsemble(
            judge_function=self._sleeping_judge(0.05), max_concurrency=1
        )
        start = time.monotonic()
        asyncio.run(ensemble.score_responses(responses))
        wall = time.monotonic() - start
        # 9 serial calls at 0.05s = 0.45s; concurrency=1 must not parallelize.
        self.assertGreaterEqual(wall, 9 * 0.05 * 0.8)

    def test_score_responses_empty_judge_models_raises(self):
        ensemble = JudgeEnsemble(judge_models=[], judge_function=_make_judge({}))
        ensemble.judge_models = []
        with self.assertRaises(ValueError):
            asyncio.run(ensemble.score_responses({"bias": "resp"}))


class GracefulDegradationTests(unittest.TestCase):
    """Verify the ensemble tolerates dropped votes without crashing."""

    def test_one_none_vote_aggregates_remaining(self):
        """One judge returns None; aggregate over the two valid votes."""
        def judge(judge_model, dimension, response, prompt):
            if judge_model == "anthropic/claude-opus-4.1":
                return None  # Simulate a dropped vote.
            return JudgeScore(
                judge_model=judge_model,
                dimension=dimension,
                score=0.8 if judge_model == "openai/gpt-5" else 0.6,
                reasoning="valid",
                confidence=0.9,
            )

        result = JudgeEnsemble(judge_function=judge).score("bias", "resp")
        # Two valid votes: 0.8 and 0.6 → median = mean = 0.7.
        self.assertAlmostEqual(result.aggregate_score, 0.7)
        self.assertEqual(len(result.judge_scores), 2)
        self.assertFalse(result.unscored)

    def test_all_none_votes_marks_unscored(self):
        """All judges return None; dimension is unscored, no exception."""
        def judge(judge_model, dimension, response, prompt):
            return None  # Every vote dropped.

        result = JudgeEnsemble(judge_function=judge).score("bias", "resp")
        self.assertTrue(result.unscored)
        self.assertIsNone(result.aggregate_score)
        self.assertIsNone(result.score_spread)
        self.assertIsNone(result.confidence)
        self.assertEqual(len(result.judge_scores), 0)

    def test_judge_exception_is_caught_and_dropped(self):
        """A judge that raises is caught, vote dropped, run continues."""
        def judge(judge_model, dimension, response, prompt):
            if judge_model == "anthropic/claude-opus-4.1":
                raise RuntimeError("Simulated judge crash")
            return JudgeScore(
                judge_model=judge_model,
                dimension=dimension,
                score=0.7,
                reasoning="ok",
                confidence=0.9,
            )

        # Capture stderr to verify the warning was logged.
        import sys
        from io import StringIO
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            result = JudgeEnsemble(judge_function=judge).score("bias", "resp")
            stderr_output = sys.stderr.getvalue()
        finally:
            sys.stderr = old_stderr

        # Two valid votes remain (openai/gpt-5 and google/gemini-2.5-pro).
        self.assertEqual(len(result.judge_scores), 2)
        self.assertAlmostEqual(result.aggregate_score, 0.7)
        # A warning was logged to stderr.
        self.assertIn("WARNING", stderr_output)
        self.assertIn("anthropic/claude-opus-4.1", stderr_output)


if __name__ == "__main__":
    unittest.main()
