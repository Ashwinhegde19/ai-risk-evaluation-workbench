"""Tests for the judge ensemble (Phase 3, Task 3.2)."""

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


if __name__ == "__main__":
    unittest.main()
