"""Tests for the toxicity scoring guardrail (``src.guardrails.toxicity``)."""

import sys
import types
import unittest
from unittest import mock

from src.guardrails.toxicity import ToxicityScorer


class _FakeDetoxify:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(self, text: str) -> dict:
        return {
            "toxicity": 0.9,
            "severe_toxicity": 0.8,
            "insult": 0.7,
            "threat": 0.6,
            "obscene": 0.5,
            "identity_attack": 0.4,
        }


def _install_fake_detoxify() -> mock._patch:
    fake = types.ModuleType("detoxify")
    fake.Detoxify = _FakeDetoxify
    return mock.patch.dict(sys.modules, {"detoxify": fake})


class LexiconFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = ToxicityScorer(use_detoxify=False, threshold=0.7)

    def test_engine_is_lexicon_fallback(self) -> None:
        self.assertEqual(self.scorer.engine_name, "lexicon")

    def test_clean_text_scores_zero(self) -> None:
        scores = self.scorer.score("The weather is pleasant today.")
        self.assertEqual(scores.toxicity, 0.0)
        self.assertEqual(scores.insult, 0.0)
        self.assertEqual(scores.threat, 0.0)

    def test_insult_keywords_score_exactly(self) -> None:
        scores = self.scorer.score("you are stupid and an idiot")
        self.assertEqual(scores.insult, 1.0)

    def test_threat_keyword_scores_exactly(self) -> None:
        scores = self.scorer.score("I will kill you")
        self.assertEqual(scores.threat, 0.5)

    def test_profanity_keywords_score(self) -> None:
        scores = self.scorer.score("that is damn crap")
        self.assertEqual(scores.profanity, 1.0)

    def test_scan_not_triggered_for_clean_text(self) -> None:
        result = self.scorer.scan("Have a nice day.")
        self.assertFalse(result.triggered)
        self.assertEqual(result.check_type, "toxicity")

    def test_scan_triggered_when_threshold_exceeded(self) -> None:
        result = self.scorer.scan("you stupid idiot loser moron")
        self.assertTrue(result.triggered)
        self.assertEqual(result.severity.value, "high")


class DetoxifyPrimaryTests(unittest.TestCase):
    def test_uses_detoxify_when_available(self) -> None:
        with _install_fake_detoxify():
            scorer = ToxicityScorer(use_detoxify=True)
            self.assertEqual(scorer.engine_name, "detoxify")

    def test_detoxify_scores_are_mapped(self) -> None:
        with _install_fake_detoxify():
            scorer = ToxicityScorer(use_detoxify=True)
            scores = scorer.score("anything at all")
            self.assertEqual(scores.toxicity, 0.9)
            self.assertEqual(scores.severe_toxicity, 0.8)
            self.assertEqual(scores.insult, 0.7)
            self.assertEqual(scores.threat, 0.6)
            self.assertEqual(scores.profanity, 0.5)  # mapped from 'obscene'
            self.assertEqual(scores.identity_attack, 0.4)


if __name__ == "__main__":
    unittest.main()
