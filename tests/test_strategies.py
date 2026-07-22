"""Tests for the red-team attack strategies in ``src.redteam.strategies``."""

import unittest

from src.redteam.strategies import (
    all_strategies,
    get_strategy,
    list_strategies,
)
from src.redteam.strategies.base import AttackStrategy, has_refusal

_REFUSAL = "I'm sorry, but I cannot help with that request."
_COMPLIANT = "Sure, here is the step-by-step explanation you asked for."


class StrategyRegistryTests(unittest.TestCase):
    def test_eight_strategies_registered(self) -> None:
        names = list_strategies()
        self.assertEqual(len(names), 8)
        self.assertIn("dan_jailbreak", names)
        self.assertIn("roleplay", names)
        self.assertIn("encoding", names)
        self.assertIn("multilingual", names)
        self.assertIn("context_overflow", names)
        self.assertIn("tool_exploit", names)
        self.assertIn("rag_poison", names)
        self.assertIn("memory_manip", names)

    def test_strategy_names_unique(self) -> None:
        names = [s.name for s in all_strategies()]
        self.assertEqual(len(names), len(set(names)))

    def test_get_strategy_returns_fresh_instance(self) -> None:
        a = get_strategy("dan_jailbreak")
        b = get_strategy("dan_jailbreak")
        self.assertIsInstance(a, AttackStrategy)
        self.assertIsNot(a, b)
        self.assertEqual(a.name, "dan_jailbreak")

    def test_get_strategy_unknown_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_strategy("does_not_exist")


class StrategyBehaviorTests(unittest.TestCase):
    def test_every_strategy_generates_valid_prompts(self) -> None:
        history: list = []
        for strategy in all_strategies():
            with self.subTest(strategy=strategy.name):
                opening = strategy.generate_prompt(1, history)
                self.assertIsInstance(opening, str)
                self.assertTrue(opening.strip(), "opening prompt must not be empty")

                escalation = strategy.get_escalation_prompt(2, history)
                self.assertIsInstance(escalation, str)
                self.assertTrue(
                    escalation.strip(), "escalation prompt must not be empty"
                )

    def test_should_escalate_is_bool_for_any_response(self) -> None:
        for strategy in all_strategies():
            with self.subTest(strategy=strategy.name):
                self.assertIsInstance(strategy.should_escalate(_REFUSAL), bool)
                self.assertIsInstance(strategy.should_escalate(_COMPLIANT), bool)

    def test_default_escalation_follows_refusal_markers(self) -> None:
        strategy = get_strategy("roleplay")
        self.assertTrue(strategy.should_escalate(_REFUSAL))
        self.assertFalse(strategy.should_escalate(_COMPLIANT))


class HelperFunctionTests(unittest.TestCase):
    def test_has_refusal_detects_markers(self) -> None:
        self.assertTrue(has_refusal("I cannot do that."))
        self.assertTrue(has_refusal("As an AI, I must decline."))
        self.assertFalse(has_refusal("Here is the information you requested."))


if __name__ == "__main__":
    unittest.main()
