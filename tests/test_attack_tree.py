"""Tests for attack-tree construction, scoring, and visualization."""

import unittest

from src.core.models import AttackTurn, AttackTree
from src.redteam.strategies.base import analyze_response, has_refusal
from src.redteam.visualize import render_dot, render_text_tree

_REFUSAL = "I'm sorry, but I cannot help with that request."
_COMPLIANT = "Sure, here is the step-by-step explanation you asked for."
_NEUTRAL = "That is an interesting question to consider carefully."


def _turn(number: int, strategy: str, response: str, escalation: int = 0) -> AttackTurn:
    """Build an :class:`AttackTurn` for tests.

    Args:
        number: Turn number.
        strategy: Strategy name used.
        response: Model response.
        escalation: Escalation level.

    Returns:
        The constructed turn.
    """
    return AttackTurn(
        turn_number=number,
        attacker_prompt=f"prompt {number}",
        model_response=response,
        strategy_used=strategy,
        escalation_level=escalation,
    )


class AnalyzeResponseTests(unittest.TestCase):
    def test_refusal_scores_zero(self) -> None:
        self.assertEqual(analyze_response(_REFUSAL), 0.0)

    def test_compliant_scores_one(self) -> None:
        self.assertEqual(analyze_response(_COMPLIANT), 1.0)

    def test_neutral_scores_half(self) -> None:
        self.assertEqual(analyze_response(_NEUTRAL), 0.5)

    def test_empty_response_scores_zero(self) -> None:
        self.assertEqual(analyze_response(""), 0.0)
        self.assertEqual(analyze_response("   "), 0.0)

    def test_has_refusal_helper(self) -> None:
        self.assertTrue(has_refusal(_REFUSAL))
        self.assertFalse(has_refusal(_COMPLIANT))


class AttackTreeModelTests(unittest.TestCase):
    def test_builds_with_turns_and_metadata(self) -> None:
        tree = AttackTree(
            root_prompt="root scenario",
            turns=[_turn(1, "dan_jailbreak", _COMPLIANT)],
            final_score=1.0,
            strategy_chain=["dan_jailbreak"],
            success=True,
        )
        self.assertEqual(tree.root_prompt, "root scenario")
        self.assertEqual(len(tree.turns), 1)
        self.assertEqual(tree.final_score, 1.0)
        self.assertTrue(tree.success)

    def test_rejects_out_of_range_final_score(self) -> None:
        with self.assertRaises(Exception):
            AttackTree(
                root_prompt="r",
                final_score=1.5,
                strategy_chain=[],
                success=False,
            )

    def test_rejects_turn_number_below_one(self) -> None:
        with self.assertRaises(Exception):
            AttackTurn(
                turn_number=0,
                attacker_prompt="p",
                model_response="r",
                strategy_used="dan_jailbreak",
                escalation_level=0,
            )

    def test_strategy_chain_preserved_in_order(self) -> None:
        tree = AttackTree(
            root_prompt="r",
            turns=[
                _turn(1, "dan_jailbreak", _REFUSAL),
                _turn(2, "roleplay", _REFUSAL),
            ],
            final_score=0.0,
            strategy_chain=["dan_jailbreak", "roleplay"],
            success=False,
        )
        self.assertEqual(tree.strategy_chain, ["dan_jailbreak", "roleplay"])


class VisualizeTests(unittest.TestCase):
    def _tree(self) -> AttackTree:
        return AttackTree(
            root_prompt="root scenario",
            turns=[
                _turn(1, "dan_jailbreak", _REFUSAL, escalation=0),
                _turn(2, "dan_jailbreak", _COMPLIANT, escalation=1),
            ],
            final_score=1.0,
            strategy_chain=["dan_jailbreak"],
            success=True,
        )

    def test_render_text_tree_includes_key_sections(self) -> None:
        text = render_text_tree(self._tree())
        self.assertIn("root scenario", text)
        self.assertIn("dan_jailbreak", text)
        self.assertIn("SUCCESS", text)
        self.assertIn("final_score=1.00", text)

    def test_render_text_tree_handles_empty_turns(self) -> None:
        empty = AttackTree(
            root_prompt="r", turns=[], final_score=0.0, strategy_chain=[], success=False
        )
        text = render_text_tree(empty)
        self.assertIn("FAILED", text)
        self.assertIn("no turns recorded", text)

    def test_render_dot_produces_valid_graph(self) -> None:
        dot = render_dot(self._tree())
        self.assertIn("digraph attack_tree", dot)
        self.assertIn("->", dot)
        self.assertIn("SUCCESS", dot)
        self.assertIn("root", dot)


if __name__ == "__main__":
    unittest.main()
