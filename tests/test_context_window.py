"""Tests for sliding-window context truncation in red-team agent."""

import unittest

from src.core.models import AttackTurn
from src.redteam.agent import estimate_tokens, history_tokens, truncate_history


class ContextWindowTruncationTest(unittest.TestCase):
    """Test sliding-window truncation logic."""

    def test_estimate_tokens_empty(self):
        """estimate_tokens should return 0 for empty string."""
        self.assertEqual(estimate_tokens(""), 0)

    def test_estimate_tokens_nonempty(self):
        """estimate_tokens should estimate ~4 chars per token."""
        # 100 chars -> ~25 tokens
        text = "a" * 100
        tokens = estimate_tokens(text)
        self.assertGreater(tokens, 0)
        self.assertLessEqual(tokens, 30)  # Allow some margin

    def test_history_tokens_empty(self):
        """history_tokens should return 0 for empty history."""
        self.assertEqual(history_tokens([]), 0)

    def test_history_tokens_sums_all_turns(self):
        """history_tokens should sum tokens across all turns."""
        history = [
            AttackTurn(
                turn_number=1,
                attacker_prompt="a" * 100,  # ~25 tokens
                model_response="b" * 100,  # ~25 tokens
                strategy_used="test",
                escalation_level=0,
            ),
            AttackTurn(
                turn_number=2,
                attacker_prompt="c" * 100,  # ~25 tokens
                model_response="d" * 100,  # ~25 tokens
                strategy_used="test",
                escalation_level=1,
            ),
        ]
        total = history_tokens(history)
        self.assertGreater(total, 80)  # At least 4 * 25 = 100, allow margin
        self.assertLess(total, 120)

    def test_truncate_history_empty(self):
        """truncate_history should return empty list for empty history."""
        result = truncate_history([], max_context_tokens=4096)
        self.assertEqual(result, [])

    def test_truncate_history_fits_budget(self):
        """truncate_history should return all turns if they fit the budget."""
        history = [
            AttackTurn(
                turn_number=i,
                attacker_prompt="a" * 100,
                model_response="b" * 100,
                strategy_used="test",
                escalation_level=0,
            )
            for i in range(1, 4)
        ]
        # Each turn is ~50 tokens, 3 turns = ~150 tokens, budget = 4096
        result = truncate_history(history, max_context_tokens=4096)
        self.assertEqual(len(result), 3)

    def test_truncate_history_drops_oldest(self):
        """truncate_history should drop oldest turns when over budget."""
        history = [
            AttackTurn(
                turn_number=i,
                attacker_prompt="a" * 1000,  # ~250 tokens
                model_response="b" * 1000,  # ~250 tokens
                strategy_used="test",
                escalation_level=0,
            )
            for i in range(1, 11)  # 10 turns, ~5000 tokens total
        ]
        # Budget = 4096 - 512 (reserve) = 3584 tokens
        # Each turn is ~500 tokens, so we can fit ~7 turns
        result = truncate_history(history, max_context_tokens=4096, reserve_tokens=512)
        self.assertLess(len(result), 10)
        self.assertGreater(len(result), 0)
        # Should keep the most recent turns (highest turn numbers)
        self.assertEqual(result[-1].turn_number, 10)
        self.assertGreater(result[0].turn_number, 1)

    def test_truncate_history_keeps_at_least_one(self):
        """truncate_history should keep at least the most recent turn."""
        history = [
            AttackTurn(
                turn_number=1,
                attacker_prompt="a" * 10000,  # ~2500 tokens
                model_response="b" * 10000,  # ~2500 tokens
                strategy_used="test",
                escalation_level=0,
            )
        ]
        # Even if the single turn exceeds budget, keep it
        result = truncate_history(history, max_context_tokens=1000, reserve_tokens=512)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].turn_number, 1)

    def test_truncate_history_preserves_order(self):
        """truncate_history should preserve turn order (oldest to newest)."""
        history = [
            AttackTurn(
                turn_number=i,
                attacker_prompt="a" * 100,
                model_response="b" * 100,
                strategy_used="test",
                escalation_level=0,
            )
            for i in range(1, 6)
        ]
        result = truncate_history(history, max_context_tokens=4096)
        turn_numbers = [t.turn_number for t in result]
        self.assertEqual(turn_numbers, sorted(turn_numbers))


if __name__ == "__main__":
    unittest.main()
