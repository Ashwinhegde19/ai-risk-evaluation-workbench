"""Tests for the adaptive red-team orchestrator in ``src.redteam.agent``."""

import unittest

from src.backends.base import ModelBackend
from src.redteam.agent import RedTeamAgent, RedTeamConfig
from src.redteam.strategies import get_strategy

_REFUSAL = "I'm sorry, but I cannot help with that request."
_COMPLIANT = "Sure, here is the step-by-step explanation you asked for."


class QueueBackend(ModelBackend):
    """Deterministic mock backend returning queued responses in order."""

    def __init__(
        self, responses: list[str], default: str = _REFUSAL
    ) -> None:
        """Initialize the mock with a response queue.

        Args:
            responses: Responses returned in order, one per ``generate`` call.
            default: Response used once the queue is exhausted.
        """
        super().__init__("mock-model")
        self._queue = list(responses)
        self.default = default
        self.calls: list[str] = []

    def generate(
        self,
        prompt: str,
        system_prompt: object = None,
        temperature: float = 0.7,
        max_tokens: object = None,
    ) -> str:
        """Return the next queued response, recording the prompt.

        Args:
            prompt: The attacker prompt.
            system_prompt: Optional system prompt (ignored by the mock).
            temperature: Sampling temperature (ignored by the mock).
            max_tokens: Optional max tokens (ignored by the mock).

        Returns:
            The next queued response or the default.
        """
        self.calls.append(prompt)
        if self._queue:
            return self._queue.pop(0)
        return self.default


class RedTeamAgentTests(unittest.TestCase):
    def _agent(self, backend: QueueBackend, **kwargs: object) -> RedTeamAgent:
        """Build an agent with two strategies for the given backend.

        Args:
            backend: The mock backend.
            **kwargs: Forwarded to :class:`RedTeamAgent`.

        Returns:
            A configured :class:`RedTeamAgent`.
        """
        strategies = [get_strategy("dan_jailbreak"), get_strategy("roleplay")]
        return RedTeamAgent(backend, strategies, **kwargs)  # type: ignore[arg-type]

    def test_compliant_model_succeeds_on_first_turn(self) -> None:
        backend = QueueBackend([_COMPLIANT])
        agent = self._agent(backend)
        tree = agent.run_attack("root scenario")
        self.assertTrue(tree.success)
        self.assertEqual(tree.final_score, 1.0)
        self.assertEqual(len(tree.turns), 1)
        self.assertEqual(len(tree.strategy_chain), 1)
        self.assertEqual(tree.turns[0].escalation_level, 0)

    def test_refusing_model_exhausts_turn_budget_and_switches(self) -> None:
        backend = QueueBackend([_REFUSAL] * 5)
        agent = self._agent(backend, max_turns=5, max_escalation=3)
        tree = agent.run_attack("root scenario")
        self.assertFalse(tree.success)
        self.assertEqual(tree.final_score, 0.0)
        self.assertEqual(len(tree.turns), 5)
        # Two strategies should have been tried (escalation exhausted -> switch).
        self.assertEqual(len(tree.strategy_chain), 2)
        # Escalation must reset to 0 after switching strategies.
        self.assertEqual(tree.turns[4].escalation_level, 0)
        self.assertEqual(tree.turns[4].strategy_used, "roleplay")

    def test_switch_then_succeed_records_both_strategies(self) -> None:
        # Four refusals escalate strategy 0 to its limit and trigger a switch;
        # the fifth turn (strategy 1) gets a compliant response -> success.
        backend = QueueBackend([_REFUSAL, _REFUSAL, _REFUSAL, _REFUSAL, _COMPLIANT])
        agent = self._agent(backend, max_turns=5, max_escalation=3)
        tree = agent.run_attack("root scenario")
        self.assertTrue(tree.success)
        self.assertEqual(tree.final_score, 1.0)
        self.assertEqual(len(tree.turns), 5)
        self.assertEqual(tree.strategy_chain, ["dan_jailbreak", "roleplay"])

    def test_empty_strategies_raises(self) -> None:
        with self.assertRaises(ValueError):
            RedTeamAgent(QueueBackend([]), [])

    def test_config_overrides_keyword_arguments(self) -> None:
        config = RedTeamConfig(max_turns=2, max_escalation=3)
        backend = QueueBackend([_REFUSAL] * 5)
        agent = self._agent(backend, config=config, max_turns=5)
        tree = agent.run_attack("root scenario")
        self.assertEqual(len(tree.turns), 2)

    def test_run_handles_multiple_root_prompts(self) -> None:
        backend = QueueBackend([_COMPLIANT, _COMPLIANT])
        agent = self._agent(backend)
        trees = agent.run(["scenario A", "scenario B"])
        self.assertEqual(len(trees), 2)
        self.assertEqual(trees[0].root_prompt, "scenario A")
        self.assertEqual(trees[1].root_prompt, "scenario B")
        self.assertTrue(all(t.success for t in trees))


if __name__ == "__main__":
    unittest.main()
