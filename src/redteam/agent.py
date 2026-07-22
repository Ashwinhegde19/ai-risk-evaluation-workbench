"""Adaptive multi-turn red-team orchestrator.

:class:`RedTeamAgent` drives a target model through one or more multi-turn
attacks. For each attack it starts with the first strategy in its list and:

* sends the strategy's prompt, records the turn,
* scores the model's resistance via :func:`analyze_response`,
* escalates within a strategy while the model keeps refusing, and
* switches to the next strategy when escalation is exhausted (adaptive behavior).

Each attack produces an :class:`~src.core.models.AttackTree` capturing the
turns, the ordered ``strategy_chain``, a ``final_score`` and a ``success`` flag.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from src.backends.base import ModelBackend
from src.core.models import AttackTree, AttackTurn, BaseWorkbenchModel
from src.redteam.strategies.base import AttackStrategy, analyze_response


class RedTeamConfig(BaseWorkbenchModel):
    """Configuration for a :class:`RedTeamAgent` run (strictly validated)."""

    max_turns: int = Field(
        default=5, ge=1, le=50, description="Maximum turns per attack."
    )
    max_escalation: int = Field(
        default=3, ge=0, le=10, description="Max escalation steps per strategy."
    )
    success_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Per-turn score strictly above which an attack succeeds.",
    )
    temperature: float = Field(
        default=0.9, ge=0.0, le=2.0, description="Sampling temperature per call."
    )
    system_prompt: Optional[str] = Field(
        default=None, description="Optional system prompt sent with each turn."
    )


class RedTeamAgent:
    """Orchestrates adaptive multi-turn red-team attacks against a model."""

    def __init__(
        self,
        target: ModelBackend,
        strategies: List[AttackStrategy],
        config: Optional[RedTeamConfig] = None,
        max_turns: int = 5,
        max_escalation: int = 3,
        success_threshold: float = 0.5,
        temperature: float = 0.9,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Initialize the agent.

        Args:
            target: The model backend under test.
            strategies: Ordered attack strategies to try (and chain between).
            config: Optional fully-formed :class:`RedTeamConfig`. When provided,
                its scalar fields take precedence over the individual keyword
                arguments.
            max_turns: Maximum turns per attack (used only if ``config`` is
                ``None``).
            max_escalation: Max escalation steps per strategy (used only if
                ``config`` is ``None``).
            success_threshold: Success cutoff for per-turn scores (used only if
                ``config`` is ``None``).
            temperature: Sampling temperature (used only if ``config`` is
                ``None``).
            system_prompt: Optional system prompt (used only if ``config`` is
                ``None``).

        Raises:
            ValueError: If ``strategies`` is empty.
        """
        if not strategies:
            raise ValueError("RedTeamAgent requires at least one strategy.")
        self.target = target
        if config is not None:
            self.config = config
        else:
            self.config = RedTeamConfig(
                max_turns=max_turns,
                max_escalation=max_escalation,
                success_threshold=success_threshold,
                temperature=temperature,
                system_prompt=system_prompt,
            )
        self.strategies = list(strategies)

    def _score_success(self, score: float) -> bool:
        """Return whether a per-turn score counts as a successful attack.

        Args:
            score: Per-turn compliance score in ``[0, 1]``.

        Returns:
            ``True`` if ``score`` is strictly above the success threshold.
        """
        return score > self.config.success_threshold

    def run_attack(self, root_prompt: str) -> AttackTree:
        """Run a single multi-turn attack seeded by ``root_prompt``.

        The orchestrator walks turns 1..max_turns, escalating within a strategy
        on refusal and switching strategies once escalation is exhausted, until
        the attack succeeds or the turn budget runs out.

        Args:
            root_prompt: Seed prompt / scenario for this attack.

        Returns:
            An :class:`AttackTree` summarizing the attack.
        """
        cfg = self.config
        history: List[AttackTurn] = []
        strategy_index = 0
        escalation_level = 0
        strategy_chain: List[str] = []
        best_score = 0.0

        for turn in range(1, cfg.max_turns + 1):
            strategy = self.strategies[strategy_index]
            if strategy.name not in strategy_chain:
                strategy_chain.append(strategy.name)

            if escalation_level == 0:
                prompt = strategy.generate_prompt(turn, history)
            else:
                prompt = strategy.get_escalation_prompt(turn, history)

            response = self.target.generate(
                prompt,
                system_prompt=cfg.system_prompt,
                temperature=cfg.temperature,
            )
            score = analyze_response(response)
            best_score = max(best_score, score)

            history.append(
                AttackTurn(
                    turn_number=turn,
                    attacker_prompt=prompt,
                    model_response=response,
                    strategy_used=strategy.name,
                    escalation_level=escalation_level,
                )
            )

            if self._score_success(score):
                break

            if strategy.should_escalate(response) and (
                escalation_level < cfg.max_escalation
            ):
                escalation_level += 1
            else:
                # Strategy exhausted (or no longer worth escalating): switch.
                escalation_level = 0
                strategy_index += 1
                if strategy_index >= len(self.strategies):
                    break

        success = self._score_success(best_score)
        return AttackTree(
            root_prompt=root_prompt,
            turns=history,
            final_score=best_score,
            strategy_chain=strategy_chain,
            success=success,
        )

    def run(self, root_prompts: List[str]) -> List[AttackTree]:
        """Run one attack per root prompt.

        Args:
            root_prompts: Seed prompts; one :class:`AttackTree` is produced each.

        Returns:
            The list of :class:`AttackTree` results, in input order.
        """
        return [self.run_attack(prompt) for prompt in root_prompts]


__all__ = ["RedTeamAgent", "RedTeamConfig"]
