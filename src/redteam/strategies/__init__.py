"""Red-team attack strategies package.

Exposes the :class:`AttackStrategy` base class, all eight concrete strategies,
and a small registry so callers (and the orchestrator) can look strategies up
by name.
"""

from __future__ import annotations

from typing import Dict, List, Type

from .base import AttackStrategy, analyze_response, has_refusal
from .context_overflow import ContextOverflowStrategy
from .dan_jailbreak import DanJailbreakStrategy
from .encoding import EncodingStrategy
from .few_shot import FewShotStrategy
from .memory_manip import MemoryManipStrategy
from .multilingual import MultilingualStrategy
from .policy_conflation import PolicyConflationStrategy
from .rag_poison import RagPoisonStrategy
from .roleplay import RoleplayStrategy
from .structured_output import StructuredOutputStrategy
from .syllogism import SyllogismStrategy
from .tool_exploit import ToolExploitStrategy

# Registry of strategy name -> zero-argument factory. Kept as factories so each
# lookup returns a fresh instance (strategies may hold per-run escalation state).
_STRATEGY_REGISTRY: Dict[str, Type[AttackStrategy]] = {
    "dan_jailbreak": DanJailbreakStrategy,
    "roleplay": RoleplayStrategy,
    "encoding": EncodingStrategy,
    "multilingual": MultilingualStrategy,
    "syllogism": SyllogismStrategy,
    "few_shot": FewShotStrategy,
    "policy_conflation": PolicyConflationStrategy,
    "structured_output": StructuredOutputStrategy,
    "context_overflow": ContextOverflowStrategy,
    "tool_exploit": ToolExploitStrategy,
    "rag_poison": RagPoisonStrategy,
    "memory_manip": MemoryManipStrategy,
}


def list_strategies() -> List[str]:
    """Return the names of all registered attack strategies.

    Returns:
        Sorted list of registered strategy names.
    """
    return sorted(_STRATEGY_REGISTRY.keys())


def get_strategy(name: str) -> AttackStrategy:
    """Instantiate a registered attack strategy by name.

    Args:
        name: One of the names returned by :func:`list_strategies`.

    Returns:
        A fresh :class:`AttackStrategy` instance.

    Raises:
        KeyError: If ``name`` is not a registered strategy.
    """
    if name not in _STRATEGY_REGISTRY:
        raise KeyError(
            f"Unknown strategy '{name}'. Available: {list_strategies()}"
        )
    return _STRATEGY_REGISTRY[name]()


def all_strategies() -> List[AttackStrategy]:
    """Instantiate every registered strategy.

    Returns:
        A list of fresh :class:`AttackStrategy` instances, one per strategy.
    """
    return [cls() for cls in _STRATEGY_REGISTRY.values()]


__all__ = [
    "AttackStrategy",
    "analyze_response",
    "has_refusal",
    "DanJailbreakStrategy",
    "RoleplayStrategy",
    "EncodingStrategy",
    "MultilingualStrategy",
    "SyllogismStrategy",
    "FewShotStrategy",
    "PolicyConflationStrategy",
    "StructuredOutputStrategy",
    "ContextOverflowStrategy",
    "ToolExploitStrategy",
    "RagPoisonStrategy",
    "MemoryManipStrategy",
    "list_strategies",
    "get_strategy",
    "all_strategies",
]
