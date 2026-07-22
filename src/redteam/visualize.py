"""Text and Graphviz rendering for red-team attack trees.

Provides a plain-text tree renderer (no external dependencies) and an optional
Graphviz DOT exporter for richer visualization in external graph tools.
"""

from __future__ import annotations

from typing import List

from src.core.models import AttackTree, AttackTurn


def render_text_tree(tree: AttackTree) -> str:
    """Render an :class:`AttackTree` as an indented, human-readable tree.

    Args:
        tree: The attack tree to render.

    Returns:
        A multi-line string with the root prompt, one line per turn (showing
        strategy, escalation level and a short response excerpt), and a footer
        summarizing the final score and success.
    """
    lines: List[str] = []
    status = "SUCCESS" if tree.success else "FAILED"
    lines.append(f"Attack root: {tree.root_prompt}")
    lines.append(f"Strategies: {' -> '.join(tree.strategy_chain) or '(none)'}")
    lines.append("")

    if not tree.turns:
        lines.append("  (no turns recorded)")
    else:
        for turn in tree.turns:
            excerpt = _excerpt(turn.model_response)
            lines.append(
                f"  [{turn.turn_number}] {turn.strategy_used} "
                f"(esc={turn.escalation_level}) -> {excerpt}"
            )

    lines.append("")
    lines.append(
        f"Result: {status} | final_score={tree.final_score:.2f}"
    )
    return "\n".join(lines)


def _excerpt(text: str, max_len: int = 60) -> str:
    """Return a trimmed, whitespace-collapsed excerpt of ``text``.

    Args:
        text: The text to excerpt.
        max_len: Maximum length of the returned excerpt.

    Returns:
        A single-line excerpt suitable for tree rendering.
    """
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= max_len:
        return collapsed or "(empty response)"
    return collapsed[: max_len - 3].rstrip() + "..."


def render_dot(tree: AttackTree) -> str:
    """Render an :class:`AttackTree` as a Graphviz DOT document.

    Args:
        tree: The attack tree to render.

    Returns:
        A ``digraph`` DOT string. Safe to write to a ``.dot`` file and compile
        with ``dot -Tpng``.
    """
    node_id = lambda t: f"turn{t.turn_number}"
    lines: List[str] = ["digraph attack_tree {", "  rankdir=TB;", "  node [shape=box];"]

    safe_root = _dot_escape(tree.root_prompt)
    lines.append(f'  root [label="ROOT: {safe_root}", style=bold];')

    prev: str = "root"
    for turn in tree.turns:
        nid = node_id(turn)
        label = (
            f"T{turn.turn_number}\\n{turn.strategy_used}\\n"
            f"esc={turn.escalation_level}"
        )
        lines.append(f'  {nid} [label="{label}"];')
        lines.append(f"  {prev} -> {nid};")
        prev = nid

    outcome = "SUCCESS" if tree.success else "FAILED"
    lines.append(
        f'  result [label="{outcome}\\nfinal={tree.final_score:.2f}", '
        f"style=filled, fillcolor={'green' if tree.success else 'red'}];"
    )
    if tree.turns:
        lines.append(f"  {prev} -> result;")

    lines.append("}")
    return "\n".join(lines)


def _dot_escape(text: str) -> str:
    """Escape characters that are special inside Graphviz double-quoted labels.

    Args:
        text: Raw label text.

    Returns:
        The escaped text.
    """
    return (text or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


__all__ = ["render_text_tree", "render_dot"]
