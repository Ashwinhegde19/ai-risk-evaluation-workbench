"""Multi-turn red-team agent package.

Public surface for running adaptive adversarial attacks against a target model
and visualizing the resulting attack trees.
"""

from __future__ import annotations

from .agent import RedTeamAgent, RedTeamConfig
from .visualize import render_dot, render_text_tree

__all__ = [
    "RedTeamAgent",
    "RedTeamConfig",
    "render_text_tree",
    "render_dot",
]
