"""Pre-generated demo artifacts for the AI Risk Evaluation Workbench.

This package produces a small, fully deterministic set of evaluation artifacts
so that reviewers can explore the workbench's outputs without configuring any
provider credentials or running a live evaluation:

* per-model evaluation results (JSON),
* a sample multi-framework compliance report (JSON + PDF),
* sample red-team attack trees (JSON + text rendering + Graphviz DOT),
* a manifest describing every generated artifact.

The entry point is :func:`src.demo.generate.generate_demo`, which writes the
artifacts into ``data/demo/`` by default. A CLI is exposed via
``python -m src.demo.generate``.
"""

from src.demo.generate import (
    DEFAULT_DEMO_DIR,
    DEMO_MODEL,
    DEMO_TIMESTAMP,
    build_demo_attack_trees,
    build_demo_eval_results,
    build_demo_report,
    generate_demo,
)

__all__ = [
    "DEFAULT_DEMO_DIR",
    "DEMO_MODEL",
    "DEMO_TIMESTAMP",
    "build_demo_attack_trees",
    "build_demo_eval_results",
    "build_demo_report",
    "generate_demo",
]
