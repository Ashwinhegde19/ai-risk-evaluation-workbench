"""Loader for on-disk workbench artifacts consumed by the dashboard.

The dashboard expects the workbench's serialized JSON outputs (produced by the
report generators and pipeline) to live in a single directory. This module
discovers those files and parses them into the :class:`DashboardData` container
used by every page.

Canonical filenames:

* ``eval_results.json``   -- a JSON list of :class:`EvalResult` objects.
* ``attack_trees.json``   -- a JSON list of :class:`AttackTree` objects.
* ``compliance_report.json`` -- a single :class:`ComplianceReport` object.
* ``scores_history.json``  -- a JSON list of run records (see
  :class:`src.dashboard.DashboardData`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from src.core.models import AttackTree, ComplianceReport, EvalResult

from src.dashboard import DashboardData

_EVAL_RESULTS_FILE = "eval_results.json"
_ATTACK_TREES_FILE = "attack_trees.json"
_COMPLIANCE_REPORT_FILE = "compliance_report.json"
_HISTORY_FILE = "scores_history.json"


def load_eval_results(path: Path) -> Optional[List[EvalResult]]:
    """Parse an ``eval_results.json`` file.

    Args:
        path: Path to the JSON file.

    Returns:
        A list of :class:`EvalResult`, or ``None`` if the file does not exist.
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EvalResult.model_validate(item) for item in data]


def load_attack_trees(path: Path) -> Optional[List[AttackTree]]:
    """Parse an ``attack_trees.json`` file.

    Args:
        path: Path to the JSON file.

    Returns:
        A list of :class:`AttackTree`, or ``None`` if the file does not exist.
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [AttackTree.model_validate(item) for item in data]


def load_compliance_report(path: Path) -> Optional[ComplianceReport]:
    """Parse a ``compliance_report.json`` file.

    Args:
        path: Path to the JSON file.

    Returns:
        A :class:`ComplianceReport`, or ``None`` if the file does not exist.
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ComplianceReport.model_validate(data)


def load_scores_history(path: Path) -> Optional[List[dict]]:
    """Parse a ``scores_history.json`` file.

    Args:
        path: Path to the JSON file.

    Returns:
        A list of run-record dicts, or ``None`` if the file does not exist.
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data)


def discover_data(data_dir: str | Path) -> Optional[DashboardData]:
    """Scan ``data_dir`` for workbench artifacts and assemble a bundle.

    Missing optional files simply yield empty collections; only when *no*
    recognized artifact is present does this return ``None`` (so callers can
    fall back to sample data).

    Args:
        data_dir: Directory to scan for the canonical JSON files.

    Returns:
        A :class:`DashboardData` if at least one artifact was found, else
        ``None``.
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        return None

    eval_results = load_eval_results(directory / _EVAL_RESULTS_FILE)
    attack_trees = load_attack_trees(directory / _ATTACK_TREES_FILE)
    report = load_compliance_report(directory / _COMPLIANCE_REPORT_FILE)
    history = load_scores_history(directory / _HISTORY_FILE)

    found_any = any(
        x is not None for x in (eval_results, attack_trees, report, history)
    )
    if not found_any:
        return None

    reports = {}
    if report is not None:
        reports[report.model_name] = report

    return DashboardData(
        eval_results=eval_results or [],
        attack_trees=attack_trees or [],
        reports=reports,
        history=history or [],
        source=str(directory),
    )


__all__ = [
    "load_eval_results",
    "load_attack_trees",
    "load_compliance_report",
    "load_scores_history",
    "discover_data",
]
