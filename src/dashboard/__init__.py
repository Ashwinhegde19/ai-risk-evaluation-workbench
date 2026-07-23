"""Streamlit dashboard for the AI Risk Evaluation Workbench.

This package exposes a five-page interactive dashboard (see
:mod:`src.dashboard.app`) together with the supporting building blocks:

* :mod:`src.dashboard.sample_data` -- deterministic demo dataset.
* :mod:`src.dashboard.data_loader` -- load workbench JSON artifacts.
* :mod:`src.dashboard.components` -- pure, testable rendering helpers.

The :class:`DashboardData` container defined here is the single object that
every page consumes, so the app can be driven equally from on-disk artifacts
or from in-memory sample data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.core.models import AttackTree, ComplianceReport, EvalResult


@dataclass
class DashboardData:
    """Aggregated evaluation artifacts consumed by every dashboard page.

    Attributes:
        eval_results: Flat list of per-model, per-dimension evaluation results.
        attack_trees: Red-team attack trees (any model).
        reports: Compliance reports keyed by model name.
        history: Time-ordered run records for trend analysis. Each record is a
            mapping with ``timestamp`` (ISO-8601), ``model`` (str) and
            ``scores`` (mapping of dimension -> score in ``[0, 1]``).
        source: Human-readable provenance label (e.g. a directory path or
            ``"sample"`` for generated demo data).
    """

    eval_results: List[EvalResult] = field(default_factory=list)
    attack_trees: List[AttackTree] = field(default_factory=list)
    reports: Dict[str, ComplianceReport] = field(default_factory=dict)
    history: List[dict] = field(default_factory=list)
    source: str = "unknown"

    @property
    def models(self) -> List[str]:
        """Return the sorted, de-duplicated list of evaluated model names."""
        seen: set[str] = set()
        for result in self.eval_results:
            seen.add(result.model_name)
        for report in self.reports.values():
            seen.add(report.model_name)
        return sorted(seen)

    @property
    def dimensions(self) -> List[str]:
        """Return the sorted, de-duplicated list of scored dimension names."""
        seen: set[str] = set()
        for result in self.eval_results:
            seen.add(result.dimension)
        for run in self.history:
            seen.update(run.get("scores", {}).keys())
        return sorted(seen)


__all__ = ["DashboardData"]
