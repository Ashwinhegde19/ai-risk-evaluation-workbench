"""Pure, framework-agnostic helpers for the Streamlit dashboard.

Every function here is side-effect free and depends only on the workbench
Pydantic models plus the standard library (and an optional, lazily-imported
``plotly`` for figure builders). Keeping this module free of ``streamlit`` and
top-level ``plotly`` imports makes the logic cheaply unit-testable without a
running Streamlit runtime.
"""

from __future__ import annotations

import csv
import io
import json
import re
from statistics import mean
from typing import Any, Dict, List, Sequence

from src.core.models import (
    AttackTree,
    ComplianceFinding,
    ComplianceReport,
    EvalResult,
    Severity,
)
from pathlib import Path

from src.redteam.visualize import render_dot


# Canonical ordering of the seven risk dimensions, used for stable axes/columns.
DIMENSION_ORDER: List[str] = [
    "hallucination",
    "bias",
    "toxicity",
    "jailbreak_resistance",
    "privacy",
    "ip_theft",
    "harmful_content",
]


def severity_from_score(score: float) -> Severity:
    """Map a normalized ``[0, 1]`` safety score to a :class:`Severity`.

    Higher scores mean safer behaviour, so lower scores map to higher
    severity.

    Args:
        score: Normalized safety score in ``[0, 1]``.

    Returns:
        The corresponding :class:`Severity` level.
    """
    if score >= 0.85:
        return Severity.LOW
    if score >= 0.70:
        return Severity.MEDIUM
    if score >= 0.50:
        return Severity.HIGH
    return Severity.CRITICAL


def aggregate_dimension_scores(results: Sequence[EvalResult]) -> Dict[str, float]:
    """Average the per-dimension scores across all supplied results.

    Args:
        results: Evaluation results that may span several models.

    Returns:
        Mapping of dimension name to the mean score for that dimension. An
        empty input yields an empty mapping.
    """
    by_dim: Dict[str, List[float]] = {}
    for result in results:
        by_dim.setdefault(result.dimension, []).append(result.score)
    return {dim: mean(scores) for dim, scores in by_dim.items()}


def aggregate_by_model(
    results: Sequence[EvalResult],
) -> Dict[str, Dict[str, float]]:
    """Average scores per dimension, grouped by model.

    Args:
        results: Evaluation results spanning one or more models.

    Returns:
        A two-level mapping ``model -> dimension -> mean score``.
    """
    by_model: Dict[str, Dict[str, List[float]]] = {}
    for result in results:
        by_model.setdefault(result.model_name, {}).setdefault(
            result.dimension, []
        ).append(result.score)
    return {
        model: {dim: mean(scores) for dim, scores in dims.items()}
        for model, dims in by_model.items()
    }


def model_comparison_rows(results: Sequence[EvalResult]) -> List[Dict[str, Any]]:
    """Flatten evaluation results into per-(model, dimension) table rows.

    Args:
        results: Evaluation results spanning one or more models.

    Returns:
        A list of row dictionaries with keys ``model``, ``dimension``,
        ``score`` and ``severity``.
    """
    rows: List[Dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "model": result.model_name,
                "dimension": result.dimension,
                "score": round(result.score, 4),
                "severity": severity_from_score(result.score).value,
            }
        )
    rows.sort(key=lambda r: (r["model"], r["dimension"]))
    return rows


def finding_rows(findings: Sequence[ComplianceFinding]) -> List[Dict[str, str]]:
    """Convert compliance findings into flat, display-friendly rows.

    Args:
        findings: Compliance findings to tabulate.

    Returns:
        A list of row dictionaries with stringified fields suitable for a
        Streamlit table or CSV export.
    """
    rows: List[Dict[str, str]] = []
    for finding in findings:
        rows.append(
            {
                "framework": finding.framework.value,
                "control_id": finding.control_id,
                "risk_tier": finding.risk_tier.value,
                "severity": finding.severity.value,
                "description": finding.description,
                "evidence": finding.evidence,
            }
        )
    rows.sort(key=lambda r: (r["framework"], r["control_id"]))
    return rows


def attack_tree_dot(tree: AttackTree) -> str:
    """Render an attack tree to a Graphviz DOT string.

    Args:
        tree: The attack tree to render.

    Returns:
        A ``digraph`` DOT document produced by
        :func:`src.redteam.visualize.render_dot`.
    """
    return render_dot(tree)


def to_csv(rows: Sequence[Dict[str, Any]]) -> str:
    """Serialize a list of row dicts to a CSV string.

    Args:
        rows: Homogeneous (or heterogeneous) dictionaries. Column order is the
            stable union of all keys, sorted alphabetically.

    Returns:
        A CSV document as a string. Returns an empty string for an empty input.
    """
    if not rows:
        return ""
    columns: List[str] = sorted({key for row in rows for key in row.keys()})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def to_json(obj: Any) -> str:
    """Serialize a value to a pretty-printed JSON string.

    Pydantic models are serialized via ``model_dump_json``; everything else
    falls back to the standard ``json`` encoder.

    Args:
        obj: The value to serialize (Pydantic model, list, dict, ...).

    Returns:
        A pretty-printed JSON string.
    """
    dump = getattr(obj, "model_dump_json", None)
    if callable(dump):
        return dump(indent=2)
    return json.dumps(obj, indent=2, default=str)


def radar_figure(
    scores_by_dim: Dict[str, float],
    title: str = "Safety Radar",
) -> "Any":
    """Build a single-trace Plotly radar (scatterpolar) figure.

    ``plotly`` is imported lazily so this module stays importable in
    environments without Plotly installed.

    Args:
        scores_by_dim: Mapping of dimension name to score in ``[0, 1]``.
        title: Chart title.

    Returns:
        A :class:`plotly.graph_objects.Figure`.

    Raises:
        ImportError: If ``plotly`` is not installed.
    """
    import plotly.graph_objects as go

    dims = DIMENSION_ORDER if all(d in scores_by_dim for d in DIMENSION_ORDER) else list(scores_by_dim.keys())
    values = [scores_by_dim.get(dim, 0.0) for dim in dims]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]] if values else [],
            theta=dims + [dims[0]] if dims else [],
            fill="toself",
            name=title,
        )
    )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
    )
    return fig


def radar_figure_multi(
    models_scores: Dict[str, Dict[str, float]],
    dimensions: Sequence[str] | None = None,
    title: str = "Model Comparison",
) -> "Any":
    """Build a multi-trace Plotly radar comparing several models.

    ``plotly`` is imported lazily so this module stays importable without it.

    Args:
        models_scores: Mapping of model name to per-dimension scores.
        dimensions: Dimension ordering for the axes. Defaults to
            :data:`DIMENSION_ORDER`.
        title: Chart title.

    Returns:
        A :class:`plotly.graph_objects.Figure`.

    Raises:
        ImportError: If ``plotly`` is not installed.
    """
    import plotly.graph_objects as go

    dims = list(dimensions) if dimensions is not None else DIMENSION_ORDER
    fig = go.Figure()
    for model, scores in models_scores.items():
        values = [scores.get(dim, 0.0) for dim in dims]
        closes = values + [values[0]]
        thetas = list(dims) + [dims[0]] if dims else []
        fig.add_trace(
            go.Scatterpolar(r=closes, theta=thetas, fill="toself", name=model)
        )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    )
    return fig


def trend_figure(
    history: Sequence[dict],
    models: Sequence[str] | None = None,
) -> "Any":
    """Build a line chart of mean safety score over time, per model.

    Each history record must contain ``timestamp`` (ISO-8601), ``model`` and
    ``scores`` (dimension -> score). The per-run mean across dimensions is
    plotted. ``plotly`` is imported lazily.

    Args:
        history: Time-ordered run records.
        models: Optional subset of models to plot. Defaults to all present.

    Returns:
        A :class:`plotly.graph_objects.Figure`.

    Raises:
        ImportError: If ``plotly`` is not installed.
    """
    import plotly.graph_objects as go

    series: Dict[str, List[dict]] = {}
    for run in history:
        model = run.get("model", "unknown")
        scores = run.get("scores", {})
        if not scores:
            continue
        series.setdefault(model, []).append(
            {"timestamp": run.get("timestamp", ""), "value": mean(scores.values())}
        )

    fig = go.Figure()
    for model, points in series.items():
        if models is not None and model not in models:
            continue
        points = sorted(points, key=lambda p: str(p["timestamp"]))
        fig.add_trace(
            go.Scatter(
                x=[p["timestamp"] for p in points],
                y=[p["value"] for p in points],
                mode="lines+markers",
                name=model,
            )
        )
    fig.update_layout(
        title="Safety Score Trends Over Time",
        xaxis_title="Run timestamp",
        yaxis_title="Mean safety score",
        yaxis=dict(range=[0, 1]),
    )
    return fig


def compliance_pdf_bytes(report: ComplianceReport) -> bytes:
    """Render a compliance report to PDF bytes in memory.

    Delegates to :func:`export_compliance_pdf`, writing to a temporary file
    and returning its contents so callers (e.g. a Streamlit download button)
    do not need to manage a destination path.

    Args:
        report: The compliance report to render.

    Returns:
        The rendered PDF document as bytes.
    """
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        export_compliance_pdf(report, tmp_path)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def export_compliance_pdf(report: ComplianceReport, path: str | object) -> str:
    """Write a compliance report to a dependency-free PDF.

    Delegates to the workbench's :func:`src.reports.compliance.write_pdf_report`,
    which uses the bundled minimal PDF writer (no ``reportlab`` dependency).

    Args:
        report: The compliance report to render.
        path: Destination ``.pdf`` path (str or :class:`pathlib.Path`).

    Returns:
        The path the PDF was written to (as a string).
    """
    from pathlib import Path

    from src.reports.compliance import write_pdf_report

    out = Path(path)
    write_pdf_report(report, out)
    return str(out)


def load_redteam_runs(data_dir: str | Path) -> List[dict]:
    """Discover and load all ``redteam_findings*.json`` artifacts.

    Files are sorted by filename (which contains timestamps) so the
    oldest run comes first. Each entry in the returned list carries
    a ``label`` key derived from the filename for display purposes.

    Args:
        data_dir: Directory to scan for redteam findings JSON files.

    Returns:
        A list of parsed redteam findings dicts, oldest first.
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        return []

    files = sorted(directory.glob("redteam_findings*.json"))
    runs: List[dict] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        # Build a human-readable label from the filename.
        # e.g. redteam_findings_20260727T181758Z.json -> 2026-07-27 18:17 UTC
        stem = f.stem  # "redteam_findings_20260727T181758Z"
        label = stem.replace("redteam_findings_", "")
        # Try to format the timestamp.
        ts_match = re.match(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z?", label)
        if ts_match:
            label = (
                f"{ts_match.group(1)}-{ts_match.group(2)}-{ts_match.group(3)} "
                f"{ts_match.group(4)}:{ts_match.group(5)} UTC"
            )
        data["label"] = label
        runs.append(data)
    return runs


def run_comparison_figure(
    runs: List[dict],
    models: List[str],
) -> "Any":
    """Build a Plotly line chart of break rates over red-team runs.

    ``plotly`` is imported lazily so this module stays importable
    without it installed.

    Args:
        runs: List of redteam findings dicts (oldest first), as
            returned by :func:`load_redteam_runs`.
        models: Subset of model names to plot.

    Returns:
        A :class:`plotly.graph_objects.Figure` with one trace per model.

    Raises:
        ImportError: If ``plotly`` is not installed.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    colors = ["#C8402A", "#7C9A8D", "#E8C24A", "#6E8A7E", "#A99E8C", "#D9B45C"]

    for idx, model in enumerate(models):
        timestamps: List[str] = []
        rates: List[float] = []
        for run in runs:
            stats = run.get("per_model", {}).get(model)
            if stats is not None:
                timestamps.append(run.get("label", ""))
                rates.append(stats["rate"])
        if not timestamps:
            continue
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=[r * 100 for r in rates],
                mode="lines+markers",
                name=model,
                line=dict(color=color, width=3),
                marker=dict(size=10),
                hovertemplate=(
                    "%{x}<br>" + model + ": %{y:.1f}% break rate<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Adversarial break rate over red-team runs",
        xaxis_title="Run",
        yaxis_title="Break rate (%)",
        yaxis=dict(range=[0, 105]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
    )
    return fig


__all__ = [
    "DIMENSION_ORDER",
    "severity_from_score",
    "aggregate_dimension_scores",
    "aggregate_by_model",
    "model_comparison_rows",
    "finding_rows",
    "attack_tree_dot",
    "to_csv",
    "to_json",
    "radar_figure",
    "radar_figure_multi",
    "trend_figure",
    "compliance_pdf_bytes",
    "export_compliance_pdf",
    "load_redteam_runs",
    "run_comparison_figure",
]
