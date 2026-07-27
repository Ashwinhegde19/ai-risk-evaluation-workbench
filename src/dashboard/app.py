"""Interactive Streamlit dashboard for the AI Risk Evaluation Workbench.

Run with::

    streamlit run src/dashboard/app.py

The app exposes five pages, selected from the sidebar:

1. **Overview**         -- radar chart of safety scores per dimension + KPIs.
2. **Model Comparison** -- side-by-side radar and per-(model, dimension) table.
3. **Red-Team Results** -- attack-tree visualization with drill-down.
4. **Compliance**       -- EU AI Act / NIST / ISO findings, gap analysis.
5. **Trends**           -- historical score tracking over time.

When no workbench artifacts are found on disk, the dashboard falls back to the
deterministic demo dataset in :mod:`src.dashboard.sample_data`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import streamlit as st

from src.dashboard import DashboardData
from src.dashboard import components
from src.dashboard.data_loader import discover_data
from src.dashboard.sample_data import generate_dashboard_data

_PAGE_NAMES: List[str] = [
    "Overview",
    "Model Comparison",
    "Red-Team Results",
    "Compliance",
    "Trends",
]

_RISK_TIER_COLORS = {
    "unacceptable": "#b30000",
    "high": "#e34a33",
    "limited": "#fa9fb5",
    "minimal": "#c2e699",
}


def _resolve_data(data_dir: str, use_demo: bool) -> DashboardData:
    """Load workbench data from disk or fall back to demo data.

    Args:
        data_dir: Directory to scan for JSON artifacts.
        use_demo: When ``True``, always use the sample dataset.

    Returns:
        A populated :class:`DashboardData`.
    """
    if use_demo:
        return generate_dashboard_data()
    data = discover_data(data_dir)
    if data is None or not (data.eval_results or data.attack_trees or data.reports):
        st.warning(
            f"No workbench artifacts found in '{data_dir}'. "
            "Showing sample demo data instead."
        )
        return generate_dashboard_data()
    return data


def _overall_mean(data: DashboardData) -> float:
    """Compute the overall mean safety score across all results.

    Args:
        data: The dashboard dataset.

    Returns:
        The mean score, or 0.0 when there are no results.
    """
    if not data.eval_results:
        return 0.0
    return sum(r.score for r in data.eval_results) / len(data.eval_results)


def page_overview(data: DashboardData) -> None:
    """Render the Overview page: KPIs and a safety radar chart.

    Args:
        data: The dashboard dataset.
    """
    st.header("Overview")
    st.markdown(
        "Aggregate safety posture across all evaluated models and risk "
        "dimensions."
    )

    dim_scores = components.aggregate_dimension_scores(data.eval_results)
    overall = _overall_mean(data)

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    col_kpi1.metric("Overall safety", f"{overall:.2f}")
    col_kpi2.metric("Models", str(len(data.models)))
    col_kpi3.metric("Red-team attacks", str(len(data.attack_trees)))
    successful = sum(1 for t in data.attack_trees if t.success)
    col_kpi4.metric("Attacks succeeded", f"{successful}/{len(data.attack_trees)}")

    if dim_scores:
        fig = components.radar_figure(dim_scores, title="Mean Safety by Dimension")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Scores by dimension")
        rows = [
            {"dimension": dim, "score": round(score, 4)}
            for dim, score in sorted(dim_scores.items())
        ]
        st.dataframe(rows, use_container_width=True)

        csv_data = components.to_csv(rows)
        st.download_button(
            "Export dimension scores (CSV)",
            data=csv_data,
            file_name="overview_dimension_scores.csv",
            mime="text/csv",
        )
    else:
        st.info("No evaluation results available to chart.")


def page_model_comparison(data: DashboardData) -> None:
    """Render the Model Comparison page.

    Args:
        data: The dashboard dataset.
    """
    st.header("Model Comparison")
    st.markdown("Side-by-side safety comparison across evaluated models.")

    by_model = components.aggregate_by_model(data.eval_results)
    if not by_model:
        st.info("No evaluation results available for comparison.")
        return

    selected = st.multiselect(
        "Models to compare",
        options=list(by_model.keys()),
        default=list(by_model.keys()),
    )
    if not selected:
        st.warning("Select at least one model.")
        return

    filtered = {m: by_model[m] for m in selected}
    fig = components.radar_figure_multi(
        filtered, dimensions=components.DIMENSION_ORDER, title="Model Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-dimension comparison")
    rows = components.model_comparison_rows(
        [r for r in data.eval_results if r.model_name in selected]
    )
    st.dataframe(rows, use_container_width=True)

    csv_data = components.to_csv(rows)
    st.download_button(
        "Export comparison (CSV)",
        data=csv_data,
        file_name="model_comparison.csv",
        mime="text/csv",
    )

    st.subheader("Response diffs (sample raw responses)")
    dimensions = data.dimensions or components.DIMENSION_ORDER
    dim = st.selectbox("Dimension", options=dimensions)
    for model in selected:
        with st.expander(f"{model} — {dim}"):
            sample = next(
                (
                    r.raw_response
                    for r in data.eval_results
                    if r.model_name == model and r.dimension == dim
                ),
                "(no response recorded)",
            )
            st.write(sample)


def page_redteam(data: DashboardData) -> None:
    """Render the Red-Team Results page with attack-tree drill-down.

    Args:
        data: The dashboard dataset.
    """
    st.header("Red-Team Results")
    st.markdown("Multi-turn attack trees and their outcomes.")

    if not data.attack_trees:
        st.info("No attack trees available.")
        return

    labels = [
        f"#{i + 1} {t.root_prompt[:40]} "
        f"({'SUCCESS' if t.success else 'FAILED'}, score={t.final_score:.2f})"
        for i, t in enumerate(data.attack_trees)
    ]
    index = st.selectbox("Select an attack tree", options=range(len(labels)), format_func=lambda i: labels[i])  # type: ignore[arg-type]
    tree = data.attack_trees[index]

    col_a, col_b = st.columns(2)
    col_a.metric("Final score", f"{tree.final_score:.2f}")
    col_b.metric("Outcome", "SUCCESS" if tree.success else "FAILED")

    st.markdown(f"**Strategy chain:** {' → '.join(tree.strategy_chain) or '(none)'}")

    st.subheader("Attack tree")
    st.graphviz_chart(components.attack_tree_dot(tree))

    st.subheader("Turn-by-turn drill-down")
    for turn in tree.turns:
        with st.expander(f"Turn {turn.turn_number} — {turn.strategy_used}"):
            st.write(f"**Escalation level:** {turn.escalation_level}")
            st.write(f"**Attacker prompt:** {turn.attacker_prompt}")
            st.write(f"**Model response:** {turn.model_response}")

    dot = components.attack_tree_dot(tree)
    st.download_button(
        "Export attack tree (DOT)",
        data=dot,
        file_name="attack_tree.dot",
        mime="text/vnd.graphviz",
    )


def page_compliance(data: DashboardData) -> None:
    """Render the Compliance page: findings, gap analysis, exports.

    Args:
        data: The dashboard dataset.
    """
    st.header("Compliance")
    st.markdown("EU AI Act, NIST AI RMF and ISO 42001 findings and gaps.")

    if not data.reports:
        st.info("No compliance reports available.")
        return

    model = st.selectbox("Model", options=list(data.reports.keys()))
    report = data.reports[model]

    st.metric("Overall risk tier", report.overall_risk_tier.value)
    st.markdown(f"**Report generated:** {report.timestamp.isoformat()}")

    rows = components.finding_rows(report.findings)
    if rows:
        st.dataframe(rows, use_container_width=True)
        csv_data = components.to_csv(rows)
        st.download_button(
            "Export findings (CSV)",
            data=csv_data,
            file_name=f"compliance_findings_{model}.csv",
            mime="text/csv",
        )
    else:
        st.success("No compliance findings (all dimensions within tolerance).")

    st.subheader("Gap analysis & recommendations")
    if report.gaps:
        for gap in report.gaps:
            st.markdown(f"- {gap}")
    else:
        st.success("No gaps identified.")

    json_data = components.to_json(report)
    st.download_button(
        "Export report (JSON)",
        data=json_data,
        file_name=f"compliance_report_{model}.json",
        mime="application/json",
    )
    pdf_bytes = components.compliance_pdf_bytes(report)
    st.download_button(
        "Export report (PDF)",
        data=pdf_bytes,
        file_name=f"compliance_report_{model}.pdf",
        mime="application/pdf",
    )


def page_trends(data: DashboardData) -> None:
    """Render the Trends page: historical score tracking over time.

    Args:
        data: The dashboard dataset.
    """
    st.header("Trends")
    st.markdown("Historical safety score tracking across evaluation runs.")

    if not data.history:
        st.info("No historical score data available.")
        return

    models = sorted({run.get("model", "unknown") for run in data.history})
    selected = st.multiselect(
        "Models", options=models, default=models
    )
    if not selected:
        st.warning("Select at least one model.")
        return

    fig = components.trend_figure(data.history, models=selected)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest run scores")
    latest: dict = {}
    for run in data.history:
        if run.get("model") in selected:
            latest[run["model"]] = run
    rows = []
    for model, run in latest.items():
        for dim, score in run.get("scores", {}).items():
            rows.append({"model": model, "dimension": dim, "score": round(score, 4)})
    st.dataframe(rows, use_container_width=True)

    csv_data = components.to_csv(rows)
    st.download_button(
        "Export latest scores (CSV)",
        data=csv_data,
        file_name="trend_latest_scores.csv",
        mime="text/csv",
    )


def main(
    page: Optional[str] = None,
    data_dir: Optional[str] = None,
    use_demo: Optional[bool] = None,
) -> None:
    """Entry point for the Streamlit dashboard.

    Args:
        page: Page to render. When ``None`` the sidebar radio selection is used.
        data_dir: Directory to scan for artifacts. Defaults to ``"results"``.
        use_demo: Force demo data when ``True``. When ``None`` the sidebar
            checkbox selection is used.
    """
    st.set_page_config(page_title="AI Risk Evaluation Workbench", layout="wide")

    st.sidebar.title("AI Risk Evaluation Workbench")
    resolved_dir = data_dir if data_dir is not None else "results"
    use_demo_flag = (
        use_demo if use_demo is not None else st.sidebar.checkbox("Use demo data", value=False)
    )
    chosen = page if page is not None else st.sidebar.radio("Page", _PAGE_NAMES)

    data = _resolve_data(resolved_dir, use_demo_flag)
    st.sidebar.caption(f"Data source: {data.source}")

    if chosen == "Overview":
        page_overview(data)
    elif chosen == "Model Comparison":
        page_model_comparison(data)
    elif chosen == "Red-Team Results":
        page_redteam(data)
    elif chosen == "Compliance":
        page_compliance(data)
    elif chosen == "Trends":
        page_trends(data)
    else:
        st.error(f"Unknown page: {chosen}")


if __name__ == "__main__":
    # Streamlit ignores argv; this guard only enables ``python app.py`` sanity.
    main(use_demo=True)
