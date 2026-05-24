"""Generate a one-page PDF evaluation report."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = ROOT / "results" / "eval_results.csv"
DEFAULT_REPORT_PATH = ROOT / "reports" / "evaluation_report.pdf"


def generate_report(
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    output_path: str | Path = DEFAULT_REPORT_PATH,
) -> Path:
    df = ensure_report_columns(pd.read_csv(results_path))
    if df.empty:
        raise ValueError("Evaluation results are empty.")

    summary = summarize(df)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("AI Assistant Risk Evaluation Report", fontsize=18, fontweight="bold")

    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.9, 1.1])
    ax_pass = fig.add_subplot(grid[0, 0])
    ax_risk = fig.add_subplot(grid[0, 1])
    ax_latency = fig.add_subplot(grid[0, 2])
    ax_table = fig.add_subplot(grid[1, :2])
    ax_recommendation = fig.add_subplot(grid[1, 2])
    ax_cases = fig.add_subplot(grid[2, :])

    plot_bar(ax_pass, summary, "model_label", "pass_rate", "Pass Rate", "Rate")
    risk_cols = ["hallucination_rate", "unsafe_rate", "bias_risk_rate"]
    summary.set_index("model_label")[risk_cols].plot(kind="bar", ax=ax_risk)
    ax_risk.set_title("Risk Rates")
    ax_risk.set_ylabel("Rate")
    ax_risk.set_xlabel("")
    ax_risk.legend(["Hallucination", "Unsafe", "Bias"], fontsize=8)
    ax_risk.tick_params(axis="x", rotation=15)

    plot_bar(ax_latency, summary, "model_label", "avg_latency_ms", "Average Latency", "ms")

    ax_table.axis("off")
    table_data = summary[
        [
            "model_label",
            "prompts",
            "pass_rate",
            "avg_risk_score",
            "avg_latency_ms",
            "cost_per_1k_requests_usd",
        ]
    ].copy()
    table_data["pass_rate"] = table_data["pass_rate"].map(lambda value: f"{value:.0%}")
    table_data["avg_risk_score"] = table_data["avg_risk_score"].map(lambda value: f"{value:.1f}")
    table_data["avg_latency_ms"] = table_data["avg_latency_ms"].map(lambda value: f"{value:.0f}")
    table_data["cost_per_1k_requests_usd"] = table_data["cost_per_1k_requests_usd"].map(
        lambda value: f"${value:.2f}"
    )
    ax_table.table(
        cellText=table_data.values,
        colLabels=["Model", "Prompts", "Pass", "Risk", "Latency", "Cost/1K"],
        loc="center",
        cellLoc="center",
    )
    ax_table.set_title("Deployment Snapshot")

    recommendation = build_recommendation(summary)
    ax_recommendation.axis("off")
    ax_recommendation.set_title("Recommendation")
    ax_recommendation.text(
        0,
        0.92,
        format_report_context(df, recommendation),
        fontsize=8.2,
        va="top",
    )

    render_notable_cases(ax_cases, df)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(output, format="pdf")
    plt.close(fig)
    return output


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("model_label")
        .agg(
            prompts=("prompt_id", "count"),
            pass_rate=("passed", "mean"),
            hallucination_rate=("hallucination_flag", "mean"),
            unsafe_rate=("unsafe_flag", "mean"),
            bias_risk_rate=("bias_risk", "mean"),
            over_refusal_rate=("over_refusal", "mean"),
            under_refusal_rate=("under_refusal", "mean"),
            avg_risk_score=("risk_score", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
            cost_per_1k_requests_usd=("cost_per_1k_requests_usd", "mean"),
        )
        .reset_index()
    )


def ensure_report_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill newer report columns when reading older eval CSV files."""

    defaults = {
        "under_refusal": 0,
        "behavior_label": "needs_review",
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
    return df


def plot_bar(ax, df: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str) -> None:
    ax.bar(df[x_col], df[y_col])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)


def build_recommendation(summary: pd.DataFrame) -> str:
    best = summary.sort_values(["avg_risk_score", "pass_rate"], ascending=[True, False]).iloc[0]
    return (
        "Recommendation: use the lowest-risk model for customer-facing or liability-sensitive "
        f"workflows. In this run, {best['model_label']} had the strongest combined risk score. "
        "The OSS model remains attractive for cost control and deployment ownership, but should "
        "be paired with guardrails, monitoring, and targeted evals before high-risk use."
    )


def format_report_context(df: pd.DataFrame, recommendation: str) -> str:
    metadata = run_metadata(df)
    lines = ["Run Metadata"]
    if metadata:
        lines.extend(metadata)
    behavior_labels = behavior_label_summary(df)
    if behavior_labels:
        lines.append("")
        lines.extend(behavior_labels)
    lines.append("")
    lines.extend(textwrap.wrap(recommendation, width=46))
    return "\n".join(lines)


def behavior_label_summary(df: pd.DataFrame) -> list[str]:
    if "behavior_label" not in df.columns:
        return []

    lines = ["Behavior Labels"]
    for model_label, group in df.groupby("model_label"):
        counts = group["behavior_label"].value_counts().head(4)
        summary = ", ".join(f"{label}={count}" for label, count in counts.items())
        lines.append(f"{shorten(str(model_label), 18)}: {shorten(summary, 44)}")
    return lines


def run_metadata(df: pd.DataFrame) -> list[str]:
    if "run_id" not in df.columns:
        return []

    first = df.iloc[0]
    models = ", ".join(sorted(str(model) for model in df["model_name"].dropna().unique()))
    return [
        f"Run: {first.get('run_id', 'unknown')}",
        f"Prompts: {df['prompt_id'].nunique()}",
        f"Temp: {first.get('temperature', 'n/a')}",
        f"Max tokens: {first.get('max_tokens', 'n/a')}",
        f"Blocking: {first.get('block_unsafe_inputs', 'n/a')}",
        f"Models: {shorten(models, 54)}",
    ]


def render_notable_cases(ax, df: pd.DataFrame) -> None:
    ax.axis("off")
    ax.set_title("Notable Eval Cases", loc="left")
    cases = select_notable_cases(df)
    if not cases:
        ax.text(0, 0.85, "No notable failures found in this run.", fontsize=9, va="top")
        return

    y = 0.88
    for index, case in enumerate(cases, start=1):
        text = format_case(index, case)
        ax.text(0, y, text, fontsize=8.2, va="top", family="monospace")
        y -= 0.43


def select_notable_cases(df: pd.DataFrame, limit: int = 2) -> list[dict[str, str]]:
    paired = []
    for prompt_id, group in df.groupby("prompt_id"):
        if group["model_label"].nunique() < 2:
            continue
        by_model = {row["model_label"]: row for _, row in group.iterrows()}
        oss = by_model.get("Open Source Assistant")
        frontier = by_model.get("Frontier Assistant")
        risks = group["risk_score"].astype(float)
        passes = group["passed"].astype(int)
        oss_risk = float(oss["risk_score"]) if oss is not None else 0.0
        frontier_risk = float(frontier["risk_score"]) if frontier is not None else 0.0
        oss_passed = int(oss["passed"]) if oss is not None else 1
        frontier_passed = int(frontier["passed"]) if frontier is not None else 0
        paired.append(
            {
                "prompt_id": prompt_id,
                "category": str(group.iloc[0]["category"]),
                "prompt": str(group.iloc[0]["prompt"]),
                "rows": by_model,
                "frontier_win": int(oss_passed == 0 and frontier_passed == 1),
                "oss_risk_gap": oss_risk - frontier_risk,
                "risk_gap": float(risks.max() - risks.min()),
                "max_risk": float(risks.max()),
                "pass_gap": int(passes.max() != passes.min()),
            }
        )
    return sorted(
        paired,
        key=lambda case: (
            case["frontier_win"],
            case["oss_risk_gap"],
            case["pass_gap"],
            case["risk_gap"],
            case["max_risk"],
        ),
        reverse=True,
    )[:limit]


def format_case(index: int, case: dict[str, str]) -> str:
    oss = case["rows"].get("Open Source Assistant")
    frontier = case["rows"].get("Frontier Assistant")
    lines = [
        f"Case {index}: {case['category']} ({case['prompt_id']})",
        f"Prompt: {shorten(case['prompt'], 132)}",
    ]
    if oss is not None:
        lines.append(
            "OSS: "
            f"pass={int(oss['passed'])}, risk={int(oss['risk_score'])}; "
            f"label={oss.get('behavior_label', 'n/a')}; "
            f"{shorten(str(oss['response']), 150)}"
        )
    if frontier is not None:
        lines.append(
            "Frontier: "
            f"pass={int(frontier['passed'])}, risk={int(frontier['risk_score'])}; "
            f"label={frontier.get('behavior_label', 'n/a')}; "
            f"{shorten(str(frontier['response']), 150)}"
        )
    wrapped_lines = []
    for line in lines:
        wrapped_lines.extend(textwrap.wrap(line, width=152, subsequent_indent="  "))
    return "\n".join(wrapped_lines)


def shorten(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3]}..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the one-page evaluation report.")
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_REPORT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = generate_report(args.results_path, args.output_path)
    print(f"Wrote report to {output}")


if __name__ == "__main__":
    main()
