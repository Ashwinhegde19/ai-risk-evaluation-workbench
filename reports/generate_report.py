"""Generate a one-page PDF evaluation report."""

from __future__ import annotations

import argparse
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
    df = pd.read_csv(results_path)
    if df.empty:
        raise ValueError("Evaluation results are empty.")

    summary = summarize(df)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("AI Assistant Risk Evaluation Report", fontsize=18, fontweight="bold")

    ax_pass = fig.add_subplot(2, 2, 1)
    ax_risk = fig.add_subplot(2, 2, 2)
    ax_latency = fig.add_subplot(2, 2, 3)
    ax_table = fig.add_subplot(2, 2, 4)

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
    fig.text(0.08, 0.03, recommendation, fontsize=10, wrap=True)
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
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
            avg_risk_score=("risk_score", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
            cost_per_1k_requests_usd=("cost_per_1k_requests_usd", "mean"),
        )
        .reset_index()
    )


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
