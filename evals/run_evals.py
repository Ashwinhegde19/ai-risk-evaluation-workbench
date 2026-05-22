"""Run the assistant risk evaluation suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from assistant import RiskAwareAssistant, SlidingWindowMemory
from evals.judge import judge_response
from evals.scoring import score_response
from models import HuggingFaceOSSClient, OpenAIModelClient

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = ROOT / "evals" / "prompts.json"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "eval_results.csv"

MODEL_LABELS = {
    "Open Source Assistant": "oss",
    "Frontier Assistant": "frontier",
}


def load_prompts(prompt_path: str | Path = DEFAULT_PROMPT_PATH) -> list[dict[str, Any]]:
    with Path(prompt_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_client(model_label: str):
    model_kind = MODEL_LABELS[model_label]
    if model_kind == "oss":
        return HuggingFaceOSSClient()
    if model_kind == "frontier":
        return OpenAIModelClient()
    raise ValueError(f"Unknown model label: {model_label}")


def run_evaluation(
    *,
    model_labels: list[str] | None = None,
    prompt_path: str | Path = DEFAULT_PROMPT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    limit: int | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    block_unsafe_inputs: bool = False,
    use_judge: bool = False,
) -> pd.DataFrame:
    load_dotenv()
    prompts = load_prompts(prompt_path)
    if limit:
        prompts = prompts[:limit]

    labels = model_labels or list(MODEL_LABELS.keys())
    rows: list[dict[str, Any]] = []

    for model_label in labels:
        client = build_client(model_label)
        for prompt in prompts:
            assistant = RiskAwareAssistant(
                client,
                memory=SlidingWindowMemory(max_messages=0),
                block_unsafe_inputs=block_unsafe_inputs,
            )
            result = assistant.respond(
                prompt["prompt"],
                temperature=temperature,
                max_tokens=max_tokens,
                log=False,
            )
            score = score_response(
                prompt,
                result.response,
                result.input_check,
                result.output_check,
                result.metadata,
            )
            row = {
                "model_label": model_label,
                "model_name": result.model_name,
                "prompt_id": prompt["id"],
                "category": prompt["category"],
                "risk_type": prompt["risk_type"],
                "prompt": prompt["prompt"],
                "expected_behavior": prompt["expected_behavior"],
                "response": result.response,
                "latency_ms": result.latency_ms,
                "input_safety": result.input_check.label,
                "input_categories": ",".join(result.input_check.categories),
                "output_safety": result.output_check.label,
                "output_categories": ",".join(result.output_check.categories),
                "blocked_before_model": bool(result.metadata.get("blocked_before_model")),
                "cost_per_1k_requests_usd": estimate_cost_per_1k(model_label),
                **score,
            }
            if use_judge:
                row.update(judge_response(prompt, result.response))
            rows.append(row)

    df = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def estimate_cost_per_1k(model_label: str) -> float:
    if model_label == "Open Source Assistant":
        return float(os.getenv("OSS_COST_PER_1K_REQUESTS_USD", "0.00"))
    return float(os.getenv("FRONTIER_COST_PER_1K_REQUESTS_USD", "0.05"))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("model_label")
        .agg(
            prompts=("prompt_id", "count"),
            pass_rate=("passed", "mean"),
            hallucination_rate=("hallucination_flag", "mean"),
            unsafe_rate=("unsafe_flag", "mean"),
            correct_refusal_rate=("correct_refusal", "mean"),
            bias_risk_rate=("bias_risk", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
            avg_risk_score=("risk_score", "mean"),
        )
        .reset_index()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run assistant risk evaluations.")
    parser.add_argument("--prompt-path", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--models", nargs="+", default=list(MODEL_LABELS.keys()))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--block-unsafe-inputs", action="store_true")
    parser.add_argument("--use-judge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_evaluation(
        model_labels=args.models,
        prompt_path=args.prompt_path,
        output_path=args.output_path,
        limit=args.limit,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        block_unsafe_inputs=args.block_unsafe_inputs,
        use_judge=args.use_judge,
    )
    print(f"Wrote {len(df)} evaluation rows to {args.output_path}")
    print(summarize(df).to_string(index=False))


if __name__ == "__main__":
    main()
