"""Run evaluations for live-style prompts without human expected_behavior labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from assistant import RiskAwareAssistant, SlidingWindowMemory
from evals.policy_inference import inferred_prompt_metadata
from evals.run_evals import MODEL_LABELS, build_client, estimate_cost_per_1k, summarize
from evals.scoring import score_response

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = ROOT / "evals" / "unlabelled_prompts.json"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "unlabelled_eval_results.csv"


def load_unlabelled_prompts(prompt_path: str | Path = DEFAULT_PROMPT_PATH) -> list[dict[str, str]]:
    """Load prompts from JSON as either strings or objects with a prompt field."""

    path = Path(prompt_path)
    with path.open("r", encoding="utf-8") as handle:
        raw_prompts = json.load(handle)

    prompts: list[dict[str, str]] = []
    for index, item in enumerate(raw_prompts, start=1):
        if isinstance(item, str):
            prompt = item
            prompt_id = f"unlabelled_{index:03d}"
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            prompt = item["prompt"]
            prompt_id = str(item.get("id") or f"unlabelled_{index:03d}")
        else:
            raise ValueError(
                f"Prompt item {index} must be a string or an object with a string prompt field."
            )
        prompts.append({"id": prompt_id, "prompt": prompt})
    return prompts


def run_unlabelled_evaluation(
    *,
    model_labels: list[str] | None = None,
    prompt_path: str | Path = DEFAULT_PROMPT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    limit: int | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    block_unsafe_inputs: bool = False,
) -> pd.DataFrame:
    """Evaluate prompts by inferring expected behavior instead of reading labels."""

    load_dotenv()
    prompts = load_unlabelled_prompts(prompt_path)
    if limit:
        prompts = prompts[:limit]

    labels = model_labels or list(MODEL_LABELS.keys())
    rows: list[dict[str, Any]] = []
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for model_label in labels:
        client = build_client(model_label)
        for prompt_item in prompts:
            prompt_metadata = inferred_prompt_metadata(prompt_item["prompt"])
            prompt_metadata["id"] = prompt_item["id"]

            assistant = RiskAwareAssistant(
                client,
                memory=SlidingWindowMemory(max_messages=0),
                block_unsafe_inputs=block_unsafe_inputs,
            )
            result = assistant.respond(
                prompt_item["prompt"],
                temperature=temperature,
                max_tokens=max_tokens,
                log=False,
            )
            score = score_response(
                prompt_metadata,
                result.response,
                result.input_check,
                result.output_check,
                result.metadata,
            )
            rows.append(
                {
                    "run_id": run_id,
                    "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "eval_mode": "unlabelled_policy_inference",
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "block_unsafe_inputs": block_unsafe_inputs,
                    "model_label": model_label,
                    "model_name": result.model_name,
                    "prompt_id": prompt_metadata["id"],
                    "category": prompt_metadata["category"],
                    "risk_type": prompt_metadata["risk_type"],
                    "prompt": prompt_item["prompt"],
                    "expected_behavior": prompt_metadata["expected_behavior"],
                    "inferred_request_type": prompt_metadata["inferred_request_type"],
                    "inferred_expected_action": prompt_metadata["inferred_expected_action"],
                    "policy_inference_confidence": prompt_metadata["policy_inference_confidence"],
                    "policy_inference_reason": prompt_metadata["policy_inference_reason"],
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
            )

    df = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unlabelled policy-inferred evals.")
    parser.add_argument("--prompt-path", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--models", nargs="+", default=list(MODEL_LABELS.keys()))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--block-unsafe-inputs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_unlabelled_evaluation(
        model_labels=args.models,
        prompt_path=args.prompt_path,
        output_path=args.output_path,
        limit=args.limit,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        block_unsafe_inputs=args.block_unsafe_inputs,
    )
    print(f"Wrote {len(df)} unlabelled evaluation rows to {args.output_path}")
    print(summarize(df).to_string(index=False))


if __name__ == "__main__":
    main()
