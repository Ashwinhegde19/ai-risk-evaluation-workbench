"""Gradio entrypoint for the AI Assistant Risk Evaluation Workbench."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from assistant import RiskAwareAssistant, SlidingWindowMemory
from evals.run_evals import run_evaluation, summarize
from models import HuggingFaceOSSClient, OpenAIModelClient
from reports.generate_report import generate_report

load_dotenv()

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "eval_results.csv"
REPORT_PATH = ROOT / "reports" / "evaluation_report.pdf"

MODEL_OPTIONS = {
    "Open Source Assistant": "oss",
    "Frontier Assistant": "frontier",
}

_MODEL_CACHE: dict[str, Any] = {}


def get_model_client(model_choice: str):
    model_key = MODEL_OPTIONS[model_choice]
    if model_key not in _MODEL_CACHE:
        if model_key == "oss":
            _MODEL_CACHE[model_key] = HuggingFaceOSSClient()
        elif model_key == "frontier":
            _MODEL_CACHE[model_key] = OpenAIModelClient()
        else:
            raise ValueError(f"Unknown model option: {model_choice}")
    return _MODEL_CACHE[model_key]


def memory_from_history(history: list[dict[str, str]] | None) -> SlidingWindowMemory:
    memory = SlidingWindowMemory(max_messages=8)
    for message in (history or [])[-8:]:
        role = message.get("role")
        content = message.get("content", "")
        if role in {"user", "assistant"} and content:
            memory.add(role, content)
    return memory


def format_trace(result) -> str:
    input_categories = ", ".join(result.input_check.categories) or "none"
    output_categories = ", ".join(result.output_check.categories) or "none"
    blocked = result.metadata.get("blocked_before_model", False)
    error = result.metadata.get("error")
    lines = [
        f"Model: {result.model_name}",
        f"Latency: {result.latency_ms} ms",
        f"Input safety: {result.input_check.label} ({input_categories})",
        f"Output safety: {result.output_check.label} ({output_categories})",
        f"Blocked before model: {blocked}",
    ]
    if error:
        lines.append(f"Backend error: {error}")
    return "\n".join(lines)


def chat(
    user_text: str,
    history: list[dict[str, str]] | None,
    model_choice: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, list[dict[str, str]], str]:
    history = history or []
    if not user_text.strip():
        return "", history, "Enter a message to start the conversation."

    assistant = RiskAwareAssistant(
        get_model_client(model_choice),
        memory=memory_from_history(history),
    )
    result = assistant.respond(
        user_text.strip(),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    updated_history = [
        *history,
        {"role": "user", "content": user_text.strip()},
        {"role": "assistant", "content": result.response},
    ]
    return "", updated_history, format_trace(result)


def reset_chat() -> tuple[list[dict[str, str]], str]:
    return [], "Conversation memory cleared."


def display_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    view = df.copy()
    for column in [
        "pass_rate",
        "hallucination_rate",
        "unsafe_rate",
        "correct_refusal_rate",
        "bias_risk_rate",
    ]:
        if column in view:
            view[column] = view[column].map(lambda value: f"{value:.0%}")
    for column in ["avg_latency_ms", "avg_risk_score"]:
        if column in view:
            view[column] = view[column].map(lambda value: f"{value:.1f}")
    return view


def run_evals_from_ui(
    selected_models: list[str],
    limit: int,
    block_unsafe_inputs: bool,
    use_judge: bool,
    eval_temperature: float,
    eval_max_tokens: int,
) -> tuple[str, pd.DataFrame, pd.DataFrame, str]:
    if not selected_models:
        empty = pd.DataFrame()
        return "Select at least one model.", empty, empty, ""

    try:
        df = run_evaluation(
            model_labels=selected_models,
            output_path=RESULTS_PATH,
            limit=int(limit) if limit else None,
            temperature=eval_temperature,
            max_tokens=int(eval_max_tokens),
            block_unsafe_inputs=block_unsafe_inputs,
            use_judge=use_judge,
        )
    except Exception as exc:
        empty = pd.DataFrame()
        return f"Evaluation failed: {exc}", empty, empty, ""

    summary_df = summarize(df)
    return (
        f"Wrote {len(df)} rows to {RESULTS_PATH}",
        display_summary(summary_df),
        df[
            [
                "model_label",
                "prompt_id",
                "category",
                "passed",
                "risk_score",
                "latency_ms",
                "score_notes",
            ]
        ],
        build_risk_summary(summary_df),
    )


def load_saved_results() -> tuple[str, pd.DataFrame, pd.DataFrame, str]:
    if not RESULTS_PATH.exists():
        empty = pd.DataFrame()
        return "No saved evaluation results yet.", empty, empty, ""

    df = pd.read_csv(RESULTS_PATH)
    summary_df = summarize(df)
    detail_cols = [
        "model_label",
        "prompt_id",
        "category",
        "passed",
        "risk_score",
        "latency_ms",
        "score_notes",
    ]
    return (
        f"Loaded {len(df)} rows from {RESULTS_PATH}",
        display_summary(summary_df),
        df[detail_cols],
        build_risk_summary(summary_df),
    )


def generate_report_from_ui() -> tuple[str, str | None]:
    if not RESULTS_PATH.exists():
        return "Run or load evaluations before generating a report.", None

    try:
        output = generate_report(RESULTS_PATH, REPORT_PATH)
    except Exception as exc:
        return f"Report generation failed: {exc}", None
    return f"Wrote report to {output}", str(output)


def build_risk_summary(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return ""

    ranked = summary_df.sort_values(["avg_risk_score", "pass_rate"], ascending=[True, False])
    best = ranked.iloc[0]
    lines = [
        "## AI Risk Summary Card",
        f"Recommended default: **{best['model_label']}**",
        "",
        "| Model | Pass rate | Hallucination | Unsafe | Bias risk | Avg latency | Avg risk |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            "| {model} | {pass_rate:.0%} | {hallucination:.0%} | {unsafe:.0%} | "
            "{bias:.0%} | {latency:.0f} ms | {risk:.1f} |".format(
                model=row["model_label"],
                pass_rate=row["pass_rate"],
                hallucination=row["hallucination_rate"],
                unsafe=row["unsafe_rate"],
                bias=row["bias_risk_rate"],
                latency=row["avg_latency_ms"],
                risk=row["avg_risk_score"],
            )
        )
    lines.extend(
        [
            "",
            "Recommendation: use the lowest-risk model for customer-facing or liability-sensitive workflows. "
            "Use the OSS model when deployment ownership and cost control matter, but pair it with guardrails, "
            "monitoring, and narrower use-case boundaries.",
        ]
    )
    return "\n".join(lines)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI Assistant Risk Evaluation Workbench") as demo:
        gr.Markdown("# AI Assistant Risk Evaluation Workbench")
        gr.Markdown("Compare OSS and frontier assistants with shared memory, safety checks, and observability.")

        with gr.Tab("Chat"):
            with gr.Row():
                model_choice = gr.Radio(
                    list(MODEL_OPTIONS.keys()),
                    value="Open Source Assistant",
                    label="Assistant",
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.2,
                    step=0.05,
                    label="Temperature",
                )
                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=1024,
                    value=512,
                    step=64,
                    label="Max tokens",
                )

            chatbot = gr.Chatbot(
                label="Conversation",
                type="messages",
                height=520,
            )
            user_text = gr.Textbox(
                label="Message",
                placeholder="Ask a factual question, test memory, or try a safety-sensitive prompt.",
                lines=3,
            )
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Reset")
            trace = gr.Textbox(
                label="Request trace",
                lines=7,
                interactive=False,
            )

            send.click(
                chat,
                inputs=[user_text, chatbot, model_choice, temperature, max_tokens],
                outputs=[user_text, chatbot, trace],
            )
            user_text.submit(
                chat,
                inputs=[user_text, chatbot, model_choice, temperature, max_tokens],
                outputs=[user_text, chatbot, trace],
            )
            clear.click(reset_chat, outputs=[chatbot, trace])

        with gr.Tab("Evaluation"):
            with gr.Row():
                selected_models = gr.CheckboxGroup(
                    list(MODEL_OPTIONS.keys()),
                    value=list(MODEL_OPTIONS.keys()),
                    label="Models",
                )
                eval_limit = gr.Slider(
                    minimum=1,
                    maximum=28,
                    value=8,
                    step=1,
                    label="Prompt limit",
                )
                eval_temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.05,
                    label="Eval temperature",
                )
                eval_max_tokens = gr.Slider(
                    minimum=64,
                    maximum=1024,
                    value=384,
                    step=64,
                    label="Eval max tokens",
                )
            with gr.Row():
                block_unsafe_inputs = gr.Checkbox(
                    value=False,
                    label="Block unsafe inputs before model call",
                )
                use_judge = gr.Checkbox(
                    value=False,
                    label="Use LLM judge",
                )
            with gr.Row():
                run_eval = gr.Button("Run Evaluation", variant="primary")
                load_eval = gr.Button("Load Saved Results")
                make_report = gr.Button("Generate PDF Report")
            eval_status = gr.Textbox(label="Evaluation status", interactive=False)
            report_file = gr.File(label="Evaluation report")
            summary_table = gr.Dataframe(label="Summary metrics", interactive=False)
            detail_table = gr.Dataframe(label="Prompt-level results", interactive=False)

        with gr.Tab("Risk Summary"):
            risk_summary = gr.Markdown()

        run_eval.click(
            run_evals_from_ui,
            inputs=[
                selected_models,
                eval_limit,
                block_unsafe_inputs,
                use_judge,
                eval_temperature,
                eval_max_tokens,
            ],
            outputs=[eval_status, summary_table, detail_table, risk_summary],
        )
        load_eval.click(
            load_saved_results,
            outputs=[eval_status, summary_table, detail_table, risk_summary],
        )
        make_report.click(
            generate_report_from_ui,
            outputs=[eval_status, report_file],
        )

    return demo


def main() -> None:
    demo = build_app()
    demo.launch()


if __name__ == "__main__":
    main()
