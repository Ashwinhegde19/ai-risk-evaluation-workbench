"""Chainlit chat app for the AI Assistant Risk Evaluation Workbench."""

from __future__ import annotations

import importlib
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch
from dotenv import load_dotenv

from assistant import RiskAwareAssistant, SlidingWindowMemory
from evals.run_evals import run_evaluation, summarize
from models import FrontierGatewayClient, HuggingFaceOSSClient
from reports.generate_report import generate_report

load_dotenv()

# Chainlit 2.x can inherit an unset local_steps context in some local runtimes.
# Giving the shared ContextVar a default keeps callbacks stable without changing app logic.
_local_steps = ContextVar("local_steps", default=None)
importlib.import_module("chainlit.context").local_steps = _local_steps
importlib.import_module("chainlit.step").local_steps = _local_steps
importlib.import_module("chainlit.message").local_steps = _local_steps

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "eval_results.csv"
REPORT_PATH = ROOT / "reports" / "evaluation_report.pdf"

MODEL_OPTIONS = {
    "Open Source Assistant": "oss",
    "Frontier Assistant": "frontier",
}

DEFAULT_SETTINGS = {
    "assistant": "Open Source Assistant",
    "temperature": 0.2,
    "max_tokens": 512,
    "block_unsafe_inputs": True,
}

_MODEL_CACHE: dict[str, Any] = {}


def get_model_client(model_choice: str):
    model_key = MODEL_OPTIONS[model_choice]
    if model_key not in _MODEL_CACHE:
        if model_key == "oss":
            _MODEL_CACHE[model_key] = HuggingFaceOSSClient()
        elif model_key == "frontier":
            _MODEL_CACHE[model_key] = FrontierGatewayClient()
        else:
            raise ValueError(f"Unknown model option: {model_choice}")
    return _MODEL_CACHE[model_key]


def get_settings() -> dict[str, Any]:
    return {**DEFAULT_SETTINGS, **(cl.user_session.get("settings") or {})}


def get_memory() -> SlidingWindowMemory:
    memory = cl.user_session.get("memory")
    if memory is None:
        memory = SlidingWindowMemory(max_messages=8)
        cl.user_session.set("memory", memory)
    return memory


def build_assistant(settings: dict[str, Any]) -> RiskAwareAssistant:
    return RiskAwareAssistant(
        get_model_client(settings["assistant"]),
        memory=get_memory(),
        block_unsafe_inputs=bool(settings.get("block_unsafe_inputs", True)),
    )


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


def settings_summary(settings: dict[str, Any]) -> str:
    return (
        f"Assistant: **{settings['assistant']}**\n"
        f"Temperature: `{settings['temperature']}`\n"
        f"Max tokens: `{int(settings['max_tokens'])}`\n"
        f"Pre-model blocking: `{bool(settings['block_unsafe_inputs'])}`"
    )


def action_bar() -> list[cl.Action]:
    return [
        cl.Action(name="reset_memory", payload={}, label="Reset Memory", icon="rotate-ccw"),
        cl.Action(name="run_smoke_eval", payload={}, label="Run 5-Prompt Eval", icon="activity"),
        cl.Action(name="generate_report", payload={}, label="Generate Report", icon="file-text"),
    ]


def render_summary_table(df) -> str:
    summary_df = summarize(df)
    lines = [
        "| Model | Prompts | Pass | Hallucination | Unsafe | Bias | Avg latency | Risk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            "| {model} | {prompts:.0f} | {pass_rate:.0%} | {hallucination:.0%} | "
            "{unsafe:.0%} | {bias:.0%} | {latency:.0f} ms | {risk:.1f} |".format(
                model=row["model_label"],
                prompts=row["prompts"],
                pass_rate=row["pass_rate"],
                hallucination=row["hallucination_rate"],
                unsafe=row["unsafe_rate"],
                bias=row["bias_risk_rate"],
                latency=row["avg_latency_ms"],
                risk=row["avg_risk_score"],
            )
        )
    return "\n".join(lines)


async def send_settings_panel(settings: dict[str, Any]) -> None:
    await cl.ChatSettings(
        [
            Select(
                id="assistant",
                label="Assistant",
                values=list(MODEL_OPTIONS.keys()),
                initial=settings["assistant"],
            ),
            Slider(
                id="temperature",
                label="Temperature",
                initial=float(settings["temperature"]),
                min=0,
                max=1,
                step=0.05,
            ),
            Slider(
                id="max_tokens",
                label="Max tokens",
                initial=float(settings["max_tokens"]),
                min=64,
                max=1024,
                step=64,
            ),
            Switch(
                id="block_unsafe_inputs",
                label="Block unsafe inputs before model call",
                initial=bool(settings["block_unsafe_inputs"]),
            ),
        ]
    ).send()


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("memory", SlidingWindowMemory(max_messages=8))
    cl.user_session.set("settings", DEFAULT_SETTINGS.copy())
    await send_settings_panel(DEFAULT_SETTINGS)
    await cl.Message(
        content=(
            "# AI Assistant Risk Evaluation Workbench\n\n"
            "Chat with the OSS or frontier assistant while the app tracks memory, latency, "
            "guardrail decisions, and safety metadata.\n\n"
            f"{settings_summary(DEFAULT_SETTINGS)}"
        ),
        actions=action_bar(),
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    merged = {**DEFAULT_SETTINGS, **settings}
    cl.user_session.set("settings", merged)
    await cl.Message(
        content=f"Settings updated.\n\n{settings_summary(merged)}",
        actions=action_bar(),
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    settings = get_settings()
    assistant = build_assistant(settings)
    response_message = cl.Message(content="")
    await response_message.send()

    result = await cl.make_async(assistant.respond)(
        message.content,
        temperature=float(settings["temperature"]),
        max_tokens=int(settings["max_tokens"]),
    )

    response_message.content = result.response
    response_message.elements = [
        cl.Text(
            name="Request trace",
            content=format_trace(result),
            display="side",
        )
    ]
    response_message.actions = action_bar()
    await response_message.update()


@cl.action_callback("reset_memory")
async def reset_memory(action: cl.Action) -> None:
    get_memory().reset()
    if hasattr(action, "remove"):
        await action.remove()
    await cl.Message(content="Conversation memory cleared.", actions=action_bar()).send()


@cl.action_callback("run_smoke_eval")
async def run_smoke_eval(action: cl.Action) -> None:
    if hasattr(action, "remove"):
        await action.remove()

    settings = get_settings()
    status = cl.Message(content=f"Running a 5-prompt eval for **{settings['assistant']}**...")
    await status.send()

    try:
        df = await cl.make_async(run_evaluation)(
            model_labels=[settings["assistant"]],
            output_path=RESULTS_PATH,
            limit=5,
            temperature=float(settings["temperature"]),
            max_tokens=int(settings["max_tokens"]),
            block_unsafe_inputs=bool(settings["block_unsafe_inputs"]),
            use_judge=False,
        )
    except Exception as exc:
        status.content = f"Evaluation failed: `{exc}`"
        await status.update()
        return

    status.content = (
        f"Smoke eval complete. Wrote `{len(df)}` rows to `{RESULTS_PATH}`.\n\n"
        f"{render_summary_table(df)}"
    )
    status.elements = [
        cl.File(name="eval_results.csv", path=str(RESULTS_PATH), display="inline")
    ]
    status.actions = action_bar()
    await status.update()


@cl.action_callback("generate_report")
async def generate_report_action(action: cl.Action) -> None:
    if hasattr(action, "remove"):
        await action.remove()

    if not RESULTS_PATH.exists():
        await cl.Message(
            content="No saved eval results found yet. Run the smoke eval or CLI eval first.",
            actions=action_bar(),
        ).send()
        return

    try:
        output = await cl.make_async(generate_report)(RESULTS_PATH, REPORT_PATH)
    except Exception as exc:
        await cl.Message(content=f"Report generation failed: `{exc}`", actions=action_bar()).send()
        return

    await cl.Message(
        content=f"Generated `{output}`.",
        elements=[cl.File(name="evaluation_report.pdf", path=str(output), display="inline")],
        actions=action_bar(),
    ).send()
