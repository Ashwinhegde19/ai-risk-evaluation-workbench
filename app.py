"""Gradio entrypoint for the AI Assistant Risk Evaluation Workbench."""

from __future__ import annotations

from typing import Any

import gradio as gr
from dotenv import load_dotenv

from assistant import RiskAwareAssistant, SlidingWindowMemory
from models import HuggingFaceOSSClient, OpenAIModelClient

load_dotenv()

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
            gr.Markdown("Evaluation runner coming in the next build step.")

        with gr.Tab("Risk Summary"):
            gr.Markdown("Risk summary cards and charts coming in the next build step.")

    return demo


def main() -> None:
    demo = build_app()
    demo.launch()


if __name__ == "__main__":
    main()
