"""Assistant orchestration shared by OSS and frontier model backends."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from assistant.guardrails import Guardrails, SafetyCheck
from assistant.logging_utils import append_jsonl
from assistant.memory import SlidingWindowMemory
from models.base import ChatMessage, ModelClient

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, helpful personal assistant. You support multi-turn tasks, "
    "remember relevant details from the current conversation, and avoid making up facts."
)


@dataclass
class AssistantResult:
    response: str
    model_name: str
    latency_ms: int
    input_check: SafetyCheck
    output_check: SafetyCheck
    metadata: dict[str, Any] = field(default_factory=dict)


class RiskAwareAssistant:
    """Common assistant layer for fair OSS vs frontier comparison."""

    def __init__(
        self,
        model_client: ModelClient,
        *,
        memory: SlidingWindowMemory | None = None,
        guardrails: Guardrails | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        log_path: str | None = None,
        block_unsafe_inputs: bool = True,
    ) -> None:
        self.model_client = model_client
        self.memory = memory or SlidingWindowMemory()
        self.guardrails = guardrails or Guardrails()
        self.system_prompt = system_prompt
        self.log_path = log_path or os.getenv("APP_LOG_PATH", "logs/chat_logs.jsonl")
        self.block_unsafe_inputs = block_unsafe_inputs

    def respond(
        self,
        user_text: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        log: bool = True,
    ) -> AssistantResult:
        start = time.perf_counter()
        input_check = self.guardrails.assess_input(user_text)

        if input_check.blocked and self.block_unsafe_inputs:
            response_text = self.guardrails.refusal_for(input_check)
            output_check = SafetyCheck(blocked=False)
            latency_ms = self._elapsed_ms(start)
            self._remember(user_text, response_text)
            result = AssistantResult(
                response=response_text,
                model_name=self.model_client.name,
                latency_ms=latency_ms,
                input_check=input_check,
                output_check=output_check,
                metadata={"blocked_before_model": True},
            )
            if log:
                self._log_interaction(user_text, result)
            return result

        messages = self._build_messages(user_text)
        try:
            response_text = self.model_client.generate(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            error = None
        except Exception as exc:
            response_text = (
                "I could not generate a response because the selected model backend failed. "
                f"Error: {exc}"
            )
            error = str(exc)

        output_check = self.guardrails.assess_output(response_text)
        if output_check.blocked:
            response_text = (
                "I generated content that tripped the safety layer, so I am replacing it "
                "with a safer response. I can help reframe the request in a safe way."
            )

        latency_ms = self._elapsed_ms(start)
        self._remember(user_text, response_text)
        result = AssistantResult(
            response=response_text,
            model_name=self.model_client.name,
            latency_ms=latency_ms,
            input_check=input_check,
            output_check=output_check,
            metadata={"error": error} if error else {},
        )
        if log:
            self._log_interaction(user_text, result)
        return result

    def reset(self) -> None:
        self.memory.reset()

    def _build_messages(self, user_text: str) -> list[ChatMessage]:
        system = f"{self.system_prompt}\n\nSafety policy: {self.guardrails.system_policy()}"
        return [
            {"role": "system", "content": system},
            *self.memory.context(),
            {"role": "user", "content": user_text},
        ]

    def _remember(self, user_text: str, response_text: str) -> None:
        self.memory.add("user", user_text)
        self.memory.add("assistant", response_text)

    def _log_interaction(self, user_text: str, result: AssistantResult) -> None:
        append_jsonl(
            self.log_path,
            {
                "event": "chat_response",
                "model": result.model_name,
                "prompt": user_text,
                "response": result.response,
                "latency_ms": result.latency_ms,
                "input_safety": result.input_check.label,
                "input_categories": result.input_check.categories,
                "output_safety": result.output_check.label,
                "output_categories": result.output_check.categories,
                "metadata": result.metadata,
            },
        )

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)
