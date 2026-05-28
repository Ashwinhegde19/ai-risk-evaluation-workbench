"""Assistant orchestration shared by OSS and frontier model backends."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from assistant.evidence import EvidenceRetriever
from assistant.guardrails import Guardrails, SafetyCheck
from assistant.logging_utils import append_jsonl
from assistant.memory import SlidingWindowMemory
from assistant.tools import AssistantTools
from models.base import ChatMessage, ModelClient

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, helpful personal assistant. You support multi-turn tasks, "
    "remember relevant details from the current conversation, and avoid making up facts."
)

GROUNDING_REQUIRED_PATTERNS = (
    r"\b(today|currently|right now|latest|recent|this week|this month|this year)\b",
    r"\b(employee count|funding|valuation|pricing|stock price|weather|ceo)\b",
    r"\bexact\b.{0,80}\b(count|number|amount|date|price)\b",
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
        enable_retrieval: bool | None = None,
        retriever: Any | None = None,
    ) -> None:
        self.model_client = model_client
        self.memory = memory or SlidingWindowMemory()
        self.guardrails = guardrails or Guardrails()
        self.system_prompt = system_prompt
        self.log_path = log_path or os.getenv("APP_LOG_PATH", "logs/chat_logs.jsonl")
        self.block_unsafe_inputs = block_unsafe_inputs
        self.tools = AssistantTools()
        self.enable_retrieval = (
            env_flag("ENABLE_RETRIEVAL") if enable_retrieval is None else enable_retrieval
        )
        self.retriever = retriever or (EvidenceRetriever() if self.enable_retrieval else None)

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

        tool_result = self.tools.run(user_text)
        if tool_result:
            response_text = tool_result.output
            output_check = self.guardrails.assess_output(response_text)
            latency_ms = self._elapsed_ms(start)
            self._remember(user_text, response_text)
            result = AssistantResult(
                response=response_text,
                model_name=self.model_client.name,
                latency_ms=latency_ms,
                input_check=input_check,
                output_check=output_check,
                metadata={
                    "used_tool": True,
                    "tool_calls": [
                        {
                            "name": tool_result.name,
                            "metadata": tool_result.metadata,
                        }
                    ],
                },
            )
            if log:
                self._log_interaction(user_text, result)
            return result

        contexts, retrieval_metadata = self._retrieve_context(user_text)
        if self.enable_retrieval and not contexts and self._requires_grounding(user_text):
            response_text = "I cannot verify this from the available trusted sources."
            output_check = SafetyCheck(blocked=False)
            latency_ms = self._elapsed_ms(start)
            self._remember(user_text, response_text)
            retrieval_metadata["retrieval_status"] = "no_context_cannot_verify"
            result = AssistantResult(
                response=response_text,
                model_name=self.model_client.name,
                latency_ms=latency_ms,
                input_check=input_check,
                output_check=output_check,
                metadata=retrieval_metadata,
            )
            if log:
                self._log_interaction(user_text, result)
            return result

        messages = self._build_messages(user_text, contexts=contexts)
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
        metadata = {"error": error} if error else {}
        metadata.update(retrieval_metadata)
        result = AssistantResult(
            response=response_text,
            model_name=self.model_client.name,
            latency_ms=latency_ms,
            input_check=input_check,
            output_check=output_check,
            metadata=metadata,
        )
        if log:
            self._log_interaction(user_text, result)
        return result

    def reset(self) -> None:
        self.memory.reset()

    def _build_messages(
        self,
        user_text: str,
        *,
        contexts: list[Any] | None = None,
    ) -> list[ChatMessage]:
        system = f"{self.system_prompt}\n\nSafety policy: {self.guardrails.system_policy()}"
        if contexts:
            system = (
                f"{system}\n\n"
                "Grounding rule: answer only using the trusted context provided by the "
                "application. If the context does not contain the answer, say you cannot verify it."
            )
            user_text = (
                "Trusted context:\n"
                f"{self._format_retrieved_context(contexts)}\n\n"
                f"Question:\n{user_text}"
            )
        return [
            {"role": "system", "content": system},
            *self.memory.context(),
            {"role": "user", "content": user_text},
        ]

    def _retrieve_context(self, user_text: str) -> tuple[list[Any], dict[str, Any]]:
        if not self.enable_retrieval or not self.retriever:
            return [], {}

        contexts = self.retriever.search(user_text)
        return contexts, {
            "retrieval_enabled": True,
            "retrieval_status": "context_found" if contexts else "no_context",
            "retrieved_context_count": len(contexts),
            "retrieval_sources": [getattr(context, "source", "") for context in contexts],
            "retrieval_matched_terms": [
                getattr(context, "matched_terms", []) for context in contexts
            ],
            "retrieval_source_types": [
                getattr(context, "source_type", "") for context in contexts
            ],
        }

    def _format_retrieved_context(self, contexts: list[Any]) -> str:
        if self.retriever:
            return self.retriever.format_context(contexts)
        return "\n\n".join(context.text for context in contexts)

    @staticmethod
    def _requires_grounding(user_text: str) -> bool:
        lowered = user_text.lower()
        return any(re.search(pattern, lowered) for pattern in GROUNDING_REQUIRED_PATTERNS)

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


def env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
