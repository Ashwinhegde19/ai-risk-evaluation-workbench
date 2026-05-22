"""Deterministic assistant tools used before model generation."""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Result returned by a deterministic assistant tool."""

    name: str
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AssistantTools:
    """Small tool router for explicit assistant utility requests."""

    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def run(self, user_text: str) -> ToolResult | None:
        calculator = self._calculator(user_text)
        if calculator:
            return calculator

        checklist = self._risk_checklist(user_text)
        if checklist:
            return checklist

        return None

    def _calculator(self, user_text: str) -> ToolResult | None:
        lowered = user_text.lower().strip()
        if not any(trigger in lowered for trigger in ("calculate", "calculator", "compute")):
            return None

        expression = self._extract_expression(user_text)
        if not expression:
            return None

        try:
            value = self._safe_eval(expression)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
            return ToolResult(
                name="calculator",
                output=f"Tool call: calculator\n\nI could not safely calculate `{expression}`. Error: {exc}",
                metadata={"expression": expression, "error": str(exc)},
            )

        return ToolResult(
            name="calculator",
            output=f"Tool call: calculator\n\n`{expression}` = **{value:g}**",
            metadata={"expression": expression, "result": value},
        )

    def _risk_checklist(self, user_text: str) -> ToolResult | None:
        lowered = user_text.lower()
        triggers = (
            "risk checklist",
            "ai risk checklist",
            "evaluate ai risk",
            "assistant risk",
            "risk controls",
            "liability controls",
        )
        if not any(trigger in lowered for trigger in triggers):
            return None

        domain = self._infer_domain(lowered)
        output = (
            "Tool call: ai_risk_checklist\n\n"
            f"Risk checklist for **{domain}**:\n\n"
            "| Area | Control | Evidence to collect |\n"
            "|---|---|---|\n"
            "| Hallucination | Ground answers in approved sources and refuse unknown facts. | Eval failures, citation/grounding logs. |\n"
            "| Prompt injection | Separate system, user, retrieved, and tool content. | Injection test cases and blocked traces. |\n"
            "| Privacy | Redact secrets/PII before model calls and logs. | Redaction samples and access policy. |\n"
            "| Bias | Test sensitive-class prompts and refusal quality. | Bias eval pass rate and notable cases. |\n"
            "| Harmful output | Refuse cyber abuse, weaponization, crisis-risk, and exploitation requests. | Safety logs and refusal examples. |\n"
            "| Liability | Route legal, medical, financial, and policy commitments to review. | Escalation rules and reviewer queue. |\n"
            "| Observability | Track latency, model, safety labels, and tool calls. | JSONL traces and eval report. |"
        )
        return ToolResult(
            name="ai_risk_checklist",
            output=output,
            metadata={"domain": domain, "rows": 7},
        )

    @classmethod
    def _extract_expression(cls, user_text: str) -> str:
        match = re.search(r"(?:calculate|calculator|compute)[:\s]+(.+)", user_text, re.IGNORECASE)
        if not match:
            return ""
        expression = match.group(1)
        expression = expression.replace("^", "**")
        expression = re.sub(r"[^0-9+\-*/().%\s]", "", expression)
        return expression.strip()

    @classmethod
    def _safe_eval(cls, expression: str) -> float:
        tree = ast.parse(expression, mode="eval")
        return float(cls._eval_node(tree.body))

    @classmethod
    def _eval_node(cls, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in cls._OPERATORS:
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponent too large")
            return float(cls._OPERATORS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in cls._OPERATORS:
            operand = cls._eval_node(node.operand)
            return float(cls._OPERATORS[type(node.op)](operand))
        raise ValueError("unsupported expression")

    @staticmethod
    def _infer_domain(lowered_text: str) -> str:
        if "insurance" in lowered_text or "underwriting" in lowered_text:
            return "AI insurance or underwriting assistants"
        if "support" in lowered_text or "customer" in lowered_text:
            return "customer-support assistants"
        if "health" in lowered_text or "medical" in lowered_text:
            return "healthcare assistants"
        if "finance" in lowered_text or "bank" in lowered_text:
            return "financial assistants"
        return "AI assistants"
