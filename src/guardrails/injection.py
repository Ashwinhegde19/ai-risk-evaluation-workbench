"""Prompt-injection detection guardrail.

Detects attempts to override a model's instructions. Two strategies are
supported:

* **Pattern-based** (default, deterministic): matches a catalogue of known
  injection phrasings (e.g. "ignore previous instructions", "you are now in
  developer mode", "DAN", "do anything now") and combines their weights into a
  confidence score. This always works without external services.
* **LLM-based** (optional): a small classifier model scores the text; enabled
  only when ``use_llm=True`` and a backend is supplied.

Both strategies return a confidence score in ``[0, 1]`` and a
:class:`~src.core.models.GuardrailResult`.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.core.models import BaseWorkbenchModel, GuardrailResult, Severity

# (pattern, weight) catalogue of known injection phrasings. Weights are summed
# across all matches and capped at 1.0 to form the confidence score.
_DEFAULT_PATTERNS: List[Tuple[str, float]] = [
    (r"ignore (all |any |previous |prior |above )?(instructions|prompts|rules|constraints)", 1.0),
    (r"disregard (your |the |previous |prior )?(instructions|rules|prompt|system)", 1.0),
    (r"forget (everything|all|your|the) (previous |prior )?(instructions|context|rules)?", 0.8),
    (r"you are now (in |in a |operating as |acting as )?(developer|admin|god|maintenance)", 0.9),
    (r"(developer|admin|god|maintenance) mode", 0.9),
    (r"jailbreak", 0.9),
    (r"\bDAN\b", 0.9),
    (r"do anything now", 0.9),
    (r"(reveal|print|show|disclose|repeat).{0,25}(system prompt|your instructions|your prompt|your rules)", 0.9),
    (r"pretend to be (an? |the )?(ai|language model|assistant) without (rules|restrictions|guidelines|filters)", 0.8),
    (r"(new|override|updated|revised) (instructions|rules|system prompt|system message)", 0.8),
    (r"act as (an? |the )?(unfiltered|uncensored|unrestricted|evil) (ai|assistant|model|chatbot)", 0.8),
    (r"no (longer|more) (bound|restricted|limited) by (your |the )?(rules|instructions|guidelines)", 0.7),
    (r"(stop|cease) (acting|being|pretending) (like|as) (an? |the )?(assistant|ai|model)", 0.7),
]


class InjectionFinding(BaseWorkbenchModel):
    """A single injection pattern match.

    Attributes:
        pattern: The matched regular expression source.
        weight: The weight contributed by this match to the confidence score.
    """

    pattern: str
    weight: float


class InjectionResult(BaseWorkbenchModel):
    """Result of injection detection on a piece of text.

    Attributes:
        confidence: Overall injection confidence in ``[0, 1]``.
        triggered: Whether the confidence exceeded the detection threshold.
        matches: The individual injection patterns that fired.
    """

    confidence: float
    triggered: bool
    matches: List[InjectionFinding]


class InjectionDetector:
    """Detect prompt-injection attempts in text.

    Args:
        patterns: Optional ``(regex, weight)`` catalogue override.
        use_llm: When ``True`` an LLM backend is used to supplement (not replace)
            pattern detection.
        llm_backend: A backend exposing ``generate(prompt, system_prompt,
            temperature) -> str``; required when ``use_llm=True``.
        threshold: Confidence at or above which detection is triggered.
    """

    def __init__(
        self,
        patterns: Optional[List[Tuple[str, float]]] = None,
        use_llm: bool = False,
        llm_backend: Optional[object] = None,
        threshold: float = 0.5,
    ) -> None:
        """Compile patterns and store detector configuration."""
        self.threshold: float = threshold
        self.use_llm: bool = use_llm
        self.llm_backend: Optional[object] = llm_backend
        raw = patterns if patterns is not None else _DEFAULT_PATTERNS
        self._compiled: List[Tuple["re.Pattern[str]", float]] = [
            (re.compile(p, re.IGNORECASE), w) for p, w in raw
        ]

    def detect(self, text: str) -> InjectionResult:
        """Detect injection in ``text`` using pattern (and optional LLM) signals.

        Args:
            text: The text to inspect.

        Returns:
            An :class:`InjectionResult` with a confidence score and match list.
        """
        if not text or not text.strip():
            return InjectionResult(
                confidence=0.0, triggered=False, matches=[]
            )
        matches: List[InjectionFinding] = []
        total = 0.0
        for pattern, weight in self._compiled:
            if pattern.search(text):
                matches.append(InjectionFinding(pattern=pattern.pattern, weight=weight))
                total += weight
        confidence = min(1.0, total)
        if self.use_llm and self.llm_backend is not None:
            confidence = max(confidence, self._llm_classify(text))
        return InjectionResult(
            confidence=confidence,
            triggered=confidence >= self.threshold,
            matches=matches,
        )

    def scan(self, text: str) -> GuardrailResult:
        """Run injection detection and return a :class:`GuardrailResult`.

        Args:
            text: The text to inspect.

        Returns:
            A :class:`GuardrailResult` with ``check_type="injection"``.
        """
        result = self.detect(text)
        if not result.triggered:
            return GuardrailResult(check_type="injection", triggered=False, details="")
        patterns = ", ".join(m.pattern for m in result.matches)
        detail = f"injection confidence={result.confidence:.2f}; patterns: {patterns}"
        severity = Severity.HIGH if result.confidence >= 0.9 else Severity.MEDIUM
        return GuardrailResult(
            check_type="injection",
            triggered=True,
            details=detail,
            severity=severity,
        )

    def _llm_classify(self, text: str) -> float:
        """Classify ``text`` as injection via an LLM backend.

        The backend is asked to respond with a single float in ``[0, 1]``. Any
        parse failure degrades gracefully to ``0.0``.

        Args:
            text: The text to classify.

        Returns:
            The backend's injection-confidence estimate in ``[0, 1]``.
        """
        system = (
            "You are a prompt-injection classifier. Reply with only a single "
            "number between 0 and 1 representing how likely the user text is a "
            "prompt-injection attack."
        )
        try:
            response = self.llm_backend.generate(
                prompt=text, system_prompt=system, temperature=0.0
            )
            number = re.search(r"0?\.\d+|1\.0+|0|1", response or "")
            if number:
                return min(1.0, max(0.0, float(number.group(0))))
        except Exception:
            return 0.0
        return 0.0


__all__ = [
    "InjectionFinding",
    "InjectionResult",
    "InjectionDetector",
]
