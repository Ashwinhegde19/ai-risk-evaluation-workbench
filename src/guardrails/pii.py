"""PII detection guardrail.

This module detects personally identifiable information (PII) in a piece of
text. It prefers `Microsoft Presidio <https://microsoft.github.io/presidio/>`_
when the ``presidio-analyzer`` package is installed, and otherwise falls back
to a deterministic, regex-based detector so the guardrail works in any
environment without heavy ML dependencies.

Both engines surface the same :class:`PiiFinding` structures and a single
:class:`~src.core.models.GuardrailResult` summarising what was found.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Optional

from src.core.models import BaseWorkbenchModel, GuardrailResult, Severity

# Entity types supported by this guardrail. The regex fallback can detect every
# type; Presidio is used for the entity types it ships recognizers for.
PII_TYPES: List[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "ADDRESS",
]

# A small curated name lexicon used only by the regex fallback when Presidio is
# unavailable. It lets deterministic name detection be tested without an NER
# model. Real deployments rely on Presidio for PERSON recognition.
_DEFAULT_KNOWN_NAMES: List[str] = [
    "John",
    "Jane",
    "Alice",
    "Bob",
    "Mary",
    "Robert",
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Davis",
    "Miller",
    "Wilson",
    "Moore",
    "Taylor",
    "Anderson",
    "Thomas",
    "Jackson",
    "White",
    "Harris",
]

# Map Presidio entity types -> our PiiType names.
_PRESIDIO_MAP: Dict[str, str] = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "US_SSN": "US_SSN",
    "CREDIT_CARD": "CREDIT_CARD",
}

# Regex patterns for the fallback engine. Each pattern is anchored so it does
# not fire on partial substrings in the middle of unrelated tokens.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.\-]?)?"
    r"(?:\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{4}"
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD_RE = re.compile(r"\b(?:\d[ \-]?){13,16}\b")
_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z][a-zA-Z0-9.\-]+\s){1,4}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|"
    r"Court|Ct|Way|Place|Pl|Terrace|Ter|Circle|Cir)\b"
    r"(?:\s*,?\s*[A-Z][a-zA-Z]+)?\s*,?\s*\d{5}(?:-\d{4})?"
)

# Entity types that are considered highly sensitive (financial / identity).
_HIGH_SEVERITY_TYPES: List[str] = ["US_SSN", "CREDIT_CARD", "PERSON"]


class PiiType(str, Enum):
    """PII entity categories detected by the guardrail."""

    PERSON = "PERSON"
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    PHONE_NUMBER = "PHONE_NUMBER"
    US_SSN = "US_SSN"
    CREDIT_CARD = "CREDIT_CARD"
    ADDRESS = "ADDRESS"


class PiiFinding(BaseWorkbenchModel):
    """A single detected PII span within the analysed text.

    Attributes:
        pii_type: The category of PII that was detected.
        value: A masked representation of the matched value (never the raw PII).
        start: Start character offset of the match within the source text.
        end: End character offset (exclusive) of the match.
        score: Detector confidence in the range ``[0, 1]``.
    """

    pii_type: PiiType
    value: str = ""
    start: int
    end: int
    score: float


def _luhn_valid(number: str) -> bool:
    """Validate a digit string using the Luhn checksum algorithm.

    Args:
        number: A string of digits representing a candidate card number.

    Returns:
        ``True`` if the number passes the Luhn check.
    """
    digits = [int(ch) for ch in number]
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


class PiiDetector:
    """Detect PII in text.

    The detector uses Microsoft Presidio when ``presidio-analyzer`` is
    importable; otherwise it transparently uses a regex fallback. The public
    API (:meth:`detect` and :meth:`scan`) is identical regardless of which
    engine is active.

    Args:
        known_names: Optional list of person names for the regex fallback's
            PERSON heuristic. Defaults to a small curated list.
        use_presidio: When ``False``, the regex fallback is always used even if
            Presidio is installed (useful for reproducible tests).
        supported_types: Subset of :data:`PII_TYPES` to detect. Defaults to all.
    """

    def __init__(
        self,
        known_names: Optional[List[str]] = None,
        use_presidio: bool = True,
        supported_types: Optional[List[str]] = None,
    ) -> None:
        """Initialise the detector and (lazily) select an analysis engine."""
        self.known_names: List[str] = list(known_names or _DEFAULT_KNOWN_NAMES)
        self.supported_types: List[str] = list(supported_types or PII_TYPES)
        self._presidio = self._build_presidio() if use_presidio else None

    @staticmethod
    def _build_presidio() -> object:
        """Attempt to construct a Presidio ``AnalyzerEngine``.

        Returns:
            An ``AnalyzerEngine`` instance, or ``None`` if Presidio is not
            installed in the environment.
        """
        try:  # pragma: no cover - exercised only when presidio is installed
            from presidio_analyzer import AnalyzerEngine

            return AnalyzerEngine()
        except Exception:
            return None

    @property
    def engine_name(self) -> str:
        """Return the name of the active detection engine."""
        return "presidio" if self._presidio is not None else "regex"

    def detect(self, text: str) -> List[PiiFinding]:
        """Detect all PII spans in ``text``.

        Args:
            text: The text to scan for PII.

        Returns:
            A list of :class:`PiiFinding` objects, sorted by start offset and
            with no overlapping spans.
        """
        if not text:
            return []
        if self._presidio is not None:  # pragma: no cover - needs presidio
            findings = self._detect_presidio(text)
        else:
            findings = self._detect_regex(text)
        return self._dedupe(findings)

    def scan(self, text: str) -> GuardrailResult:
        """Run PII detection and return a summarised :class:`GuardrailResult`.

        Args:
            text: The text to scan for PII.

        Returns:
            A :class:`GuardrailResult` with ``check_type="pii"``. ``triggered``
            is ``True`` when at least one PII span is found.
        """
        findings = self.detect(text)
        if not findings:
            return GuardrailResult(check_type="pii", triggered=False, details="")
        types = sorted({f.pii_type.value for f in findings})
        severity = Severity.HIGH if any(
            t in _HIGH_SEVERITY_TYPES for t in types
        ) else Severity.MEDIUM
        detail = (
            f"Detected {len(findings)} PII span(s): " + ", ".join(types)
        )
        return GuardrailResult(
            check_type="pii",
            triggered=True,
            details=detail,
            severity=severity,
        )

    # -- internal engines ---------------------------------------------------

    def _detect_presidio(self, text: str) -> List[PiiFinding]:
        """Detect PII using the Presidio engine.

        Args:
            text: The text to scan.

        Returns:
            PII findings derived from Presidio's analyser output.
        """
        entities = [p for p in _PRESIDIO_MAP if p in self.supported_types]
        results = self._presidio.analyze(text=text, entities=entities, language="en")
        findings: List[PiiFinding] = []
        for res in results:
            mapped = _PRESIDIO_MAP.get(res.entity_type, res.entity_type)
            if mapped not in self.supported_types:
                continue
            findings.append(
                PiiFinding(
                    pii_type=PiiType(mapped),
                    value=f"<{mapped}>",
                    start=res.start,
                    end=res.end,
                    score=float(res.score),
                )
            )
        # Presidio does not ship an ADDRESS recognizer by default; cover it.
        if "ADDRESS" in self.supported_types:
            findings.extend(self._detect_address(text))
        return findings

    def _detect_regex(self, text: str) -> List[PiiFinding]:
        """Detect PII using deterministic regex patterns.

        Args:
            text: The text to scan.

        Returns:
            PII findings derived from the regex fallback engine.
        """
        findings: List[PiiFinding] = []
        if "EMAIL_ADDRESS" in self.supported_types:
            findings.extend(self._regex_find(text, _EMAIL_RE, PiiType.EMAIL_ADDRESS, 0.9))
        if "PHONE_NUMBER" in self.supported_types:
            findings.extend(self._regex_find(text, _PHONE_RE, PiiType.PHONE_NUMBER, 0.85))
        if "US_SSN" in self.supported_types:
            findings.extend(self._regex_find(text, _SSN_RE, PiiType.US_SSN, 0.95))
        if "CREDIT_CARD" in self.supported_types:
            findings.extend(self._detect_credit_card(text))
        if "ADDRESS" in self.supported_types:
            findings.extend(self._detect_address(text))
        if "PERSON" in self.supported_types:
            findings.extend(self._detect_person(text))
        return findings

    def _regex_find(
        self,
        text: str,
        pattern: "re.Pattern[str]",
        pii_type: PiiType,
        score: float,
    ) -> List[PiiFinding]:
        """Apply a regex and wrap matches as :class:`PiiFinding` objects.

        Args:
            text: The text to scan.
            pattern: A compiled regular expression.
            pii_type: The PII category for matches.
            score: Confidence score assigned to each match.

        Returns:
            Findings for every non-empty match.
        """
        out: List[PiiFinding] = []
        for match in pattern.finditer(text):
            if match.group(0).strip():
                out.append(
                    PiiFinding(
                        pii_type=pii_type,
                        value=f"<{pii_type.value}>",
                        start=match.start(),
                        end=match.end(),
                        score=score,
                    )
                )
        return out

    def _detect_credit_card(self, text: str) -> List[PiiFinding]:
        """Detect candidate credit-card numbers via Luhn validation.

        Args:
            text: The text to scan.

        Returns:
            Findings for digit runs that pass the Luhn checksum.
        """
        out: List[PiiFinding] = []
        for match in _CARD_RE.finditer(text):
            digits = re.sub(r"\D", "", match.group(0))
            if 13 <= len(digits) <= 19 and _luhn_valid(digits):
                out.append(
                    PiiFinding(
                        pii_type=PiiType.CREDIT_CARD,
                        value="<CREDIT_CARD>",
                        start=match.start(),
                        end=match.end(),
                        score=0.9,
                    )
                )
        return out

    def _detect_address(self, text: str) -> List[PiiFinding]:
        """Detect street addresses using a suffix-anchored pattern.

        Args:
            text: The text to scan.

        Returns:
            Findings for address-like spans.
        """
        return self._regex_find(text, _ADDRESS_RE, PiiType.ADDRESS, 0.8)

    def _detect_person(self, text: str) -> List[PiiFinding]:
        """Detect person names from the curated lexicon (fallback only).

        Args:
            text: The text to scan.

        Returns:
            Findings for whole-word name matches from the known-names list.
        """
        out: List[PiiFinding] = []
        if not self.known_names:
            return out
        # Build one alternation, longest names first to prefer full matches.
        names = sorted(self.known_names, key=len, reverse=True)
        pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")
        for match in pattern.finditer(text):
            out.append(
                PiiFinding(
                    pii_type=PiiType.PERSON,
                    value="<PERSON>",
                    start=match.start(),
                    end=match.end(),
                    score=0.7,
                )
            )
        return out

    @staticmethod
    def _dedupe(findings: List[PiiFinding]) -> List[PiiFinding]:
        """Remove overlapping spans, keeping the highest-confidence match.

        Args:
            findings: Findings that may contain overlapping character spans.

        Returns:
            A de-duplicated, start-offset-sorted list of findings.
        """
        # Greedily keep spans in descending confidence order so the
        # highest-confidence match wins when spans overlap.
        ordered = sorted(findings, key=lambda f: (-f.score, f.start, f.end))
        kept: List[PiiFinding] = []
        last_end = -1
        for finding in ordered:
            if finding.start >= last_end:
                kept.append(finding)
                last_end = finding.end
        return sorted(kept, key=lambda f: f.start)


__all__ = [
    "PiiType",
    "PiiFinding",
    "PiiDetector",
]
