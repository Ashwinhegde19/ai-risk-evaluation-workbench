"""Tests for the PII detection guardrail (``src.guardrails.pii``)."""

import re
import sys
import types
import unittest
from unittest import mock

from src.guardrails.pii import PiiDetector, PiiFinding, PiiType


class _FakePresidioResult:
    def __init__(self, entity_type: str, start: int, end: int, score: float) -> None:
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.score = score


class _FakePresidioAnalyzer:
    def analyze(self, text: str, entities=None, language: str = "en"):
        out = []
        if "EMAIL_ADDRESS" in (entities or []):
            m = re.search(r"[\w.+-]+@[\w.-]+\.\w+", text)
            if m:
                out.append(_FakePresidioResult("EMAIL_ADDRESS", m.start(), m.end(), 0.99))
        if "PHONE_NUMBER" in (entities or []):
            m = re.search(r"\d{3}-\d{3}-\d{4}", text)
            if m:
                out.append(_FakePresidioResult("PHONE_NUMBER", m.start(), m.end(), 0.95))
        return out


def _install_fake_presidio() -> mock._patch:
    fake = types.ModuleType("presidio_analyzer")
    fake.AnalyzerEngine = _FakePresidioAnalyzer
    return mock.patch.dict(sys.modules, {"presidio_analyzer": fake})


class RegexFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = PiiDetector(use_presidio=False)

    def test_engine_is_regex_fallback(self) -> None:
        self.assertEqual(self.detector.engine_name, "regex")

    def test_detects_email(self) -> None:
        findings = self.detector.detect("Reach me at jane.doe@example.com please")
        types = {f.pii_type for f in findings}
        self.assertIn(PiiType.EMAIL_ADDRESS, types)

    def test_detects_phone(self) -> None:
        findings = self.detector.detect("Call 123-456-7890 now")
        self.assertTrue(any(f.pii_type == PiiType.PHONE_NUMBER for f in findings))

    def test_detects_ssn(self) -> None:
        findings = self.detector.detect("SSN 123-45-6789 on file")
        self.assertTrue(any(f.pii_type == PiiType.US_SSN for f in findings))

    def test_detects_valid_credit_card_via_luhn(self) -> None:
        findings = self.detector.detect("Card 4111 1111 1111 1111 expired")
        self.assertTrue(any(f.pii_type == PiiType.CREDIT_CARD for f in findings))

    def test_rejects_invalid_credit_card(self) -> None:
        findings = self.detector.detect("Card 4111 1111 1111 1112 expired")
        self.assertFalse(any(f.pii_type == PiiType.CREDIT_CARD for f in findings))

    def test_detects_address(self) -> None:
        findings = self.detector.detect("Ship to 123 Main Street, Springfield 12345")
        self.assertTrue(any(f.pii_type == PiiType.ADDRESS for f in findings))

    def test_detects_person_from_lexicon(self) -> None:
        findings = self.detector.detect("John Smith sent the message")
        person_types = [f for f in findings if f.pii_type == PiiType.PERSON]
        self.assertEqual(len(person_types), 2)

    def test_no_pii_returns_empty(self) -> None:
        self.assertEqual(self.detector.detect("The weather is pleasant today."), [])

    def test_scan_not_triggered_when_clean(self) -> None:
        result = self.detector.scan("Nothing sensitive here.")
        self.assertFalse(result.triggered)
        self.assertEqual(result.check_type, "pii")

    def test_scan_triggered_with_high_severity_for_ssn(self) -> None:
        result = self.detector.scan("My SSN is 123-45-6789")
        self.assertTrue(result.triggered)
        self.assertEqual(result.severity.value, "high")


class DedupeTests(unittest.TestCase):
    def test_overlapping_spans_keep_highest_confidence(self) -> None:
        a = PiiFinding(pii_type=PiiType.EMAIL_ADDRESS, start=0, end=5, score=0.5)
        b = PiiFinding(pii_type=PiiType.EMAIL_ADDRESS, start=2, end=8, score=0.9)
        kept = PiiDetector._dedupe([a, b])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].score, 0.9)

    def test_non_overlapping_spans_both_kept_sorted(self) -> None:
        a = PiiFinding(pii_type=PiiType.PHONE_NUMBER, start=10, end=20, score=0.8)
        b = PiiFinding(pii_type=PiiType.EMAIL_ADDRESS, start=0, end=5, score=0.8)
        kept = PiiDetector._dedupe([a, b])
        self.assertEqual([f.start for f in kept], [0, 10])


class PresidioPrimaryTests(unittest.TestCase):
    def test_uses_presidio_when_available(self) -> None:
        with _install_fake_presidio():
            detector = PiiDetector(use_presidio=True)
            self.assertEqual(detector.engine_name, "presidio")

    def test_presidio_detects_email_and_phone(self) -> None:
        with _install_fake_presidio():
            detector = PiiDetector(use_presidio=True)
            findings = detector.detect("a@b.com or 123-456-7890")
            types = {f.pii_type for f in findings}
            self.assertIn(PiiType.EMAIL_ADDRESS, types)
            self.assertIn(PiiType.PHONE_NUMBER, types)


if __name__ == "__main__":
    unittest.main()
