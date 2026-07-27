"""Tests for the compliance report generator in ``src.reports.compliance``."""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.core.models import (
    ComplianceFramework,
    ComplianceReport,
    EvalResult,
    RiskTier,
    Severity,
)
from src.reports.compliance import (
    ComplianceReportGenerator,
    generate_compliance_report,
    write_json_report,
    write_pdf_report,
)


def _result(dimension: str, score: float, severity: Severity) -> EvalResult:
    """Build an EvalResult for use in tests."""
    return EvalResult(
        model_name="report-model",
        dimension=dimension,
        score=score,
        severity=severity,
        raw_response="sample response",
    )


def _sample_results() -> list[EvalResult]:
    """A representative mix of eval results spanning risk tiers."""
    return [
        _result("social_scoring", 0.1, Severity.CRITICAL),
        _result("bias", 0.3, Severity.HIGH),
        _result("privacy", 0.4, Severity.MEDIUM),
        _result("hallucination", 0.2, Severity.MEDIUM),
        _result("toxicity", 0.6, Severity.LOW),
    ]


class ReportBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ts = datetime(2026, 7, 23, 12, 0, 0, tzinfo=timezone.utc)
        self.report = generate_compliance_report(
            "report-model", _sample_results(), timestamp=self.ts
        )

    def test_returns_compliance_report(self) -> None:
        self.assertIsInstance(self.report, ComplianceReport)

    def test_finds_across_all_three_frameworks(self) -> None:
        frameworks = {f.framework for f in self.report.findings}
        self.assertEqual(
            frameworks,
            {
                ComplianceFramework.EU_AI_ACT,
                ComplianceFramework.NIST_RMF,
                ComplianceFramework.ISO_42001,
            },
        )

    def test_overall_tier_is_highest(self) -> None:
        # social_scoring is UNACCEPTABLE -> overall must be UNACCEPTABLE.
        self.assertEqual(self.report.overall_risk_tier, RiskTier.UNACCEPTABLE)

    def test_overall_tier_minimal_when_no_findings(self) -> None:
        clean = generate_compliance_report(
            "clean-model", [_result("toxicity", 0.9, Severity.LOW)], timestamp=self.ts
        )
        self.assertEqual(clean.overall_risk_tier, RiskTier.MINIMAL)

    def test_timestamp_preserved(self) -> None:
        self.assertEqual(self.report.timestamp, self.ts)

    def test_gaps_are_populated_and_deduped(self) -> None:
        self.assertTrue(len(self.report.gaps) > 0)
        self.assertEqual(len(self.report.gaps), len(set(self.report.gaps)))


class GeneratorHelperTests(unittest.TestCase):
    def test_executive_summary_mentions_model_and_tier(self) -> None:
        gen = ComplianceReportGenerator("m", _sample_results())
        summary = gen.executive_summary()
        self.assertIn("m", summary)
        self.assertIn("unacceptable", summary.lower())

    def test_gap_analysis_structure(self) -> None:
        gen = ComplianceReportGenerator("m", _sample_results())
        analysis = gen.gap_analysis()
        self.assertEqual(analysis["model_name"], "m")
        self.assertIn("by_framework", analysis)
        self.assertIn("by_tier", analysis)
        self.assertIn("gaps", analysis)

    def test_findings_cached(self) -> None:
        gen = ComplianceReportGenerator("m", _sample_results())
        first = gen.build_findings()
        second = gen.build_findings()
        self.assertIs(first, second)


class JsonReportTests(unittest.TestCase):
    def test_to_json_round_trips(self) -> None:
        gen = ComplianceReportGenerator("m", _sample_results())
        payload = json.loads(gen.to_json())
        self.assertEqual(payload["model_name"], "m")
        self.assertEqual(payload["overall_risk_tier"], "unacceptable")
        self.assertGreater(len(payload["findings"]), 0)

    def test_write_json_report_creates_file(self) -> None:
        report = generate_compliance_report("m", _sample_results())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            written = write_json_report(report, path)
            self.assertTrue(written.exists())
            data = json.loads(written.read_text(encoding="utf-8"))
            self.assertEqual(data["model_name"], "m")


class PdfReportTests(unittest.TestCase):
    def test_write_pdf_report_creates_valid_pdf(self) -> None:
        report = generate_compliance_report("m", _sample_results())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.pdf"
            written = write_pdf_report(report, path)
            self.assertTrue(written.exists())
            blob = written.read_bytes()
            self.assertTrue(blob.startswith(b"%PDF-"))
            self.assertIn(b"%%EOF", blob)
            # A non-trivial report should span more than a single empty page.
            self.assertGreater(len(blob), 500)

    def test_pdf_includes_findings_text(self) -> None:
        report = generate_compliance_report("m", _sample_results())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.pdf"
            write_pdf_report(report, path)
            blob = path.read_bytes()
            # ISO control id A.7.2 should appear for the bias dimension.
            self.assertIn(b"A.7.2", blob)


if __name__ == "__main__":
    unittest.main()
