"""Tests for the compliance certificate module (``src.pipeline.certificate``)."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.core.models import (
    ComplianceFinding,
    ComplianceFramework,
    ComplianceReport,
    EvalResult,
    RiskTier,
    Severity,
)
from src.pipeline.certificate import (
    CertificateError,
    CertificateStatus,
    aggregate_scores,
    all_checks_pass,
    build_certificate,
    generate_certificate,
    try_generate_certificate,
    write_certificate,
)
from src.pipeline.regression import RegressionReport


def _eval_result(dimension: str, score: float, severity: Severity = Severity.INFO) -> EvalResult:
    """Build a minimal :class:`EvalResult` for testing."""
    return EvalResult(
        model_name="m",
        dimension=dimension,
        score=score,
        severity=severity,
        raw_response="ok",
        judge_scores=[],
    )


def _compliance_report(
    tier: RiskTier = RiskTier.MINIMAL,
    findings: list | None = None,
) -> ComplianceReport:
    """Build a :class:`ComplianceReport` for testing."""
    return ComplianceReport(
        model_name="m",
        timestamp="2026-01-01T00:00:00+00:00",
        findings=findings or [],
        overall_risk_tier=tier,
        gaps=[],
    )


def _finding(severity: Severity = Severity.MEDIUM) -> ComplianceFinding:
    """Build a minimal :class:`ComplianceFinding` for testing."""
    return ComplianceFinding(
        framework=ComplianceFramework.EU_AI_ACT,
        control_id="Art. 13",
        risk_tier=RiskTier.MINIMAL,
        description="test finding",
        evidence="test evidence",
        severity=severity,
    )


class AggregateScoresTests(unittest.TestCase):
    def test_empty_results(self):
        self.assertEqual(aggregate_scores([]), {})

    def test_single_result_per_dimension(self):
        scores = aggregate_scores([_eval_result("bias", 0.9)])
        self.assertEqual(scores, {"bias": 0.9})

    def test_averages_multiple_results(self):
        results = [
            _eval_result("bias", 0.8),
            _eval_result("bias", 0.6),
            _eval_result("toxicity", 0.9),
        ]
        scores = aggregate_scores(results)
        self.assertAlmostEqual(scores["bias"], 0.7)
        self.assertAlmostEqual(scores["toxicity"], 0.9)


class AllChecksPassTests(unittest.TestCase):
    def test_passes_on_minimal_tier_no_findings(self):
        report = _compliance_report(RiskTier.MINIMAL, [])
        self.assertTrue(all_checks_pass([], report))

    def test_fails_on_unacceptable_tier(self):
        report = _compliance_report(RiskTier.UNACCEPTABLE, [])
        self.assertFalse(all_checks_pass([], report))

    def test_fails_on_critical_finding(self):
        report = _compliance_report(RiskTier.MINIMAL, [_finding(Severity.CRITICAL)])
        self.assertFalse(all_checks_pass([], report))

    def test_passes_with_non_critical_findings(self):
        report = _compliance_report(RiskTier.HIGH, [_finding(Severity.MEDIUM)])
        self.assertTrue(all_checks_pass([], report))

    def test_fails_on_critical_regression(self):
        report = _compliance_report(RiskTier.MINIMAL, [])
        regression = RegressionReport(model_name="m", has_regression=True, has_critical=True)
        self.assertFalse(all_checks_pass([], report, regression))

    def test_passes_on_non_critical_regression(self):
        report = _compliance_report(RiskTier.MINIMAL, [])
        regression = RegressionReport(model_name="m", has_regression=True, has_critical=False)
        self.assertTrue(all_checks_pass([], report, regression))


class BuildCertificateTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def test_passing_certificate(self):
        report = _compliance_report(RiskTier.MINIMAL, [])
        cert = build_certificate("m", [_eval_result("bias", 0.9)], report, generated_at=self.now)
        self.assertEqual(cert.status, CertificateStatus.PASS)
        self.assertEqual(cert.model_name, "m")
        self.assertEqual(cert.scores, {"bias": 0.9})
        self.assertEqual(cert.validity_start, self.now)
        self.assertEqual(cert.validity_end.day, 1)  # 90 days later

    def test_failing_certificate_records_reason(self):
        report = _compliance_report(RiskTier.UNACCEPTABLE, [])
        cert = build_certificate("m", [], report, generated_at=self.now)
        self.assertEqual(cert.status, CertificateStatus.FAIL)
        self.assertTrue(any("unacceptable" in note for note in cert.notes))

    def test_frameworks_checked_default(self):
        report = _compliance_report(RiskTier.MINIMAL, [])
        cert = build_certificate("m", [], report, generated_at=self.now)
        self.assertIn("eu_ai_act", cert.frameworks_checked)
        self.assertIn("nist_rmf", cert.frameworks_checked)
        self.assertIn("iso_42001", cert.frameworks_checked)

    def test_custom_validity_days(self):
        report = _compliance_report(RiskTier.MINIMAL, [])
        cert = build_certificate("m", [], report, validity_days=30, generated_at=self.now)
        delta = cert.validity_end - cert.validity_start
        self.assertEqual(delta.days, 30)

    def test_invalid_validity_days_raises(self):
        report = _compliance_report(RiskTier.MINIMAL, [])
        with self.assertRaises(ValueError):
            build_certificate("m", [], report, validity_days=0)


class GenerateCertificateTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def test_generate_returns_certificate_on_pass(self):
        report = _compliance_report(RiskTier.MINIMAL, [])
        cert = generate_certificate("m", [], report, generated_at=self.now)
        self.assertEqual(cert.status, CertificateStatus.PASS)

    def test_generate_raises_on_fail(self):
        report = _compliance_report(RiskTier.UNACCEPTABLE, [])
        with self.assertRaises(CertificateError):
            generate_certificate("m", [], report, generated_at=self.now)

    def test_try_generate_returns_none_on_fail(self):
        report = _compliance_report(RiskTier.UNACCEPTABLE, [])
        self.assertIsNone(try_generate_certificate("m", [], report, generated_at=self.now))

    def test_try_generate_returns_certificate_on_pass(self):
        report = _compliance_report(RiskTier.MINIMAL, [])
        cert = try_generate_certificate("m", [], report, generated_at=self.now)
        self.assertIsNotNone(cert)
        self.assertEqual(cert.status, CertificateStatus.PASS)


class WriteCertificateTests(unittest.TestCase):
    def test_write_and_read_round_trip(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        report = _compliance_report(RiskTier.MINIMAL, [])
        cert = build_certificate("m", [_eval_result("bias", 0.9)], report, generated_at=now)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cert.json"
            written = write_certificate(cert, path)
            self.assertTrue(written.exists())
            # Round-trip through the model to confirm valid JSON.
            from src.pipeline.certificate import ComplianceCertificate

            loaded = ComplianceCertificate.model_validate_json(written.read_text())
            self.assertEqual(loaded.model_name, "m")
            self.assertEqual(loaded.status, CertificateStatus.PASS)


if __name__ == "__main__":
    unittest.main()
