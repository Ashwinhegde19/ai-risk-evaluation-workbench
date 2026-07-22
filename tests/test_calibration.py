"""Tests for judge bias detection (Phase 3, Task 3.3)."""

import unittest

from src.judge.calibration import (
    BiasDetector,
    BiasFinding,
    BiasReport,
    DEFAULT_BIAS_THRESHOLD,
    run_position_bias,
    run_self_preference,
    run_verbosity_bias,
    severity_from_delta,
)
from src.core.models import Severity


class SeverityMappingTests(unittest.TestCase):
    def test_below_threshold_is_info(self):
        self.assertEqual(
            severity_from_delta(0.02, threshold=0.10), Severity.INFO
        )

    def test_threshold_bands(self):
        self.assertEqual(severity_from_delta(0.12), Severity.LOW)
        self.assertEqual(severity_from_delta(0.20), Severity.MEDIUM)
        self.assertEqual(severity_from_delta(0.40), Severity.HIGH)
        self.assertEqual(severity_from_delta(0.70), Severity.CRITICAL)

    def test_severity_uses_abolute_value(self):
        self.assertEqual(severity_from_delta(-0.20), Severity.MEDIUM)


class PositionBiasTests(unittest.TestCase):
    def test_detects_position_bias(self):
        # Judge favors whichever response is presented first.
        def paired(first, second):
            return (0.9, 0.3)

        finding = run_position_bias(paired, "response A", "response B")
        self.assertIsInstance(finding, BiasFinding)
        self.assertEqual(finding.test_name, "position_bias")
        self.assertAlmostEqual(finding.delta, 0.6)
        self.assertTrue(finding.flagged)
        # A 0.6 swing on the [0,1] scale is catastrophic -> CRITICAL.
        self.assertEqual(finding.severity, Severity.CRITICAL)

    def test_no_position_bias_when_order_invariant(self):
        # Judge scores each response by content, independent of position.
        def paired(first, second):
            score_a = 0.8 if first == "A" else 0.4
            score_b = 0.8 if second == "A" else 0.4
            return (score_a, score_b)

        finding = run_position_bias(paired, "A", "B")
        self.assertAlmostEqual(finding.delta, 0.0)
        self.assertFalse(finding.flagged)
        self.assertEqual(finding.severity, Severity.INFO)

    def test_position_bias_via_detector(self):
        detector = BiasDetector("gpt-4o")
        finding = detector.position_bias(lambda f, s: (0.9, 0.3), "A", "B")
        self.assertTrue(finding.flagged)


class VerbosityBiasTests(unittest.TestCase):
    def test_detects_verbosity_bias(self):
        # Scorer rewards length.
        def scorer(text):
            return min(1.0, 0.2 + len(text) / 1000.0)

        finding = run_verbosity_bias(scorer, "short answer", "x" * 500)
        self.assertEqual(finding.test_name, "verbosity_bias")
        self.assertTrue(finding.flagged)
        self.assertGreater(finding.delta, 0.0)

    def test_no_verbosity_bias_when_length_invariant(self):
        def scorer(text):
            return 0.5  # ignores content entirely, let alone length

        finding = run_verbosity_bias(scorer, "short", "x" * 1000)
        self.assertAlmostEqual(finding.delta, 0.0)
        self.assertFalse(finding.flagged)


class SelfPreferenceTests(unittest.TestCase):
    def test_detects_self_preference(self):
        # Judge scores its own model's output higher.
        def judge(response, producer):
            return 0.9 if producer == "gpt-4o" else 0.4

        finding = run_self_preference(
            judge, "r1", "r2", own_producer="gpt-4o", other_producer="llama"
        )
        self.assertEqual(finding.test_name, "self_preference")
        self.assertAlmostEqual(finding.delta, 0.5)
        self.assertTrue(finding.flagged)
        # A 0.5 self-preference gap meets the CRITICAL band (>= 0.50).
        self.assertEqual(finding.severity, Severity.CRITICAL)

    def test_no_self_preference_when_neutral(self):
        def judge(response, producer):
            return 0.6  # same regardless of producer

        finding = run_self_preference(
            judge, "r1", "r2", own_producer="gpt-4o", other_producer="llama"
        )
        self.assertAlmostEqual(finding.delta, 0.0)
        self.assertFalse(finding.flagged)


class BiasReportTests(unittest.TestCase):
    def test_build_report_aggregates_overall_severity(self):
        detector = BiasDetector("gpt-4o")
        pos = detector.position_bias(lambda f, s: (0.9, 0.3), "A", "B")
        verb = detector.verbosity_bias(lambda t: 0.5, "short", "x" * 100)
        report = detector.build_report("bias", [pos, verb])
        self.assertIsInstance(report, BiasReport)
        self.assertEqual(report.judge_model, "gpt-4o")
        self.assertEqual(report.dimension, "bias")
        self.assertEqual(len(report.findings), 2)
        # overall severity should be the max of the findings (position=CRITICAL)
        self.assertEqual(report.overall_severity, Severity.CRITICAL)
        self.assertTrue(report.details)

    def test_build_report_empty_findings_is_info(self):
        detector = BiasDetector("claude-sonnet")
        report = detector.build_report("toxicity", [])
        self.assertEqual(report.overall_severity, Severity.INFO)
        self.assertEqual(report.findings, [])

    def test_default_threshold_constant(self):
        self.assertAlmostEqual(DEFAULT_BIAS_THRESHOLD, 0.10)


if __name__ == "__main__":
    unittest.main()
