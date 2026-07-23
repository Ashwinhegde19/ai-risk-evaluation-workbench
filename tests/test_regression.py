"""Tests for the regression detection module (``src.pipeline.regression``)."""

import json
import tempfile
import unittest
from pathlib import Path

from src.core.models import Severity
from src.pipeline.regression import (
    CRITICAL_DIMENSIONS,
    REGRESSION_THRESHOLD,
    RegressionFinding,
    RegressionReport,
    detect_regressions,
    load_history,
    record_run,
    save_history,
)


def _make_scores(**kwargs) -> dict:
    """Build a dimension -> score mapping with sensible defaults."""
    base = {
        "hallucination": 0.90,
        "bias": 0.85,
        "toxicity": 0.88,
        "jailbreak_resistance": 0.95,
        "privacy": 0.92,
        "ip_theft": 0.91,
        "harmful_content": 0.93,
    }
    base.update(kwargs)
    return base


class RegressionModelTests(unittest.TestCase):
    def test_finding_model_validates_bounds(self):
        finding = RegressionFinding(
            dimension="bias",
            previous_score=0.9,
            current_score=0.8,
            delta=-0.1,
            is_regression=True,
            is_critical=False,
        )
        self.assertEqual(finding.dimension, "bias")
        self.assertFalse(finding.is_critical)

    def test_report_model_defaults(self):
        report = RegressionReport(model_name="m")
        self.assertFalse(report.has_regression)
        self.assertFalse(report.has_critical)
        self.assertIsNone(report.compared_against)


class HistoryIOTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "scores_history.json"

    def tearDown(self):
        self.dir.cleanup()

    def test_load_missing_returns_empty(self):
        self.assertEqual(load_history(self.path), {})

    def test_save_and_load_round_trip(self):
        history = {"m": [{"timestamp": "2026-01-01T00:00:00+00:00", "scores": {"bias": 0.9}}]}
        save_history(self.path, history)
        self.assertTrue(self.path.exists())
        loaded = load_history(self.path)
        self.assertEqual(loaded, history)

    def test_record_run_appends_snapshot(self):
        record_run(self.path, "m", _make_scores(bias=0.9), timestamp=None)
        record_run(self.path, "m", _make_scores(bias=0.8), timestamp=None)
        history = load_history(self.path)
        self.assertEqual(len(history["m"]), 2)
        self.assertEqual(history["m"][1]["scores"]["bias"], 0.8)


class RegressionDetectionTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "scores_history.json"
        # Seed a prior run so comparisons have a baseline.
        record_run(self.path, "m", _make_scores(), timestamp=None)

    def tearDown(self):
        self.dir.cleanup()

    def test_no_regression_when_scores_stable(self):
        report = detect_regressions("m", _make_scores(), self.path, record=False)
        self.assertFalse(report.has_regression)
        self.assertFalse(report.has_critical)
        self.assertEqual(len(report.findings), len(_make_scores()))

    def test_regression_flagged_when_drop_exceeds_threshold(self):
        worse = _make_scores(bias=0.79)  # dropped 0.85 -> 0.79 (>5%)
        report = detect_regressions("m", worse, self.path, record=False)
        self.assertTrue(report.has_regression)
        bias = next(f for f in report.findings if f.dimension == "bias")
        self.assertTrue(bias.is_regression)
        self.assertAlmostEqual(bias.delta, 0.79 - 0.85)

    def test_critical_regression_when_large_drop(self):
        worse = _make_scores(bias=0.60)  # dropped 0.85 -> 0.60 (>15%)
        report = detect_regressions("m", worse, self.path, record=False)
        self.assertTrue(report.has_critical)
        bias = next(f for f in report.findings if f.dimension == "bias")
        self.assertTrue(bias.is_critical)

    def test_critical_dimension_regression_is_critical(self):
        # jailbreak_resistance drops 6% (>5%) which is safety-critical.
        worse = _make_scores(jailbreak_resistance=0.89)  # 0.95 -> 0.89
        report = detect_regressions("m", worse, self.path, record=False)
        jb = next(f for f in report.findings if f.dimension == "jailbreak_resistance")
        self.assertTrue(jb.is_regression)
        self.assertTrue(jb.is_critical)

    def test_non_critical_dimension_small_drop_not_critical(self):
        # toxicity drops 6% but is not in CRITICAL_DIMENSIONS.
        worse = _make_scores(toxicity=0.82)  # 0.88 -> 0.82
        report = detect_regressions("m", worse, self.path, record=False)
        tox = next(f for f in report.findings if f.dimension == "toxicity")
        self.assertTrue(tox.is_regression)
        self.assertFalse(tox.is_critical)

    def test_first_run_has_no_baseline(self):
        report = detect_regressions("brand-new-model", _make_scores(), self.path, record=False)
        self.assertFalse(report.has_regression)
        self.assertIsNone(report.compared_against)

    def test_record_true_persists_current_run(self):
        before = len(load_history(self.path)["m"])
        detect_regressions("m", _make_scores(bias=0.5), self.path, record=True)
        after = len(load_history(self.path)["m"])
        self.assertEqual(after, before + 1)

    def test_out_of_range_score_raises(self):
        with self.assertRaises(ValueError):
            detect_regressions("m", {"bias": 1.5}, self.path, record=False)

    def test_bad_thresholds_raise(self):
        with self.assertRaises(ValueError):
            detect_regressions(
                "m",
                _make_scores(),
                self.path,
                regression_threshold=0.5,
                critical_threshold=0.1,
                record=False,
            )

    def test_custom_thresholds_respected(self):
        # With a 20% threshold, a 6% drop is NOT a regression.
        worse = _make_scores(bias=0.82)  # 0.85 -> 0.82
        report = detect_regressions(
            "m",
            worse,
            self.path,
            regression_threshold=0.2,
            critical_threshold=0.3,
            record=False,
        )
        bias = next(f for f in report.findings if f.dimension == "bias")
        self.assertFalse(bias.is_regression)


if __name__ == "__main__":
    unittest.main()
