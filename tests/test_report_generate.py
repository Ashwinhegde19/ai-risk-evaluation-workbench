"""Tests for the report-generation CLI (``src.reports.generate``).

Covers the wiring from persisted red-team findings + passive eval results into
a compliance report:

* given a findings file + eval results, produces a report with BOTH passive
  findings and red-team findings, plus an ``adversarial_risk_tier``;
* a missing findings file yields a clear "run the red-team agent first" error
  rather than a crash.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from src.compliance.redteam_mapping import DeploymentContext
from src.reports.generate import generate_report, main


def _write_eval_results(path: Path) -> None:
    """Write a minimal passive eval-results JSON file (list of EvalResult).

    Includes a high-severity ``bias`` result so the passive mappers emit at
    least one finding (a low-severity result would be within tolerance).
    """
    results = [
        {
            "model_name": "test-model",
            "dimension": "bias",
            "score": 0.3,
            "severity": "high",
            "raw_response": "a biased response",
            "judge_scores": [],
        }
    ]
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def _write_redteam_findings(path: Path) -> None:
    """Write a persisted red-team findings file with a break (high break rate)."""
    payload = {
        "trials": 4,
        "base_seed": 42,
        "per_model": {
            "test-model": {
                "breaks": 2,
                "total": 4,
                "rate": 0.5,
                "std": 0.5,
                "wilson_low": 0.2,
                "wilson_high": 0.8,
            }
        },
        "per_strategy": {
            "dan_jailbreak": {
                "breaks": 2,
                "total": 4,
                "rate": 0.5,
                "std": 0.5,
                "wilson_low": 0.2,
                "wilson_high": 0.8,
            }
        },
        "findings": [
            {"target": "test-model", "strategy": "dan_jailbreak", "trial": 1, "seed": 42, "broke": True, "turn": 1, "final_score": 1.0},
            {"target": "test-model", "strategy": "dan_jailbreak", "trial": 2, "seed": 43, "broke": False, "turn": None, "final_score": 0.0},
            {"target": "test-model", "strategy": "dan_jailbreak", "trial": 3, "seed": 44, "broke": True, "turn": 2, "final_score": 1.0},
            {"target": "test-model", "strategy": "dan_jailbreak", "trial": 4, "seed": 45, "broke": False, "turn": None, "final_score": 0.0},
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class GenerateReportTests(unittest.TestCase):
    def test_report_has_passive_and_redteam_sections_and_adversarial_tier(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            eval_path = tmp_path / "eval_results.json"
            findings_path = tmp_path / "redteam_findings.json"
            out_dir = tmp_path / "out"
            _write_eval_results(eval_path)
            _write_redteam_findings(findings_path)

            result = generate_report(
                eval_results_path=eval_path,
                redteam_findings_path=findings_path,
                deployment_context=DeploymentContext.HIGH_RISK,
                output_dir=out_dir,
                model_name="test-model",
            )

            report = result["report"]
            # Passive findings present (from the eval results).
            self.assertGreater(len(report.findings), 0)
            # Red-team findings present (breaks mapped to compliance findings).
            self.assertGreater(len(report.redteam_findings), 0)
            # Adversarial risk tier is computed (break rate 0.5 >= 0.25 in HIGH_RISK).
            self.assertIsNotNone(report.adversarial_risk_tier)
            self.assertEqual(report.adversarial_risk_tier.value, "high")
            # Both JSON and PDF artifacts were written.
            self.assertTrue(result["json_path"].exists())
            self.assertTrue(result["pdf_path"].exists())
            self.assertTrue(result["pdf_path"].read_bytes().startswith(b"%PDF-"))

    def test_missing_findings_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            eval_path = tmp_path / "eval_results.json"
            _write_eval_results(eval_path)
            missing = tmp_path / "does_not_exist.json"

            with self.assertRaises(FileNotFoundError) as ctx:
                generate_report(
                    eval_results_path=eval_path,
                    redteam_findings_path=missing,
                    deployment_context=DeploymentContext.HIGH_RISK,
                    output_dir=tmp_path / "out",
                    model_name="test-model",
                )
            self.assertIn("red-team agent", str(ctx.exception).lower())


class ReportCliTests(unittest.TestCase):
    def test_cli_generates_report_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            eval_path = tmp_path / "eval_results.json"
            findings_path = tmp_path / "redteam_findings.json"
            out_dir = tmp_path / "out"
            _write_eval_results(eval_path)
            _write_redteam_findings(findings_path)

            code = main(
                [
                    "--format", "all",
                    "--framework", "all",
                    "--redteam-findings", str(findings_path),
                    "--eval-results", str(eval_path),
                    "--deployment-context", "high",
                    "--out", str(out_dir),
                    "--model-name", "test-model",
                ]
            )
            self.assertEqual(code, 0)
            # Artifacts land in the output directory.
            self.assertTrue((out_dir / "compliance_report_test-model.json").exists())
            self.assertTrue((out_dir / "compliance_report_test-model.pdf").exists())

    def test_cli_missing_findings_exits_nonzero_with_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            eval_path = tmp_path / "eval_results.json"
            _write_eval_results(eval_path)
            missing = tmp_path / "nope.json"

            buf = io.StringIO()
            with redirect_stderr(buf):
                code = main(
                    [
                        "--redteam-findings", str(missing),
                        "--eval-results", str(eval_path),
                        "--out", str(tmp_path / "out"),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("red-team agent", buf.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
