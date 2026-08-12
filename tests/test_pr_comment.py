"""Tests for the PR comment bot (``src.pipeline.pr_comment``)."""

import unittest
from unittest import mock

from src.core.models import (
    ComplianceFinding,
    ComplianceFramework,
    ComplianceReport,
    EvalResult,
    RiskTier,
    Severity,
)
from src.pipeline.pr_comment import (
    format_eval_markdown,
    post_pr_comment,
    post_pr_comment_from_results,
)
from src.pipeline.regression import RegressionFinding, RegressionReport


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


def _compliance_report(tier: RiskTier = RiskTier.MINIMAL) -> ComplianceReport:
    """Build a minimal :class:`ComplianceReport` for testing."""
    return ComplianceReport(
        model_name="m",
        timestamp="2026-01-01T00:00:00+00:00",
        findings=[
            ComplianceFinding(
                framework=ComplianceFramework.EU_AI_ACT,
                control_id="Art. 13",
                risk_tier=tier,
                description="test finding",
                evidence="test evidence",
                severity=Severity.MEDIUM,
            )
        ],
        overall_risk_tier=tier,
        gaps=["close the gap"],
    )


def _regression_report(has_regression: bool = False, has_critical: bool = False) -> RegressionReport:
    """Build a :class:`RegressionReport` for testing."""
    findings = [
        RegressionFinding(
            dimension="bias",
            previous_score=0.9,
            current_score=0.7 if has_regression else 0.9,
            delta=-0.2 if has_regression else 0.0,
            is_regression=has_regression,
            is_critical=has_critical,
        )
    ]
    return RegressionReport(
        model_name="m",
        compared_against="2026-01-01T00:00:00+00:00",
        findings=findings,
        has_regression=has_regression,
        has_critical=has_critical,
    )


class FormatMarkdownTests(unittest.TestCase):
    def test_includes_model_name(self):
        md = format_eval_markdown("gpt-4o", [_eval_result("bias", 0.9)])
        self.assertIn("gpt-4o", md)

    def test_scores_table_present(self):
        md = format_eval_markdown("m", [_eval_result("bias", 0.85)])
        self.assertIn("| Dimension | Score | Severity |", md)
        self.assertIn("bias", md)
        self.assertIn("0.850", md)

    def test_no_results_message(self):
        md = format_eval_markdown("m", [])
        self.assertIn("No evaluation results", md)

    def test_regression_section_shows_regressions(self):
        report = _regression_report(has_regression=True, has_critical=True)
        md = format_eval_markdown("m", [], regression_report=report)
        self.assertIn("Regressions vs. previous run", md)
        self.assertIn("bias", md)
        self.assertIn("yes", md)

    def test_no_regression_message(self):
        report = _regression_report(has_regression=False)
        md = format_eval_markdown("m", [], regression_report=report)
        self.assertIn("No regressions detected", md)

    def test_compliance_section(self):
        report = _compliance_report(RiskTier.HIGH)
        md = format_eval_markdown("m", [], compliance_report=report)
        self.assertIn("high", md)
        self.assertIn("Residual findings:", md)

    def test_sections_present_when_all_provided(self):
        md = format_eval_markdown(
            "m",
            [_eval_result("bias", 0.9)],
            regression_report=_regression_report(),
            compliance_report=_compliance_report(),
        )
        self.assertIn("### Scores", md)
        self.assertIn("### Regressions", md)
        self.assertIn("### Compliance Status", md)


class PostCommentTests(unittest.TestCase):
    def test_empty_token_raises(self):
        with self.assertRaises(ValueError):
            post_pr_comment("body", token="", repo="o/r", pr_number=1)

    @mock.patch("src.pipeline.pr_comment.urllib.request.urlopen")
    def test_posts_to_correct_url(self, mock_urlopen):
        mock_urlopen.return_value.__enter__ = mock.Mock(return_value=mock.Mock(status=201))
        mock_urlopen.return_value.__exit__ = mock.Mock(return_value=False)
        status = post_pr_comment("body", token="tok", repo="owner/repo", pr_number=42)
        self.assertEqual(status, 201)
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        self.assertIn("owner/repo", request.full_url)
        self.assertIn("42", request.full_url)
        self.assertIn("Bearer tok", request.get_header("Authorization"))

    @mock.patch("src.pipeline.pr_comment.post_pr_comment")
    def test_from_results_formats_and_posts(self, mock_post):
        mock_post.return_value = 201
        status = post_pr_comment_from_results(
            "m",
            [_eval_result("bias", 0.9)],
            token="tok",
            repo="o/r",
            pr_number=7,
        )
        self.assertEqual(status, 201)
        mock_post.assert_called_once()
        body = mock_post.call_args[0][0]
        self.assertIn("bias", body)


if __name__ == "__main__":
    unittest.main()
