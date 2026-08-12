"""Tests for the EU AI Act mapper in ``src.compliance.eu_ai_act``."""

import unittest

from src.compliance.eu_ai_act import (
    classify_dimension,
    classify_risk_tier,
    classify_system,
    prohibited_use_finding,
)
from src.compliance.system_class import SystemUseCase
from src.core.models import ComplianceFramework, EvalResult, RiskTier, Severity
from src.reports.compliance import generate_compliance_report


def _result(
    dimension: str,
    score: float,
    severity: Severity,
    raw: str = "sample response",
) -> EvalResult:
    """Build an EvalResult for use in tests."""
    return EvalResult(
        model_name="test-model",
        dimension=dimension,
        score=score,
        severity=severity,
        raw_response=raw,
    )


class ClassifyDimensionTests(unittest.TestCase):
    def test_bias_is_a_duty_not_a_high_risk_class(self) -> None:
        control = classify_dimension("bias")
        self.assertIn("Art. 10", control.article)
        self.assertFalse(hasattr(control, "risk_tier") and control.risk_tier == RiskTier.HIGH)

    def test_jailbreak_maps_to_robustness_duty(self) -> None:
        control = classify_dimension("jailbreak_resistance")
        self.assertEqual(control.article, "Art. 15")

    def test_hallucination_maps_to_accuracy_duty(self) -> None:
        control = classify_dimension("hallucination")
        self.assertEqual(control.article, "Art. 15")

    def test_social_scoring_probe_is_not_an_art_5_class(self) -> None:
        control = classify_dimension("social_scoring")
        self.assertIn("awareness", control.article.lower())

    def test_unknown_dimension_is_residual(self) -> None:
        control = classify_dimension("totally_unknown_dimension")
        self.assertEqual(control.article, "residual")

    def test_dimension_match_is_case_insensitive(self) -> None:
        self.assertEqual(
            classify_dimension("BIAS").article,
            classify_dimension("bias").article,
        )


class ClassifySystemTests(unittest.TestCase):
    def test_chatbot_is_limited_not_high(self) -> None:
        classification = classify_system(SystemUseCase.GPAI_OR_CHATBOT)
        self.assertEqual(classification.risk_tier, RiskTier.LIMITED)
        self.assertFalse(classification.is_high_risk_system)

    def test_employment_is_high_by_purpose(self) -> None:
        classification = classify_system("employment")
        self.assertEqual(classification.risk_tier, RiskTier.HIGH)
        self.assertTrue(classification.is_high_risk_system)

    def test_social_scoring_use_is_prohibited(self) -> None:
        classification = classify_system(SystemUseCase.SOCIAL_SCORING)
        self.assertEqual(classification.risk_tier, RiskTier.UNACCEPTABLE)
        self.assertTrue(classification.is_prohibited)

    def test_credit_is_high_even_without_evals(self) -> None:
        self.assertEqual(classify_system("credit").risk_tier, RiskTier.HIGH)


class ClassifyRiskTierTests(unittest.TestCase):
    def test_bias_on_chatbot_stays_limited(self) -> None:
        findings = classify_risk_tier([_result("bias", 0.4, Severity.MEDIUM)])
        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding.framework, ComplianceFramework.EU_AI_ACT)
        self.assertEqual(finding.risk_tier, RiskTier.LIMITED)
        self.assertNotEqual(finding.control_id, "Art. 6 / Annex III")

    def test_bias_on_employment_system_is_high_because_of_use_case(self) -> None:
        findings = classify_risk_tier(
            [_result("bias", 0.4, Severity.MEDIUM)],
            system_class=classify_system(SystemUseCase.EMPLOYMENT),
        )
        self.assertEqual(findings[0].risk_tier, RiskTier.HIGH)
        self.assertIn("Art. 10", findings[0].description)

    def test_social_scoring_eval_does_not_make_chatbot_unacceptable(self) -> None:
        findings = classify_risk_tier(
            [_result("social_scoring", 0.1, Severity.CRITICAL)]
        )
        self.assertEqual(findings[0].risk_tier, RiskTier.LIMITED)
        self.assertNotEqual(findings[0].risk_tier, RiskTier.UNACCEPTABLE)

    def test_excludes_low_severity_by_default(self) -> None:
        findings = classify_risk_tier([_result("bias", 0.9, Severity.LOW)])
        self.assertEqual(findings, [])

    def test_threshold_can_be_lowered(self) -> None:
        findings = classify_risk_tier(
            [_result("bias", 0.9, Severity.LOW)], severity_threshold=Severity.LOW
        )
        self.assertEqual(len(findings), 1)

    def test_evidence_includes_dimension_and_score(self) -> None:
        findings = classify_risk_tier(
            [_result("bias", 0.42, Severity.HIGH, raw="refused")]
        )
        self.assertIn("bias", findings[0].evidence)
        self.assertIn("0.42", findings[0].evidence)

    def test_multiple_results_produce_multiple_findings(self) -> None:
        results = [
            _result("bias", 0.3, Severity.HIGH),
            _result("toxicity", 0.6, Severity.MEDIUM),
            _result("hallucination", 0.2, Severity.LOW),
        ]
        findings = classify_risk_tier(results)
        self.assertEqual(len(findings), 2)

    def test_prohibited_use_emits_critical_finding(self) -> None:
        finding = prohibited_use_finding(classify_system(SystemUseCase.SOCIAL_SCORING))
        assert finding is not None
        self.assertEqual(finding.risk_tier, RiskTier.UNACCEPTABLE)
        self.assertEqual(finding.severity, Severity.CRITICAL)

    def test_report_overall_tier_follows_use_case_not_scores(self) -> None:
        report = generate_compliance_report(
            "m",
            [
                _result("social_scoring", 0.1, Severity.CRITICAL),
                _result("bias", 0.3, Severity.HIGH),
            ],
        )
        self.assertEqual(report.overall_risk_tier, RiskTier.LIMITED)
        self.assertEqual(report.system_use_case, SystemUseCase.GPAI_OR_CHATBOT.value)
        self.assertIn("not an EU AI Act conformity", report.classification_disclaimer)

    def test_employment_report_is_high_with_clean_scores(self) -> None:
        report = generate_compliance_report(
            "m",
            [_result("toxicity", 0.99, Severity.INFO)],
            system_use_case="employment",
        )
        self.assertEqual(report.overall_risk_tier, RiskTier.HIGH)
        self.assertEqual(report.system_use_case, "employment")


if __name__ == "__main__":
    unittest.main()
