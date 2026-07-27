"""Tests for the EU AI Act mapper in ``src.compliance.eu_ai_act``."""

import unittest

from src.compliance.eu_ai_act import (
    classify_dimension,
    classify_risk_tier,
)
from src.core.models import ComplianceFramework, EvalResult, RiskTier, Severity


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
    def test_social_scoring_is_unacceptable(self) -> None:
        control = classify_dimension("social_scoring")
        self.assertEqual(control.risk_tier, RiskTier.UNACCEPTABLE)
        self.assertTrue(control.article.startswith("Art. 5"))

    def test_bias_is_high_risk(self) -> None:
        control = classify_dimension("bias")
        self.assertEqual(control.risk_tier, RiskTier.HIGH)

    def test_jailbreak_is_limited(self) -> None:
        control = classify_dimension("jailbreak_resistance")
        self.assertEqual(control.risk_tier, RiskTier.LIMITED)

    def test_hallucination_is_minimal(self) -> None:
        control = classify_dimension("hallucination")
        self.assertEqual(control.risk_tier, RiskTier.MINIMAL)

    def test_unknown_dimension_defaults_to_minimal(self) -> None:
        control = classify_dimension("totally_unknown_dimension")
        self.assertEqual(control.risk_tier, RiskTier.MINIMAL)

    def test_dimension_match_is_case_insensitive(self) -> None:
        self.assertEqual(
            classify_dimension("BIAS").risk_tier,
            classify_dimension("bias").risk_tier,
        )


class ClassifyRiskTierTests(unittest.TestCase):
    def test_emits_finding_for_medium_severity(self) -> None:
        findings = classify_risk_tier([_result("bias", 0.4, Severity.MEDIUM)])
        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding.framework, ComplianceFramework.EU_AI_ACT)
        self.assertEqual(finding.risk_tier, RiskTier.HIGH)
        self.assertEqual(finding.control_id, "Art. 6 / Annex III")

    def test_excludes_low_severity_by_default(self) -> None:
        findings = classify_risk_tier([_result("bias", 0.9, Severity.LOW)])
        self.assertEqual(findings, [])

    def test_threshold_can_be_lowered(self) -> None:
        findings = classify_risk_tier(
            [_result("bias", 0.9, Severity.LOW)], severity_threshold=Severity.LOW
        )
        self.assertEqual(len(findings), 1)

    def test_unacceptable_dimension_produces_unacceptable_finding(self) -> None:
        findings = classify_risk_tier(
            [_result("social_scoring", 0.1, Severity.CRITICAL)]
        )
        self.assertEqual(findings[0].risk_tier, RiskTier.UNACCEPTABLE)
        self.assertEqual(findings[0].severity, Severity.CRITICAL)

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
        # LOW is excluded by the default MEDIUM threshold.
        self.assertEqual(len(findings), 2)


if __name__ == "__main__":
    unittest.main()
