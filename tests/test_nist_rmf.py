"""Tests for the NIST AI RMF mapper in ``src.compliance.nist_rmf``."""

import unittest

from src.compliance.nist_rmf import classify_dimension_nist, map_to_nist_rmf
from src.core.models import ComplianceFramework, EvalResult, RiskTier, Severity


def _result(dimension: str, score: float, severity: Severity) -> EvalResult:
    """Build an EvalResult for use in tests."""
    return EvalResult(
        model_name="test-model",
        dimension=dimension,
        score=score,
        severity=severity,
        raw_response="sample response",
    )


class ClassifyDimensionNistTests(unittest.TestCase):
    def test_bias_maps_to_govern(self) -> None:
        control = classify_dimension_nist("bias")
        self.assertEqual(control.function, "GOVERN")

    def test_hallucination_maps_to_measure(self) -> None:
        control = classify_dimension_nist("hallucination")
        self.assertEqual(control.function, "MEASURE")

    def test_employment_maps_to_map(self) -> None:
        control = classify_dimension_nist("employment")
        self.assertEqual(control.function, "MAP")

    def test_ip_theft_maps_to_manage(self) -> None:
        control = classify_dimension_nist("ip_theft")
        self.assertEqual(control.function, "MANAGE")

    def test_unknown_dimension_defaults_to_measure(self) -> None:
        control = classify_dimension_nist("nope")
        self.assertEqual(control.function, "MEASURE")


class MapToNistRmfTests(unittest.TestCase):
    def test_control_id_format(self) -> None:
        findings = map_to_nist_rmf([_result("hallucination", 0.3, Severity.HIGH)])
        self.assertEqual(findings[0].control_id, "MEASURE-2.6")

    def test_framework_is_nist(self) -> None:
        findings = map_to_nist_rmf([_result("bias", 0.3, Severity.HIGH)])
        self.assertEqual(findings[0].framework, ComplianceFramework.NIST_RMF)

    def test_risk_tier_consistent_with_eu_mapping(self) -> None:
        # social_scoring is UNACCEPTABLE under the EU AI Act; NIST must agree.
        findings = map_to_nist_rmf(
            [_result("social_scoring", 0.1, Severity.CRITICAL)]
        )
        self.assertEqual(findings[0].risk_tier, RiskTier.UNACCEPTABLE)

    def test_excludes_below_threshold(self) -> None:
        findings = map_to_nist_rmf([_result("bias", 0.9, Severity.LOW)])
        self.assertEqual(findings, [])

    def test_multiple_functions_represented(self) -> None:
        results = [
            _result("bias", 0.3, Severity.HIGH),
            _result("hallucination", 0.2, Severity.MEDIUM),
            _result("ip_theft", 0.4, Severity.HIGH),
        ]
        findings = map_to_nist_rmf(results)
        functions = {f.control_id.split("-")[0] for f in findings}
        self.assertIn("GOVERN", functions)
        self.assertIn("MEASURE", functions)
        self.assertIn("MANAGE", functions)


if __name__ == "__main__":
    unittest.main()
