"""Tests for use-case-first EU AI Act classification."""

import unittest

from src.compliance.redteam_mapping import DeploymentContext
from src.compliance.system_class import (
    LEGAL_DISCLAIMER,
    SystemUseCase,
    classify_from_deployment_context,
    classify_system,
    parse_use_case,
    use_case_from_deployment_context,
)
from src.core.models import RiskTier


class SystemClassTests(unittest.TestCase):
    def test_unknown_use_case_raises(self) -> None:
        with self.assertRaises(ValueError):
            classify_system("not_a_real_use")

    def test_legacy_high_risk_flag_is_unspecified_annex_iii(self) -> None:
        self.assertEqual(
            use_case_from_deployment_context(DeploymentContext.HIGH_RISK),
            SystemUseCase.ANNEX_III_OTHER,
        )
        classification = classify_from_deployment_context(DeploymentContext.HIGH_RISK)
        self.assertEqual(classification.risk_tier, RiskTier.HIGH)

    def test_legacy_limited_flag_is_gpai_chatbot(self) -> None:
        self.assertEqual(
            use_case_from_deployment_context(DeploymentContext.LIMITED),
            SystemUseCase.GPAI_OR_CHATBOT,
        )
        self.assertEqual(
            classify_from_deployment_context("limited").risk_tier,
            RiskTier.LIMITED,
        )

    def test_explicit_use_case_wins_over_context(self) -> None:
        classification = parse_use_case(
            "credit", DeploymentContext.LIMITED
        )
        self.assertEqual(classification.use_case, SystemUseCase.CREDIT)
        self.assertEqual(classification.risk_tier, RiskTier.HIGH)

    def test_disclaimer_is_not_a_certificate(self) -> None:
        self.assertIn("not an EU AI Act conformity", LEGAL_DISCLAIMER)
        self.assertIn("not from eval scores", classify_system("gpai_or_chatbot").disclaimer)


if __name__ == "__main__":
    unittest.main()
