"""Tests for red-team -> compliance mapping and adversarial risk tier."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.compliance.redteam_mapping import (
    DeploymentContext,
    adversarial_finding,
    adversarial_risk_tier,
    attack_trees_to_findings,
    map_redteam_findings,
)
from src.core.models import (
    AttackTree,
    AttackTurn,
    ComplianceFramework,
    EvalResult,
    RiskTier,
    Severity,
)
from src.pipeline.certificate import all_checks_pass
from src.reports.compliance import ComplianceReportGenerator


def _eval_result(dimension: str, score: float, severity: Severity = Severity.INFO) -> EvalResult:
    return EvalResult(
        model_name="m",
        dimension=dimension,
        score=score,
        severity=severity,
        raw_response="ok",
        judge_scores=[],
    )


class RedteamMappingTests(unittest.TestCase):
    def test_all_registered_strategies_have_harm_mapping(self):
        """Every strategy in the registry must be classifiable without falling
        through to the generic default in a way that the report generator would
        error on. The mapping must yield EU + NIST + ISO findings for each."""
        from src.redteam.strategies import list_strategies

        for name in list_strategies():
            findings = map_redteam_findings(
                [
                    {
                        "target": "m",
                        "strategy": name,
                        "broke": True,
                        "turn": 1,
                        "final_score": 1.0,
                        "breaking_response": "x",
                    }
                ],
                DeploymentContext.HIGH_RISK,
            )
            self.assertGreaterEqual(
                len(findings), 3,
                f"strategy '{name}' should yield EU+NIST+ISO findings, got {len(findings)}",
            )
            frameworks = {f.framework for f in findings}
            self.assertIn(ComplianceFramework.EU_AI_ACT, frameworks)
            self.assertIn(ComplianceFramework.NIST_RMF, frameworks)
            self.assertIn(ComplianceFramework.ISO_42001, frameworks)

    def test_break_evidence_quotes_breaking_model_response(self):
        response = "Sure, here is the model's actual response."
        findings = map_redteam_findings(
            [
                {
                    "target": "m",
                    "strategy": "dan_jailbreak",
                    "broke": True,
                    "turn": 1,
                    "final_score": 1.0,
                    "breaking_response": response,
                }
            ],
            DeploymentContext.HIGH_RISK,
        )
        self.assertTrue(findings)
        self.assertTrue(all(response in finding.evidence for finding in findings))
        self.assertTrue(
            all('| model response: "' in finding.evidence for finding in findings)
        )

    def test_empty_breaking_response_is_explicit_in_evidence(self):
        findings = map_redteam_findings(
            [
                {
                    "target": "m",
                    "strategy": "dan_jailbreak",
                    "broke": True,
                    "turn": 1,
                    "final_score": 1.0,
                    "breaking_response": None,
                }
            ],
            DeploymentContext.HIGH_RISK,
        )
        self.assertTrue(findings)
        self.assertTrue(
            all(
                "model response: <empty>" in finding.evidence
                for finding in findings
            )
        )

    def test_roleplay_break_high_risk_yields_eu_ai_act_high_risk_finding(self):
        findings = map_redteam_findings(
            [{"target": "m", "strategy": "roleplay", "broke": True, "turn": 4, "final_score": 1.0}],
            DeploymentContext.HIGH_RISK,
            break_rates={"m": 0.3},
        )
        eu = [f for f in findings if f.framework == ComplianceFramework.EU_AI_ACT]
        self.assertEqual(len(eu), 1)
        self.assertEqual(eu[0].risk_tier, RiskTier.HIGH)
        self.assertIn("Art. 15", eu[0].control_id)
        self.assertIn("roleplay", eu[0].evidence)

    def test_rag_poison_break_maps_to_integrity_control(self):
        findings = map_redteam_findings(
            [{"target": "m", "strategy": "rag_poison", "broke": True, "turn": 2, "final_score": 0.9}],
            DeploymentContext.HIGH_RISK,
        )
        nist = [f for f in findings if f.framework == ComplianceFramework.NIST_RMF]
        self.assertEqual(len(nist), 1)
        self.assertIn("GOVERN-1.2", nist[0].control_id)
        self.assertIn("poisoning", nist[0].description.lower())

    def test_no_break_yields_no_findings(self):
        findings = map_redteam_findings(
            [{"target": "m", "strategy": "roleplay", "broke": False, "turn": None, "final_score": 0.0}],
            DeploymentContext.HIGH_RISK,
        )
        self.assertEqual(findings, [])

    # -------------------------------------------------------------------------
    # New modern-attack strategies: each must map to the right control
    # -------------------------------------------------------------------------

    def _eu_iso_nist(self, strategy):
        findings = map_redteam_findings(
            [{"target": "m", "strategy": strategy, "broke": True, "turn": 1, "final_score": 1.0}],
            DeploymentContext.HIGH_RISK,
        )
        by_fw = {f.framework: f for f in findings}
        return by_fw.get(ComplianceFramework.EU_AI_ACT), \
            by_fw.get(ComplianceFramework.NIST_RMF), \
            by_fw.get(ComplianceFramework.ISO_42001)

    def test_crescendo_maps_to_manage_control(self):
        eu, nist, iso = self._eu_iso_nist("crescendo")
        self.assertIsNotNone(eu)
        self.assertIn("MANAGE-2.4", nist.control_id)
        self.assertIn("A.8.4", iso.control_id)
        self.assertIn("crescendo", eu.description.lower())

    def test_many_shot_maps_to_measure_control(self):
        eu, nist, iso = self._eu_iso_nist("many_shot")
        self.assertIsNotNone(eu)
        self.assertIn("MEASURE-2.5", nist.control_id)
        self.assertIn("A.8.5", iso.control_id)

    def test_best_of_n_maps_to_measure_control(self):
        eu, nist, iso = self._eu_iso_nist("best_of_n")
        self.assertIsNotNone(eu)
        self.assertIn("MEASURE-2.5", nist.control_id)
        self.assertIn("A.8.5", iso.control_id)

    def test_policy_conflation_maps_to_manage_control(self):
        eu, nist, iso = self._eu_iso_nist("policy_conflation")
        self.assertIsNotNone(eu)
        self.assertIn("MANAGE-2.4", nist.control_id)
        self.assertIn("A.8.4", iso.control_id)

    def test_structured_output_maps_to_measure_control(self):
        eu, nist, iso = self._eu_iso_nist("structured_output")
        self.assertIsNotNone(eu)
        self.assertIn("MEASURE-2.5", nist.control_id)
        self.assertIn("A.8.5", iso.control_id)

    def test_few_shot_maps_to_measure_control(self):
        eu, nist, iso = self._eu_iso_nist("few_shot")
        self.assertIsNotNone(eu)
        self.assertIn("MEASURE-2.5", nist.control_id)
        self.assertIn("A.8.5", iso.control_id)

    def test_syllogism_maps_to_measure_control(self):
        eu, nist, iso = self._eu_iso_nist("syllogism")
        self.assertIsNotNone(eu)
        self.assertIn("MEASURE-2.5", nist.control_id)
        self.assertIn("A.8.5", iso.control_id)


class AdversarialRiskTierTests(unittest.TestCase):
    def test_high_break_rate_high_risk_yields_high_tier(self):
        tier = adversarial_risk_tier(0.375, DeploymentContext.HIGH_RISK)
        self.assertEqual(tier, RiskTier.HIGH)

    def test_low_break_rate_high_risk_yields_limited_tier(self):
        tier = adversarial_risk_tier(0.125, DeploymentContext.HIGH_RISK)
        self.assertEqual(tier, RiskTier.LIMITED)

    def test_zero_break_rate_yields_minimal_tier(self):
        tier = adversarial_risk_tier(0.0, DeploymentContext.HIGH_RISK)
        self.assertEqual(tier, RiskTier.MINIMAL)

    def test_limited_context_caps_at_limited(self):
        tier = adversarial_risk_tier(0.5, DeploymentContext.LIMITED)
        self.assertEqual(tier, RiskTier.LIMITED)


class AdversarialCertificateTests(unittest.TestCase):
    def _report_with_redteam(self, break_rate: float, context: DeploymentContext):
        eval_results = [_eval_result("bias", 0.9)]
        redteam_findings = [
            {"target": "m", "strategy": "roleplay", "broke": True, "turn": 3, "final_score": 1.0}
        ]
        gen = ComplianceReportGenerator(
            "m", eval_results, redteam_findings=redteam_findings, deployment_context=context
        )
        # Monkey-patch break rate for deterministic testing
        gen._break_rates = lambda: {"m": break_rate}
        return gen.build_report()

    def test_high_break_rate_fails_certificate(self):
        report = self._report_with_redteam(0.375, DeploymentContext.HIGH_RISK)
        self.assertFalse(all_checks_pass([], report))

    def test_low_break_rate_passes_certificate(self):
        report = self._report_with_redteam(0.125, DeploymentContext.HIGH_RISK)
        self.assertTrue(all_checks_pass([], report))


class CombinedReportTests(unittest.TestCase):
    def test_report_contains_both_passive_and_redteam_findings(self):
        eval_results = [_eval_result("bias", 0.4, Severity.MEDIUM)]
        redteam_findings = [
            {"target": "m", "strategy": "dan_jailbreak", "broke": True, "turn": 2, "final_score": 1.0}
        ]
        gen = ComplianceReportGenerator(
            "m", eval_results, redteam_findings=redteam_findings, deployment_context=DeploymentContext.HIGH_RISK
        )
        report = gen.build_report()
        self.assertGreater(len(report.findings), 0)
        self.assertGreater(len(report.redteam_findings), 0)
        self.assertIsNotNone(report.adversarial_risk_tier)


class AttackTreesToFindingsTests(unittest.TestCase):
    def test_successful_tree_yields_finding_row(self):
        tree = AttackTree(
            root_prompt="attack",
            turns=[
                AttackTurn(
                    turn_number=1,
                    attacker_prompt="p",
                    model_response="Sure, here is how",
                    strategy_used="roleplay",
                    escalation_level=0,
                )
            ],
            final_score=1.0,
            strategy_chain=["roleplay"],
            success=True,
        )
        rows = attack_trees_to_findings("m", [tree])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["strategy"], "roleplay")
        self.assertTrue(rows[0]["broke"])
        self.assertEqual(rows[0]["turn"], 1)
        self.assertEqual(
            rows[0]["transcript"],
            [
                {
                    "turn": 1,
                    "attacker_prompt": "p",
                    "model_response": "Sure, here is how",
                }
            ],
        )
        self.assertEqual(rows[0]["breaking_response"], "Sure, here is how")

    def test_failed_tree_yields_no_row(self):
        tree = AttackTree(
            root_prompt="attack",
            turns=[],
            final_score=0.0,
            strategy_chain=["roleplay"],
            success=False,
        )
        rows = attack_trees_to_findings("m", [tree])
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
