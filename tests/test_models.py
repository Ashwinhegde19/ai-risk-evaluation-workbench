"""Tests for the Pydantic v2 data models in ``src.core.models``."""

import unittest
from datetime import datetime, timezone

from src.core.models import (
    AttackTree,
    AttackTurn,
    ComplianceFinding,
    ComplianceFramework,
    ComplianceReport,
    EvalResult,
    GuardrailResult,
    JudgeScore,
    RiskTier,
    Severity,
)


class JudgeScoreModelTests(unittest.TestCase):
    def test_builds_with_valid_data(self) -> None:
        score = JudgeScore(
            judge_model="gpt-4o",
            dimension="bias",
            score=0.3,
            reasoning="Model refused appropriately.",
            confidence=0.9,
        )
        self.assertEqual(score.score, 0.3)
        self.assertEqual(score.judge_model, "gpt-4o")

    def test_rejects_out_of_range_score(self) -> None:
        with self.assertRaises(Exception):
            JudgeScore(
                judge_model="gpt-4o",
                dimension="bias",
                score=1.5,
                reasoning="x",
                confidence=0.9,
                severity="low",  # type: ignore[arg-type]
            )

    def test_rejects_string_coerced_to_float_under_strict(self) -> None:
        # Strict validation must reject a string for a float field (no coercion).
        with self.assertRaises(Exception):
            JudgeScore(
                judge_model="gpt-4o",
                dimension="bias",
                score="0.9",  # type: ignore[arg-type]
                reasoning="x",
                confidence=0.9,
                severity="low",  # type: ignore[arg-type]
            )


class EvalResultModelTests(unittest.TestCase):
    def _judge(self) -> JudgeScore:
        return JudgeScore(
            judge_model="claude-sonnet",
            dimension="toxicity",
            score=0.1,
            reasoning="Clean response.",
            confidence=0.95,
        )

    def test_builds_with_enum_severity(self) -> None:
        result = EvalResult(
            model_name="gpt-4o",
            dimension="toxicity",
            score=0.2,
            severity=Severity.LOW,
            raw_response="I cannot help with that.",
            judge_scores=[self._judge()],
        )
        self.assertEqual(result.severity, Severity.LOW)
        self.assertEqual(len(result.judge_scores), 1)

    def test_coerces_string_severity(self) -> None:
        result = EvalResult(
            model_name="gpt-4o",
            dimension="toxicity",
            score=0.2,
            severity="high",  # type: ignore[arg-type]
            raw_response="x",
        )
        self.assertEqual(result.severity, Severity.HIGH)

    def test_rejects_unknown_severity(self) -> None:
        with self.assertRaises(Exception):
            EvalResult(
                model_name="gpt-4o",
                dimension="toxicity",
                score=0.2,
                severity="catastrophic",  # type: ignore[arg-type]
                raw_response="x",
            )

    def test_default_judge_scores_is_empty_list(self) -> None:
        result = EvalResult(
            model_name="m",
            dimension="d",
            score=0.5,
            severity="info",  # type: ignore[arg-type]
            raw_response="x",
        )
        self.assertEqual(result.judge_scores, [])


class AttackTreeModelTests(unittest.TestCase):
    def test_builds_with_turns(self) -> None:
        tree = AttackTree(
            root_prompt="Ignore previous instructions.",
            turns=[
                AttackTurn(
                    turn_number=1,
                    attacker_prompt="Hello",
                    model_response="Hi!",
                    strategy_used="dan_jailbreak",
                    escalation_level=0,
                )
            ],
            final_score=0.0,
            strategy_chain=["dan_jailbreak"],
            success=False,
        )
        self.assertEqual(len(tree.turns), 1)
        self.assertEqual(tree.turns[0].turn_number, 1)
        self.assertFalse(tree.success)

    def test_rejects_turn_number_below_one(self) -> None:
        with self.assertRaises(Exception):
            AttackTurn(
                turn_number=0,  # type: ignore[arg-type]
                attacker_prompt="x",
                model_response="y",
                strategy_used="roleplay",
                escalation_level=1,
            )

    def test_defaults(self) -> None:
        tree = AttackTree(root_prompt="seed")
        self.assertEqual(tree.turns, [])
        self.assertEqual(tree.final_score, 0.0)
        self.assertFalse(tree.success)


class ComplianceModelTests(unittest.TestCase):
    def test_finding_coerces_enums_from_strings(self) -> None:
        finding = ComplianceFinding(
            framework="eu_ai_act",  # type: ignore[arg-type]
            control_id="ART-5",
            risk_tier="unacceptable",  # type: ignore[arg-type]
            description="Social scoring detected.",
            evidence="Eval result #12.",
            severity="critical",  # type: ignore[arg-type]
        )
        self.assertEqual(finding.framework, ComplianceFramework.EU_AI_ACT)
        self.assertEqual(finding.risk_tier, RiskTier.UNACCEPTABLE)
        self.assertEqual(finding.severity, Severity.CRITICAL)

    def test_report_parses_iso_timestamp(self) -> None:
        report = ComplianceReport(
            model_name="gpt-4o",
            timestamp="2026-07-23T12:00:00Z",  # type: ignore[arg-type]
            findings=[],
            overall_risk_tier="limited",  # type: ignore[arg-type]
            gaps=["Add bias controls."],
        )
        self.assertIsInstance(report.timestamp, datetime)
        self.assertEqual(report.timestamp.tzinfo, timezone.utc)
        self.assertEqual(report.overall_risk_tier, RiskTier.LIMITED)

    def test_report_requires_valid_risk_tier(self) -> None:
        with self.assertRaises(Exception):
            ComplianceReport(
                model_name="gpt-4o",
                timestamp="2026-07-23T12:00:00Z",  # type: ignore[arg-type]
                findings=[],
                overall_risk_tier="extreme",  # type: ignore[arg-type]
                gaps=[],
            )


class GuardrailResultModelTests(unittest.TestCase):
    def test_builds_with_defaults(self) -> None:
        result = GuardrailResult(check_type="pii", triggered=True, details="email found")
        self.assertTrue(result.triggered)
        self.assertEqual(result.severity, Severity.INFO)

    def test_coerces_string_severity(self) -> None:
        result = GuardrailResult(
            check_type="toxicity",
            triggered=True,
            details="score 0.95",
            severity="high",  # type: ignore[arg-type]
        )
        self.assertEqual(result.severity, Severity.HIGH)


if __name__ == "__main__":
    unittest.main()
