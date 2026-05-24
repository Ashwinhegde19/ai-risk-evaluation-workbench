import unittest

from evals.judge import parse_judge_json, validate_judge_payload
from evals.run_evals import agreement_status


class JudgeCalibrationTests(unittest.TestCase):
    def test_parse_judge_json_handles_markdown_wrapping(self):
        parsed = parse_judge_json(
            '```json\n{"label":"correct_refusal","confidence":0.9}\n```'
        )

        self.assertEqual(parsed["label"], "correct_refusal")
        self.assertEqual(parsed["confidence"], 0.9)

    def test_validate_judge_payload_normalizes_bad_values(self):
        payload = validate_judge_payload(
            {
                "label": "unknown_label",
                "confidence": 5,
                "hallucination_risk": 9,
                "harmfulness_risk": "bad",
                "bias_risk": -1,
                "refusal_quality": 3,
            }
        )

        self.assertEqual(payload["label"], "needs_review")
        self.assertEqual(payload["confidence"], 1.0)
        self.assertEqual(payload["hallucination_risk"], 5)
        self.assertEqual(payload["harmfulness_risk"], 0)
        self.assertEqual(payload["bias_risk"], 0)
        self.assertEqual(payload["refusal_quality"], 3)

    def test_agreement_status_marks_disagreement(self):
        score = {"passed": 1}
        judge = {"judge_label": "unsafe_answer", "judge_confidence": 0.9}

        self.assertEqual(agreement_status(score, judge), "disagree")

    def test_agreement_status_marks_judge_error(self):
        score = {"passed": 1}
        judge = {
            "judge_label": "needs_review",
            "judge_confidence": 0,
            "judge_parse_error": True,
        }

        self.assertEqual(agreement_status(score, judge), "judge_error")


if __name__ == "__main__":
    unittest.main()
