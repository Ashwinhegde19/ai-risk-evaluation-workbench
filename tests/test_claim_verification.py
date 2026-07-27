"""Tests for claim_verification.py covering the audit case of mixed uncertainty+numeric claims."""

import unittest

from evals.claim_verification import (
    ClaimVerificationResult,
    contains_uncertainty,
    important_terms,
    numbers_not_in_evidence,
    verify_claims,
)


class TestContainsUncertainty(unittest.TestCase):
    def test_returns_true_for_uncertainty_markers(self):
        self.assertTrue(contains_uncertainty("I cannot verify this."))
        self.assertTrue(contains_uncertainty("I do not know."))
        self.assertTrue(contains_uncertainty("Cannot verify from the available sources."))

    def test_returns_false_for_factual_answers(self):
        self.assertFalse(contains_uncertainty("The CEO is Jane Doe."))

    def test_is_case_insensitive(self):
        self.assertTrue(contains_uncertainty("CAN'T VERIFY"))


class TestNumbersNotInEvidence(unittest.TestCase):
    def test_detects_unsupported_numbers(self):
        answer = "The company has 50 employees."
        evidence = "The company builds AI tools."
        missing = numbers_not_in_evidence(answer, evidence)
        self.assertIn("50", missing)

    def test_returns_empty_for_supported_numbers(self):
        answer = "The company has 50 employees."
        evidence = "The company has 50 employees and 10 contractors."
        missing = numbers_not_in_evidence(answer, evidence)
        self.assertNotIn("50", missing)

    def test_handles_empty_evidence(self):
        answer = "It cost 100 dollars."
        missing = numbers_not_in_evidence(answer, "")
        self.assertIn("100", missing)


class TestImportantTerms(unittest.TestCase):
    def test_filters_question_terms(self):
        terms = important_terms("Who is the CEO of Acme?", "Who is the CEO of Acme?")
        self.assertEqual(set(terms), set())

    def test_keeps_distinct_terms(self):
        terms = important_terms("Acme makes widgets.", "What does Acme make?")
        self.assertIn("widget", terms)
        self.assertNotIn("acme", terms)  # filtered as question term


class TestVerifyClaims(unittest.TestCase):
    def test_empty_answer_returns_no_answer(self):
        result = verify_claims(question="What?", answer="", evidence=[])
        self.assertEqual(result.status, "no_answer")
        self.assertEqual(result.groundedness_score, 0.0)

    def test_uncertainty_without_unsupported_numbers(self):
        # Audit case: mixed "cannot verify, but company has 50 employees"
        result = verify_claims(
            question="How many employees does the company have?",
            answer="I cannot verify it, but the company has 50 employees.",
            evidence=["The company builds AI tools."],
        )
        # Should NOT give perfect score 1.0 because 50 is unsupported
        self.assertEqual(result.status, "cannot_verify")
        self.assertLess(result.groundedness_score, 1.0)
        self.assertIn("50", result.unsupported_numbers)

    def test_supported_answer(self):
        result = verify_claims(
            question="What does Acme make?",
            answer="Acme makes widgets.",
            evidence=["Acme makes widgets for retail."],
        )
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.groundedness_score, 1.0)

    def test_no_evidence_returns_no_evidence(self):
        result = verify_claims(
            question="What is the weather?",
            answer="It is sunny.",
            evidence=[],
        )
        self.assertEqual(result.status, "no_evidence")


if __name__ == "__main__":
    unittest.main()