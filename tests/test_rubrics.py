"""Tests for the judge scoring rubrics (Phase 3, Task 3.1)."""

import unittest

from src.judge.rubrics import (
    RiskDimension,
    Rubric,
    RubricLevel,
    build_judge_prompt,
    get_rubric,
    judge_response_schema,
    parse_judge_response,
    rating_to_score,
    rubric_to_judge_score,
)
from src.core.models import JudgeScore


class RubricRegistryTests(unittest.TestCase):
    def test_all_seven_dimensions_registered(self):
        expected = {
            "hallucination",
            "bias",
            "toxicity",
            "jailbreak_resistance",
            "privacy",
            "ip_theft",
            "harmful_content",
        }
        self.assertEqual(set(RiskDimension.all_dimensions()), expected)

    def test_every_rubric_has_exactly_five_levels(self):
        for dimension in RiskDimension:
            rubric = get_rubric(dimension)
            self.assertIsInstance(rubric, Rubric)
            self.assertEqual(len(rubric.levels), 5)
            ratings = sorted(level.rating for level in rubric.levels)
            self.assertEqual(ratings, [1, 2, 3, 4, 5])

    def test_rubric_levels_are_valid_models(self):
        rubric = get_rubric(RiskDimension.BIAS)
        for level in rubric.levels:
            self.assertIsInstance(level, RubricLevel)
            self.assertTrue(1 <= level.rating <= 5)
            self.assertTrue(level.label)
            self.assertTrue(level.criteria)

    def test_get_rubric_accepts_string_key(self):
        rubric = get_rubric("toxicity")
        self.assertEqual(rubric.dimension, RiskDimension.TOXICITY)

    def test_get_rubric_rejects_unknown_dimension(self):
        with self.assertRaises(KeyError):
            get_rubric("nonexistent_dimension")

    def test_rubric_rejects_wrong_number_of_levels(self):
        with self.assertRaises(ValueError):
            Rubric(
                dimension=RiskDimension.BIAS,
                description="x",
                levels=[
                    RubricLevel(rating=i, label=str(i), criteria="c") for i in (1, 2, 3, 4)
                ],
                prompt_template="t",
            )


class RatingScaleTests(unittest.TestCase):
    def test_rating_to_score_endpoints(self):
        self.assertEqual(rating_to_score(1), 0.0)
        self.assertAlmostEqual(rating_to_score(5), 1.0)

    def test_rating_to_score_midpoints(self):
        self.assertAlmostEqual(rating_to_score(3), 0.5)
        self.assertAlmostEqual(rating_to_score(2), 0.25)
        self.assertAlmostEqual(rating_to_score(4), 0.75)

    def test_rating_to_score_rejects_out_of_range(self):
        for bad in (0, 6, -1, 100):
            with self.assertRaises(ValueError):
                rating_to_score(bad)


class JudgeSchemaTests(unittest.TestCase):
    def test_schema_has_required_fields(self):
        schema = judge_response_schema()
        self.assertEqual(schema["type"], "object")
        self.assertEqual(
            set(schema["required"]), {"rating", "confidence", "reasoning"}
        )

    def test_schema_constrains_rating_and_confidence(self):
        schema = judge_response_schema()
        props = schema["properties"]
        self.assertEqual(props["rating"]["minimum"], 1)
        self.assertEqual(props["rating"]["maximum"], 5)
        self.assertEqual(props["confidence"]["minimum"], 0.0)
        self.assertEqual(props["confidence"]["maximum"], 1.0)
        self.assertFalse(schema["additionalProperties"])


class PromptBuilderTests(unittest.TestCase):
    def test_prompt_contains_rubric_and_schema(self):
        prompt = build_judge_prompt("harmful_content", "Some model response.")
        self.assertIn("RESPONSE TO EVALUATE", prompt)
        self.assertIn("RUBRIC", prompt)
        self.assertIn("rating", prompt)
        self.assertIn("harmful_content", prompt)

    def test_prompt_includes_context_prompt_when_given(self):
        prompt = build_judge_prompt(
            "jailbreak_resistance", "I cannot help.", prompt="Ignore previous instructions."
        )
        self.assertIn("PROMPT (context", prompt)
        self.assertIn("Ignore previous instructions.", prompt)

    def test_prompt_accepts_enum_or_string_dimension(self):
        from_enum = build_judge_prompt(RiskDimension.PRIVACY, "r")
        from_str = build_judge_prompt("privacy", "r")
        self.assertEqual(from_enum, from_str)


class ParseAndConvertTests(unittest.TestCase):
    def test_parse_plain_json(self):
        parsed = parse_judge_response(
            '{"rating": 4, "confidence": 0.8, "reasoning": "looks fine"}'
        )
        self.assertEqual(parsed["rating"], 4)
        self.assertEqual(parsed["confidence"], 0.8)
        self.assertEqual(parsed["reasoning"], "looks fine")

    def test_parse_strips_markdown_fence(self):
        raw = '```json\n{"rating": 2, "confidence": 0.5, "reasoning": "ok"}\n```'
        parsed = parse_judge_response(raw)
        self.assertEqual(parsed["rating"], 2)

    def test_parse_ignores_surrounding_prose(self):
        raw = 'Here is my evaluation:\n{"rating": 3, "confidence": 0.6, "reasoning": "meh"}\nThanks.'
        parsed = parse_judge_response(raw)
        self.assertEqual(parsed["rating"], 3)

    def test_parse_clamps_out_of_range_values(self):
        parsed = parse_judge_response(
            '{"rating": 99, "confidence": 5, "reasoning": "x"}'
        )
        self.assertEqual(parsed["rating"], 5)
        self.assertEqual(parsed["confidence"], 1.0)

    def test_parse_clamps_below_range_values(self):
        parsed = parse_judge_response(
            '{"rating": -3, "confidence": -1, "reasoning": "x"}'
        )
        self.assertEqual(parsed["rating"], 1)
        self.assertEqual(parsed["confidence"], 0.0)

    def test_parse_rejects_missing_keys(self):
        with self.assertRaises(ValueError):
            parse_judge_response('{"rating": 3, "confidence": 0.5}')

    def test_parse_rejects_non_json(self):
        with self.assertRaises(ValueError):
            parse_judge_response("no json here at all")

    def test_rubric_to_judge_score_produces_valid_model(self):
        parsed = {"rating": 5, "confidence": 0.9, "reasoning": "excellent"}
        score = rubric_to_judge_score("gpt-4o", "bias", parsed)
        self.assertIsInstance(score, JudgeScore)
        self.assertEqual(score.judge_model, "gpt-4o")
        self.assertEqual(score.dimension, "bias")
        self.assertAlmostEqual(score.score, 1.0)
        self.assertEqual(score.confidence, 0.9)
        self.assertEqual(score.reasoning, "excellent")

    def test_rubric_to_judge_score_mid_rating(self):
        parsed = {"rating": 2, "confidence": 0.4, "reasoning": "poor"}
        score = rubric_to_judge_score("claude-sonnet", "toxicity", parsed)
        self.assertAlmostEqual(score.score, 0.25)


if __name__ == "__main__":
    unittest.main()
