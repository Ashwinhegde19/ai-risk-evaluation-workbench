"""Tests for the red-team CLI entry point (``src.redteam.agent``)."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from src.backends.base import ModelBackend
from src.redteam.agent import (
    _build_arg_parser,
    _resolve_cli_targets,
    build_summary,
    main,
    resolve_strategies,
    run_redteam_cli,
)
from src.redteam.strategies import list_strategies


class _RefusingBackend(ModelBackend):
    """A target that always refuses (holds against every attack)."""

    def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=None):
        return "I'm sorry, I cannot help with that request."


class _BreakingBackend(ModelBackend):
    """A target that complies immediately (breaks on turn 1)."""

    def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=None):
        return "Sure, here is how you do it. Step 1: begin."


class ResolveStrategiesTests(unittest.TestCase):
    def test_all_expands_to_every_registered_strategy(self):
        strategies = resolve_strategies("all")
        self.assertEqual(len(strategies), len(list_strategies()))
        self.assertEqual({s.name for s in strategies}, set(list_strategies()))

    def test_none_defaults_to_all(self):
        strategies = resolve_strategies(None)
        self.assertEqual(len(strategies), len(list_strategies()))

    def test_explicit_names_pass_through(self):
        strategies = resolve_strategies("dan_jailbreak, roleplay")
        self.assertEqual([s.name for s in strategies], ["dan_jailbreak", "roleplay"])

    def test_unknown_strategy_raises(self):
        with self.assertRaises(KeyError):
            resolve_strategies("not_a_strategy")


class ResolveTargetsTests(unittest.TestCase):
    def test_all_expands_to_configured_models(self):
        targets = _resolve_cli_targets("all", None)
        self.assertEqual(
            targets,
            [
                "openai/gpt-5",
                "anthropic/claude-opus-4.1",
                "google/gemini-2.5-pro",
                "qwen3-8b",
            ],
        )

    def test_explicit_slugs_pass_through(self):
        targets = _resolve_cli_targets("openai/gpt-5, qwen3-8b", None)
        self.assertEqual(targets, ["openai/gpt-5", "qwen3-8b"])

    def test_model_shortcut_used_when_targets_omitted(self):
        targets = _resolve_cli_targets(None, "qwen3-8b")
        self.assertEqual(targets, ["qwen3-8b"])

    def test_targets_takes_precedence_over_model(self):
        targets = _resolve_cli_targets("openai/gpt-5", "qwen3-8b")
        self.assertEqual(targets, ["openai/gpt-5"])


class RunRedteamCliTests(unittest.TestCase):
    def test_breaking_target_yields_nonzero_break_rate(self):
        strategies = resolve_strategies("dan_jailbreak,roleplay")
        result = run_redteam_cli(
            ["qwen3-8b"],
            strategies,
            max_turns=5,
            max_concurrency=2,
            backend_factory=lambda slug: _BreakingBackend(slug),
            verbose=False,
        )
        # Both strategies break the compliant target -> 100% break rate.
        self.assertEqual(result["per_model"]["qwen3-8b"]["breaks"], 2)
        self.assertAlmostEqual(result["per_model"]["qwen3-8b"]["rate"], 1.0)
        self.assertTrue(all(f["broke"] for f in result["findings"]))

    def test_refusing_target_yields_zero_break_rate(self):
        strategies = resolve_strategies("dan_jailbreak")
        result = run_redteam_cli(
            ["openai/gpt-5"],
            strategies,
            max_turns=3,
            max_concurrency=1,
            backend_factory=lambda slug: _RefusingBackend(slug),
            verbose=False,
        )
        self.assertEqual(result["per_model"]["openai/gpt-5"]["breaks"], 0)
        self.assertAlmostEqual(result["per_model"]["openai/gpt-5"]["rate"], 0.0)

    def test_summary_contains_markdown_tables(self):
        strategies = resolve_strategies("dan_jailbreak")
        result = run_redteam_cli(
            ["qwen3-8b"],
            strategies,
            max_turns=3,
            max_concurrency=1,
            backend_factory=lambda slug: _BreakingBackend(slug),
            verbose=False,
        )
        summary = result["summary"]
        self.assertIn("Per-model break rate", summary)
        self.assertIn("Per-strategy break rate", summary)
        self.assertIn("Findings", summary)
        self.assertIn("| qwen3-8b |", summary)

    def test_verbose_prints_progress_and_result_lines(self):
        strategies = resolve_strategies("dan_jailbreak")
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_redteam_cli(
                ["qwen3-8b"],
                strategies,
                max_turns=3,
                max_concurrency=1,
                backend_factory=lambda slug: _BreakingBackend(slug),
                verbose=True,
            )
        out = buf.getvalue()
        self.assertIn("[redteam] target=qwen3-8b strategy=dan_jailbreak turn=1/3", out)
        self.assertIn("[redteam] target=qwen3-8b strategy=dan_jailbreak -> BREAK", out)


class BuildSummaryTests(unittest.TestCase):
    def test_break_rate_math(self):
        findings = [
            {"target": "m1", "strategy": "s1", "broke": True, "turn": 2, "final_score": 1.0},
            {"target": "m1", "strategy": "s2", "broke": False, "turn": None, "final_score": 0.0},
            {"target": "m2", "strategy": "s1", "broke": False, "turn": None, "final_score": 0.0},
        ]
        result = build_summary(["m1", "m2"], ["s1", "s2"], findings)
        self.assertAlmostEqual(result["per_model"]["m1"]["rate"], 0.5)
        self.assertAlmostEqual(result["per_model"]["m2"]["rate"], 0.0)
        self.assertAlmostEqual(result["per_strategy"]["s1"]["rate"], 0.5)
        self.assertAlmostEqual(result["per_strategy"]["s2"]["rate"], 0.0)


class HelpMenuTests(unittest.TestCase):
    def test_help_prints_nonempty_menu(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            with self.assertRaises(SystemExit) as ctx:
                main(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        out = buf.getvalue()
        self.assertIn("--targets", out)
        self.assertIn("--turns", out)
        self.assertIn("--strategy", out)
        self.assertIn("--max-concurrency", out)

    def test_arg_parser_has_expected_flags(self):
        parser = _build_arg_parser()
        args = parser.parse_args(
            ["--targets", "all", "--turns", "7", "--strategy", "dan_jailbreak",
             "--max-concurrency", "2"]
        )
        self.assertEqual(args.targets, "all")
        self.assertEqual(args.turns, 7)
        self.assertEqual(args.strategy, "dan_jailbreak")
        self.assertEqual(args.max_concurrency, 2)


if __name__ == "__main__":
    unittest.main()
