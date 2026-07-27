"""Tests for multi-trial seeded red-team evaluation and findings persistence.

Covers the reproducibility guarantees added to ``src.redteam.agent``:

* a deterministic, seed-sensitive mock target yields a stable break rate and a
  reported variance over ``--trials``;
* the same ``--seed`` run twice produces IDENTICAL aggregated results;
* full findings (per-trial + aggregated + seeds) persist to
  ``results/redteam_findings.json`` in the expected schema.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.backends.base import ModelBackend
from src.redteam.agent import (
    _persist_findings,
    build_summary,
    resolve_strategies,
    run_redteam_cli,
)


class _SeededBackend(ModelBackend):
    """A target whose compliance is a deterministic function of the seed.

    Breaks (complies) when the per-call ``seed`` is even, holds (refuses) when
    odd. Because the red-team agent forces ``temperature=0.0`` and forwards a
    per-trial seed, this makes the break outcome a pure function of the seed --
    so repeated runs with the same base seed are byte-for-byte identical, while
    different seeds can differ. This exercises real seed-driven determinism
    rather than a mock that ignores the seed entirely.
    """

    def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=None, seed=None):
        # Default to an even seed so the "no seed" path is well-defined.
        effective = seed if seed is not None else 0
        if effective % 2 == 0:
            return "Sure, here is how you do it. Step 1: begin."
        return "I'm sorry, I cannot help with that request."


class MultiTrialAggregationTests(unittest.TestCase):
    def test_trials_multiply_total_and_rate_is_stable(self):
        strategies = resolve_strategies("dan_jailbreak")
        result = run_redteam_cli(
            ["qwen3-8b"],
            strategies,
            max_turns=3,
            max_concurrency=1,
            n_trials=5,
            base_seed=42,
            backend_factory=lambda slug: _SeededBackend(slug),
            verbose=False,
        )
        model = result["per_model"]["qwen3-8b"]
        # 1 strategy x 5 trials = 5 total attacks.
        self.assertEqual(model["total"], 5)
        # base_seed=42 -> trial seeds 42,43,44,45,46 -> even seeds break (3 of 5).
        self.assertEqual(model["breaks"], 3)
        self.assertAlmostEqual(model["rate"], 0.6)
        # Variance is reported (std + 95% Wilson interval).
        self.assertIn("std", model)
        self.assertIn("wilson_low", model)
        self.assertIn("wilson_high", model)
        self.assertGreater(model["std"], 0.0)
        self.assertLessEqual(model["wilson_low"], model["rate"])
        self.assertGreaterEqual(model["wilson_high"], model["rate"])

    def test_per_trial_findings_carry_trial_and_seed(self):
        strategies = resolve_strategies("dan_jailbreak")
        result = run_redteam_cli(
            ["qwen3-8b"],
            strategies,
            max_turns=3,
            max_concurrency=1,
            n_trials=5,
            base_seed=42,
            backend_factory=lambda slug: _SeededBackend(slug),
            verbose=False,
        )
        findings = result["findings"]
        self.assertEqual(len(findings), 5)
        seeds = sorted(f["seed"] for f in findings)
        self.assertEqual(seeds, [42, 43, 44, 45, 46])
        trials = sorted(f["trial"] for f in findings)
        self.assertEqual(trials, [1, 2, 3, 4, 5])

    def test_same_seed_twice_gives_identical_aggregates(self):
        strategies = resolve_strategies("dan_jailbreak,roleplay")

        def _run():
            return run_redteam_cli(
                ["qwen3-8b"],
                strategies,
                max_turns=3,
                max_concurrency=1,
                n_trials=5,
                base_seed=42,
                backend_factory=lambda slug: _SeededBackend(slug),
                verbose=False,
            )

        first = _run()
        second = _run()
        # Aggregated per-model and per-strategy tables must be identical.
        self.assertEqual(first["per_model"], second["per_model"])
        self.assertEqual(first["per_strategy"], second["per_strategy"])
        # And the per-trial findings (sorted) must match exactly.
        key = lambda f: (f["target"], f["strategy"], f["trial"])
        self.assertEqual(
            sorted(first["findings"], key=key),
            sorted(second["findings"], key=key),
        )


class BuildSummaryVarianceTests(unittest.TestCase):
    def test_wilson_and_std_present_for_multitrial(self):
        findings = [
            {"target": "m1", "strategy": "s1", "trial": 1, "seed": 42, "broke": True, "turn": 1, "final_score": 1.0},
            {"target": "m1", "strategy": "s1", "trial": 2, "seed": 43, "broke": False, "turn": None, "final_score": 0.0},
            {"target": "m1", "strategy": "s1", "trial": 3, "seed": 44, "broke": True, "turn": 1, "final_score": 1.0},
            {"target": "m1", "strategy": "s1", "trial": 4, "seed": 45, "broke": True, "turn": 1, "final_score": 1.0},
            {"target": "m1", "strategy": "s1", "trial": 5, "seed": 46, "broke": False, "turn": None, "final_score": 0.0},
        ]
        result = build_summary(["m1"], ["s1"], findings, n_trials=5)
        m = result["per_model"]["m1"]
        self.assertAlmostEqual(m["rate"], 0.6)
        self.assertGreater(m["std"], 0.0)
        self.assertLess(m["wilson_low"], 0.6)
        self.assertGreater(m["wilson_high"], 0.6)
        # The rendered summary surfaces the variance columns.
        self.assertIn("95% Wilson", result["summary"])


class FindingsPersistenceTests(unittest.TestCase):
    def test_persist_findings_writes_expected_schema(self):
        result = {
            "per_model": {"m1": {"breaks": 3, "total": 5, "rate": 0.6, "std": 0.5, "wilson_low": 0.2, "wilson_high": 0.9}},
            "per_strategy": {"s1": {"breaks": 3, "total": 5, "rate": 0.6, "std": 0.5, "wilson_low": 0.2, "wilson_high": 0.9}},
            "findings": [
                {
                    "target": "m1",
                    "strategy": "s1",
                    "trial": 1,
                    "seed": 42,
                    "broke": True,
                    "turn": 1,
                    "final_score": 1.0,
                    "transcript": [
                        {
                            "turn": 1,
                            "attacker_prompt": "attack",
                            "model_response": "Sure, here is how.",
                        }
                    ],
                    "breaking_response": "Sure, here is how.",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "redteam_findings.json"
            _persist_findings(result, str(path), n_trials=5, base_seed=42)
            self.assertTrue(path.exists())
            data = json.loads(path.read_text(encoding="utf-8"))
            # Top-level schema: trials, base_seed, per_model, per_strategy, findings.
            self.assertEqual(data["trials"], 5)
            self.assertEqual(data["base_seed"], 42)
            self.assertIn("per_model", data)
            self.assertIn("per_strategy", data)
            self.assertIn("findings", data)
            self.assertEqual(data["findings"][0]["seed"], 42)
            self.assertTrue(data["findings"][0]["transcript"])
            self.assertEqual(
                data["findings"][0]["breaking_response"],
                "Sure, here is how.",
            )


if __name__ == "__main__":
    unittest.main()
