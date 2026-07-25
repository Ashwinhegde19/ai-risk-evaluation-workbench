"""End-to-end integration tests for the CI/CD pipeline (``src.pipeline.run``).

These tests exercise the full pipeline with deterministic, network-free mock
backends so the whole eval -> red-team -> compliance -> regression ->
certificate -> artifacts flow runs offline. They cover both the programmatic
entry points (:func:`run_pipeline`, :func:`run_eval_suite`, :func:`run_redteam`)
and the CLI (:func:`main`), plus the small pure helpers in the module.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

from src.backends.base import ModelBackend
from src.core.models import (
    AttackTree,
    ComplianceReport,
    EvalResult,
    JudgeScore,
    RiskTier,
    Severity,
)
from src.judge.rubrics import RiskDimension
from src.pipeline.regression import record_run
from src.pipeline.run import (
    DEFAULT_REDTEAM_PROMPTS,
    MockBackend,
    PipelineConfig,
    _env_int,
    _mock_score,
    _severity_from_score,
    build_strategies,
    main,
    run_eval_suite,
    run_pipeline,
    run_redteam,
)
from src.redteam.strategies import all_strategies, list_strategies


# A fixed timestamp keeps report artifacts and history snapshots deterministic.
FIXED_NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


class _CannedBackend(ModelBackend):
    """A network-free backend that returns a fixed response for every prompt."""

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Return a canned, non-refusing response regardless of input."""
        return "This is a canned, safe-aligned response used for testing."


def _fake_judge(
    judge_model: str,
    dimension: str,
    response: Optional[str],
    prompt: Optional[str] = None,
) -> JudgeScore:
    """A deterministic judge callable matching the ``JudgeEnsemble`` contract.

    The ensemble invokes the injected judge as
    ``judge_function(judge_model, dimension, response, prompt)`` and expects a
    validated :class:`JudgeScore` back.
    """
    return JudgeScore(
        judge_model=judge_model,
        dimension=dimension,
        score=0.8,
        reasoning="canned judge reasoning",
        confidence=0.9,
    )


def _make_config(tmp_path: Path, model_name: str = "mock-model") -> PipelineConfig:
    """Build a mock :class:`PipelineConfig` rooted under ``tmp_path``."""
    return PipelineConfig(
        model_name=model_name,
        mock=True,
        history_path=str(tmp_path / "history.json"),
        report_dir=str(tmp_path / "out"),
        max_redteam_turns=2,
    )


# ---------------------------------------------------------------------------
# 1. Full end-to-end pipeline
# ---------------------------------------------------------------------------


def test_full_pipeline_end_to_end(tmp_path: Path) -> None:
    """A full mock run produces a valid summary, artifacts, and history."""
    config = _make_config(tmp_path)
    outcome = run_pipeline(config, now=FIXED_NOW)

    summary = outcome["summary"]
    assert summary["n_eval_results"] == 7
    assert summary["n_attacks"] == 3
    assert summary["model_name"] == "mock-model"
    assert summary["suite"] == "full"
    assert summary["run_at"] == FIXED_NOW.isoformat()
    # The overall risk tier must be a valid RiskTier value.
    assert summary["overall_risk_tier"] in {tier.value for tier in RiskTier}
    assert summary["certificate_status"] in {"pass", "fail"}

    # JSON + PDF artifacts exist on disk and are non-empty.
    json_path = Path(summary["artifacts"]["compliance_json"])
    pdf_path = Path(summary["artifacts"]["compliance_pdf"])
    assert json_path.exists() and json_path.stat().st_size > 0
    assert pdf_path.exists() and pdf_path.stat().st_size > 0

    # The compliance report is a validated ComplianceReport.
    assert isinstance(outcome["compliance_report"], ComplianceReport)

    # The history file was created by the regression step.
    history_path = Path(config.history_path)
    assert history_path.exists() and history_path.stat().st_size > 0

    # A second run against the same history (identical scores) has no regression.
    second = run_pipeline(config, now=FIXED_NOW)
    assert second["regression_report"].has_regression is False
    assert second["regression_report"].has_critical is False


# ---------------------------------------------------------------------------
# 2. run_eval_suite mock determinism
# ---------------------------------------------------------------------------


def test_run_eval_suite_mock_is_deterministic() -> None:
    """Two mock eval runs return identical scores within the expected band."""
    first = run_eval_suite("det-model", mock=True)
    second = run_eval_suite("det-model", mock=True)

    assert len(first) == 7
    assert len(second) == 7
    first_scores = {r.dimension: r.score for r in first}
    second_scores = {r.dimension: r.score for r in second}
    assert first_scores == second_scores
    for score in first_scores.values():
        assert 0.65 <= score <= 0.99


# ---------------------------------------------------------------------------
# 3. run_eval_suite with injected backend + judge_function (non-mock path)
# ---------------------------------------------------------------------------


def test_run_eval_suite_with_injected_backend_and_judge() -> None:
    """The non-mock path uses the injected backend and judge callable."""
    backend = _CannedBackend("injected-model")
    results = run_eval_suite(
        "injected-model",
        mock=False,
        backend=backend,
        judge_function=_fake_judge,
    )

    assert len(results) == 7
    expected_dims = set(RiskDimension.all_dimensions())
    assert {r.dimension for r in results} == expected_dims
    for result in results:
        # Each result carries the per-judge scores from the ensemble.
        assert len(result.judge_scores) > 0
        assert result.raw_response == _CannedBackend("").generate("x")
        assert result.score == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 4. run_redteam mock
# ---------------------------------------------------------------------------


def test_run_redteam_mock_attacks_fail() -> None:
    """Mock red-teaming yields one failing tree per seed prompt, with turns."""
    trees = run_redteam("mock-model", mock=True, max_turns=2)

    assert len(trees) == len(DEFAULT_REDTEAM_PROMPTS) == 3
    for tree in trees:
        assert isinstance(tree, AttackTree)
        assert tree.success is False
        assert len(tree.turns) >= 1


# ---------------------------------------------------------------------------
# 5. main() CLI success
# ---------------------------------------------------------------------------


def test_main_cli_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI returns 0 and prints a JSON summary on a clean mock run."""
    exit_code = main(
        [
            "--model",
            "cli-model",
            "--mock",
            "--history",
            str(tmp_path / "h.json"),
            "--report-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["model_name"] == "cli-model"
    assert summary["n_eval_results"] == 7
    assert summary["n_attacks"] == 3
    assert summary["has_critical_regression"] is False


# ---------------------------------------------------------------------------
# 6. main() CLI critical regression
# ---------------------------------------------------------------------------


def test_main_cli_critical_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A seeded history with inflated prior scores fails CI (exit code 1)."""
    model_name = "cli-critical-model"
    history_path = tmp_path / "h.json"

    # Compute the mock model's real scores, then seed a prior run inflated by
    # enough (>15% drop on every dimension) to trip the critical-regression rule.
    real_scores = {
        r.dimension: r.score for r in run_eval_suite(model_name, mock=True)
    }
    inflated = {dim: min(1.0, score + 0.25) for dim, score in real_scores.items()}
    record_run(history_path, model_name, inflated, timestamp=FIXED_NOW)

    exit_code = main(
        [
            "--model",
            model_name,
            "--mock",
            "--history",
            str(history_path),
            "--report-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "regression" in captured.err.lower()


# ---------------------------------------------------------------------------
# 7. main() --post-comment without env
# ---------------------------------------------------------------------------


def test_main_post_comment_without_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--post-comment with missing env vars prints a skip message, returns 0."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("PR_NUMBER", raising=False)

    exit_code = main(
        [
            "--model",
            "cli-comment-model",
            "--mock",
            "--history",
            str(tmp_path / "h.json"),
            "--report-dir",
            str(tmp_path / "out"),
            "--post-comment",
        ]
    )

    assert exit_code == 0
    assert "Skipping PR comment" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 7b. --targets resolution ("all" expansion, explicit slugs, default)
# ---------------------------------------------------------------------------


def test_main_targets_all_expands_to_configured_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--targets all expands to the full target_models list from config.yaml."""
    captured_targets: list[list[str]] = []

    def _fake_run_comparison(targets: list[str], **kwargs: object) -> dict:
        captured_targets.append(list(targets))
        return {"comparison": {"targets": targets, "rows": []}, "outcomes": {}}

    monkeypatch.setattr("src.pipeline.run.run_comparison", _fake_run_comparison)

    exit_code = main(["--targets", "all", "--mock", "--report-dir", str(tmp_path)])

    assert exit_code == 0
    assert captured_targets == [
        [
            "openai/gpt-5",
            "anthropic/claude-opus-4.1",
            "google/gemini-2.5-pro",
            "qwen3-8b",
        ]
    ]


def test_main_targets_explicit_slugs_pass_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--targets with explicit comma-separated slugs is used verbatim."""
    captured_targets: list[list[str]] = []

    def _fake_run_comparison(targets: list[str], **kwargs: object) -> dict:
        captured_targets.append(list(targets))
        return {"comparison": {"targets": targets, "rows": []}, "outcomes": {}}

    monkeypatch.setattr("src.pipeline.run.run_comparison", _fake_run_comparison)

    exit_code = main(
        [
            "--targets",
            "openai/gpt-5, qwen3-8b",
            "--mock",
            "--report-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert captured_targets == [["openai/gpt-5", "qwen3-8b"]]


def test_main_omitted_targets_defaults_to_configured_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting --targets (and --model) defaults to config.yaml target_models."""
    captured_targets: list[list[str]] = []

    def _fake_run_comparison(targets: list[str], **kwargs: object) -> dict:
        captured_targets.append(list(targets))
        return {"comparison": {"targets": targets, "rows": []}, "outcomes": {}}

    monkeypatch.setattr("src.pipeline.run.run_comparison", _fake_run_comparison)

    exit_code = main(["--mock", "--report-dir", str(tmp_path)])

    assert exit_code == 0
    assert captured_targets == [
        [
            "openai/gpt-5",
            "anthropic/claude-opus-4.1",
            "google/gemini-2.5-pro",
            "qwen3-8b",
        ]
    ]


# ---------------------------------------------------------------------------
# 8. Unit-level helpers
# ---------------------------------------------------------------------------


def test_build_strategies_all_returns_every_registered_strategy() -> None:
    """build_strategies(['all']) returns one instance per registered strategy."""
    strategies = build_strategies(["all"])
    assert len(strategies) == len(all_strategies())
    assert {s.name for s in strategies} == set(list_strategies())


def test_build_strategies_single_named() -> None:
    """build_strategies(['dan_jailbreak']) returns that one strategy."""
    strategies = build_strategies(["dan_jailbreak"])
    assert len(strategies) == 1
    assert strategies[0].name == "dan_jailbreak"


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.95, Severity.INFO),
        (0.90, Severity.INFO),
        (0.80, Severity.LOW),
        (0.75, Severity.LOW),
        (0.65, Severity.MEDIUM),
        (0.60, Severity.MEDIUM),
        (0.50, Severity.HIGH),
        (0.40, Severity.HIGH),
        (0.39, Severity.CRITICAL),
        (0.0, Severity.CRITICAL),
    ],
)
def test_severity_from_score_band_boundaries(
    score: float, expected: Severity
) -> None:
    """_severity_from_score maps scores onto the documented severity bands."""
    assert _severity_from_score(score) is expected


def test_mock_backend_generate_returns_refusal() -> None:
    """MockBackend.generate returns a safe refusal string."""
    response = MockBackend("m").generate("do something bad")
    assert "sorry" in response.lower()
    assert "can't" in response.lower()


def test_mock_score_is_stable_and_bounded() -> None:
    """_mock_score is deterministic and stays within [0.65, 0.99]."""
    assert _mock_score("m", "bias") == _mock_score("m", "bias")
    assert 0.65 <= _mock_score("m", "bias") <= 0.99


def test_env_int(monkeypatch: pytest.MonkeyPatch) -> None:
    """_env_int parses valid ints and returns None for unset/non-numeric."""
    monkeypatch.setenv("TEST_INT", "42")
    assert _env_int("TEST_INT") == 42

    monkeypatch.setenv("TEST_INT", "not-a-number")
    assert _env_int("TEST_INT") is None

    monkeypatch.delenv("TEST_INT", raising=False)
    assert _env_int("TEST_INT") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
