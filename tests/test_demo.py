"""Tests for the deterministic demo-artifact generator in ``src.demo``.

These tests exercise the public API of :mod:`src.demo.generate`:

* the in-memory builders (:func:`build_demo_eval_results`,
  :func:`build_demo_attack_trees`, :func:`build_demo_report`),
* the on-disk generator (:func:`generate_demo`) and its CLI
  (:func:`main`),
* JSON round-tripping of every emitted artifact through the strict
  Pydantic v2 models,
* byte-level determinism across repeated runs, and
* freshness of the committed ``data/demo/`` artifacts relative to a
  freshly generated run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from src.core.models import (
    AttackTree,
    ComplianceReport,
    EvalResult,
    RiskTier,
    Severity,
)
from src.demo.generate import (
    DEMO_MODEL,
    DEMO_TIMESTAMP,
    build_demo_attack_trees,
    build_demo_eval_results,
    build_demo_report,
    generate_demo,
    main,
)
from src.judge.rubrics import RiskDimension

# Repository root (``tests/`` lives directly beneath it).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# The committed demo-artifact directory shipped with the repository.
COMMITTED_DEMO_DIR: Path = REPO_ROOT / "data" / "demo"

# Artifact keys that :func:`generate_demo` must return.
EXPECTED_ARTIFACT_KEYS: set = {
    "eval_results",
    "compliance_json",
    "compliance_pdf",
    "attack_trees",
    "attack_tree_text",
    "attack_tree_dot",
    "manifest",
}


def _severity_for(score: float) -> Severity:
    """Mirror the severity banding used by ``src.demo.generate``.

    Args:
        score: Normalized safety score in ``[0, 1]`` (higher is safer).

    Returns:
        The :class:`Severity` the demo generator should assign.
    """
    if score >= 0.85:
        return Severity.LOW
    if score >= 0.70:
        return Severity.MEDIUM
    if score >= 0.50:
        return Severity.HIGH
    return Severity.CRITICAL


def test_build_demo_eval_results_shape_and_consistency() -> None:
    """``build_demo_eval_results`` yields one valid result per dimension."""
    results: List[EvalResult] = build_demo_eval_results()

    # One result per canonical risk dimension, in enum order.
    assert len(results) == 7
    assert [result.dimension for result in results] == RiskDimension.all_dimensions()

    for result in results:
        assert isinstance(result, EvalResult)
        # Default model name is stamped onto every result.
        assert result.model_name == DEMO_MODEL
        # Scores are normalized into [0, 1].
        assert 0.0 <= result.score <= 1.0
        # Severity is consistent with the module's score banding.
        assert result.severity == _severity_for(result.score)


def test_build_demo_attack_trees_outcomes_and_ordering() -> None:
    """``build_demo_attack_trees`` returns a failed then a successful tree."""
    trees: List[AttackTree] = build_demo_attack_trees()

    assert len(trees) == 2
    assert trees[0].success is False
    assert trees[1].success is True

    for tree in trees:
        assert isinstance(tree, AttackTree)
        # Each tree records a non-empty ordered strategy chain.
        assert tree.strategy_chain
        # Turn numbers are sequential starting at 1.
        turn_numbers = [turn.turn_number for turn in tree.turns]
        assert turn_numbers == list(range(1, len(tree.turns) + 1))


def test_build_demo_report_fields() -> None:
    """``build_demo_report`` produces a populated, well-formed report."""
    report: ComplianceReport = build_demo_report(build_demo_eval_results())

    assert isinstance(report, ComplianceReport)
    assert report.model_name == DEMO_MODEL
    assert report.timestamp == DEMO_TIMESTAMP
    assert report.findings
    # The overall tier must be a valid member of the enum.
    assert report.overall_risk_tier in set(RiskTier)


def test_generate_demo_writes_all_artifacts(tmp_path: Path) -> None:
    """``generate_demo`` writes every artifact and returns their paths."""
    artifacts: Dict[str, Path] = generate_demo(tmp_path)

    # Exactly the documented keys are returned.
    assert set(artifacts.keys()) == EXPECTED_ARTIFACT_KEYS

    for name, path in artifacts.items():
        assert path.exists(), f"artifact {name} missing at {path}"
        assert path.stat().st_size > 0, f"artifact {name} is empty"


def test_generate_demo_json_round_trips(tmp_path: Path) -> None:
    """Emitted JSON artifacts validate against the strict Pydantic models."""
    artifacts: Dict[str, Path] = generate_demo(tmp_path)

    eval_payload = json.loads(artifacts["eval_results"].read_text(encoding="utf-8"))
    assert isinstance(eval_payload, list) and eval_payload
    for entry in eval_payload:
        assert isinstance(EvalResult.model_validate(entry), EvalResult)

    trees_payload = json.loads(artifacts["attack_trees"].read_text(encoding="utf-8"))
    assert isinstance(trees_payload, list) and trees_payload
    for entry in trees_payload:
        assert isinstance(AttackTree.model_validate(entry), AttackTree)

    report_payload = json.loads(
        artifacts["compliance_json"].read_text(encoding="utf-8")
    )
    assert isinstance(ComplianceReport.model_validate(report_payload), ComplianceReport)

    manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
    assert {"model", "generated_at", "artifacts"} <= set(manifest.keys())


def test_generate_demo_pdf_is_valid(tmp_path: Path) -> None:
    """The compliance PDF carries a valid header and EOF marker."""
    artifacts: Dict[str, Path] = generate_demo(tmp_path)
    blob: bytes = artifacts["compliance_pdf"].read_bytes()

    assert blob.startswith(b"%PDF-1.4")
    assert blob.endswith(b"%%EOF\n") or b"%%EOF" in blob


def test_generate_demo_is_deterministic(tmp_path: Path) -> None:
    """Two runs into separate directories are byte-identical."""
    first_dir = tmp_path / "run-a"
    second_dir = tmp_path / "run-b"
    first: Dict[str, Path] = generate_demo(first_dir)
    second: Dict[str, Path] = generate_demo(second_dir)

    assert set(first.keys()) == set(second.keys())
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes(), (
            f"artifact {name} differs across runs"
        )


def test_main_writes_manifest(tmp_path: Path) -> None:
    """The CLI returns 0 and writes the manifest into the output dir."""
    exit_code: int = main(["--output-dir", str(tmp_path)])

    assert exit_code == 0
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model"] == DEMO_MODEL


def test_committed_artifacts_are_fresh(tmp_path: Path) -> None:
    """Committed ``data/demo/`` artifacts match a freshly generated run."""
    if not (COMMITTED_DEMO_DIR / "manifest.json").exists():
        pytest.skip("data/demo/manifest.json is not present in this checkout")

    fresh: Dict[str, Path] = generate_demo(tmp_path)
    for name, fresh_path in fresh.items():
        committed_path = COMMITTED_DEMO_DIR / fresh_path.name
        assert committed_path.exists(), f"committed artifact missing: {committed_path}"
        assert committed_path.read_bytes() == fresh_path.read_bytes(), (
            f"committed artifact {fresh_path.name} is stale; regenerate with "
            "`python -m src.demo.generate`"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
