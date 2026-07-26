"""Report generation CLI for the AI Risk Evaluation Workbench.

Wires passive evaluation results and red-team findings into a unified
compliance report. Loads:

* **Passive eval results** from ``--eval-results`` (default: latest in
  ``results/``), and
* **Red-team findings** from ``--redteam-findings`` (default:
  ``results/redteam_findings.json``).

Maps both onto the EU AI Act, NIST AI RMF, and ISO/IEC 42001 frameworks,
computes an adversarially-aware risk tier, and emits JSON + PDF reports with
separate passive and "Red-Team (Adversarial)" sections.

Usage::

    python -m src.reports.generate --format all --framework all
    python -m src.reports.generate --redteam-findings results/redteam_findings.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from src.compliance.redteam_mapping import DeploymentContext
from src.core.models import EvalResult
from src.reports.compliance import (
    ComplianceReportGenerator,
    write_json_report,
    write_pdf_report,
)


def _load_eval_results(path: Path) -> List[EvalResult]:
    """Load passive evaluation results from a JSON file.

    Args:
        path: Path to the eval results JSON (a list of EvalResult objects).

    Returns:
        A list of :class:`EvalResult` objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is malformed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Eval results file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of eval results in {path}")
    return [EvalResult.model_validate(item) for item in data]


def _load_redteam_findings(path: Path) -> Optional[List[dict]]:
    """Load red-team findings from a JSON file.

    Args:
        path: Path to the red-team findings JSON (produced by the red-team CLI).

    Returns:
        A list of finding dicts, or ``None`` if the file does not exist.
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("findings", [])


def _find_latest_eval_results(results_dir: Path) -> Optional[Path]:
    """Find the most recent eval results file in a directory.

    Searches for files matching ``eval_results*.json`` and returns the one with
    the most recent modification time.

    Args:
        results_dir: Directory to search.

    Returns:
        The path to the latest eval results file, or ``None`` if none found.
    """
    if not results_dir.is_dir():
        return None
    candidates = list(results_dir.glob("eval_results*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def generate_report(
    eval_results_path: Optional[Path],
    redteam_findings_path: Path,
    deployment_context: DeploymentContext,
    output_dir: Path,
    model_name: str = "model",
) -> dict:
    """Generate compliance reports from passive and red-team results.

    Args:
        eval_results_path: Path to passive eval results JSON (or ``None`` to
            skip passive evaluation).
        redteam_findings_path: Path to red-team findings JSON.
        deployment_context: Deployment context for adversarial risk scaling.
        output_dir: Directory for output artifacts.
        model_name: Name of the model being reported on.

    Returns:
        A dict with ``json_path`` and ``pdf_path`` keys pointing to the
        generated artifacts.

    Raises:
        FileNotFoundError: If required input files are missing.
    """
    # Load passive eval results (if provided).
    eval_results: List[EvalResult] = []
    if eval_results_path is not None:
        eval_results = _load_eval_results(eval_results_path)

    # Load red-team findings.
    redteam_findings = _load_redteam_findings(redteam_findings_path)
    if redteam_findings is None:
        raise FileNotFoundError(
            f"Red-team findings not found at {redteam_findings_path}. "
            "Run the red-team agent first: python -m src.redteam.agent"
        )

    # Build the compliance report.
    generator = ComplianceReportGenerator(
        model_name=model_name,
        eval_results=eval_results,
        redteam_findings=redteam_findings,
        deployment_context=deployment_context,
    )
    report = generator.build_report()

    # Write outputs.
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"compliance_report_{model_name}.json"
    pdf_path = output_dir / f"compliance_report_{model_name}.pdf"

    write_json_report(report, json_path)
    write_pdf_report(report, pdf_path)

    return {
        "json_path": json_path,
        "pdf_path": pdf_path,
        "report": report,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for report generation.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code: ``0`` on success, ``1`` on error.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.reports.generate",
        description="Generate compliance reports from passive eval + red-team findings.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "pdf", "all"],
        default="all",
        help="Output format (default: all).",
    )
    parser.add_argument(
        "--framework",
        choices=["all", "eu_ai_act", "nist_rmf", "iso_42001"],
        default="all",
        help="Compliance framework to report on (default: all).",
    )
    parser.add_argument(
        "--redteam-findings",
        type=Path,
        default=Path("results/redteam_findings.json"),
        help="Path to red-team findings JSON (default: results/redteam_findings.json).",
    )
    parser.add_argument(
        "--eval-results",
        type=Path,
        default=None,
        help=(
            "Path to passive eval results JSON. If omitted, searches for the "
            "latest eval_results*.json in the results directory."
        ),
    )
    parser.add_argument(
        "--deployment-context",
        choices=["high", "limited", "minimal"],
        default="high",
        help="Deployment context for adversarial risk scaling (default: high).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results"),
        help="Output directory for generated reports (default: results/).",
    )
    parser.add_argument(
        "--model-name",
        default="model",
        help="Name of the model being reported on (default: model).",
    )

    args = parser.parse_args(argv)

    # Resolve eval results path.
    eval_results_path = args.eval_results
    if eval_results_path is None:
        eval_results_path = _find_latest_eval_results(Path("results"))
        if eval_results_path is None:
            print(
                "Warning: no passive eval results found; generating red-team-only report.",
                file=sys.stderr,
            )

    # Map deployment context string to enum.
    context_map = {
        "high": DeploymentContext.HIGH_RISK,
        "limited": DeploymentContext.LIMITED,
        "minimal": DeploymentContext.MINIMAL,
    }
    deployment_context = context_map[args.deployment_context]

    try:
        result = generate_report(
            eval_results_path=eval_results_path,
            redteam_findings_path=args.redteam_findings,
            deployment_context=deployment_context,
            output_dir=args.out,
            model_name=args.model_name,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error generating report: {exc}", file=sys.stderr)
        return 1

    print(f"✓ Generated compliance report for '{args.model_name}'")
    print(f"  JSON: {result['json_path']}")
    print(f"  PDF:  {result['pdf_path']}")
    print(f"  Overall risk tier: {result['report'].overall_risk_tier.value}")
    if result["report"].adversarial_risk_tier is not None:
        print(f"  Adversarial risk tier: {result['report'].adversarial_risk_tier.value}")
    print(f"  Findings: {len(result['report'].findings)} passive, {len(result['report'].redteam_findings)} red-team")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
