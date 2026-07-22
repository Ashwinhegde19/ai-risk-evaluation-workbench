"""Compliance report generation.

Aggregates evaluation results across the three regulatory frameworks
(EU AI Act, NIST AI RMF, ISO/IEC 42001) into an audit-ready
:class:`~src.core.models.ComplianceReport`, and serializes it to JSON and PDF.

The public entry points are:

    * :class:`ComplianceReportGenerator` -- stateful builder with helpers for
      the executive summary, gap analysis, and recommendations.
    * :func:`generate_compliance_report` -- one-shot builder.
    * :func:`write_json_report` / :func:`write_pdf_report` -- serializers.
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from src.core.models import (
    ComplianceFinding,
    ComplianceFramework,
    ComplianceReport,
    EvalResult,
    RiskTier,
)
from src.compliance._common import max_risk_tier
from src.compliance.eu_ai_act import classify_risk_tier
from src.compliance.iso_42001 import map_to_iso_42001
from src.compliance.nist_rmf import map_to_nist_rmf
from src.reports._pdf import write_pdf

# Width (chars) used when wrapping long lines for the PDF / text rendering.
_WRAP_WIDTH = 95


class ComplianceReportGenerator:
    """Build and serialize a multi-framework compliance report for one model.

    The generator is lazy: findings are computed once (on first access) and
    reused by :meth:`build_report`, :meth:`executive_summary`,
    :meth:`gap_analysis`, and :meth:`recommendations`.
    """

    def __init__(
        self,
        model_name: str,
        eval_results: List[EvalResult],
        timestamp: datetime | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            model_name: Name of the model being reported on.
            eval_results: Raw evaluation results to map onto the frameworks.
            timestamp: Report time (UTC). Defaults to "now" if omitted.
        """
        self.model_name = model_name
        self.eval_results: List[EvalResult] = list(eval_results)
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self._findings: List[ComplianceFinding] | None = None

    def build_findings(self) -> List[ComplianceFinding]:
        """Compute findings across all three frameworks (cached).

        Returns:
            The combined list of findings (EU AI Act + NIST AI RMF + ISO 42001).
        """
        if self._findings is None:
            self._findings = (
                classify_risk_tier(self.eval_results)
                + map_to_nist_rmf(self.eval_results)
                + map_to_iso_42001(self.eval_results)
            )
        return self._findings

    def recommendations(self) -> List[str]:
        """Derive de-duplicated compliance recommendations (gaps) from findings.

        Returns:
            One recommendation string per unique (framework, control_id) pair,
            ordered by framework.
        """
        findings = self.build_findings()
        seen: set = set()
        gaps: List[str] = []
        for finding in findings:
            key = (finding.framework, finding.control_id)
            if key in seen:
                continue
            seen.add(key)
            gaps.append(
                f"[{finding.framework.value}/{finding.control_id}] "
                f"({finding.severity.value}) {finding.description}"
            )
        return gaps

    def build_report(self) -> ComplianceReport:
        """Assemble the :class:`ComplianceReport` for this model.

        Returns:
            A fully populated, validated compliance report.
        """
        findings = self.build_findings()
        overall = max_risk_tier([f.risk_tier for f in findings])
        return ComplianceReport(
            model_name=self.model_name,
            timestamp=self.timestamp,
            findings=findings,
            overall_risk_tier=overall,
            gaps=self.recommendations(),
        )

    def executive_summary(self, report: ComplianceReport | None = None) -> str:
        """Produce a short prose executive summary for the report.

        Args:
            report: Optional pre-built report; built on demand if omitted.

        Returns:
            A multi-line executive summary string.
        """
        report = report or self.build_report()
        per_framework: Dict[str, int] = {}
        per_tier: Dict[str, int] = {}
        for finding in report.findings:
            per_framework[finding.framework.value] = (
                per_framework.get(finding.framework.value, 0) + 1
            )
            per_tier[finding.risk_tier.value] = (
                per_tier.get(finding.risk_tier.value, 0) + 1
            )
        framework_line = ", ".join(
            f"{name}={count}" for name, count in per_framework.items()
        ) or "none"
        tier_line = ", ".join(
            f"{tier}={count}" for tier, count in per_tier.items()
        ) or "none"
        return (
            f"Compliance assessment for model '{report.model_name}' "
            f"generated {report.timestamp.isoformat()}.\n"
            f"Overall risk tier: {report.overall_risk_tier.value}.\n"
            f"Findings by framework: {framework_line}.\n"
            f"Findings by risk tier: {tier_line}."
        )

    def gap_analysis(self, report: ComplianceReport | None = None) -> Dict[str, object]:
        """Summarize findings and gaps for an at-a-glance gap analysis.

        Args:
            report: Optional pre-built report; built on demand if omitted.

        Returns:
            A dict with ``model_name``, ``overall_risk_tier``,
            ``total_findings``, ``by_framework``, ``by_tier`` and ``gaps``.
        """
        report = report or self.build_report()
        by_framework: Dict[str, int] = {}
        by_tier: Dict[str, int] = {}
        for finding in report.findings:
            by_framework[finding.framework.value] = (
                by_framework.get(finding.framework.value, 0) + 1
            )
            by_tier[finding.risk_tier.value] = (
                by_tier.get(finding.risk_tier.value, 0) + 1
            )
        return {
            "model_name": report.model_name,
            "overall_risk_tier": report.overall_risk_tier.value,
            "total_findings": len(report.findings),
            "by_framework": by_framework,
            "by_tier": by_tier,
            "gaps": report.gaps,
        }

    def to_json(self, report: ComplianceReport | None = None) -> str:
        """Serialize the report to a pretty-printed JSON string.

        Args:
            report: Optional pre-built report; built on demand if omitted.

        Returns:
            A JSON string representation of the report.
        """
        report = report or self.build_report()
        return report.model_dump_json(indent=2)

    def to_pdf(self, path: str | Path, report: ComplianceReport | None = None) -> Path:
        """Render the report to a minimal dependency-free PDF.

        Args:
            path: Destination ``.pdf`` path.
            report: Optional pre-built report; built on demand if omitted.

        Returns:
            The path the PDF was written to.
        """
        report = report or self.build_report()
        lines: List[str] = []
        lines += textwrap.wrap(self.executive_summary(report), _WRAP_WIDTH) or [""]
        lines.append("")
        lines.append("=== Compliance Findings ===")
        if not report.findings:
            lines.append("No compliance findings (all dimensions within tolerance).")
        for finding in report.findings:
            lines.append(
                f"[{finding.framework.value}] {finding.control_id} "
                f"({finding.risk_tier.value}, {finding.severity.value})"
            )
            for wrapped in textwrap.wrap(finding.description, _WRAP_WIDTH):
                lines.append(f"    {wrapped}")
            for wrapped in textwrap.wrap(f"Evidence: {finding.evidence}", _WRAP_WIDTH):
                lines.append(f"    {wrapped}")
        lines.append("")
        lines.append("=== Gap Analysis & Recommendations ===")
        if report.gaps:
            for gap in report.gaps:
                for wrapped in textwrap.wrap(gap, _WRAP_WIDTH):
                    lines.append(f"- {wrapped}")
        else:
            lines.append("No gaps identified.")
        title = (
            f"AI Compliance Report -- {report.model_name} "
            f"({report.overall_risk_tier.value})"
        )
        return write_pdf(path, title, lines)


def generate_compliance_report(
    model_name: str,
    eval_results: List[EvalResult],
    timestamp: datetime | None = None,
) -> ComplianceReport:
    """One-shot builder returning a populated :class:`ComplianceReport`.

    Args:
        model_name: Name of the model under assessment.
        eval_results: Raw evaluation results to map.
        timestamp: Report time (UTC). Defaults to "now" if omitted.

    Returns:
        The assembled compliance report.
    """
    return ComplianceReportGenerator(
        model_name=model_name, eval_results=eval_results, timestamp=timestamp
    ).build_report()


def write_json_report(
    report: ComplianceReport, path: str | Path
) -> Path:
    """Write a report to a JSON file.

    Args:
        report: The report to serialize.
        path: Destination ``.json`` path.

    Returns:
        The path the JSON was written to.
    """
    path = Path(path)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_pdf_report(
    report: ComplianceReport, path: str | Path
) -> Path:
    """Write a report to a PDF file using the dependency-free writer.

    Args:
        report: The report to render.
        path: Destination ``.pdf`` path.

    Returns:
        The path the PDF was written to.
    """
    return ComplianceReportGenerator(
        model_name=report.model_name,
        eval_results=[],
        timestamp=report.timestamp,
    ).to_pdf(path, report)
