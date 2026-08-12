"""Compliance report generation.

Aggregates evaluation results across the four regulatory frameworks
(EU AI Act, NIST AI RMF, ISO/IEC 42001, OWASP LLM Top 10) into an audit-ready
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
from src.compliance.eu_ai_act import classify_risk_tier, prohibited_use_finding
from src.compliance.iso_42001 import map_to_iso_42001
from src.compliance.nist_rmf import map_to_nist_rmf
from src.compliance.redteam_mapping import (
    DeploymentContext,
    adversarial_finding,
    adversarial_risk_tier,
    map_redteam_findings,
)
from src.compliance.system_class import (
    LEGAL_DISCLAIMER,
    SystemClassification,
    parse_use_case,
)
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
        redteam_findings: List[dict] | None = None,
        deployment_context: DeploymentContext = DeploymentContext.LIMITED,
        per_model: List[Dict[str, object]] | None = None,
        system_use_case: str | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            model_name: Name of the model being reported on.
            eval_results: Raw evaluation results to map onto the frameworks.
            timestamp: Report time (UTC). Defaults to "now" if omitted.
            redteam_findings: Optional red-team finding rows (``{target,
                strategy, broke, turn, final_score, snippet?}``). When supplied,
                breaks are mapped to compliance findings and the report becomes
                adversarially-aware.
            deployment_context: Legacy high/limited/minimal flag used when
                ``system_use_case`` is omitted.
            per_model: Optional combined per-model summary rows (``{model,
                passive_tier, adversarial_tier, break_rate, certificate}``)
                rendered as a table in combined multi-model reports.
            system_use_case: Explicit EU AI Act use case. This — not eval
                scores — sets the report's legal class.
        """
        self.model_name = model_name
        self.eval_results: List[EvalResult] = list(eval_results)
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.redteam_findings: List[dict] = list(redteam_findings or [])
        self.deployment_context = deployment_context
        self.per_model: List[Dict[str, object]] = list(per_model or [])
        self.system_class: SystemClassification = parse_use_case(
            system_use_case, deployment_context
        )
        self._findings: List[ComplianceFinding] | None = None
        self._redteam_compliance: List[ComplianceFinding] | None = None

    def build_findings(self) -> List[ComplianceFinding]:
        """Compute passive-eval findings across the three canonical frameworks.

        Returns:
            The combined list of findings (EU AI Act + NIST AI RMF + ISO 42001).
            OWASP LLM Top 10 findings are derived from red-team breaks in
            :meth:`build_redteam_findings`.
        """
        if self._findings is None:
            prohibited = prohibited_use_finding(self.system_class)
            self._findings = (
                ([prohibited] if prohibited is not None else [])
                + classify_risk_tier(
                    self.eval_results, system_class=self.system_class
                )
                + map_to_nist_rmf(
                    self.eval_results, system_class=self.system_class
                )
                + map_to_iso_42001(
                    self.eval_results, system_class=self.system_class
                )
            )
        return self._findings

    def build_redteam_findings(self) -> List[ComplianceFinding]:
        """Map red-team breaks to compliance findings (cached).

        Returns:
            Compliance findings derived from red-team breaks, plus an aggregate
            adversarial-risk finding when the break rate trips the policy.
        """
        if self._redteam_compliance is None:
            break_rates = self._break_rates()
            snippets = {
                f"{f.get('target')}::{f.get('strategy')}": f.get("snippet", "")
                for f in self.redteam_findings
                if f.get("snippet")
            }
            mapped = map_redteam_findings(
                self.redteam_findings,
                self.deployment_context,
                break_rates=break_rates,
                snippets=snippets,
            )
            # Add the aggregate adversarial-risk finding per model (critical when
            # the break rate trips the policy in a high-risk context).
            for target, rate in break_rates.items():
                agg = adversarial_finding(target, rate, self.deployment_context)
                if agg is not None:
                    mapped.append(agg)
            self._redteam_compliance = mapped
        return self._redteam_compliance

    def _break_rates(self) -> Dict[str, float]:
        """Compute the per-model break rate from the red-team findings.

        Returns:
            A ``{target: break_rate}`` mapping over every target that appears in
            the red-team findings (breaks / total attacks for that target).
        """
        totals: Dict[str, int] = {}
        breaks: Dict[str, int] = {}
        for f in self.redteam_findings:
            target = str(f.get("target", ""))
            totals[target] = totals.get(target, 0) + 1
            if f.get("broke"):
                breaks[target] = breaks.get(target, 0) + 1
        return {
            target: (breaks.get(target, 0) / totals[target]) if totals[target] else 0.0
            for target in totals
        }

    def recommendations(self) -> List[str]:
        """Derive de-duplicated compliance recommendations (gaps) from findings.

        Includes both passive-eval and red-team findings so the gap list reflects
        adversarial weaknesses too.

        Returns:
            One recommendation string per unique (framework, control_id) pair,
            ordered by framework.
        """
        findings = self.build_findings() + self.build_redteam_findings()
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

        The report carries passive-eval findings and red-team findings as
        separate lists; ``overall_risk_tier`` is the highest tier across *both*,
        and ``adversarial_risk_tier`` reflects the red-team break rate in the
        deployment context (so a passive "pass" that is fragile under attack is
        surfaced).

        Returns:
            A fully populated, validated compliance report.
        """
        findings = self.build_findings()
        redteam = self.build_redteam_findings()
        # Legal class is the declared use case. Never take max(eval tiers).
        overall = self.system_class.risk_tier

        break_rates = self._break_rates()
        adv_tier: RiskTier | None = None
        if self.redteam_findings:
            # Residual robustness label — not a legal reclassification.
            adv_tier = max_risk_tier(
                [
                    adversarial_risk_tier(rate, self.deployment_context)
                    for rate in break_rates.values()
                ]
            )

        return ComplianceReport(
            model_name=self.model_name,
            timestamp=self.timestamp,
            findings=findings,
            redteam_findings=redteam,
            overall_risk_tier=overall,
            system_use_case=self.system_class.use_case.value,
            classification_disclaimer=LEGAL_DISCLAIMER,
            adversarial_risk_tier=adv_tier,
            per_model=self.per_model,
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
            f"Evaluation record for model '{report.model_name}' "
            f"generated {report.timestamp.isoformat()}.\n"
            f"Declared use case: {report.system_use_case} "
            f"(legal class: {report.overall_risk_tier.value}). "
            f"{self.system_class.rationale}\n"
            f"Residual findings by framework: {framework_line}.\n"
            f"Residual findings by risk tier: {tier_line}.\n"
            f"{report.classification_disclaimer}"
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
        lines.append("=== Legal class (use case, not eval scores) ===")
        lines.append(f"Use case: {report.system_use_case}")
        lines.append(f"EU AI Act class: {report.overall_risk_tier.value}")
        for wrapped in textwrap.wrap(self.system_class.rationale, _WRAP_WIDTH):
            lines.append(wrapped)
        lines.append("")
        if report.per_model:
            lines.append("=== Per-Model Summary (Passive + Adversarial) ===")
            lines.append(
                "Model | Passive tier | Mean safety | Adversarial tier | Break rate "
                "| 95% Wilson | Certificate"
            )
            for row in report.per_model:
                wilson_low = row.get("wilson_low", "-")
                wilson_high = row.get("wilson_high", "-")
                if isinstance(wilson_low, (int, float)) and isinstance(
                    wilson_high, (int, float)
                ):
                    wilson = f"[{wilson_low:.1%}, {wilson_high:.1%}]"
                else:
                    wilson = "-"
                lines.append(
                    f"{row.get('model', '-')} | {row.get('passive_tier', '-')} "
                    f"| {row.get('mean_safety', '-')} "
                    f"| {row.get('adversarial_tier', '-')} "
                    f"| {row.get('break_rate', '-')} | {wilson} "
                    f"| {row.get('certificate', '-')}"
                )
            lines.append("")
        lines.append("=== Residual eval findings (not a risk-class change) ===")
        lines.append(f"Declared legal class: {report.overall_risk_tier.value}")
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
        lines.append("=== Red-Team (Adversarial) Findings ===")
        if report.adversarial_risk_tier is not None:
            lines.append(f"Adversarial risk tier: {report.adversarial_risk_tier.value}")
        if not report.redteam_findings:
            lines.append("No red-team findings (no adversarial breaks mapped).")
        for finding in report.redteam_findings:
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
            f"AI Evaluation Record -- {report.model_name} "
            f"(use case: {report.system_use_case}, "
            f"class: {report.overall_risk_tier.value})"
        )
        return write_pdf(path, title, lines)


def generate_compliance_report(
    model_name: str,
    eval_results: List[EvalResult],
    timestamp: datetime | None = None,
    redteam_findings: List[dict] | None = None,
    deployment_context: DeploymentContext = DeploymentContext.LIMITED,
    system_use_case: str | None = None,
) -> ComplianceReport:
    """One-shot builder returning a populated :class:`ComplianceReport`.

    Args:
        model_name: Name of the model under assessment.
        eval_results: Raw evaluation results to map.
        timestamp: Report time (UTC). Defaults to "now" if omitted.
        redteam_findings: Optional red-team finding rows to map.
        deployment_context: Scales the red-team risk tier (default LIMITED).

    Returns:
        The assembled compliance report.
    """
    return ComplianceReportGenerator(
        model_name=model_name,
        eval_results=eval_results,
        timestamp=timestamp,
        redteam_findings=redteam_findings,
        deployment_context=deployment_context,
        system_use_case=system_use_case,
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
    report: ComplianceReport,
    path: str | Path,
    per_model_stats: Optional[dict] = None,
    per_strategy_stats: Optional[dict] = None,
    raw_findings: Optional[List[dict]] = None,
) -> Path:
    """Write a report to a PDF file.

    Prefers the reportlab-based writer (richer layout with tables & transcripts).
    Falls back to the dependency-free writer when reportlab is unavailable.

    Args:
        report: The report to render.
        path: Destination ``.pdf`` path.
        per_model_stats: Optional ``{model: {breaks, total, rate, wilson_low,
            wilson_high}}`` for the results table.
        per_strategy_stats: Optional ``{strategy: {breaks, total, rate,
            wilson_low, wilson_high}}`` for the results table.
        raw_findings: Optional list of raw finding dicts (with ``transcript``
            keys) for the representative breaking transcripts section.

    Returns:
        The path the PDF was written to.
    """
    try:
        from src.reports._pdf_reportlab import write_pdf_reportlab
    except ImportError:
        write_pdf_reportlab = None

    if write_pdf_reportlab is not None:
        return write_pdf_reportlab(
            report, path,
            per_model_stats=per_model_stats,
            per_strategy_stats=per_strategy_stats,
            raw_findings=raw_findings,
        )

    # Fallback: dependency-free minimal writer.
    return ComplianceReportGenerator(
        model_name=report.model_name,
        eval_results=[],
        timestamp=report.timestamp,
    ).to_pdf(path, report)
