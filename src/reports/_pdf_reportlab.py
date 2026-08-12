"""Reportlab-based PDF writer for compliance reports.

Produces a multi-section PDF with:

    * Executive summary
    * Method (targets, strategies, trials, seed, judge model)
    * Results tables (per-model & per-strategy break rates with Wilson CIs)
    * Per-framework control mapping + compliance narrative
    * Representative breaking transcripts (truncated, labeled)
    * Remediation recommendations prioritized by strategy break rate

Falls back to the dependency-free :mod:`src.reports._pdf` writer when reportlab
is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

# Max characters for a breaking transcript excerpt in the PDF.
_TRANSCRIPT_MAX_LEN = 500


def _truncate(text: str, max_len: int = _TRANSCRIPT_MAX_LEN) -> str:
    """Truncate text to ``max_len`` chars, appending ``...`` when clipped."""
    if not text:
        return ""
    return text if len(text) <= max_len else text[:max_len] + "..."


def write_pdf_reportlab(
    report: object,
    path: str | Path,
    per_model_stats: Optional[dict] = None,
    per_strategy_stats: Optional[dict] = None,
    raw_findings: Optional[List[dict]] = None,
) -> Path:
    """Render a compliance report to PDF using reportlab.

    Args:
        report: A :class:`~src.core.models.ComplianceReport` (duck-typed).
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
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    path = Path(path)
    per_model_stats = per_model_stats or {}
    per_strategy_stats = per_strategy_stats or {}

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h1_style = styles["Heading1"]
    h2_style = styles["Heading2"]
    body_style = styles["BodyText"]
    # Monospace-ish small style for transcript excerpts.
    mono_style = ParagraphStyle(
        "Mono", parent=body_style, fontName="Courier", fontSize=7,
        leading=9,
    )

    doc = SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        pageCompression=0,
    )

    story: List[object] = []

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    model_name = getattr(report, "model_name", "model")
    overall_tier = getattr(
        getattr(report, "overall_risk_tier", None), "value", "unknown"
    )
    use_case = getattr(report, "system_use_case", "gpai_or_chatbot")
    story.append(Paragraph(
        f"AI Evaluation Record &mdash; {model_name} "
        f"(use case: {use_case}, class: {overall_tier})",
        title_style,
    ))
    story.append(Spacer(1, 0.2 * inch))

    # ------------------------------------------------------------------
    # Section 1: Executive Summary
    # ------------------------------------------------------------------
    story.append(Paragraph("Executive Summary", h1_style))
    passive_count = len(getattr(report, "findings", []) or [])
    redteam_count = len(getattr(report, "redteam_findings", []) or [])
    adv_tier = getattr(
        getattr(report, "adversarial_risk_tier", None), "value", "N/A"
    )
    ts = getattr(report, "timestamp", None)
    ts_str = ts.isoformat() if ts is not None else "N/A"
    disclaimer = getattr(report, "classification_disclaimer", "")
    summary_text = (
        f"Evaluation record for model <b>'{model_name}'</b> generated "
        f"{ts_str}.<br/>"
        f"Declared use case: <b>{use_case}</b>. "
        f"Legal class (from use case, not scores): <b>{overall_tier}</b>.<br/>"
        f"Residual robustness label: <b>{adv_tier}</b>.<br/>"
        f"Residual findings: {passive_count} eval, {redteam_count} red-team.<br/>"
        f"{disclaimer}"
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 0.2 * inch))

    # ------------------------------------------------------------------
    # Section 2: Method
    # ------------------------------------------------------------------
    story.append(Paragraph("Method", h1_style))
    targets = sorted(per_model_stats.keys()) if per_model_stats else []
    strategies = sorted(per_strategy_stats.keys()) if per_strategy_stats else []
    total_trials = sum(s["total"] for s in per_model_stats.values()) if per_model_stats else 0
    method_text = (
        f"Targets assessed: <b>{', '.join(targets) if targets else 'N/A'}</b>.<br/>"
        f"Attack strategies: <b>{', '.join(strategies) if strategies else 'N/A'}</b>.<br/>"
        f"Total trials: <b>{total_trials}</b>.<br/>"
        f"Judge model: <b>LLM-as-Judge ensemble</b>.<br/>"
        f"Break adjudication: calibrated LLM-as-Judge scoring with semantic "
        f"break adjudication."
    )
    story.append(Paragraph(method_text, body_style))
    story.append(Spacer(1, 0.2 * inch))

    # ------------------------------------------------------------------
    # Section 3: Results Tables
    # ------------------------------------------------------------------
    story.append(Paragraph("Results", h1_style))

    story.append(Paragraph("Per-Model Break Rate", h2_style))
    if per_model_stats:
        model_rows = [["Model", "Breaks", "Total", "Rate", "Std", "95% Wilson"]]
        for m in sorted(per_model_stats):
            s = per_model_stats[m]
            model_rows.append([
                m,
                str(s["breaks"]),
                str(s["total"]),
                f"{s['rate']:.2%}",
                f"{s.get('std', 0):.3f}",
                f"[{s['wilson_low']:.2%}, {s['wilson_high']:.2%}]",
            ])
        story.append(_make_table(model_rows))
    else:
        story.append(Paragraph("No per-model results.", body_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Per-Strategy Break Rate", h2_style))
    if per_strategy_stats:
        strat_rows = [["Strategy", "Breaks", "Total", "Rate", "Std", "95% Wilson"]]
        for st in sorted(per_strategy_stats):
            s = per_strategy_stats[st]
            strat_rows.append([
                st,
                str(s["breaks"]),
                str(s["total"]),
                f"{s['rate']:.2%}",
                f"{s.get('std', 0):.3f}",
                f"[{s['wilson_low']:.2%}, {s['wilson_high']:.2%}]",
            ])
        story.append(_make_table(strat_rows))
    else:
        story.append(Paragraph("No per-strategy results.", body_style))
    story.append(Spacer(1, 0.2 * inch))

    # ------------------------------------------------------------------
    # Section 4: Per-Framework Control Mapping
    # ------------------------------------------------------------------
    story.append(Paragraph("Per-Framework Control Mapping", h1_style))
    findings = getattr(report, "findings", []) or []
    redteam_findings = getattr(report, "redteam_findings", []) or []

    passive_by_fw: dict[str, list] = {}
    for f in findings:
        fw = getattr(getattr(f, "framework", None), "value", str(getattr(f, "framework", "?")))
        passive_by_fw.setdefault(fw, []).append(f)

    story.append(Paragraph("Passive Compliance", h2_style))
    if passive_by_fw:
        for fw in sorted(passive_by_fw):
            story.append(Paragraph(fw, h2_style))
            for finding in passive_by_fw[fw]:
                cid = getattr(finding, "control_id", "?")
                tier = getattr(getattr(finding, "risk_tier", None), "value", "?")
                sev = getattr(getattr(finding, "severity", None), "value", "?")
                desc = getattr(finding, "description", "")
                story.append(Paragraph(
                    f"<b>[{cid}]</b> ({tier}, {sev}): {_esc(desc)}",
                    body_style,
                ))
    else:
        story.append(Paragraph("No passive compliance findings.", body_style))

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Red-Team (Adversarial) Findings", h2_style))
    redteam_by_fw: dict[str, list] = {}
    for f in redteam_findings:
        fw = getattr(getattr(f, "framework", None), "value", str(getattr(f, "framework", "?")))
        redteam_by_fw.setdefault(fw, []).append(f)

    if redteam_by_fw:
        for fw in sorted(redteam_by_fw):
            story.append(Paragraph(fw, h2_style))
            for finding in redteam_by_fw[fw][:20]:  # Cap to avoid runaway PDFs.
                cid = getattr(finding, "control_id", "?")
                tier = getattr(getattr(finding, "risk_tier", None), "value", "?")
                sev = getattr(getattr(finding, "severity", None), "value", "?")
                desc = getattr(finding, "description", "")
                story.append(Paragraph(
                    f"<b>[{cid}]</b> ({tier}, {sev}): {_esc(desc)}",
                    body_style,
                ))
    else:
        story.append(Paragraph("No red-team findings mapped.", body_style))
    story.append(Spacer(1, 0.2 * inch))

    # ------------------------------------------------------------------
    # Section 5: Representative Breaking Transcripts
    # ------------------------------------------------------------------
    story.append(Paragraph("Representative Breaking Transcripts", h1_style))
    breaking: List[dict] = []
    if raw_findings:
        breaking = [f for f in raw_findings if f.get("broke")]
    if breaking:
        for bf in breaking[:5]:  # Cap representative transcripts.
            target = bf.get("target", "?")
            strategy = bf.get("strategy", "?")
            trial = bf.get("trial", "?")
            turn = bf.get("turn", "?")
            transcript = bf.get("transcript") or []
            story.append(Paragraph(
                f"<b>Target: {target} | Strategy: {strategy} | "
                f"Trial: {trial} | Turn: {turn}</b>",
                body_style,
            ))
            for t in transcript:
                t_num = t.get("turn", "?") if isinstance(t, dict) else "?"
                prompt = _esc(_truncate(t.get("attacker_prompt", "") if isinstance(t, dict) else "", 200))
                response = _esc(_truncate(t.get("model_response", "") if isinstance(t, dict) else "", 200))
                story.append(Paragraph(
                    f"Turn {t_num}:<br/>"
                    f"&nbsp;&nbsp;Prompt: {prompt}<br/>"
                    f"&nbsp;&nbsp;Response: {response}",
                    mono_style,
                ))
            story.append(Spacer(1, 0.1 * inch))
    else:
        story.append(Paragraph(
            "No breaking transcripts available (no adversarial breaks recorded).",
            body_style,
        ))
    story.append(Spacer(1, 0.2 * inch))

    # ------------------------------------------------------------------
    # Section 6: Remediation Recommendations
    # ------------------------------------------------------------------
    story.append(Paragraph("Remediation Recommendations", h1_style))
    gaps = getattr(report, "gaps", []) or []
    # Prioritize by strategy break rate: order gaps by the break rate of the
    # strategy they reference (approximate via per_strategy_stats ordering).
    if per_strategy_stats:
        # Sort strategy names by break rate descending.
        sorted_strats = sorted(
            per_strategy_stats,
            key=lambda s: per_strategy_stats[s]["rate"],
            reverse=True,
        )
        # Reorder gaps so that those mentioning higher-break-rate strategies come first.
        def _gap_priority(gap: str) -> int:
            for i, st in enumerate(sorted_strats):
                if st in gap.lower():
                    return i
            return len(sorted_strats)
        gaps = sorted(gaps, key=_gap_priority)

    if gaps:
        for gap in gaps:
            story.append(Paragraph(f"&bull; {_esc(gap)}", body_style))
    else:
        story.append(Paragraph("No gaps identified.", body_style))

    doc.build(story)
    return path


def _esc(text: str) -> str:
    """Escape text for reportlab Paragraph XML."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _make_table(rows: List[List[str]]) -> Table:
    """Build a styled reportlab Table from a list of string rows."""
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    t = Table(rows, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t
