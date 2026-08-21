"""Merge a single-model red-team run into the canonical findings JSON.

Takes a per-run findings file (as produced by ``python -m src.redteam.agent
--findings-out ...``) and folds its per-model/per-strategy statistics and
per-trial findings into the canonical ``results/redteam_findings.json`` so the
VERDICT console picks the model up via its live fetch.

Usage:
    python3 scripts/merge_redteam_run.py \
        --run results/redteam_findings_oxalpha_15strat.json \
        --canonical results/redteam_findings.json

The merge is idempotent per target: an existing block for the run's target is
replaced before the new one is folded in.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval — mirrors src/redteam/agent.py exactly."""
    if total == 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denominator = 1 + z * z / total
    centre = (p_hat + z * z / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(
        (p_hat * (1 - p_hat) + z * z / (4 * total)) / total
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _aggregate(findings: List[dict]) -> Dict[str, dict]:
    """Aggregate per-trial findings into the per-model stats shape."""
    breaks = sum(1 for f in findings if f["broke"])
    total = len(findings)
    rate = breaks / total if total else 0.0
    if total > 1:
        variance = sum(
            (1 - rate) ** 2 if f["broke"] else (0 - rate) ** 2 for f in findings
        ) / (total - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    lo, hi = _wilson_interval(breaks, total)
    return {
        "breaks": breaks,
        "total": total,
        "rate": round(rate, 4),
        "std": round(std, 4),
        "wilson_low": round(lo, 4),
        "wilson_high": round(hi, 4),
    }


def main() -> None:
    """Fold the run file into the canonical findings JSON (in place)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Single-model run findings JSON")
    parser.add_argument("--canonical", required=True, help="Canonical findings JSON to update")
    args = parser.parse_args()

    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    canonical = json.loads(Path(args.canonical).read_text(encoding="utf-8"))

    targets = list(run["per_model"].keys())
    if len(targets) != 1:
        raise SystemExit(f"expected exactly one target in run file, got {targets}")
    target = targets[0]

    # Drop any stale block for this target (idempotent re-merge).
    canonical["findings"] = [
        f for f in canonical["findings"] if f["target"] != target
    ]
    canonical["findings"].extend(run["findings"])
    canonical["per_model"][target] = run["per_model"][target]

    # Re-aggregate per-strategy stats across every model present.
    strategies = sorted({f["strategy"] for f in canonical["findings"]})
    per_strategy: Dict[str, dict] = {}
    for strategy in strategies:
        rows = [f for f in canonical["findings"] if f["strategy"] == strategy]
        per_strategy[strategy] = _aggregate(rows)
    canonical["per_strategy"] = per_strategy

    Path(args.canonical).write_text(
        json.dumps(canonical, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    total = len(canonical["findings"])
    models = len(canonical["per_model"])
    print(f"merged '{target}' -> {args.canonical}: {total} findings, {models} models")
    for name, stats in sorted(per_strategy.items()):
        print(f"  {name:18s} {stats['breaks']:>3}/{stats['total']}")


if __name__ == "__main__":
    main()
