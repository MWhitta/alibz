"""Aggregate profile.jsonl + summary.csv into failure and timing tables.

The scoreboard that makes "fails often" and "is slow" measurable::

    python scripts/failure_report.py <data_dir>

Reads ``<data_dir>/profile.jsonl`` (per-spectrum stage timings, counters,
failure attribution written by ``analyze_directory``) and prints:

- failure rate by ``failure_reason`` and by ``failure_stage``;
- guard-fire rates (``guard_triggered`` vocabulary);
- wall-time by stage (median / p90 / share of total);
- counter totals (objective evals, overlap rebuilds vs cache hits, ...).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_profile(data_dir: Path) -> list:
    path = data_dir / "profile.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no profile.jsonl in {data_dir}")
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def report(data_dir: Path) -> str:
    rows = load_profile(data_dir)
    n = len(rows)
    ok = [r for r in rows if (r.get("status") or "").startswith("ok")]
    bad = [r for r in rows if not (r.get("status") or "").startswith("ok")]
    out = [f"{data_dir}: {n} spectra, {len(ok)} ok, {len(bad)} failed "
           f"({100.0 * len(bad) / max(n, 1):.1f}%)", ""]

    if bad:
        out.append("failures by reason:")
        for reason, count in Counter(
                r.get("failure_reason") or "?" for r in bad).most_common():
            out.append(f"  {reason:28s} {count}")
        out.append("failures by stage:")
        for stg, count in Counter(
                r.get("failure_stage") or "<between stages>"
                for r in bad).most_common():
            out.append(f"  {stg:28s} {count}")
        out.append("")

    guards = Counter()
    for r in rows:
        for g in filter(None, (r.get("guard_triggered") or "").split(";")):
            guards[g] += 1
    if guards:
        out.append("guard fires (per spectrum analyzed):")
        for g, count in guards.most_common():
            out.append(f"  {g:28s} {count}  ({100.0 * count / max(n, 1):.0f}%)")
        out.append("")

    stage_t = defaultdict(list)
    totals = []
    for r in rows:
        for name, rec in (r.get("stages") or {}).items():
            stage_t[name].append(float(rec["s"]))
        if r.get("t_total_s"):
            totals.append(float(r["t_total_s"]))
    if stage_t:
        wall = sum(totals) or 1.0
        out.append(f"stage wall time (total {wall:.0f}s across {n} spectra):")
        agg = sorted(((sum(v), name, v) for name, v in stage_t.items()),
                     reverse=True)
        for tot, name, v in agg:
            out.append(f"  {name:18s} total {tot:8.1f}s  share {tot / wall:5.1%}"
                       f"  median {np.median(v):7.2f}s  p90 {np.quantile(v, 0.9):7.2f}s")
        out.append("")

    counters = Counter()
    for r in rows:
        for name, val in (r.get("counters") or {}).items():
            counters[name] += int(val)
    if counters:
        out.append("counters (totals):")
        for name, val in counters.most_common():
            out.append(f"  {name:28s} {val}")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data_dir", type=Path)
    args = p.parse_args(argv)
    print(report(args.data_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
