"""Suite M — adjacent-shot coherence scoreboard for scan data.

On a continuous line scan (e.g. the MW2-112 stratigraphic core, 100 um
pitch) the composition of adjacent shots should be nearly identical, so
the rate at which the DOMINANT element flips between neighbours is a
direct, ground-truth-free measure of pipeline stability.  Measured
baseline on MW2-112: ~77% of adjacent shots flipped dominant element —
the number this program must drive below 10%.

Works on any ``summary.csv`` produced by ``alibz-analyze`` (or the
MW2-112 runner).  Rows are ordered by ``test_id`` when present, else by
filename.  Usage::

    python -m bench.coherence <summary.csv> [--out report.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

#: summary.csv columns that are not element-fraction columns.
_META_PREFIXES = ("test_id", "height_um", "file", "sample", "status",
                  "n_peaks", "shift_", "response_", "T_K", "log_ne",
                  "r_squared", "sa_converged", "qc_", "flags", "t_total_s",
                  "failure_", "guard_triggered")


def _is_element_column(name: str) -> bool:
    if name.endswith("_unc"):
        return False
    return not any(name == p or name.startswith(p) for p in _META_PREFIXES)


def load_fractions(summary_csv: Path):
    """-> (ordered row keys, elements, fractions array [n_rows, n_el])."""
    with open(summary_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    ok = [r for r in rows if r.get("status") == "ok"]
    elements = [c for c in (rows[0].keys() if rows else [])
                if _is_element_column(c)]

    def _key(r):
        tid = r.get("test_id") or ""
        return (0, float(tid)) if tid else (1, r.get("file", ""))

    ok.sort(key=_key)
    frac = np.zeros((len(ok), len(elements)))
    for i, r in enumerate(ok):
        for j, el in enumerate(elements):
            v = r.get(el) or ""
            frac[i, j] = float(v) if v else 0.0
    keys = [r.get("test_id") or r.get("file", "") for r in ok]
    return keys, elements, frac


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def coherence_report(summary_csv: Path) -> dict:
    keys, elements, frac = load_fractions(Path(summary_csv))
    n = frac.shape[0]
    if n < 2:
        return dict(n_rows=n, error="need at least 2 ok rows")
    dominant = np.argmax(frac, axis=1)
    has_mass = frac.sum(axis=1) > 0
    valid_pairs = has_mass[1:] & has_mass[:-1]
    flips = (dominant[1:] != dominant[:-1]) & valid_pairs
    l1 = np.abs(np.diff(frac, axis=0)).sum(axis=1)

    per_element = {}
    for j, el in enumerate(elements):
        col = frac[:, j]
        if np.count_nonzero(col) < max(3, n // 10):
            continue
        per_element[el] = dict(
            lag1_spearman=round(_spearman(col[:-1], col[1:]), 4),
            median_adjacent_delta=round(float(np.median(np.abs(np.diff(col)))),
                                        5),
            mean_fraction=round(float(np.mean(col)), 5),
        )

    return dict(
        n_rows=n,
        n_valid_pairs=int(valid_pairs.sum()),
        dominant_flip_rate=round(float(flips.sum() / max(valid_pairs.sum(), 1)),
                                 4),
        median_adjacent_l1=round(float(np.median(l1[valid_pairs])), 5)
        if valid_pairs.any() else None,
        per_element=per_element,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("summary_csv")
    p.add_argument("--out", default=None, help="write JSON report here")
    args = p.parse_args(argv)
    report = coherence_report(Path(args.summary_csv))
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
