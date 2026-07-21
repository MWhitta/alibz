"""Suite R — composition sensitivity under nuisance perturbations.

The MW2-112 pilot documented the core reproducibility failure: doubling
``n_calls`` or moving the 620 nm gain fallback within its own corpus IQR
produced LARGE composition changes.  This harness turns that anecdote
into a regression number: for each spectrum, analyze at a reference
config and under each perturbation, and report the composition L1
distance, dominant-element flips, and plasma-state deltas.  A pinned
baseline JSON turns regressions into nonzero exit codes.

Perturbations (the measured instability axes):
- ``n_calls_x2``      — optimizer budget doubled
- ``response_q25/q75``— 620 nm fallback at its corpus IQR endpoints
- ``gp_seed``         — GP seed 42 -> 7 (a reproducible pipeline should
                        not care)
- ``wavelength_jitter``— spectrum shifted by +-1 native pixel

Usage::

    python -m bench.sensitivity <data_dir> [--n-calls 40] [--limit 5]
        [--out report.json] [--baseline pinned.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

from alibz.pipeline import (AnalysisConfig, _analyze_file, _worker_init,
                            resolve_dbpath)

#: 620 nm segment-response corpus IQR endpoints (corrections/detector.json).
RESPONSE_Q25, RESPONSE_Q75 = 3.0, 5.2


def _perturbations(cfg: AnalysisConfig) -> dict:
    return {
        "n_calls_x2": dataclasses.replace(cfg, n_calls=2 * cfg.n_calls),
        "response_q25": dataclasses.replace(
            cfg, segment_response_fallback_ratio=RESPONSE_Q25,
            segment_response_fallback_source="sensitivity_q25"),
        "response_q75": dataclasses.replace(
            cfg, segment_response_fallback_ratio=RESPONSE_Q75,
            segment_response_fallback_source="sensitivity_q75"),
        "gp_seed": dataclasses.replace(cfg, gp_seed=7),
    }


def _fractions(row: dict) -> dict:
    return {el: float(f) for el, f in (row.get("fractions") or {}).items()
            if f > 0}


def _compare(ref: dict, alt: dict) -> dict:
    elements = set(ref) | set(alt)
    l1 = sum(abs(ref.get(el, 0.0) - alt.get(el, 0.0)) for el in elements)
    dom_ref = max(ref, key=ref.get) if ref else None
    dom_alt = max(alt, key=alt.get) if alt else None
    return dict(l1=round(l1, 4),
                dominant_flip=bool(dom_ref != dom_alt),
                dominant=f"{dom_ref}->{dom_alt}")


def _jitter_spectrum(path: str, out_dir: Path, sign: int) -> str:
    """Copy the spectrum with wavelengths shifted by one native pixel."""
    from alibz.pipeline import load_spectrum_csv
    x, y = load_spectrum_csv(path)
    pitch = float(np.median(np.diff(x)))
    out = out_dir / f"jitter{'+' if sign > 0 else '-'}_{Path(path).name}"
    np.savetxt(out, np.column_stack([x + sign * pitch, y]), delimiter=",",
               header="wavelength,intensity", comments="")
    return str(out)


def run_sensitivity(data_dir: str, dbpath: str = None, n_calls: int = 40,
                    draws: int = 8, timeout_s: int = 900, limit: int = 5,
                    jitter: bool = True, pattern: str = "*.csv",
                    progress=print) -> dict:
    dbpath = resolve_dbpath(dbpath)
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    files = [f for f in files if os.path.basename(f) not in
             ("summary.csv", "detections.csv")][:limit]
    if not files:
        raise FileNotFoundError(f"no spectra in {data_dir}")
    _worker_init()

    cfg = AnalysisConfig(dbpath=dbpath, n_calls=n_calls, draws=draws,
                         timeout_s=timeout_s)
    report = {"config": dataclasses.asdict(cfg), "spectra": {}}
    scratch = Path(data_dir) / ".sensitivity_jitter"
    for path in files:
        name = os.path.basename(path)
        progress(f"[sensitivity] {name}: reference")
        ref_row = _analyze_file((path, cfg))
        ref = _fractions(ref_row)
        entry = {"reference_status": ref_row["status"],
                 "reference_fractions": ref,
                 "reference_T_K": ref_row.get("T_K"),
                 "reference_log_ne": ref_row.get("log_ne"),
                 "perturbations": {}}
        if ref_row["status"] == "ok":
            for pname, pcfg in _perturbations(cfg).items():
                progress(f"[sensitivity] {name}: {pname}")
                row = _analyze_file((path, pcfg))
                entry["perturbations"][pname] = (
                    _compare(ref, _fractions(row))
                    if row["status"] == "ok"
                    else {"error": row["status"]})
            if jitter:
                scratch.mkdir(exist_ok=True)
                for sign in (+1, -1):
                    pname = f"wavelength_jitter{'+' if sign > 0 else '-'}"
                    progress(f"[sensitivity] {name}: {pname}")
                    jpath = _jitter_spectrum(path, scratch, sign)
                    row = _analyze_file((jpath, cfg))
                    entry["perturbations"][pname] = (
                        _compare(ref, _fractions(row))
                        if row["status"] == "ok"
                        else {"error": row["status"]})
                    os.unlink(jpath)
        report["spectra"][name] = entry

    l1s = [p["l1"] for e in report["spectra"].values()
           for p in e["perturbations"].values() if "l1" in p]
    flips = [p["dominant_flip"] for e in report["spectra"].values()
             for p in e["perturbations"].values() if "dominant_flip" in p]
    report["summary"] = dict(
        n_spectra=len(files),
        n_perturbation_runs=len(l1s),
        max_l1=round(max(l1s), 4) if l1s else None,
        median_l1=round(float(np.median(l1s)), 4) if l1s else None,
        dominant_flip_count=int(sum(flips)),
    )
    return report


def check_against_baseline(report: dict, baseline: dict) -> list:
    """-> list of regression strings (empty = pass).

    Baseline schema: ``{"max_l1": float, "dominant_flip_count": int}`` —
    measured-then-pinned; tighten as fix stages land.
    """
    failures = []
    s = report["summary"]
    if baseline.get("max_l1") is not None and s["max_l1"] is not None:
        if s["max_l1"] > baseline["max_l1"] + 1e-9:
            failures.append(
                f"max_l1 {s['max_l1']} exceeds baseline {baseline['max_l1']}")
    if baseline.get("dominant_flip_count") is not None:
        if s["dominant_flip_count"] > baseline["dominant_flip_count"]:
            failures.append(
                f"dominant_flip_count {s['dominant_flip_count']} exceeds "
                f"baseline {baseline['dominant_flip_count']}")
    return failures


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data_dir")
    p.add_argument("--db", default=None)
    p.add_argument("--n-calls", type=int, default=40)
    p.add_argument("--draws", type=int, default=8)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--no-jitter", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--baseline", default=None,
                   help="pinned baseline JSON; regressions -> exit 1")
    args = p.parse_args(argv)
    report = run_sensitivity(args.data_dir, dbpath=args.db,
                             n_calls=args.n_calls, draws=args.draws,
                             timeout_s=args.timeout, limit=args.limit,
                             jitter=not args.no_jitter)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        failures = check_against_baseline(report, baseline)
        for f in failures:
            print(f"SENSITIVITY REGRESSION: {f}", file=sys.stderr)
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
