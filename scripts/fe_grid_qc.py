#!/usr/bin/env python
"""Purity, repeatability, and surface-cleanup QC for a repeated grid of shots.

Written for the Z300 nominal-99.98%-Fe checkout: a 4x4 grid of locations,
one shot per location, with the whole grid repeated N times (default 3) so
that surface contamination (which should burn off) can be separated from
instrument repeatability (which should not).

MEASUREMENT ONLY.  This script does not fit anything: it consumes the
``summary.csv`` (or ``summary_raw.csv``) and ``detections.csv`` that
``alibz-analyze`` already wrote for a directory of per-shot spectra, and
reduces them along the (rep, site) axes of the acquisition design.

It answers the three questions the checkout was run to answer:

1. **Purity / contamination** -- which non-Fe elements are detected, in how
   many shots, at what level; and for the ones that are *not* detected, what
   upper limit the data supports.  For a 99.98% pure target every non-Fe
   element is expected either absent or at trace level, so a confident
   non-Fe detection is either real contamination or a pipeline artefact.

2. **Repeatability** -- shot-to-shot RSD of the Fe fraction and of the fitted
   plasma state (T_K, log_ne) within each rep, and site-to-site RSD across
   the grid.  On a homogeneous target this is the instrument's precision
   floor; a site-to-site RSD well above the within-rep RSD points at a
   spatial effect (focus, tilt, height) rather than counting statistics.

3. **Rep-to-rep cleanup** -- per-site contaminant level as a function of rep
   index.  A surface layer that the laser ablates through shows up as a
   monotone decrease from rep 1 to rep N at most sites; a flat profile says
   the signal is bulk, not surface.

Shot-to-(rep, site) mapping is set by ``--layout``; see ``site_of``.  The
mapping is an assumption about acquisition order, not something recoverable
from the spectra, so it is echoed into the output for checking.

Usage::

    python scripts/fe_grid_qc.py RUN_DIR --grid 4x4 --reps 3 \
        --layout sequential --out-dir RUN_DIR/qc
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MAJOR = "Fe"

# summary.csv columns that are per-shot scalars worth tracking for repeatability
STATE_COLUMNS = ("T_K", "log_ne", "r_squared", "response_620", "shift_pm")

SHOT_RE = re.compile(r"Shot\s*\(?(\d+)\)?", re.IGNORECASE)
# Decoder output: one file per shot, naming the rep and the position within it.
REP_POS_RE = re.compile(r"rep(\d+)[_-]pos(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------
# acquisition layout
# --------------------------------------------------------------------------

def shot_number(filename: str) -> Optional[int]:
    """Return the instrument shot index parsed from a Z300 export filename.

    The Z300 writes ``Shot(12).csv`` per shot inside a per-test folder.
    Returns None when the name carries no shot index.
    """
    match = SHOT_RE.search(os.path.basename(filename))
    return int(match.group(1)) if match else None


def rep_pos(filename: str) -> Optional[Tuple[int, int]]:
    """Return 0-based ``(rep, position)`` parsed from a ``repR_posNN`` name."""
    match = REP_POS_RE.search(os.path.basename(filename))
    if not match:
        return None
    return int(match.group(1)) - 1, int(match.group(2)) - 1


def site_of(index: int, sites: int, reps: int, layout: str) -> Tuple[int, int]:
    """Map a 0-based acquisition index to ``(rep, site)``, both 0-based.

    ``sequential``  the whole grid is shot rep-by-rep: sites 0..S-1 are rep 0,
                    the next S are rep 1, and so on.  This is what a single
                    48-shot test on a 16-point grid repeated 3x produces.
    ``interleaved`` each site is revisited immediately: site 0 is shot ``reps``
                    times, then site 1, and so on.
    """
    if layout == "sequential":
        return index // sites, index % sites
    if layout == "interleaved":
        return index % reps, index // reps
    raise ValueError(f"unknown layout: {layout}")


def grid_rowcol(site: int, ncols: int) -> Tuple[int, int]:
    """Return ``(row, col)`` for a 0-based site index in row-major order."""
    return site // ncols, site % ncols


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------

def _read_csv(path: str) -> List[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def find_summary(run_dir: str) -> str:
    """Return the summary table path, preferring the raw composition table."""
    for name in ("summary.csv", "summary_raw.csv"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    raise SystemExit(f"no summary.csv or summary_raw.csv in {run_dir}")


# Metadata columns that also carry a ``_unc`` companion and so would
# otherwise be mistaken for elements.
NON_ELEMENT_UNC_COLUMNS = frozenset({"response_620"})


def element_columns(header: Sequence[str]) -> List[str]:
    """Return the element fraction columns of a summary header.

    Element columns are those with a matching ``<El>_unc`` companion, less
    the metadata columns that happen to share that shape.
    """
    names = set(header)
    return [c for c in header
            if not c.endswith("_unc")
            and f"{c}_unc" in names
            and c not in NON_ELEMENT_UNC_COLUMNS]


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def rsd(values: Sequence[float]) -> Optional[float]:
    """Relative standard deviation (%) using the sample sd; None if undefined."""
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if len(clean) < 2:
        return None
    mean = float(np.mean(clean))
    if mean == 0:
        return None
    return 100.0 * float(np.std(clean, ddof=1)) / abs(mean)


def summarize(values: Sequence[float]) -> dict:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return {"n": 0, "mean": None, "sd": None, "median": None,
                "min": None, "max": None, "rsd_pct": None}
    arr = np.asarray(clean, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "rsd_pct": rsd(clean),
    }


def theil_sen_slope(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Median of pairwise slopes -- robust to a single bad rep."""
    pairs = [(a, b) for a, b in zip(x, y)
             if a is not None and b is not None and math.isfinite(b)]
    if len(pairs) < 2:
        return None
    slopes = [(pairs[j][1] - pairs[i][1]) / (pairs[j][0] - pairs[i][0])
              for i in range(len(pairs)) for j in range(i + 1, len(pairs))
              if pairs[j][0] != pairs[i][0]]
    return float(np.median(slopes)) if slopes else None


# --------------------------------------------------------------------------
# reductions
# --------------------------------------------------------------------------

def build_shots(summary_rows: List[dict], elements: Sequence[str],
                sites: int, reps: int, layout: str,
                ncols: int, order: str,
                partial_reps: Sequence[int] = ()) -> List[dict]:
    """Attach (rep, site, row, col) to each summary row.

    With ``layout='filename'`` the rep and position come from a ``repR_posNN``
    filename, which is the only safe option when the reps have different shot
    counts.  Otherwise ``order`` selects what defines acquisition order:
    ``shot`` uses the ``Shot(N)`` index in the filename, ``file`` uses sorted
    file order, and the position is derived from a fixed-size design.

    ``partial_reps`` (0-based) names reps that returned fewer shots than the
    design, so position within the rep is an acquisition order and *not* a
    known grid site.  Those shots get ``site_known = False`` and are excluded
    from per-site pairing downstream.
    """
    rows = []
    for row in summary_rows:
        path = row.get("file") or row.get("sample") or ""
        rows.append({"row": row, "file": path, "shot": shot_number(path),
                     "rp": rep_pos(path)})

    if layout == "filename":
        if not all(r["rp"] is not None for r in rows):
            missing = [r["file"] for r in rows if r["rp"] is None][:3]
            raise SystemExit("layout=filename needs repR_posNN names; "
                             f"e.g. unparsable: {missing}")
        rows.sort(key=lambda r: r["rp"])
    elif order == "shot" and all(r["shot"] is not None for r in rows):
        rows.sort(key=lambda r: r["shot"])
    else:
        rows.sort(key=lambda r: r["file"])

    shots = []
    for index, entry in enumerate(rows):
        if layout == "filename":
            rep, site = entry["rp"]
        else:
            rep, site = site_of(index, sites, reps, layout)
        grid_row, grid_col = grid_rowcol(site, ncols)
        record = {
            "index": index,
            "file": entry["file"],
            "shot": entry["shot"],
            "rep": rep,
            "site": site,
            "grid_row": grid_row,
            "grid_col": grid_col,
            "site_known": rep not in set(partial_reps),
            "status": entry["row"].get("status", ""),
            "qc_status": entry["row"].get("qc_status", ""),
        }
        for column in STATE_COLUMNS:
            record[column] = _float(entry["row"].get(column))
        for element in elements:
            record[f"frac_{element}"] = _float(entry["row"].get(element))
            record[f"unc_{element}"] = _float(entry["row"].get(f"{element}_unc"))
        shots.append(record)
    return shots


def purity_table(shots: List[dict], detections: List[dict],
                 elements: Sequence[str]) -> List[dict]:
    """Per-element detection rate and level across all shots.

    Detection status comes from ``detections.csv`` when it is available --
    that table is the pipeline's own adjudication, including upper limits
    for non-detections -- and falls back to a nonzero fraction otherwise.
    """
    by_element: Dict[str, List[dict]] = defaultdict(list)
    for row in detections:
        by_element[row.get("element", "")].append(row)

    n_shots = len(shots)
    out = []
    for element in elements:
        fractions = [s.get(f"frac_{element}") for s in shots]
        present = [f for f in fractions if f is not None and f > 0]
        rows = by_element.get(element, [])
        statuses = [r.get("status", "") for r in rows]
        n_detected = sum(1 for s in statuses if s == "detected")
        upper_limits = [_float(r.get("upper_limit")) for r in rows]
        upper_limits = [u for u in upper_limits if u is not None]
        z_scores = [_float(r.get("z")) for r in rows]
        z_scores = [z for z in z_scores if z is not None]
        stats = summarize(present)
        out.append({
            "element": element,
            "n_shots": n_shots,
            "n_detected": n_detected,
            "detect_rate": (n_detected / n_shots) if n_shots else None,
            "n_nonzero_fraction": len(present),
            "median_fraction": stats["median"],
            "max_fraction": stats["max"],
            "rsd_pct": stats["rsd_pct"],
            "median_upper_limit": (float(np.median(upper_limits))
                                   if upper_limits else None),
            "max_z": (max(z_scores) if z_scores else None),
            "status_counts": json.dumps(
                {s: statuses.count(s) for s in sorted(set(statuses)) if s}),
        })
    out.sort(key=lambda r: (-(r["n_detected"] or 0),
                            -(r["median_fraction"] or 0.0)))
    return out


def repeatability_table(shots: List[dict], elements: Sequence[str],
                        reps: int) -> List[dict]:
    """Within-rep (shot-to-shot) and across-site dispersion of key quantities."""
    tracked = [f"frac_{MAJOR}"] + list(STATE_COLUMNS)
    out = []
    for key in tracked:
        for rep in range(reps):
            values = [s.get(key) for s in shots if s["rep"] == rep]
            stats = summarize(values)
            out.append({"quantity": key, "scope": f"rep{rep + 1}", **stats})
        stats = summarize([s.get(key) for s in shots])
        out.append({"quantity": key, "scope": "all", **stats})

        # site-mean dispersion: averages out shot noise, leaves spatial spread
        by_site: Dict[int, List[float]] = defaultdict(list)
        for shot in shots:
            value = shot.get(key)
            if value is not None:
                by_site[shot["site"]].append(value)
        site_means = [float(np.mean(v)) for v in by_site.values() if v]
        stats = summarize(site_means)
        out.append({"quantity": key, "scope": "site_means", **stats})
    return out


def cleanup_rollup(shots: List[dict], reps: int,
                   contaminants: Sequence[str]) -> List[dict]:
    """Rep-level contaminant medians -- valid even without site pairing.

    This is the headline cleanup result: it needs only the rep each shot
    belongs to, so a rep that lost a location still contributes.
    """
    out = []
    for element in contaminants:
        key = f"frac_{element}"
        record = {"element": element}
        medians = []
        for rep in range(reps):
            values = [s[key] for s in shots
                      if s["rep"] == rep and s.get(key) is not None]
            stats = summarize(values)
            record[f"rep{rep + 1}_median"] = stats["median"]
            record[f"rep{rep + 1}_n"] = stats["n"]
            record[f"rep{rep + 1}_detect_n"] = sum(1 for v in values if v > 0)
            medians.append(stats["median"])
        first, last = medians[0], medians[-1]
        record["delta_first_last"] = (None if first is None or last is None
                                      else last - first)
        record["frac_drop_pct"] = (None if not first or last is None
                                   else 100.0 * (first - last) / first)
        record["slope_per_rep"] = theil_sen_slope(
            [float(i) for i, m in enumerate(medians) if m is not None],
            [m for m in medians if m is not None])
        out.append(record)
    out.sort(key=lambda r: -(r["frac_drop_pct"] or -1e9))
    return out


def cleanup_table(shots: List[dict], ncols: int,
                  reps: int, contaminants: Sequence[str]) -> List[dict]:
    """Per-site contaminant level by rep, with a robust trend slope.

    Only shots whose grid site is actually known are paired; a rep that
    returned fewer shots than the design has position-not-site and is
    excluded here (it still appears in :func:`cleanup_rollup`).
    """
    out = []
    paired = [s for s in shots if s.get("site_known", True)]
    for element in contaminants:
        key = f"frac_{element}"
        by_site: Dict[int, Dict[int, float]] = defaultdict(dict)
        for shot in paired:
            value = shot.get(key)
            if value is not None:
                by_site[shot["site"]][shot["rep"]] = value
        for site in sorted(by_site):
            per_rep = by_site[site]
            xs = sorted(per_rep)
            ys = [per_rep[r] for r in xs]
            reps_present = sorted(per_rep)
            first = per_rep.get(reps_present[0]) if reps_present else None
            last = per_rep.get(reps_present[-1]) if reps_present else None
            grid_row, grid_col = grid_rowcol(site, ncols)
            record = {
                "element": element,
                "site": site,
                "grid_row": grid_row,
                "grid_col": grid_col,
                "slope_per_rep": theil_sen_slope([float(x) for x in xs], ys),
                "first_rep": first,
                "last_rep": last,
                "delta_first_last": (None if first is None or last is None
                                     else last - first),
                "frac_drop_pct": (None if not first or last is None
                                  else 100.0 * (first - last) / first),
            }
            for rep in range(reps):
                record[f"rep{rep + 1}"] = per_rep.get(rep)
            out.append(record)

        # element-level rollup across sites
        slopes = [r["slope_per_rep"] for r in out
                  if r["element"] == element and r["slope_per_rep"] is not None]
        drops = [r["frac_drop_pct"] for r in out
                 if r["element"] == element and r["frac_drop_pct"] is not None]
        out.append({
            "element": element,
            "site": "ALL",
            "slope_per_rep": (float(np.median(slopes)) if slopes else None),
            "frac_drop_pct": (float(np.median(drops)) if drops else None),
            "n_sites_declining": sum(1 for s in slopes if s < 0),
            "n_sites": len(slopes),
        })
    return out


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def write_csv(rows: Sequence[dict], path: str) -> None:
    if not rows:
        return
    header: List[str] = []
    for row in rows:
        for key in row:
            if key not in header:
                header.append(key)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="directory holding summary.csv/detections.csv")
    parser.add_argument("--grid", default="4x4",
                        help="grid as ROWSxCOLS (default 4x4)")
    parser.add_argument("--reps", type=int, default=3,
                        help="number of times the grid was repeated")
    parser.add_argument("--layout", default="sequential",
                        choices=("sequential", "interleaved", "filename"),
                        help="acquisition order; 'filename' reads repR_posNN names")
    parser.add_argument("--partial-reps", default="",
                        help="comma-separated 1-based reps that lost shots, so "
                             "position is acquisition order and not a grid site; "
                             "these are excluded from per-site pairing")
    parser.add_argument("--order", default="shot", choices=("shot", "file"),
                        help="what defines acquisition order")
    parser.add_argument("--major", default=MAJOR,
                        help="expected matrix element (default Fe)")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default RUN_DIR/qc)")
    args = parser.parse_args(argv)

    try:
        nrows, ncols = (int(v) for v in args.grid.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--grid must look like 4x4, got {args.grid!r}")
    sites = nrows * ncols

    summary_path = find_summary(args.run_dir)
    summary_rows = _read_csv(summary_path)
    if not summary_rows:
        raise SystemExit(f"{summary_path} has no rows")
    elements = element_columns(list(summary_rows[0].keys()))
    if not elements:
        raise SystemExit(f"no element columns found in {summary_path}")

    detections_path = os.path.join(args.run_dir, "detections.csv")
    detections = _read_csv(detections_path) if os.path.isfile(detections_path) else []

    partial = tuple(int(v) - 1 for v in args.partial_reps.split(",") if v.strip())

    expected = sites * args.reps
    if len(summary_rows) != expected and args.layout != "filename":
        print(f"WARNING: {len(summary_rows)} spectra but the "
              f"{args.grid} x {args.reps} design expects {expected}; "
              "the (rep, site) mapping below is therefore suspect.",
              file=sys.stderr)

    shots = build_shots(summary_rows, elements, sites, args.reps,
                        args.layout, ncols, args.order, partial)

    major = args.major
    contaminants = [e for e in elements if e != major]

    out_dir = args.out_dir or os.path.join(args.run_dir, "qc")
    os.makedirs(out_dir, exist_ok=True)

    purity = purity_table(shots, detections, elements)
    repeat = repeatability_table(shots, elements, args.reps)
    cleanup = cleanup_table(shots, ncols, args.reps, contaminants)
    rollup = cleanup_rollup(shots, args.reps, contaminants)

    write_csv(shots, os.path.join(out_dir, "shots.csv"))
    write_csv(purity, os.path.join(out_dir, "purity.csv"))
    write_csv(repeat, os.path.join(out_dir, "repeatability.csv"))
    write_csv(cleanup, os.path.join(out_dir, "cleanup.csv"))
    write_csv(rollup, os.path.join(out_dir, "cleanup_by_rep.csv"))

    manifest = {
        "run_dir": os.path.abspath(args.run_dir),
        "summary": os.path.basename(summary_path),
        "n_spectra": len(summary_rows),
        "grid": args.grid,
        "sites": sites,
        "reps": args.reps,
        "layout": args.layout,
        "order": args.order,
        "partial_reps_1based": [p + 1 for p in partial],
        "shots_per_rep": {f"rep{r + 1}": sum(1 for s in shots if s["rep"] == r)
                          for r in range(args.reps)},
        "major": major,
        "elements": elements,
        "detections_rows": len(detections),
    }
    with open(os.path.join(out_dir, "qc_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    # console digest
    print(f"{len(shots)} shots, {sites} sites x {args.reps} reps, "
          f"layout={args.layout}")
    major_stats = summarize([s.get(f"frac_{major}") for s in shots])
    if major_stats["n"]:
        print(f"{major}: median={major_stats['median']:.4f} "
              f"RSD={major_stats['rsd_pct']:.2f}%" if major_stats["rsd_pct"]
              is not None else f"{major}: median={major_stats['median']:.4f}")
    print("\ntop non-matrix elements by detection count:")
    for row in purity:
        if row["element"] == major:
            continue
        if not row["n_detected"]:
            continue
        median = row["median_fraction"]
        print(f"  {row['element']:<3s} detected in {row['n_detected']:>2d}/"
              f"{row['n_shots']} shots  median_frac="
              f"{median:.5f}" if median is not None else
              f"  {row['element']:<3s} detected in {row['n_detected']:>2d}/"
              f"{row['n_shots']} shots")
    print("\ncleanup by rep (median fraction):")
    for row in rollup[:6]:
        cells = " -> ".join(
            f"{row.get(f'rep{r + 1}_median'):.5f}"
            if row.get(f"rep{r + 1}_median") is not None else "  n/a  "
            for r in range(args.reps))
        drop = row.get("frac_drop_pct")
        print(f"  {row['element']:<3s} {cells}"
              + (f"   drop={drop:+.1f}%" if drop is not None else ""))

    print(f"\nwrote {out_dir}/"
          "{shots,purity,repeatability,cleanup,cleanup_by_rep}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
