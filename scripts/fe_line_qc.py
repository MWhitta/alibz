#!/usr/bin/env python
"""Line-intensity QC for a repeated grid of shots on a nominally pure target.

This is the fallback path for when the full CF-LIBS inversion does not
survive its own quality gates.  On the 2026-09-21 Z300 Fe checkout
``alibz-analyze`` returned ``qc_status = fail`` for 46 of 47 spectra
(r^2 median 0.42, ``electron-density-at-bound`` on 43, ``composition-collapse``
on 5), so its atom fractions are not usable.  The *spectra* are fine; only
the inversion collapsed.  Everything here is measured straight off the
spectrum and needs no plasma model:

* **presence** -- is a diagnostic line there, above local noise?
* **strength** -- background-subtracted peak area, and its ratio to Fe,
  which cancels shot-to-shot coupling variation
* **repeatability** -- RSD of those quantities across shots in a rep
* **cleanup** -- how the contaminant/Fe ratio moves from rep to rep

A line ratio is not a concentration.  Relative *changes* in a ratio across
shots of the same material on the same instrument are meaningful; the
absolute value is not, and nothing here should be read as a mass fraction.

Usage::

    python scripts/fe_line_qc.py SPECTRA_DIR --out-dir DIR \
        --pattern 'rep*_pos*.csv'
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REP_POS_RE = re.compile(r"rep(\d+)[_-]pos(\d+)", re.IGNORECASE)

# Diagnostic lines, NIST air wavelengths in nm.  Chosen to be strong and
# reasonably isolated at this instrument's resolution.  ``Fe-ref`` is the
# normalising set: several Fe I/II lines spread across the detector so the
# ratio is not hostage to one segment's response.
LINES: Dict[str, List[float]] = {
    # matrix
    "Fe II": [238.204, 239.562, 259.940, 261.187, 273.955, 274.932],
    "Fe I": [371.993, 373.486, 374.556, 382.043, 385.991, 404.581, 438.354],
    # surface / handling contamination
    "Na I": [588.995, 589.592, 818.326, 819.482],
    "K I": [766.490, 769.896],
    "Li I": [670.791],
    "Ca II": [393.366, 396.847],
    "Ca I": [422.673],
    "Mg II": [279.553, 280.270],
    "Mg I": [285.213],
    "H I": [656.279],
    "O I": [777.194, 777.417, 777.539],
    # plausible steel/alloy impurities
    "C I": [247.856],
    "Si I": [251.611, 288.158],
    "Mn I": [403.076, 403.307, 403.449],
    "Cr I": [425.435, 427.480, 428.972],
    "Ni I": [341.476, 352.454],
    "Cu I": [324.754, 327.396],
    "Al I": [394.401, 396.152],
    "Ti II": [334.941, 336.121],
}

FE_REFERENCE = ("Fe II", "Fe I")

# Integration half-width and the offsets of the two background shoulders.
HALF_WIDTH_NM = 0.25
BACKGROUND_OFFSET_NM = 0.9
BACKGROUND_HALF_NM = 0.35
DETECT_SIGMA = 3.0


def load_spectrum(path: str) -> Tuple[np.ndarray, np.ndarray]:
    wavelength, intensity = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            wavelength.append(x)
            intensity.append(y)
    return np.asarray(wavelength), np.asarray(intensity)


def rep_pos(path: str) -> Tuple[Optional[int], Optional[int]]:
    m = REP_POS_RE.search(os.path.basename(path))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def measure_line(wavelength: np.ndarray, intensity: np.ndarray,
                 center: float) -> Optional[dict]:
    """Background-subtracted area and SNR for one line.

    The background is the median of two shoulders either side of the line;
    the noise is their combined MAD-based sigma, which is robust to a
    neighbouring line clipping a shoulder.
    """
    peak = np.abs(wavelength - center) <= HALF_WIDTH_NM
    left = (np.abs(wavelength - (center - BACKGROUND_OFFSET_NM)) <= BACKGROUND_HALF_NM)
    right = (np.abs(wavelength - (center + BACKGROUND_OFFSET_NM)) <= BACKGROUND_HALF_NM)
    shoulders = left | right
    if peak.sum() < 3 or shoulders.sum() < 5:
        return None

    baseline = float(np.median(intensity[shoulders]))
    mad = float(np.median(np.abs(intensity[shoulders] - baseline)))
    sigma = 1.4826 * mad
    if sigma <= 0:
        sigma = float(np.std(intensity[shoulders])) or 1e-9

    corrected = intensity[peak] - baseline
    step = float(np.median(np.diff(wavelength[peak]))) if peak.sum() > 1 else 0.1
    area = float(np.sum(corrected) * step)
    height = float(np.max(corrected))
    return {
        "area": area,
        "height": height,
        "baseline": baseline,
        "sigma": sigma,
        "snr": height / sigma,
        "n_peak": int(peak.sum()),
    }


def measure_species(wavelength: np.ndarray, intensity: np.ndarray,
                    centers: Sequence[float]) -> dict:
    """Aggregate a species' lines: summed area over the lines that are present."""
    per_line = [measure_line(wavelength, intensity, c) for c in centers]
    present = [m for m in per_line if m is not None]
    if not present:
        return {"area": None, "n_lines": 0, "n_detected": 0, "max_snr": None}
    detected = [m for m in present if m["snr"] >= DETECT_SIGMA]
    return {
        # sum only detected lines; a line that is absent should add nothing
        # rather than adding its (noisy, possibly negative) integral
        "area": float(sum(m["area"] for m in detected)) if detected else 0.0,
        "n_lines": len(present),
        "n_detected": len(detected),
        "max_snr": max(m["snr"] for m in present),
    }


def rsd(values: Sequence[float]) -> Optional[float]:
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
        return {"n": 0, "median": None, "mean": None, "sd": None,
                "min": None, "max": None, "rsd_pct": None}
    arr = np.asarray(clean, float)
    return {"n": int(arr.size), "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max()),
            "rsd_pct": rsd(clean)}


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
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spectra_dir")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.spectra_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"no spectra matching {args.pattern} in {args.spectra_dir}")

    out_dir = args.out_dir or os.path.join(args.spectra_dir, "line_qc")
    os.makedirs(out_dir, exist_ok=True)

    shots = []
    for path in paths:
        wavelength, intensity = load_spectrum(path)
        rep, pos = rep_pos(path)
        record = {"file": os.path.basename(path), "rep": rep, "pos": pos,
                  "max_intensity": float(np.max(intensity))}
        species = {name: measure_species(wavelength, intensity, centers)
                   for name, centers in LINES.items()}
        fe_area = sum(species[k]["area"] or 0.0 for k in FE_REFERENCE)
        record["Fe_area"] = fe_area
        for name, m in species.items():
            key = name.replace(" ", "")
            record[f"{key}_area"] = m["area"]
            record[f"{key}_ndet"] = m["n_detected"]
            record[f"{key}_snr"] = None if m["max_snr"] is None else round(m["max_snr"], 2)
            if name not in FE_REFERENCE:
                record[f"{key}_ratio"] = (m["area"] / fe_area) if fe_area > 0 and m["area"] is not None else None
        shots.append(record)
    write_csv(shots, os.path.join(out_dir, "line_shots.csv"))

    reps = sorted({s["rep"] for s in shots if s["rep"] is not None})
    species_names = [n for n in LINES if n not in FE_REFERENCE]

    # presence / strength across all shots
    presence = []
    n_shots = len(shots)
    for name in list(LINES):
        key = name.replace(" ", "")
        det = [s for s in shots if (s.get(f"{key}_ndet") or 0) > 0]
        ratios = [s.get(f"{key}_ratio") for s in shots]
        snrs = [s.get(f"{key}_snr") for s in shots if s.get(f"{key}_snr") is not None]
        stats = summarize([r for r in ratios if r is not None])
        presence.append({
            "species": name,
            "n_shots": n_shots,
            "n_shots_detected": len(det),
            "detect_rate": len(det) / n_shots,
            "max_snr": max(snrs) if snrs else None,
            "median_ratio_to_Fe": stats["median"],
            "max_ratio_to_Fe": stats["max"],
            "ratio_rsd_pct": stats["rsd_pct"],
        })
    presence.sort(key=lambda r: -r["n_shots_detected"])
    write_csv(presence, os.path.join(out_dir, "line_presence.csv"))

    # repeatability within each rep
    repeat = []
    for quantity in ["Fe_area", "max_intensity"] + [f"{n.replace(' ', '')}_ratio"
                                                    for n in species_names]:
        for rep in reps:
            vals = [s.get(quantity) for s in shots if s["rep"] == rep]
            repeat.append({"quantity": quantity, "scope": f"rep{rep}",
                           **summarize(vals)})
        repeat.append({"quantity": quantity, "scope": "all",
                       **summarize([s.get(quantity) for s in shots])})
    write_csv(repeat, os.path.join(out_dir, "line_repeatability.csv"))

    # cleanup trend by rep
    cleanup = []
    for name in species_names:
        key = name.replace(" ", "")
        row = {"species": name}
        medians = []
        for rep in reps:
            vals = [s.get(f"{key}_ratio") for s in shots if s["rep"] == rep]
            stats = summarize(vals)
            row[f"rep{rep}_median_ratio"] = stats["median"]
            row[f"rep{rep}_ndet"] = sum(1 for s in shots if s["rep"] == rep
                                        and (s.get(f"{key}_ndet") or 0) > 0)
            row[f"rep{rep}_n"] = stats["n"]
            medians.append(stats["median"])
        first, last = medians[0], medians[-1]
        row["drop_pct_first_to_last"] = (None if not first or last is None
                                         else 100.0 * (first - last) / first)
        cleanup.append(row)
    cleanup.sort(key=lambda r: -(r["drop_pct_first_to_last"] or -1e9))
    write_csv(cleanup, os.path.join(out_dir, "line_cleanup.csv"))

    with open(os.path.join(out_dir, "line_qc_manifest.json"), "w") as fh:
        json.dump({"spectra_dir": os.path.abspath(args.spectra_dir),
                   "n_spectra": n_shots,
                   "reps": reps,
                   "shots_per_rep": {f"rep{r}": sum(1 for s in shots if s["rep"] == r)
                                     for r in reps},
                   "half_width_nm": HALF_WIDTH_NM,
                   "detect_sigma": DETECT_SIGMA,
                   "lines": LINES}, fh, indent=2, sort_keys=True)

    # console digest
    print(f"{n_shots} spectra; shots per rep: "
          + ", ".join(f"rep{r}={sum(1 for s in shots if s['rep'] == r)}" for r in reps))
    print("\nspecies present (by shots detected):")
    for row in presence:
        if not row["n_shots_detected"]:
            continue
        ratio = row["median_ratio_to_Fe"]
        print(f"  {row['species']:<6s} {row['n_shots_detected']:>3d}/{n_shots}"
              f"  maxSNR={row['max_snr']:>7.1f}"
              + (f"  median ratio/Fe={ratio:.4f}" if ratio is not None else ""))
    print("\ncontaminant/Fe ratio by rep:")
    for row in cleanup:
        if not any(row.get(f"rep{r}_ndet") for r in reps):
            continue
        cells = " -> ".join(
            f"{row.get(f'rep{r}_median_ratio'):.4f}"
            if row.get(f"rep{r}_median_ratio") is not None else " n/a "
            for r in reps)
        drop = row.get("drop_pct_first_to_last")
        print(f"  {row['species']:<6s} {cells}"
              + (f"   drop={drop:+.0f}%" if drop is not None else ""))
    print(f"\nwrote {out_dir}/line_{{shots,presence,repeatability,cleanup}}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
