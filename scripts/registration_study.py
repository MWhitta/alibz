#!/usr/bin/env python
"""Wavelength-registration measurement study over the Fe/V acquisition runs.

Regenerates the numbers and figures behind
``reports/2026-09-23-wavelength-registration.md``: per-run ambient and element
registrations (10-shot means), per-segment shift versus warm-up time, the
vendor base->current calibration change, within-segment wavelength dependence,
and the post-registration residual.

Data live under a scratchpad acquisition tree (``--acq``) with per-run
``shots/shot-N.csv`` and two ledgers (Fe: ``ledger.json``; V: ``ledger_v.json``)
providing ``created_at`` (UTC) and the element.

Calibration epochs.  The per-run ``calibrationTime`` lives only in the analyzer
shot records on Moissanite (not available offline); the ledger gives only
``created_at`` (UTC).  The brief establishes the grouping: the three earliest Fe
runs carry the 2026-09-21 calibration; run-126d0c96 (16:41:37 UTC) onward carry
the 2026-09-22 13:39 calibration.  The analyzer local clock is known to be
mis-set (~+3 h from UTC), so the absolute warm-up zero is uncertain -- but the
DRIFT SLOPE (the owner's actual question) is invariant to a constant offset in
the epoch, so we anchor each epoch consistently and report the slope.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timezone

import numpy as np

from alibz.utils.database import Database
from alibz.pipeline import load_spectrum_csv
from alibz import wavelength_registration as wr

# --- calibration epochs (UTC), documented above ---------------------------
# run-126d0c96 (first 2026-09-22 13:39-cal run) is at 16:41:37 UTC; the brief
# places it ~4 min after calibration, so anchor cal-A at 16:37:37 UTC.
CAL_A_EPOCH = datetime(2026, 9, 22, 16, 37, 37, tzinfo=timezone.utc)
CAL_A_ID = "2026-09-22T13:39:21 (analyzer local)"
# The 2026-09-21 calibration ("2:55:07 PM" local + ~3 h analyzer offset).
CAL_B_EPOCH = datetime(2026, 9, 21, 17, 55, 7, tzinfo=timezone.utc)
CAL_B_ID = "2026-09-21T14:55:07 (analyzer local)"
# runs created before this UTC instant used the 2026-09-21 calibration
CAL_A_FIRST_RUN_UTC = datetime(2026, 9, 22, 16, 41, 37, tzinfo=timezone.utc)

# vendor calibration cubics (calibration.json base, currentwlcalibration.json)
VENDOR_BASE = [[3.67951969E2, -7.78662478E-2, -5.99133112E-6, 4.94396034E-10],
               [6.25593006E2, -1.11792094E-1, -9.35107472E-6, 6.53116663E-10],
               [9.48170506E2, -1.55936742E-1, -1.40437163E-5, 9.01628935E-10],
               [961, -0.0004, 1E-12, 1E-12]]
VENDOR_CURRENT = [[368.0025857351851, -0.0778498710316756, -5.990071027925716E-6, 4.942920530128833E-10],
                  [625.4823570464079, -0.11177738525661182, -9.349844377553242E-6, 6.530307309358004E-10],
                  [947.9356738154337, -0.1559247671299144, -1.404263783910583E-5, 9.0155969609438E-10],
                  [961.0, -4.0E-4, 1.0E-12, 1.0E-12]]
VENDOR_KNOTS = [180, 365, 620, 960, 961]


def _parse(ts):
    return datetime.fromisoformat(ts)


def load_runs(acq_dir, fe_ledger, v_ledger):
    runs = []
    for path, element in ((fe_ledger, "Fe"), (v_ledger, "V")):
        for rec in json.load(open(path)):
            run = rec["run"]
            shots = sorted(glob.glob(os.path.join(acq_dir, run, "shots", "shot-*.csv")))
            if not shots:
                continue
            runs.append({"run": run, "element": element,
                         "created_at": rec["created_at"], "n_shots": len(shots),
                         "shots": shots})
    return runs


def mean_spectrum(shots):
    ys = [load_spectrum_csv(s)[1] for s in shots]
    x = load_spectrum_csv(shots[0])[0]
    return x, np.mean(np.asarray(ys), axis=0)


def warmup_minutes(created_at):
    t = _parse(created_at)
    if t < CAL_A_FIRST_RUN_UTC:
        return (t - CAL_B_EPOCH).total_seconds() / 60.0, CAL_B_ID
    return (t - CAL_A_EPOCH).total_seconds() / 60.0, CAL_A_ID


def _resid_pm(ele):
    out = []
    for name in wr.SEGMENT_NAMES:
        s = ele["segments"][name]
        if s["model"] and s["n_lines"] >= 5:
            out.extend(1000.0 * np.asarray(s["model"]["residuals_nm"]))
    return out


def paired_subpixel_residuals(x, y, db, composition, cfg):
    """Gaussian vs parabolic centring on the SAME detected peaks.

    For every high-SNR local maximum, compute both the Gaussian and the
    parabolic sub-pixel centre, match to the nearest strong database line
    within 0.12 nm, subtract the per-segment median offset, and return the
    residual scatter (pm) for each method on the identical peak set.
    """
    from scipy.ndimage import percentile_filter
    from alibz.utils.peakfit import gaussian_subpixel_center, instrument_sigma_px
    dbwl, _ = wr._db_strong_lines(db, composition, cfg)
    if dbwl.size == 0:
        return [], []
    size = max(15, int(round(2.0 / float(np.median(np.diff(x))))))
    base = percentile_filter(y, 25, size=size)
    resid = y - base
    noise = max(1.4826 * float(np.median(np.abs(resid - np.median(resid)))), 1e-9)
    ismax = np.zeros(x.size, dtype=bool)
    ismax[1:-1] = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
    cand = np.where(ismax & (resid >= 5.0 * noise))[0]
    by_seg = {}
    for i in cand:
        g = gaussian_subpixel_center(x, y, int(i), sigma_px=instrument_sigma_px(float(x[i])))
        if not g.get("ok") or g["snr"] < 8:
            continue
        cg = g["center_nm"]
        cp = wr._parabolic_center(resid, x, int(min(max(i, 1), x.size - 2)), x[i])
        j = int(np.argmin(np.abs(dbwl - cg)))
        if abs(dbwl[j] - cg) > 0.12:
            continue
        seg = wr.SEGMENT_NAMES[int(wr._segment_of(cg))]
        by_seg.setdefault(seg, []).append((cg - dbwl[j], cp - dbwl[j]))
    resg, resp = [], []
    for rows in by_seg.values():
        if len(rows) < 5:
            continue
        g = np.array([r[0] for r in rows]); p = np.array([r[1] for r in rows])
        resg.extend(1000.0 * (g - np.median(g)))
        resp.extend(1000.0 * (p - np.median(p)))
    return resg, resp


def analyse_run(x, y, db, element, subpixel_compare=True):
    amb = wr.ambient_registration(x, y, db)
    ele = wr.element_registration(x, y, db, [element])
    comb = wr.combined_registration(amb, ele)
    resid_gauss, resid_parab = [], []
    if subpixel_compare:
        resid_gauss, resid_parab = paired_subpixel_residuals(
            x, y, db, [element], wr._config(None))
    seg_shifts, seg_sigma, seg_n, seg_q, seg_slope = {}, {}, {}, {}, {}
    golden = {}
    for name in wr.SEGMENT_NAMES:
        s = ele["segments"][name]
        seg_shifts[name] = s["shift_nm"] if s["quality"] == "ok" else None
        seg_sigma[name] = s["sigma_nm"]
        seg_n[name] = s["n_lines"]
        seg_q[name] = s["quality"]
        seg_slope[name] = s.get("slope_nm_per_nm")
        golden[name] = [(l["db_nm"], l["shift_nm"], l.get("residual_nm"))
                        for l in s.get("lines", []) if l.get("matched")]
    amb_nir = amb["segments"]["NIR"]["shift_nm"]
    dis = comb["nir_disagreement"]
    return {"ambient": amb, "element": ele, "combined": comb,
            "segment_shifts": seg_shifts, "segment_sigma": seg_sigma,
            "segment_n": seg_n, "segment_quality": seg_q,
            "segment_slope": seg_slope, "golden": golden,
            "ambient_nir_shift_nm": amb_nir,
            "ambient_species": amb["species_detected"],
            "nir_disagreement": dis,
            "resid_gauss_pm": resid_gauss, "resid_parab_pm": resid_parab}


def fe_v_transfer(records):
    """(3) Validation: fit Δλ(λ) from fresh-Fe golden lines per segment and
    apply it to the V golden lines; report the V residual (median |res|) and
    whether it drops toward the 30–50 pm level with no wavelength trend."""
    out = {}
    fe = [r for r in records if r["element"] == "Fe"
          and "2026-09-22" in r["calibration_id"]]
    vv = [r for r in records if r["element"] == "V"]
    for name in wr.SEGMENT_NAMES:
        fpts = [(w[0], w[1]) for r in fe for w in r["result"]["golden"][name]]
        vpts = [(w[0], w[1]) for r in vv for w in r["result"]["golden"][name]]
        if len(fpts) < 5 or len(vpts) < 5:
            continue
        fl = np.array([p[0] for p in fpts]); fs = np.array([p[1] for p in fpts])
        ref = float(np.median(fl))
        coeffs, _ = wr._huber_polyfit(fl - ref, fs, 1)
        vl = np.array([p[0] for p in vpts]); vs = np.array([p[1] for p in vpts])
        vres = 1000.0 * (vs - np.polyval(coeffs, vl - ref))
        trend = wr._linfit_with_sigma(vl, vs - np.polyval(coeffs, vl - ref))
        out[name] = {"n_fe": len(fpts), "n_v": len(vpts),
                     "v_residual_median_pm": float(np.median(np.abs(vres))),
                     "v_residual_rms_pm": float(np.sqrt(np.mean(vres ** 2))),
                     "v_residual_slope_pm_per_nm": (1000.0 * trend["slope"]
                                                    if trend else None)}
    return out


def run_to_run_sd(records):
    """(3) Per-segment run-to-run SD of the element shift within each
    (element, calibration epoch) group; target <= 30 pm per segment."""
    out = {}
    for element in ("Fe", "V"):
        for epoch in ("2026-09-22", "2026-09-21"):
            grp = [r for r in records if r["element"] == element
                   and epoch in r["calibration_id"]]
            if len(grp) < 3:
                continue
            for name in wr.SEGMENT_NAMES:
                vals = [1000 * r["segment_shifts"][name] for r in grp
                        if r["segment_shifts"][name] is not None]
                if len(vals) >= 3:
                    out[f"{element}/{epoch[-5:]}/{name}"] = {
                        "n": len(vals), "mean_pm": float(np.mean(vals)),
                        "sd_pm": float(np.std(vals))}
    return out


def fe_v_reconciliation(records):
    """(2) The SAME instrument function must reconcile both metals: pooled
    per-segment window modes for Fe vs V and the mean Fe-minus-V offset."""
    out = {}
    for name in wr.SEGMENT_NAMES:
        per = {}
        for element in ("Fe", "V"):
            pts = [(w[0], 1000 * w[1]) for r in records
                   if r["element"] == element for w in r["result"]["golden"][name]]
            per[element] = pts
        if per["Fe"] and per["V"]:
            fe_m = float(np.median([p[1] for p in per["Fe"]]))
            v_m = float(np.median([p[1] for p in per["V"]]))
            out[name] = {"fe_median_pm": fe_m, "v_median_pm": v_m,
                         "fe_minus_v_pm": fe_m - v_m,
                         "n_fe": len(per["Fe"]), "n_v": len(per["V"])}
    return out


def within_segment_slope_significance(records):
    """(c) Wavelength dependence within a segment, from the golden-line shifts.

    Pool the matched golden-line (db_nm, shift) pairs per element+segment and
    fit shift vs wavelength; report the slope and its sigma.
    """
    out = {}
    for element in ("Fe", "V"):
        for name in wr.SEGMENT_NAMES:
            lam, dl = [], []
            for r in records:
                if r["element"] != element:
                    continue
                for w in r["result"]["golden"][name]:
                    lam.append(w[0]); dl.append(w[1])
            if len(lam) < 6:
                continue
            lam = np.asarray(lam); dl = np.asarray(dl)
            fit = wr._linfit_with_sigma(lam, dl)
            if fit is None:
                continue
            out[f"{element}/{name}"] = {
                "n_lines": int(lam.size),
                "slope_pm_per_nm": 1000.0 * fit["slope"],
                "slope_sigma_pm_per_nm": 1000.0 * fit["slope_sigma"],
                "significant": abs(fit["slope"]) > 2 * fit["slope_sigma"],
                "residual_rms_pm": 1000.0 * fit["rms_nm"],
            }
    return out


def post_registration_residual(records):
    """(d) Residual scatter of per-line shifts about the per-segment model."""
    res = {"Fe": [], "V": []}
    for r in records:
        for name in wr.SEGMENT_NAMES:
            seg = r["result"]["element"]["segments"][name]
            if seg["model"] and seg["n_lines"] >= 5:
                res[r["element"]].extend(seg["model"]["residuals_nm"])
    summary = {}
    for el, v in res.items():
        if v:
            v = np.asarray(v)
            summary[el] = {"n": int(v.size),
                           "median_abs_pm": 1000.0 * float(np.median(np.abs(v))),
                           "rms_pm": 1000.0 * float(np.sqrt(np.mean(v ** 2)))}
    return summary


def make_figures(records, thermal, fig_dir, epoch_a_range=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)

    # (a) shift vs minutes, per segment, both metals
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    colors = {"Fe": "#1f77b4", "V": "#d62728"}
    for ax, name in zip(axes, wr.SEGMENT_NAMES):
        for el in ("Fe", "V"):
            m = [r["warmup_min"] for r in records
                 if r["element"] == el and r["segment_shifts"][name] is not None]
            s = [1000 * r["segment_shifts"][name] for r in records
                 if r["element"] == el and r["segment_shifts"][name] is not None]
            ax.scatter(m, s, s=18, alpha=0.7, color=colors[el], label=el)
        f = thermal["segments"][name].get("vs_minutes")
        if f:
            rng = epoch_a_range or (min(r["warmup_min"] for r in records),
                                    max(r["warmup_min"] for r in records))
            xs = np.array(rng)
            res = "resolved" if thermal["segments"][name]["resolved"] else "n.s."
            ax.plot(xs, 1000 * (f["intercept"] + f["slope"] * xs), "k--", lw=1,
                    label=f"{1000*f['slope']*60:+.1f} pm/h ({res})")
        ax.set_title(f"{name} segment"); ax.set_xlabel("warm-up (min)")
        ax.axhline(0, color="0.7", lw=0.6); ax.legend(fontsize=8)
    axes[0].set_ylabel("element shift (pm)")
    fig.suptitle("Per-segment wavelength shift vs warm-up time")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "shift_vs_minutes.png"), dpi=110)
    plt.close(fig)

    # (b) ambient vs element NIR shift scatter
    fig, ax = plt.subplots(figsize=(5, 5))
    for el in ("Fe", "V"):
        pts = [(1000 * r["result"]["element"]["segments"]["NIR"]["shift_nm"],
                1000 * r["ambient_nir_shift_nm"]) for r in records
               if r["element"] == el
               and r["result"]["element"]["segments"]["NIR"]["shift_nm"] is not None
               and r["ambient_nir_shift_nm"] is not None]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=22, alpha=0.7, color=colors[el], label=el)
    lim = [-400, 400]
    ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("element NIR shift (pm)"); ax.set_ylabel("ambient NIR shift (pm)")
    ax.set_title("Ambient vs element NIR shift"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "ambient_vs_element_nir.png"), dpi=110)
    plt.close(fig)

    # (c) golden-line shift vs lambda, one Fe + one V
    for el in ("Fe", "V"):
        cand = [r for r in records if r["element"] == el
                and sum(len(r["result"]["golden"][n]) for n in wr.SEGMENT_NAMES) >= 6]
        if not cand:
            continue
        r = max(cand, key=lambda z: sum(len(z["result"]["golden"][n])
                                        for n in wr.SEGMENT_NAMES))
        fig, ax = plt.subplots(figsize=(8, 4))
        for name in wr.SEGMENT_NAMES:
            ws = r["result"]["golden"][name]
            if not ws:
                continue
            lam = np.array([w[0] for w in ws]); sh = np.array([1000 * w[1] for w in ws])
            ax.scatter(lam, sh, s=30, color=colors[el], label=name)
        ax.axhline(0, color="0.6", lw=0.6)
        for e in wr.SEGMENT_EDGES:
            ax.axvline(e, color="0.85", lw=0.8)
        ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("golden-line shift (pm)")
        ax.set_title(f"{el} run {r['run'][:14]}: golden-line shifts")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"residual_vs_lambda_{el}.png"), dpi=110)
        plt.close(fig)

    # (d) Fe vs V reconciliation: golden-line shifts on one axis
    fig, ax = plt.subplots(figsize=(9, 4))
    for el in ("Fe", "V"):
        pts = [(w[0], 1000 * w[1]) for r in records if r["element"] == el
               for name in wr.SEGMENT_NAMES for w in r["result"]["golden"][name]]
        if pts:
            lam, sh = zip(*pts)
            ax.scatter(lam, sh, s=10, alpha=0.4, color=colors[el], label=el)
    for e in wr.SEGMENT_EDGES:
        ax.axvline(e, color="0.85", lw=0.8)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("golden-line shift (pm)")
    ax.set_title("Fe vs V golden-line shifts (should share one instrument function)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fe_v_reconciliation.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    default_acq = ("/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/"
                   "39a75126-564e-4be9-a01b-a385e7a2834d/scratchpad")
    ap.add_argument("--acq", default=os.path.join(default_acq, "acq"))
    ap.add_argument("--fe-ledger", default=os.path.join(default_acq, "ledger.json"))
    ap.add_argument("--v-ledger", default=os.path.join(default_acq, "ledger_v.json"))
    ap.add_argument("--out", default="reports/registration-study-20260923.json")
    ap.add_argument("--fig-dir", default="reports/figures/registration-20260923")
    ap.add_argument("--dbpath", default="db")
    ap.add_argument("--limit", type=int, default=None, help="first N runs (debug)")
    ap.add_argument("--no-subpixel", action="store_true", help="skip the slow paired sub-pixel comparison")
    args = ap.parse_args()

    db = Database(args.dbpath)
    runs = load_runs(args.acq, args.fe_ledger, args.v_ledger)
    if args.limit:
        runs = runs[: args.limit]
    print(f"{len(runs)} runs")

    records = []
    for i, run in enumerate(runs):
        x, y = mean_spectrum(run["shots"])
        result = analyse_run(x, y, db, run["element"], subpixel_compare=not args.no_subpixel)
        wu, cal_id = warmup_minutes(run["created_at"])
        rec = {"run": run["run"], "element": run["element"],
               "created_at": run["created_at"], "n_shots": run["n_shots"],
               "warmup_min": wu, "calibration_id": cal_id,
               "segment_shifts": result["segment_shifts"],
               "segment_sigma": result["segment_sigma"],
               "segment_n": result["segment_n"],
               "ambient_nir_shift_nm": result["ambient_nir_shift_nm"],
               "ambient_species": result["ambient_species"],
               "nir_disagreement": result["nir_disagreement"],
               "result": result}
        records.append(rec)
        print(f"  [{i+1}/{len(runs)}] {run['run'][:14]} {run['element']} "
              f"wu={wu:6.1f}min UV/VIS/NIR="
              + "/".join(str(result["segment_n"][n]) for n in wr.SEGMENT_NAMES))

    def _thermal_records(subset):
        return [{"minutes_since_calibration": r["warmup_min"],
                 "segment_shifts": r["segment_shifts"], "temperature_c": None,
                 "ambient_nir_shift_nm": r["ambient_nir_shift_nm"],
                 "calibration_id": r["calibration_id"]} for r in subset]

    # Fit the drift WITHIN a single calibration epoch: runs from different
    # calibrations have different fixed offsets, so pooling them across the
    # ~15 h gap between epochs (3 old-cal runs at ~1330 min vs 50 at 4-414 min)
    # would let the high-leverage old-cal points mask a real within-epoch
    # drift.  Epoch A (2026-09-22 13:39 cal) is the only epoch with warm-up
    # spread, so it is the primary drift result; the pooled fit is kept for
    # transparency.
    epoch_a = [r for r in records if "2026-09-22" in r["calibration_id"]] or records
    thermal = wr.thermal_drift_model(_thermal_records(epoch_a))
    thermal_pooled = wr.thermal_drift_model(_thermal_records(records))
    epoch_a_range = (min(r["warmup_min"] for r in epoch_a),
                     max(r["warmup_min"] for r in epoch_a))
    vendor = wr.vendor_calibration_shift(VENDOR_BASE, VENDOR_CURRENT, VENDOR_KNOTS)
    slopes = within_segment_slope_significance(records)
    residual = post_registration_residual(records)
    r2r = run_to_run_sd(records)
    reconcile = fe_v_reconciliation(records)
    transfer = fe_v_transfer(records)
    golden_counts = {comp: {nm: len(wr.golden_lines(db, [comp], nm))
                            for nm in wr.SEGMENT_NAMES} for comp in ("Fe", "V")}

    # parabolic vs Gaussian sub-pixel centring, same runs (measurable benefit)
    g = np.abs(np.concatenate([np.asarray(r["result"]["resid_gauss_pm"])
                               for r in records if r["result"]["resid_gauss_pm"]]
                              or [np.array([])]))
    p = np.abs(np.concatenate([np.asarray(r["result"]["resid_parab_pm"])
                               for r in records if r["result"]["resid_parab_pm"]]
                              or [np.array([])]))
    subpixel_cmp = {
        "gaussian": {"n": int(g.size), "median_abs_pm": float(np.median(g)) if g.size else None,
                     "rms_pm": float(np.sqrt(np.mean(g ** 2))) if g.size else None},
        "parabolic": {"n": int(p.size), "median_abs_pm": float(np.median(p)) if p.size else None,
                      "rms_pm": float(np.sqrt(np.mean(p ** 2))) if p.size else None},
    }

    make_figures(records, thermal, args.fig_dir, epoch_a_range)

    # strip the bulky nested 'result' before serialising the per-run table
    table = []
    for r in records:
        t = {k: v for k, v in r.items() if k != "result"}
        table.append(t)
    summary = {
        "n_runs": len(records),
        "calibration_epochs": {"cal_A": {"id": CAL_A_ID, "epoch_utc": CAL_A_EPOCH.isoformat()},
                               "cal_B": {"id": CAL_B_ID, "epoch_utc": CAL_B_EPOCH.isoformat()}},
        "per_run": table,
        "thermal_drift_epoch_A": {name: dict(thermal["segments"][name])
                                  for name in wr.SEGMENT_NAMES},
        "thermal_drift_pooled_all_epochs": {name: dict(thermal_pooled["segments"][name])
                                            for name in wr.SEGMENT_NAMES},
        "thermal_resolved_epoch_A": thermal["resolved"],
        "epoch_A_warmup_range_min": list(epoch_a_range),
        "n_epoch_A": len(epoch_a), "n_epoch_B": len(records) - len(epoch_a),
        "run_to_run_sd": r2r,
        "fe_v_reconciliation": reconcile,
        "fe_to_v_transfer": transfer,
        "golden_line_counts": golden_counts,
        "vendor_calibration_change": vendor,
        "within_segment_slopes": slopes,
        "post_registration_residual": residual,
        "subpixel_comparison": subpixel_cmp,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(wr.registration_to_json(summary), fh, indent=1)
    print("wrote", args.out)

    # console verdict digest
    print(f"\n=== thermal drift, epoch A only (n={len(epoch_a)}, "
          f"warm-up {epoch_a_range[0]:.0f}-{epoch_a_range[1]:.0f} min) ===")
    for name in wr.SEGMENT_NAMES:
        f = thermal["segments"][name].get("vs_minutes")
        if f:
            print(f"  {name}: {1000*f['slope']*60:+.1f} pm/h "
                  f"(2sigma limit {thermal['segments'][name]['detection_limit_nm_per_hour']*1000:.1f} pm/h) "
                  f"resolved={thermal['segments'][name]['resolved']} n={f['n']}")
    print("=== thermal drift, pooled across both epochs (masks drift) ===")
    for name in wr.SEGMENT_NAMES:
        f = thermal_pooled["segments"][name].get("vs_minutes")
        if f:
            print(f"  {name}: {1000*f['slope']*60:+.1f} pm/h "
                  f"resolved={thermal_pooled['segments'][name]['resolved']} n={f['n']}")
    print("\n=== vendor base->current change (pm) ===")
    for name, s in vendor["segments"].items():
        print(f"  {name}: {1000*s['delta_nm_at_lo']:+.0f}..{1000*s['delta_nm_at_hi']:+.0f} "
              f"mean {1000*s['delta_nm_mean']:+.0f}")
    print("\n=== run-to-run SD within (element,epoch) group [target <=30 pm] ===")
    for k, v in r2r.items():
        print(f"  {k}: mean {v['mean_pm']:+.0f} SD {v['sd_pm']:.0f} pm (n={v['n']})")
    print("=== golden-line counts /segment ===", golden_counts)
    print("=== Fe->V transfer (Fe-fit function applied to V golden lines) ===")
    for name, v in transfer.items():
        print(f"  {name}: V residual median {v['v_residual_median_pm']:.0f} pm "
              f"rms {v['v_residual_rms_pm']:.0f} trend {v['v_residual_slope_pm_per_nm']} "
              f"(nFe={v['n_fe']} nV={v['n_v']})")
    print("=== Fe vs V reconciliation (median golden-line shift per segment) ===")
    for name, v in reconcile.items():
        print(f"  {name}: Fe {v['fe_median_pm']:+.0f} V {v['v_median_pm']:+.0f} "
              f"Fe-V {v['fe_minus_v_pm']:+.0f} pm (nFe={v['n_fe']} nV={v['n_v']})")
    print("=== within-segment wavelength dependence (golden lines) ===")
    for k, v in slopes.items():
        print(f"  {k}: {v['slope_pm_per_nm']:+.1f}+-{v['slope_sigma_pm_per_nm']:.1f} pm/nm "
              f"sig={v['significant']} resid {v['residual_rms_pm']:.0f}pm n={v['n_lines']}")
    print("\n=== post-registration residual ===", residual)
    print("=== sub-pixel comparison (same detected peaks) ===")
    for meth in ("gaussian", "parabolic"):
        s = subpixel_cmp[meth]
        med = f"{s['median_abs_pm']:.1f}" if s['median_abs_pm'] is not None else "n/a"
        rms = f"{s['rms_pm']:.1f}" if s['rms_pm'] is not None else "n/a"
        print(f"  {meth}: median|res| {med} pm rms {rms} pm n={s['n']}")


if __name__ == "__main__":
    main()
