#!/usr/bin/env python
"""Build the empirical pixel-integrated LSF (ePSF) of the Z300 native grid.

Deterministic.  Stages (each cached as a pickle in --workdir):

  catalog   load native run means (Fe acq + V run2), register each run and
            segment by PATTERN matching in pixel space (the vendor pixToNm is
            off by ~18.3 px on these files; see the report), identify lines
            with db-isolation + position criteria, extract fit windows.
  lamcv     5-fold held-out chi2 vs the smoothness weight (sets LAM).
  build     per-segment instrument ePSF from the reference class (thin,
            isolated, identified, delay >= 5 us), with 5-fold
            cross-validation by db line, and 3 wavelength sub-bands per
            segment (each with its own 5 CV folds).  The recommended shape at a
            wavelength is the linear interpolation (B-spline coefficients) of
            the sub-band shapes at their median db wavelengths: the LSF is
            closer to constant in nm than in px (report section 2).
  write     cal/data/epsf-z300-<tag>.json with shapes and provenance
            (tag from --dataset: 20260923 uncorrected native, 20260924 corrected axis).

Validation lives in cal/scripts/validate_epsf.py.

Usage:
    .venv/bin/python cal/scripts/build_epsf.py [--workdir DIR] [--out JSON]
        [--stage all|catalog|build|write] [--jobs N]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker (determinism)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from epsf_common import (REPO, FE_ACQ, V_ACQ, FE_LEDGER, V_LEDGER, KB_EV, DATASETS,  # noqa: E402
                         load_all, db_table, pattern_offset, find_peaks_u,
                         identify, running_median)
from alibz.utils.epsf import (EPSF, LineData, build_epsf, fit_pixint_gauss,  # noqa: E402
                              gauss_epsf, fit_line)

SEG_NAMES = ("UV", "VIS", "NIR")
CONTAM = ("Ar", "Ca", "Na", "K", "H", "O", "N")
ID_WEIGHTS_OTHER = {"Ar": 1.0, "Ca": 0.05, "Na": 0.05, "K": 0.05, "H": 0.05,
                    "O": 0.05, "N": 0.05}
T_EXC = 9000.0          # K, for relative strengths / optical-depth proxy
WIN = 7                 # px either side of the peak pixel stored per line
N_FOLDS = 5
ID_DOM = 3.0            # db dominance within +-2 px required for an identification
ID_REL_MIN = 0.005      # min predicted strength rel. to the segment's strongest (per element)
REF_TAU_MAX = 0.3       # reference class: tau_rel below this (see report)
REF_EI_MIN = 0.12       # eV; excludes ground-term (resonance) lines
REF_RESID_MAX = 0.35    # px; |obs - db - registration| (rejects mis-identifications)
REF_DELAY_MIN = 5       # us
REF_SNR = (15.0, 3000.0)


# ---------------------------------------------------------------------------
def tau_rel_table(elements):
    """Optical-depth proxy per line, normalised per (element, ion) over the
    whole 185-950 nm range:  tau ~ f_ik * lambda * n_i ~ gA * lambda^3 *
    exp(-E_i / kT) (g_i f_ik = 1.499e-16 lambda^2 g_k A_ki; n_i ~ g_i e^-Ei/kT)."""
    tab = db_table(elements, T=T_EXC)
    t = tab["gA"] * tab["wl"] ** 3 * np.exp(-tab["Ei"] / (KB_EV * T_EXC))
    out = np.zeros_like(t)
    for el in np.unique(tab["el"]):
        for ion in (1, 2):
            m = (tab["el"] == el) & (tab["ion"] == ion)
            if m.any():
                out[m] = t[m] / t[m].max()
    tab["tau_rel"] = out
    return tab


def catalog(runs, register="pattern"):
    tabs = {fam: tau_rel_table([fam] + list(CONTAM)) for fam in ("Fe", "V")}
    lines = []
    reg = []
    for r in runs:
        tab = tabs[r.family]
        w = dict(ID_WEIGHTS_OTHER); w[r.family] = 1.0
        for sn in SEG_NAMES:
            seg = r.segs[sn]
            own = {k: v[tab["el"] == r.family] for k, v in tab.items()}
            du0_pat, z, z2 = pattern_offset(seg, own)
            # 'pattern': start the registration from the pattern offset (needed on
            # the uncorrected native files); 'zero': the axis is corrected, the
            # pattern offset is only a CHECK and registration starts from 0.
            du0 = du0_pat if register == "pattern" else 0.0
            pk5 = find_peaks_u(seg, 5.0)
            P = []
            for k, snr in pk5:
                if snr < 8.0:
                    continue
                g = fit_pixint_gauss(seg.u, seg.y, seg.sigma, c0=float(k), half=3.5)
                if g.get("ok") and abs(g["center"] - k) < 0.8:
                    P.append((k, snr, g["center"]))
            ids, poly = identify(seg, tab, du0, P, weights=w, dom=ID_DOM, rel_min=ID_REL_MIN)
            reg.append(dict(run=r.id, family=r.family, seg=sn, du0=du0, du_pattern=du0_pat,
                            axis_check=(r.meta.get("axis_check") or {}).get("offset_px"), z=z, z2=z2,
                            poly=[float(x) for x in poly], n_id=len(ids)))
            base = running_median(seg.y, 41)
            yb = seg.y - base
            pku = np.array([p[0] for p in pk5]); pkh = np.array([yb[p[0]] for p in pk5])
            for d in ids:
                k = d["k"]
                if k - WIN < 0 or k + WIN >= seg.n:
                    continue
                # isolation: no other local max (SNR>=5) within +-2 px above
                # 3 % of this line, none within +-4 px taller than the line;
                # local maxima at 2-7.5 px are fitted as neighbour components
                nb4 = (np.abs(pku - k) <= 4) & (pku != k)
                nb2 = (np.abs(pku - k) <= 2) & (pku != k)
                iso_ok = not (np.any(pkh[nb4] > 1.0 * yb[k]) or np.any(pkh[nb2] > 0.03 * yb[k]))
                nbrs = [float(x) for x in pku[(np.abs(pku - k) > 2) & (np.abs(pku - k) <= 7.5)]]
                j = np.where((tab["el"] == d["el"]) & (np.abs(tab["wl"] - d["wl_db"]) < 1e-6))[0]
                tau_rel = float(tab["tau_rel"][j[0]]) if j.size else np.nan
                sl = slice(k - WIN, k + WIN + 1)
                d.update(run=r.id, family=r.family, seg=sn, delay=r.delay, period=r.period,
                         u=seg.u[sl].copy(), y=seg.y[sl].copy(), sigma=seg.sigma[sl].copy(),
                         pitch=float(seg.pitch(k)), wl_k=float(seg.wl[k]),
                         iso_ok=bool(iso_ok), tau_rel=tau_rel, neighbors=nbrs,
                         line_key=f"{d['el']}{d['ion']}-{d['wl_db']:.3f}")
                lines.append(d)
    return lines, reg


def is_reference(d):
    return (d["iso_ok"] and d["tau_rel"] < REF_TAU_MAX and d["Ei"] > REF_EI_MIN
            and d["el"] not in ("Na", "Ca", "K", "H")
            and (d["delay"] is not None and d["delay"] >= REF_DELAY_MIN)
            and REF_SNR[0] <= d["snr"] <= REF_SNR[1]
            and abs(d["resid"]) < REF_RESID_MAX)


def sub_bands(R, nb=3):
    """Split reference windows into nb wavelength bands with equal numbers of
    distinct db lines (a line never straddles two bands)."""
    keys = sorted({d["line_key"] for d in R}, key=lambda k: float(k.split("-")[-1]))
    edges = [keys[int(round(i * len(keys) / nb))] if i < nb else None for i in range(nb + 1)]
    out = []
    for b in range(nb):
        lo = float(edges[b].split("-")[-1])
        hi = float(edges[b + 1].split("-")[-1]) if edges[b + 1] else np.inf
        out.append([d for d in R if lo <= d["wl_db"] < hi])
    return out


def fold_of(key, keys_sorted):
    return keys_sorted.index(key) % N_FOLDS


def to_linedata(d):
    return LineData(d["u"], d["y"], d["sigma"], c0=float(d["uc"]), key=(d["run"], d["seg"], d["k"]),
                    info=dict(neighbors=d.get("neighbors", []), group=d["line_key"],
                              run=d["run"], upos=float(d["u_db"])))


# smoothness weight per segment from 5-fold held-out chi2 per sample over LAM_GRID
# (--stage lamcv, 2026-09-23; UV 7.09/6.64/6.78/6.96, VIS 4.90/5.07/4.97/5.49,
# NIR 12.98/12.84/12.23/11.92 for 0.03/0.1/0.3/1.0).  The UV/VIS curves are flat to
# ~3 % over 0.03-0.3 (4 of 60 CV builds stopped early on a degenerate solve; guarded since),
# so 0.3 is used for both; NIR prefers the smoothest shape (few lines).
LAM_BY_DATASET = {
    "native-20260923": {"UV": 0.3, "VIS": 0.3, "NIR": 1.0},
    # corrected axis (lamcv 2026-09-24; chi2/sample for 0.03/0.1/0.3/1.0):
    # UV 6.08/6.97/6.27/5.92, VIS 4.49/4.40/4.28/4.92, NIR 7.06/7.73/7.84/7.12.
    # UV -> 1.0 (minimum), VIS -> 0.3 (minimum), NIR -> 1.0 (within 1 % of the
    # 0.03 minimum; the smoother shape is preferred with 21 distinct lines).
    "corrected-20260924": {"UV": 1.0, "VIS": 0.3, "NIR": 1.0},
}
LAM = dict(LAM_BY_DATASET["corrected-20260924"])
LAM_GRID = (0.03, 0.1, 0.3, 1.0)


def _build_job(args):
    name, L, lam = args[0], args[1], (args[2] if len(args) > 2 else LAM[args[0].split("/")[0]])
    try:
        ep, fits, hist = build_epsf(L, init=gauss_epsf(0.55), lam=lam, n_iter=40,
                                    warp_mode="dither_struct", run_slope=True)
    except Exception as exc:  # keep the other builds
        return name, None, None, [dict(failed=repr(exc))]
    return name, ep.to_dict(), [f.get("center") if f.get("ok") else None for f in fits], hist


def _heldout_chi2(args):
    epd, L = args
    if epd is None:
        return np.inf, 1
    ep = EPSF.from_dict(epd)
    tot, n = 0.0, 0
    for x in L:
        f = fit_line(ep, x.u, x.y, x.sigma, c0=x.c0, neighbors=x.info.get("neighbors"), fallback=False)
        if f.get("ok"):
            tot += float(np.sum((f["resid"] / x.sigma[np.abs(x.u - x.c0) <= 4.5]) ** 2))
            n += int(f["resid"].size)
    return tot, n


def lam_cv(lines, jobs=4):
    """Held-out chi2 of the reference lines vs the smoothness weight."""
    ref = [d for d in lines if is_reference(d)]
    keys = sorted({d["line_key"] for d in ref}, key=lambda s: float(s.split("-")[-1]))
    for d in ref:
        d["fold"] = fold_of(d["line_key"], keys)
    tasks, held = [], {}
    for sn in SEG_NAMES:
        R = [d for d in ref if d["seg"] == sn]
        for lam in LAM_GRID:
            for f in range(N_FOLDS):
                tasks.append((f"{sn}/{lam}/{f}", [to_linedata(d) for d in R if d["fold"] != f], lam))
                held[f"{sn}/{lam}/{f}"] = [to_linedata(d) for d in R if d["fold"] == f]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        built = {}
        for name, epd, _, hist in ex.map(_build_job, tasks):
            built[name] = epd
            fails = [h for h in hist if h.get("failed")]
            if fails or epd is None:
                print("  note", name, fails[:1], flush=True)
        names = list(built)
        chis = list(ex.map(_heldout_chi2, [(built[k], held[k]) for k in names]))
    res = {}
    for (k, (c, n)) in zip(names, chis):
        sn, lam, f = k.split("/")
        r = res.setdefault(sn, {}).setdefault(lam, [0.0, 0])
        r[0] += c; r[1] += n
    out = {sn: {lam: dict(chi2=v[0], n=v[1], chi2_per_sample=v[0] / max(v[1], 1)) for lam, v in d.items()}
           for sn, d in res.items()}
    for sn, d in out.items():
        print(sn, {lam: round(v["chi2_per_sample"], 4) for lam, v in d.items()})
    return out


def build(lines, jobs=4):
    ref = [d for d in lines if is_reference(d)]
    keys = sorted({d["line_key"] for d in ref}, key=lambda s: float(s.split("-")[-1]))
    for d in ref:
        d["fold"] = fold_of(d["line_key"], keys)
    tasks = []
    for sn in SEG_NAMES:
        R = [d for d in ref if d["seg"] == sn]
        tasks.append((f"{sn}/full", [to_linedata(d) for d in R]))
        for f in range(N_FOLDS):
            tasks.append((f"{sn}/cv{f}", [to_linedata(d) for d in R if d["fold"] != f]))
        # 3 sub-bands with equal numbers of distinct reference lines (+ 5 CV folds each)
        for b, sub in enumerate(sub_bands(R)):
            tasks.append((f"{sn}/band{b}", [to_linedata(x) for x in sub]))
            for f in range(N_FOLDS):
                tasks.append((f"{sn}/band{b}cv{f}", [to_linedata(x) for x in sub if x["fold"] != f]))
    out = {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for name, epd, cen, hist in ex.map(_build_job, tasks):
            out[name] = dict(epsf=epd, centers=cen, history=hist)
            if epd is None:
                print(f"  FAILED {name}: {hist}", flush=True)
                continue
            fails = [h for h in hist if h.get("failed")]
            print(f"  built {name}: n={epd['meta'].get('n_lines')} fwhm={epd['fwhm_px']:.3f} px "
                  f"iters={len(hist)} warp={hist[-1].get('warp_amp', float('nan')):.4f}"
                  + (f" NOTE {fails[0]['failed']}" if fails else ""), flush=True)
    return ref, out


def git_rev():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


LAMCV_PATH = None
DATASET = "native-20260923"


def write_json(path, lines, reg, ref, built):
    seg_info = {}
    for sn in SEG_NAMES:
        R = [d for d in ref if d["seg"] == sn]
        pitch = np.array([d["pitch"] for d in R])
        folds = [EPSF.from_dict(built[f"{sn}/cv{f}"]["epsf"]) for f in range(N_FOLDS)]
        fw = np.array([e.fwhm() for e in folds])
        jk = float(np.sqrt((N_FOLDS - 1) / N_FOLDS * np.sum((fw - fw.mean()) ** 2)))
        bands = []
        for b, sub in enumerate(sub_bands(R)):
            e = built[f"{sn}/band{b}"]["epsf"]
            pb = np.array([d["pitch"] for d in sub]) if sub else np.array([np.nan])
            wl = np.array([d["wl_db"] for d in sub]) if sub else np.array([np.nan])
            bands.append(dict(band=b, n_lines=e["meta"].get("n_lines"),
                              wl_center_nm=float(np.median(wl)),
                              cv_folds=[built[f"{sn}/band{b}cv{f}"]["epsf"] for f in range(N_FOLDS)],
                              n_db_lines=len({d["line_key"] for d in sub}),
                              wl_range_nm=[float(np.nanmin(wl)), float(np.nanmax(wl))],
                              fwhm_px=e["fwhm_px"], median_pitch_nm=float(np.nanmedian(pb)),
                              fwhm_nm=float(e["fwhm_px"] * np.nanmedian(pb)), epsf=e))
        full = built[f"{sn}/full"]["epsf"]
        seg_info[sn] = dict(
            reference=dict(n_lines=len(R), n_runs=len({d["run"] for d in R}),
                           n_db_lines=len({d["line_key"] for d in R}),
                           species=sorted({f"{d['el']}{'I' * d['ion']}" for d in R}),
                           median_pitch_nm=float(np.median(pitch)),
                           criteria=dict(tau_rel_max=REF_TAU_MAX, Ei_min_eV=REF_EI_MIN,
                                         delay_min_us=REF_DELAY_MIN, snr=REF_SNR,
                                         resid_max_px=REF_RESID_MAX, id_dom=ID_DOM, id_rel_min=ID_REL_MIN,
                                         iso="no local max >3% within +-2 px, none taller within +-4 px; "
                                             "maxima at 2-7.5 px fitted as neighbour components")),
            epsf=full, fwhm_px=full["fwhm_px"], fwhm_px_jackknife_sd=jk,
            fwhm_nm_at_median_pitch=float(full["fwhm_px"] * np.median(pitch)),
            cv_folds=[built[f"{sn}/cv{f}"]["epsf"] for f in range(N_FOLDS)],
            sub_bands=bands,
            build_history=built[f"{sn}/full"]["history"])
    data_files = sorted({d["run"] for d in lines})
    h = hashlib.sha256()
    for d in sorted(lines, key=lambda x: (x["run"], x["seg"], x["k"])):
        h.update(np.asarray(d["y"], float).tobytes())
    doc = dict(
        schema="alibz.epsf/1",
        created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        instrument="SciAps Z300 (native API grid, 7,914 samples; 4th channel excluded)",
        units=dict(d="pixels of the local native pitch (consecutive samples within a segment, ascending wavelength)",
                   area="counts * px (multiply by local pitch for counts * nm)"),
        model="y_k = A * Psi(u_k - c) + b0 + b1 (u_k - c); Psi pixel-integrated, unit integral, "
              "compact support |d|<R, partition of unity sum_k Psi(k+x)=1, centroid(|d|<=2)=0, "
              "uniform-phase warp applied during the build (alibz/utils/epsf.py)",
        segments=seg_info,
        registration=dict(note="pattern pixel offset du0 (observed u = u_vendor(db) + du) per run/segment; "
                               "vendor pixToNm is ~18.3 px off on these files",
                          per_run=reg),
        smoothness_lam=LAM,
        lam_cv=(json.load(open(LAMCV_PATH)) if LAMCV_PATH and os.path.exists(LAMCV_PATH) else None),
        provenance=dict(script="cal/scripts/build_epsf.py", git_head=git_rev(),
                        dataset=DATASET, dataset_cfg=DATASETS[DATASET],
                        data=dict(fe_acq=FE_ACQ, fe_ledger=FE_LEDGER, v_acq=V_ACQ, v_ledger=V_LEDGER,
                                  n_runs=len(data_files), window_sha256=h.hexdigest()),
                        db="alibz/utils/database.Database('db') (air wavelengths)",
                        T_exc_K=T_EXC),
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, default=float)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=os.path.join(tempfile.gettempdir(), "alibz-epsf-work"),
                    help="cache directory for stage pickles (default $TMPDIR/alibz-epsf-work)")
    ap.add_argument("--dataset", default="corrected-20260924", choices=sorted(DATASETS),
                    help="data preset (epsf_common.DATASETS); sets the default --out tag")
    ap.add_argument("--out", default=None, help="default cal/data/epsf-z300-<tag>.json")
    ap.add_argument("--stage", default="all", choices=["all", "catalog", "lamcv", "build", "write"])
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    pr = os.path.join(a.workdir, "runs.pkl")
    pc = os.path.join(a.workdir, "catalog.pkl")
    pb = os.path.join(a.workdir, "built.pkl")
    global LAMCV_PATH, DATASET
    DATASET = a.dataset
    LAM.clear(); LAM.update(LAM_BY_DATASET[a.dataset])
    if a.out is None:
        a.out = os.path.join(REPO, "cal", "data", f"epsf-z300-{DATASETS[a.dataset]['tag']}.json")
    LAMCV_PATH = os.path.join(a.workdir, "lamcv.json")
    runs = load_all(cache=pr, dataset=a.dataset)
    if a.stage in ("all", "catalog") or not os.path.exists(pc):
        lines, reg = catalog(runs, DATASETS[a.dataset]["register"])
        pickle.dump((lines, reg), open(pc, "wb"))
        print(f"catalog: {len(lines)} identified line windows")
    lines, reg = pickle.load(open(pc, "rb"))
    if a.stage == "catalog":
        return
    if a.stage == "lamcv":
        res = lam_cv(lines, a.jobs)
        json.dump(res, open(os.path.join(a.workdir, "lamcv.json"), "w"), indent=1)
        return
    if a.stage in ("all", "build") or not os.path.exists(pb):
        ref, built = build(lines, a.jobs)
        pickle.dump((ref, built), open(pb, "wb"))
    ref, built = pickle.load(open(pb, "rb"))
    if a.stage in ("all", "write", "build"):
        write_json(a.out, lines, reg, ref, built)


if __name__ == "__main__":
    main()
