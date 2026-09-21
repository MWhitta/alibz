#!/usr/bin/env python
"""Stage-by-stage least-squares residual audit for alibz.pipeline.analyze_spectrum.

MEASUREMENT ONLY.  This script does not alter any pipeline behaviour: it
calls the exact same functions in the exact same order as
``alibz.pipeline.analyze_spectrum`` (mirroring the stage walk already
implemented in ``scripts/fit_inspector.py::capture``), and additionally
instruments the internal steps of ``PeakyFinder.fit_spectrum``
(fit_peaks -> fit_shoulders -> fit_all -> filters) via a thin tracing
subclass that overrides those three methods to snapshot their output
before returning it unchanged -- the real, un-modified ``fit_spectrum``
method (inherited, not reimplemented) still runs and still performs the
filter step exactly as shipped.

For every stage it computes, against ``y_bgsub = y - fit['background']``:
  - the model = multi_voigt(x, peaks) evaluated on the native x grid,
  - SSE, RMS, and a noise-normalised chi2 (PeakyFinder._noise_scale_local
    with DEFAULT_SEGMENT_EDGES = (365, 620)), per segment (UV <365,
    VIS 365-620, NIR >620) and total,
  - an "effective" chi2 = chi2 / 4, an indicative correction for
    pixel-to-pixel correlation (oversampled ~3-5x, rho(1) ~ 0.8-0.9 on
    this export grid -- NOT a rigorous effective-N; see report caveats),
  - peak count (area > 0).

For every stage transition with an action record (refine_fit decision,
minor_lines record, deblend_shoulders record, or an internal filter
drop/clip), it computes the window (+/-1 nm around the action's center)
delta-SSE = SSE_after - SSE_before in that window, and keeps the top
windows with the largest positive delta for attribution.

Usage:
    python scripts/audit_stage_residuals.py --file data/remote_samples/REE_01.csv
    python scripts/audit_stage_residuals.py --all   # all 4 files, sequential

Writes per-sample stage-metrics and window-attribution CSVs to
--outdir (default: scratchpad), and prints a summary table.
"""
import argparse
import copy
import json
import os
import sys
import time
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from alibz.peaky_finder import PeakyFinder                        # noqa: E402
from alibz.minor_lines import (recover_residual_lines,            # noqa: E402
                               seed_minor_lines)
from alibz.peaky_indexer_v3 import PeakyIndexerV3                 # noqa: E402
from alibz.profiles import (analyze_peak_profiles, deblend_shoulders,  # noqa: E402
                            recover_sa_areas)
from alibz.refinement import refine_fit                           # noqa: E402
from alibz.pipeline import (CONFIDENT_MIN_REFS, DEEPEN_BARS,      # noqa: E402
                            ESTABLISHED_MIN_FRACTION, SUPPORT_GA_FLOOR,
                            SUPPORT_TOL_NM, composition_collapsed,
                            load_spectrum_csv, sample_name, _halpha_ne,
                            resolve_dbpath)
from alibz.inspection import estimate_peak_uncertainties          # noqa: E402
from alibz.detector import (correct_segment_response,             # noqa: E402
                            estimate_segment_response,
                            segment_response_fallback)
from alibz.utils.database import Database                         # noqa: E402
from alibz.utils.voigt import multi_voigt, voigt_width             # noqa: E402
from alibz.utils.wavelength import (estimate_wavelength_shift,    # noqa: E402
                                    estimate_wavelength_shift_segments,
                                    shift_at)

DEFAULT_SEGMENT_EDGES = PeakyFinder.DEFAULT_SEGMENT_EDGES  # (365.0, 620.0)


# ===========================================================================
# Tracing subclass: intercepts fit_peaks / fit_shoulders / fit_all, snapshots
# their output, and returns it UNCHANGED so fit_spectrum's own (unmodified,
# inherited) logic -- including the post-fit filter block -- runs exactly as
# shipped.
# ===========================================================================
class TracedFinder(PeakyFinder):
    def fit_peaks(self, x, y, peak_indices, plot=False, fast: bool = False):
        out = PeakyFinder.fit_peaks(self, x, y, peak_indices, plot=plot, fast=fast)
        self.trace["fit_peaks"] = copy.deepcopy(out)
        return out

    def fit_shoulders(self, x, y, peak_indices, residuals, peak_dictionary, rng=5):
        out_dict, out_idx = PeakyFinder.fit_shoulders(
            self, x, y, peak_indices, residuals, peak_dictionary, rng=rng)
        self.trace["fit_shoulders"] = copy.deepcopy(out_dict)
        return out_dict, out_idx

    def fit_all(self, x, y, peak_dictionary, plot=True):
        out = PeakyFinder.fit_all(self, x, y, peak_dictionary, plot=plot)
        self.trace["fit_all"] = copy.deepcopy(out)
        return out


def _new_traced_finder():
    f = TracedFinder.__new__(TracedFinder)
    f.trace = {}
    return f


# ===========================================================================
# Residual measurement
# ===========================================================================
def seg_of(x):
    """UV / VIS / NIR label per pixel, per DEFAULT_SEGMENT_EDGES = (365, 620)."""
    e0, e1 = DEFAULT_SEGMENT_EDGES
    lab = np.full(x.shape, "VIS", dtype=object)
    lab[x < e0] = "UV"
    lab[x >= e1] = "NIR"
    return lab


def dict_to_params(d):
    """peak_dictionary -> (n,4) array via PeakyFinder._parameter_array."""
    return PeakyFinder._parameter_array(d)


def model_from_peaks(x, peaks):
    peaks = np.atleast_2d(np.asarray(peaks, dtype=float))
    if peaks.size == 0:
        return np.zeros_like(x)
    keep = peaks[:, 0] > 0
    if not keep.any():
        return np.zeros_like(x)
    return multi_voigt(x, np.ravel(peaks[keep][:, :4]))


def measure_stage(x, y_bgsub, peaks, noise_local, labels):
    """SSE/RMS/chi2/chi2_eff per segment + total, and peak count."""
    peaks = np.atleast_2d(np.asarray(peaks, dtype=float)) if np.size(peaks) else \
        np.empty((0, 4))
    n_peaks = int(np.sum(peaks[:, 0] > 0)) if peaks.size else 0
    model = model_from_peaks(x, peaks)
    resid = y_bgsub - model
    out = {}
    for seg in ("UV", "VIS", "NIR", "total"):
        m = np.ones_like(x, dtype=bool) if seg == "total" else (labels == seg)
        n = int(np.sum(m))
        if n == 0:
            out[seg] = dict(n_pixels=0, sse=0.0, rms=0.0, chi2=0.0, chi2_eff=0.0)
            continue
        r = resid[m]
        sse = float(np.sum(r ** 2))
        rms = float(np.sqrt(sse / n))
        nl = np.maximum(noise_local[m], 1e-12)
        chi2 = float(np.sum((r / nl) ** 2))
        out[seg] = dict(n_pixels=n, sse=sse, rms=rms, chi2=chi2, chi2_eff=chi2 / 4.0)
    out["n_peaks"] = n_peaks
    out["model"] = model  # kept for window deltas; stripped before CSV export
    return out


def window_delta_sse(x, y_bgsub, model_before, model_after, center, half=1.0):
    m = (x >= center - half) & (x <= center + half)
    n = int(np.sum(m))
    if n == 0:
        return 0.0, 0
    rb = y_bgsub[m] - model_before[m]
    ra = y_bgsub[m] - model_after[m]
    return float(np.sum(ra ** 2) - np.sum(rb ** 2)), n


# ===========================================================================
# Filter-block re-derivation (for classifying drops/clips between the
# fit_all trace and the final post-filter blind fit) -- copied logic from
# PeakyFinder.fit_spectrum lines ~1330-1362, read-only / diagnostic use.
# ===========================================================================
def classify_filter_action(before, after, inc, median_fwhm, min_height):
    """Explain why a fit_all-stage component changed to its post-filter state."""
    b2, b3 = float(before[2]), float(before[3])
    a0, a2, a3 = float(after[0]), float(after[2]), float(after[3])
    reasons = []
    if b2 < inc / 2 and a2 == 0.0:
        reasons.append("sigma snap-to-zero (<0.5 px)")
    if b3 < inc / 2 and a3 == 0.0:
        reasons.append("gamma snap-to-zero (<0.5 px)")
    if a0 == 0.0 and float(before[0]) > 0.0:
        if b2 > 100 * median_fwhm or b3 > 100 * median_fwhm:
            reasons.append("width > 100x median FWHM (baseline-remnant drop)")
        elif b2 == 0.0 and b3 == 0.0:
            reasons.append("zero-width drop (sigma==gamma==0)")
        else:
            reasons.append("height-vs-local-noise gate drop "
                           f"(min_height={min_height:.4g})")
    return "; ".join(reasons) if reasons else "refit (no filter action)"


# ===========================================================================
# Stage capture: mirrors scripts/fit_inspector.py::capture, extended with
# (a) the internal fit_spectrum sub-steps, (b) per-round deepening
# breakdown, (c) action records attached to every stage transition so
# window-level delta-SSE attribution is possible.
# ===========================================================================
def audit_capture(x, y, dbpath, db):
    """Returns (stages, transitions, meta).

    ``stages``: list of dicts with name, peaks (n,4 array), and metrics
    from measure_stage (minus the raw model array).
    ``transitions``: list of dicts (from_stage, to_stage, actions) where
    actions is a list of dicts with center, kind, action/verdict, and any
    extra fields (tau_a, n_components, ...).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    finder = _new_traced_finder()
    t0 = time.time()
    fit = finder.fit_spectrum(x, y, subtract_background=True, plot=False, n_sigma=0)
    t_blind = time.time() - t0
    bg0 = np.asarray(fit.get("background", np.zeros_like(y)), dtype=float)
    y_bgsub = y - bg0
    labels = seg_of(x)
    noise_local = PeakyFinder._noise_scale_local(
        y_bgsub, segment_indices=np.searchsorted(x, np.sort(np.asarray(DEFAULT_SEGMENT_EDGES)))
        if x.size else None)

    stages = []       # (name, peaks_array)
    transitions = []  # (from_idx, to_idx, actions[list of dict])

    def add_stage(name, peaks):
        m = measure_stage(x, y_bgsub, peaks, noise_local, labels)
        stages.append(dict(name=name, peaks=np.atleast_2d(np.asarray(peaks, dtype=float))
                           if np.size(peaks) else np.empty((0, 4)), metrics=m))
        return len(stages) - 1

    def add_transition(i_from, i_to, actions):
        transitions.append(dict(i_from=i_from, i_to=i_to, actions=actions))

    # --- internal fit_spectrum sub-steps -----------------------------------
    inc = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    tr = finder.trace
    i_peaks = add_stage("1a . fit_peaks (windowed Voigt seeds)",
                        dict_to_params(tr.get("fit_peaks", {})))
    i_shoulders = add_stage("1b . fit_shoulders (+residual components)",
                            dict_to_params(tr.get("fit_shoulders", {})))
    i_all = add_stage("1c . fit_all (block-coordinate joint refit)",
                      dict_to_params(tr.get("fit_all", {})))

    # shoulders-added actions: keys present in fit_shoulders trace but not
    # in fit_peaks trace
    added_keys = sorted(set(tr.get("fit_shoulders", {}).keys())
                        - set(tr.get("fit_peaks", {}).keys()))
    shoulder_actions = []
    for k in added_keys:
        v = tr["fit_shoulders"][k]
        shoulder_actions.append(dict(center=float(v[1]), kind="fit_shoulders",
                                     action="added", key=int(k)))
    add_transition(i_peaks, i_shoulders, shoulder_actions)

    # fit_all: pure refit of the shoulders set (no add/drop), still an
    # action worth tracking as "refit" windows (params changed materially)
    refit_actions = []
    common_keys = set(tr.get("fit_shoulders", {}).keys()) & set(tr.get("fit_all", {}).keys())
    for k in sorted(common_keys):
        vb = np.asarray(tr["fit_shoulders"][k], dtype=float)
        va = np.asarray(tr["fit_all"][k], dtype=float)
        if not np.allclose(vb, va, rtol=0.02, atol=1e-6):
            refit_actions.append(dict(center=float(va[1]), kind="fit_all",
                                      action="refit", key=int(k)))
    add_transition(i_shoulders, i_all, refit_actions)

    # filters: fit['peak_dictionary'] holds the SAME (mutated in-place)
    # arrays fit_all returned, now post-filter -- compare against the
    # pristine fit_all trace snapshot to classify each change.
    peak_params = dict_to_params(tr.get("fit_all", {}))
    median_fwhm = 0.0
    if peak_params.size:
        widths = voigt_width(peak_params[:, 2], peak_params[:, 3])
        span = float(x[-1] - x[0]) if x.size > 1 else 0.0
        narrow = widths[np.isfinite(widths) & (widths > 0)]
        if span > 0:
            line_like = narrow[narrow < span / 50.0]
            if line_like.size:
                narrow = line_like
        median_fwhm = float(np.median(narrow)) if narrow.size else 0.0
    filter_actions = []
    final_dict = fit.get("peak_dictionary", {})
    for k, before in tr.get("fit_all", {}).items():
        after = final_dict.get(k)
        if after is None:
            continue
        before = np.asarray(before, dtype=float)
        after = np.asarray(after, dtype=float)
        if np.allclose(before, after, rtol=1e-9, atol=1e-12):
            continue
        min_h = PeakyFinder.MIN_HEIGHT_SIGMA * float(
            noise_local[int(np.clip(k, 0, noise_local.size - 1))])
        reason = classify_filter_action(before, after, inc, median_fwhm, min_h)
        filter_actions.append(dict(center=float(before[1]), kind="filter",
                                   action=reason, key=int(k)))
    i_blind = add_stage("1d . post-fit filters (= 1 . blind fit)",
                        fit["sorted_parameter_array"])
    add_transition(i_all, i_blind, filter_actions)

    # detector segment-response correction (amplitude-space only; does not
    # touch the peak table / pixel-space model -- recorded for the report,
    # no transition needed since peaks are byte-identical)
    _noise_glob = 1.4826 * float(np.median(np.abs(y_bgsub - np.median(y_bgsub))))
    seg_response = estimate_segment_response(
        x, bg0, edges=(620.0,), noise_scale=_noise_glob,
        fallback=segment_response_fallback(edges=(620.0,)))

    def _amp_sigma(peaks_arr, seg_resp=seg_response):
        sig = estimate_peak_uncertainties(x, y - bg0, peaks_arr)[:, 0]
        seg = np.searchsorted(np.asarray((620.0,)), np.atleast_2d(peaks_arr)[:, 1])
        return sig / np.asarray(seg_resp, dtype=float)[seg]

    def dbf(p):
        o = np.asarray(p, dtype=float).copy()
        o[:, 1] -= shift_at(SHIFT, o[:, 1])
        return correct_segment_response(o, seg_response, edges=(620.0,))

    # --- 2 . wavelength shift (no peak-table change) -----------------------
    shift0, _ = estimate_wavelength_shift(fit["sorted_parameter_array"], db)

    # --- 3a . data-only refine (asym deferred) ------------------------------
    refined, dec_data = refine_fit(x, y, fit, db=db, shift_nm=shift0, asymmetric="defer")
    SHIFT, _ = estimate_wavelength_shift_segments(refined["sorted_parameter_array"], db)
    i_3a = add_stage("3a . data-only refine (asym deferred)",
                     refined["sorted_parameter_array"])
    actions_3a = [dict(center=float(d["center"]), kind="refine_fit",
                       verdict=d.get("verdict"), action=d.get("action"),
                       n_components=len(d.get("indices", ())) + len(d.get("absorbed", ())),
                       tau_a=d.get("tau_a"))
                 for d in dec_data]
    add_transition(i_blind, i_3a, actions_3a)

    rp = refined["sorted_parameter_array"]
    ne_init, ne_bounds = _halpha_ne(rp)
    run_kw = dict(sa_doublets=True, n_calls=40, verbose=False)
    idx_kw = dict(dbpath=dbpath)
    if ne_init is not None:
        idx_kw["ne_init"] = ne_init
        run_kw["ne_bounds"] = ne_bounds
    idx_kw["amp_sigma_floor"] = _noise_glob

    t1 = time.time()
    idx1 = PeakyIndexerV3(dbf(rp), **idx_kw)
    idx1._amp_sigma = _amp_sigma(rp)
    res1 = idx1.run(**run_kw)
    t_pass1 = time.time() - t1
    # pass-1 indexer: no peak-table change -> same stage/peaks as 3a, but
    # tag the r2 for reporting
    r2_pass1 = float(res1.r_squared)
    established = sorted([e for e, f in res1.element_fractions.items()
                          if f >= ESTABLISHED_MIN_FRACTION])

    # --- 3b . self-absorption tags ------------------------------------------
    posterior = sorted({sp.element for sp in res1.species})
    refined, dec_phys = refine_fit(x, y, refined, db=db, elements=posterior or None,
                                   shift_nm=SHIFT, asymmetric="only")
    i_3b = add_stage("3b . self-absorption tags", refined["sorted_parameter_array"])
    actions_3b = [dict(center=float(d["center"]), kind="refine_fit",
                       verdict=d.get("verdict"), action=d.get("action"),
                       n_components=len(d.get("indices", ())) + len(d.get("absorbed", ())),
                       tau_a=d.get("tau_a"))
                 for d in dec_phys]
    add_transition(i_3a, i_3b, actions_3b)

    decisions = dec_data + dec_phys
    sa_zones, sa_merges = [], []
    for dec in decisions:
        if (dec.get("action") == "sa-tag" and str(dec.get("verdict", "")).startswith("asymmetric")
                and dec.get("params_asym") is not None):
            pS = dec.get("params_single")
            if pS is None:
                pS = dec["params_asym"]
            hw = 1.5 * max(float(voigt_width(max(pS[2], 1e-6), max(pS[3], 1e-6))), 0.15)
            sa_zones.append((float(pS[1]), hw))
            obs = float(dec.get("observed_area") or 0.0)
            if obs > 0.0 and dec.get("emission_area"):
                sa_merges.append(dict(center_nm=float(pS[1]),
                                      factor=float(dec["emission_area"]) / obs,
                                      tau_a=float(dec.get("tau_a", 0.0)),
                                      observed_area=obs,
                                      emission_area=float(dec["emission_area"])))

    # --- 5 . Boltzmann-seeded minor lines ------------------------------------
    final = refined
    seed_records = []
    if established:
        final, seed_records = seed_minor_lines(x, y, refined, db, established,
                                               shift_nm=SHIFT, exclude=tuple(sa_zones))
    i_5 = add_stage("5 . Boltzmann-seeded minor lines", final["sorted_parameter_array"])
    actions_5 = [dict(center=float(r.get("center", r.get("predicted_center_nm", float("nan")))),
                      kind="seed_minor_lines", action=r.get("action"),
                      element=r.get("element"))
                for r in seed_records if r.get("action") == "added"]
    add_transition(i_3b, i_5, actions_5)

    # --- 6 . residual recovery ------------------------------------------------
    prev = final
    final, recover_records = recover_residual_lines(x, y, final, exclude=tuple(sa_zones))
    i_6 = add_stage("6 . residual recovery", final["sorted_parameter_array"])
    actions_6 = [dict(center=float(r.get("center", r.get("center_nm", float("nan")))),
                      kind="recover_residual_lines", action=r.get("action"))
                for r in recover_records if r.get("action") == "added"]
    add_transition(i_5, i_6, actions_6)

    # --- 7 . shoulder deblends -------------------------------------------------
    prof = analyze_peak_profiles(x, y, final)
    final, deblend_records = deblend_shoulders(x, y, final, prof, exclude=tuple(sa_zones))
    i_7 = add_stage("7 . shoulder deblends", final["sorted_parameter_array"])
    actions_7 = [dict(center=float(r.get("center_nm", float("nan"))),
                      kind="deblend_shoulders", action=r.get("action"))
                for r in deblend_records if r.get("action") == "deblended"]
    add_transition(i_6, i_7, actions_7)

    fp = final["sorted_parameter_array"]
    t2 = time.time()
    idx2 = PeakyIndexerV3(dbf(fp), dbpath=dbpath, amp_sigma_floor=_noise_glob,
                          temperature_init=res1.temperature, ne_init=res1.ne)
    idx2._amp_sigma = _amp_sigma(fp)
    res2 = idx2.run(**run_kw)
    t_pass2 = time.time() - t2
    r2_pass2 = float(res2.r_squared)

    # --- 9 . iterative deepening, PER ROUND -----------------------------------
    confirmed = sorted([e for e, f in res2.element_fractions.items()
                        if f >= ESTABLISHED_MIN_FRACTION])
    result, fidx = res2, idx2
    r2_deepen_rounds = []
    i_prev = i_7
    work = final
    if confirmed:
        from alibz.minor_lines import match_and_scale
        scales, _ = match_and_scale(fp, db, confirmed, shift_nm=SHIFT)
        confident = sorted({e for (e, _s), i in scales.items()
                            if i["n_ref"] >= CONFIDENT_MIN_REFS})
        sup = []
        for el in confident:
            if el in db.no_lines:
                continue
            arr = db.lines(el)
            if arr.size == 0:
                continue
            mk = (arr[:, 0].astype(float) <= 2) & (arr[:, 3].astype(float) >= SUPPORT_GA_FLOOR)
            wl = arr[mk, 1].astype(float)
            if wl.size:
                sup.append(wl + shift_at(SHIFT, wl))
        supported = np.concatenate(sup) if sup else np.empty(0)
        for rnd, bar in enumerate(DEEPEN_BARS):
            if not confident:
                break
            prev_round = work
            work, corr = seed_minor_lines(x, y, work, db, confident, shift_nm=SHIFT,
                                          accept_snr=bar, min_expected_snr=bar,
                                          robust_elements=set(confident),
                                          exclude=tuple(sa_zones))
            work, rec = recover_residual_lines(x, y, work, exclude=tuple(sa_zones),
                                               supported_lines=supported,
                                               snr_min_supported=bar, accept_snr_supported=bar,
                                               support_tol_nm=SUPPORT_TOL_NM)
            n_added = (sum(1 for r in corr if r.get("action") == "added")
                      + sum(1 for r in rec if r.get("action") == "added"))
            if n_added == 0:
                continue
            i_round = add_stage(f"9.{rnd} . deepening round bar={bar}",
                                work["sorted_parameter_array"])
            actions_round = (
                [dict(center=float(r.get("center", r.get("center_nm", float("nan")))),
                     kind="seed_minor_lines(deepen)", action="added", bar=bar)
                for r in corr if r.get("action") == "added"]
                + [dict(center=float(r.get("center", r.get("center_nm", float("nan")))),
                       kind="recover_residual_lines(deepen)", action="added", bar=bar)
                  for r in rec if r.get("action") == "added"])
            add_transition(i_prev, i_round, actions_round)
            i_prev = i_round

            idxN = PeakyIndexerV3(dbf(work["sorted_parameter_array"]), dbpath=dbpath,
                                  amp_sigma_floor=_noise_glob,
                                  temperature_init=res2.temperature, ne_init=res2.ne)
            idxN._amp_sigma = _amp_sigma(work["sorted_parameter_array"])
            idxN.build_candidate_matrix(sa_doublets=True)
            resN = idxN.solve_at(res2.temperature, res2.ne, res2.sigma, res2.gamma)
            r2_deepen_rounds.append((bar, float(resN.r_squared)))
            if composition_collapsed(res2.element_fractions, resN.element_fractions):
                r2_deepen_rounds[-1] = (bar, float(resN.r_squared), "collapsed-rejected")
                break
            final, result, fidx = work, resN, idxN
    r2_final = float(result.r_squared)

    # --- 10 . profiles + recover_sa_areas (amplitude-space only) ------------
    profiles_final = analyze_peak_profiles(x, y, final)
    try:
        result2, sa_records, sa_used = recover_sa_areas(
            fidx, result, x, y, final, profiles_final, exclude=tuple(sa_zones),
            premeasured=tuple(sa_merges))
        r2_sa = float(result2.r_squared)
    except Exception:
        sa_used, r2_sa, sa_records = False, r2_final, []
    # peaks unchanged by recover_sa_areas -- same peak table as the last
    # deepening stage (i_prev); record for the report only.
    i_final = add_stage("10 . final (post SA-area recovery; peaks unchanged)",
                        final["sorted_parameter_array"])
    add_transition(i_prev, i_final, [])  # no peak-table action; SSE identical

    meta = dict(
        r2_pass1=r2_pass1, r2_pass2=r2_pass2, r2_final=r2_final, r2_sa=r2_sa,
        r2_deepen_rounds=r2_deepen_rounds, sa_used=bool(sa_used),
        n_sa_records=len(sa_records),
        timings=dict(blind_fit_s=t_blind, pass1_s=t_pass1, pass2_s=t_pass2),
        final_peak_dict=fit["peak_dictionary"], median_fwhm_blind=median_fwhm,
        noise_local=noise_local, labels=labels, y_bgsub=y_bgsub, x=x,
        finder=finder, blind_peak_indices=list(fit.get("spectrum_dictionary", {}).keys()),
    )
    return stages, transitions, meta


# ===========================================================================
# Degeneracy stats (task 3)
# ===========================================================================
def degeneracy_stats(finder, x, y_bgsub, blind_peaks_dict, labels):
    idxs = list(blind_peaks_dict.keys())
    if not idxs:
        return dict(per_segment={}, gaussian_fraction=dict(median=float("nan"), iqr=float("nan")))
    fwhm_cap = finder._fwhm_cap(x, y_bgsub, np.array(idxs))
    rows = []
    for k in idxs:
        v = np.asarray(blind_peaks_dict[k], dtype=float)
        area, mu, sig, gam = v
        cap = fwhm_cap * (PeakyFinder.H_CAP_RELAX
                          if any(abs(float(x[int(k)]) - h) < 3.0
                                 for h in PeakyFinder.H_BALMER_NM) else 1.0)
        sigma_max = cap / 2.3548200450309493
        gamma_max = cap / 2.0
        seg = labels[int(np.clip(k, 0, len(labels) - 1))]
        at_bound = (sig >= 0.995 * sigma_max) or (gam >= 0.995 * gamma_max)
        rows.append(dict(seg=seg, sigma=sig, gamma=gam, at_bound=at_bound))
    out = {}
    for seg in ("UV", "VIS", "NIR"):
        sub = [r for r in rows if r["seg"] == seg]
        n = len(sub)
        if n == 0:
            out[seg] = dict(n=0)
            continue
        n_sig0 = sum(1 for r in sub if r["sigma"] == 0.0)
        n_gam0 = sum(1 for r in sub if r["gamma"] == 0.0)
        n_bound = sum(1 for r in sub if r["at_bound"])
        out[seg] = dict(n=n, frac_sigma0=n_sig0 / n, frac_gamma0=n_gam0 / n,
                        frac_at_bound=n_bound / n)
    gf = []
    for r in rows:
        s, g = r["sigma"], r["gamma"]
        if s + g > 0:
            gf.append(s / (s + g))
    gf = np.asarray(gf)
    gfstat = dict(median=float(np.median(gf)) if gf.size else float("nan"),
                 iqr=float(np.percentile(gf, 75) - np.percentile(gf, 25)) if gf.size else float("nan"),
                 n=int(gf.size))
    return dict(per_segment=out, gaussian_fraction=gfstat, n_total=len(idxs))


# ===========================================================================
# Attribution justification / quantification lookup (task 2)
# ===========================================================================
JUSTIFICATION = {
    "merge": ("statistical: symmetric-single beats richer models by noise-"
             "rescaled BIC (refinement.py classify_feature); no explicit "
             "cost accounting in the returned decision record."),
    "split": ("statistical + database: >=2 distinct db lines match the two "
             "fitted centers with consistent separation, BIC margin >= "
             "BIC_MARGIN (refinement.py); no cost accounting in the record."),
    "sa-tag": ("physical: resonance-capable lower level (Ei<=0.2 eV) + "
              "self-absorption growth-curve model A beats S/B by BIC "
              "(refinement.py, docs/fit_pipeline.md sec 5a); "
              "docs/fit_pipeline.md quantifies the residual-RMS cost for "
              "MW2-112 archetype lines (not for the files audited here) "
              "and states the merge trades pointwise fidelity in crowded "
              "windows."),
    "added": ("statistical: matched-filter SNR + BIC gate on the predicted "
             "(minor_lines) or residual (recover_residual_lines) line; no "
             "cost accounting in the record."),
    "deblended": ("shape-based: one-sided residual flank inconsistent with "
                 "symmetric Voigt (profiles.py); no cost accounting in the "
                 "record."),
    "filter": ("engineering safeguard: sub-pixel width snap-to-zero, "
              ">100x median-FWHM baseline-remnant drop, or height-vs-"
              "local-noise significance gate (peaky_finder.py fit_spectrum "
              "~1330-1362); no cost accounting -- amplitude is just zeroed."),
    "refit": ("no verdict/gate -- plain re-optimisation of existing "
             "components in fit_all's windowed block-coordinate refit."),
}


def justify(action_kind, action_label):
    if action_kind == "refine_fit":
        return JUSTIFICATION.get(action_label, "unclassified refine_fit action")
    if action_kind in ("seed_minor_lines", "recover_residual_lines",
                      "seed_minor_lines(deepen)", "recover_residual_lines(deepen)"):
        return JUSTIFICATION["added"]
    if action_kind == "deblend_shoulders":
        return JUSTIFICATION["deblended"]
    if action_kind == "filter":
        return JUSTIFICATION["filter"]
    if action_kind == "fit_all":
        return JUSTIFICATION["refit"]
    return "no stated justification found"


# ===========================================================================
# Driver
# ===========================================================================
def run_one(path, dbpath, db, outdir, half_window=1.0, top_n=10):
    sample = sample_name(path)
    x, y = load_spectrum_csv(path)
    t0 = time.time()
    stages, transitions, meta = audit_capture(x, y, dbpath, db)
    elapsed = time.time() - t0

    # --- stage metrics table -> CSV ---
    rows = []
    for i, st in enumerate(stages):
        m = st["metrics"]
        row = dict(sample=sample, stage_idx=i, stage_name=st["name"], n_peaks=m["n_peaks"])
        for seg in ("UV", "VIS", "NIR", "total"):
            for k in ("n_pixels", "sse", "rms", "chi2", "chi2_eff"):
                row[f"{seg}_{k}"] = m[seg][k]
        rows.append(row)
    import csv
    metrics_csv = os.path.join(outdir, f"stage_metrics_{sample}.csv")
    with open(metrics_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- window attribution table -> CSV ---
    win_rows = []
    x_arr, y_bgsub = meta["x"], meta["y_bgsub"]
    for tr in transitions:
        i_from, i_to = tr["i_from"], tr["i_to"]
        mb = stages[i_from]["metrics"]["model"]
        ma = stages[i_to]["metrics"]["model"]
        sse_before = stages[i_from]["metrics"]["total"]["sse"]
        sse_after = stages[i_to]["metrics"]["total"]["sse"]
        sse_delta_total = sse_after - sse_before
        # per-segment increase flags
        seg_incr = {seg: (stages[i_to]["metrics"][seg]["sse"]
                          - stages[i_from]["metrics"][seg]["sse"]) > 0
                   for seg in ("UV", "VIS", "NIR")}
        increased = (sse_delta_total > 0) or any(seg_incr.values())
        per_window = []
        for act in tr["actions"]:
            c = act.get("center")
            if c is None or not np.isfinite(c):
                continue
            d, n = window_delta_sse(x_arr, y_bgsub, mb, ma, c, half=half_window)
            per_window.append((d, n, act))
        per_window.sort(key=lambda t: -t[0])
        for rank, (d, n, act) in enumerate(per_window[:top_n]):
            if d <= 0:
                continue
            kind = act.get("kind")
            label = act.get("action") or act.get("verdict") or ""
            win_rows.append(dict(
                sample=sample, stage_from=stages[i_from]["name"],
                stage_to=stages[i_to]["name"], transition_sse_increased=increased,
                window_center_nm=round(act["center"], 4), n_pixels=n,
                delta_sse=d, action_kind=kind, action_label=label,
                n_components=act.get("n_components"), tau_a=act.get("tau_a"),
                bar=act.get("bar"), justification=justify(kind, label),
            ))
    if win_rows:
        win_csv = os.path.join(outdir, f"window_attribution_{sample}.csv")
        with open(win_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(win_rows[0].keys()))
            w.writeheader()
            w.writerows(win_rows)
    else:
        win_csv = None

    # --- degeneracy stats ---
    deg = degeneracy_stats(meta["finder"], meta["x"], meta["y_bgsub"],
                           meta["final_peak_dict"], meta["labels"])

    summary = dict(
        sample=sample, path=path, elapsed_s=elapsed, n_stages=len(stages),
        metrics_csv=metrics_csv, window_csv=win_csv,
        r2_pass1=meta["r2_pass1"], r2_pass2=meta["r2_pass2"],
        r2_final=meta["r2_final"], r2_sa=meta["r2_sa"],
        r2_deepen_rounds=meta["r2_deepen_rounds"],
        degeneracy=deg, timings=meta["timings"],
        stage_names=[s["name"] for s in stages],
        first_last_total_sse=(stages[0]["metrics"]["total"]["sse"],
                              stages[-1]["metrics"]["total"]["sse"]),
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data", default=os.path.join(REPO, "data", "remote_samples"))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--dbpath", default=None)
    ap.add_argument("--summary-json", default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.getcwd()
    os.makedirs(outdir, exist_ok=True)
    dbpath = resolve_dbpath(args.dbpath)
    db = Database(dbpath)

    if args.file:
        files = [args.file]
    elif args.all:
        files = sorted(os.path.join(args.data, f) for f in os.listdir(args.data)
                       if f.endswith(".csv"))
    else:
        raise SystemExit("pass --file FILE.csv or --all")

    results = []
    for f in files:
        sample = sample_name(f)
        print(f"[{sample}] starting ...", flush=True)
        try:
            summ = run_one(f, dbpath, db, outdir)
            print(f"[{sample}] done in {summ['elapsed_s']:.1f}s; "
                 f"stages={summ['n_stages']} r2(pass1/pass2/final)="
                 f"{summ['r2_pass1']:.3f}/{summ['r2_pass2']:.3f}/{summ['r2_final']:.3f}",
                 flush=True)
            results.append(summ)
        except Exception as e:
            print(f"[{sample}] FAILED: {e}", flush=True)
            traceback.print_exc()
            results.append(dict(sample=sample, path=f, error=str(e),
                                traceback=traceback.format_exc()))

    if args.summary_json:
        def _default(o):
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            return str(o)
        with open(args.summary_json, "w") as fh:
            json.dump(results, fh, indent=2, default=_default)
        print(f"summary JSON -> {args.summary_json}")

    print("\n=== SUMMARY ===")
    for r in results:
        if "error" in r:
            print(f"{r['sample']}: FAILED ({r['error']})")
            continue
        print(f"{r['sample']}: r2 pass1={r['r2_pass1']:.3f} pass2={r['r2_pass2']:.3f} "
             f"final={r['r2_final']:.3f} sa={r['r2_sa']:.3f}  "
             f"SSE[0]={r['first_last_total_sse'][0]:.4g} -> "
             f"SSE[-1]={r['first_last_total_sse'][1]:.4g}  "
             f"({r['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
