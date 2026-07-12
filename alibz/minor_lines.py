"""Prior-driven fitting of minor lines from established elements.

Once an element is established from its strong lines, every other line
of that element has a PREDICTED intensity: within one ion stage the
Boltzmann factor fixes all line ratios up to a single per-(element,
stage) scale, which is measured from the matched strong lines
themselves.  Those predictions let us

1. fit LOW-INTENSITY peaks the blind first pass rejected — the prior
   "a line of an established element belongs exactly here, this strong"
   justifies a detection threshold well below the blind 3–5 sigma;
2. DEBLEND established lines: a minor line of element A sitting under a
   strong line of element B is fitted jointly with it (center pinned at
   the database position), and the strong line's area correction is
   reported as its contamination.

The physics is deliberately same-stage only: predicting across ion
stages would drag in Saha/nₑ, which is the indexer's job.  Scales are
medians over ISOLATED matched lines, so a single self-absorbed or
blended reference cannot corrupt them; self-absorption biases scales
low, making predictions conservative.

Caveats: expected areas inherit the detector response of the segment
the reference lines live in — run after ``correct_segment_response``
(or accept cross-segment predictions as order-of-magnitude); line
ratios assume optically thin emission at a single temperature.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.special import voigt_profile as _voigt

from alibz.refinement import SA_ROBUST_F_SCALE as _SA_F_SCALE
from alibz.refinement import _fit, _weighted_median
from alibz.utils.voigt import multi_voigt as _multi_voigt
from alibz.utils.voigt import voigt_width as _voigt_width
from alibz.utils.wavelength import shift_at as _shift_at

#: Default plasma temperature for Boltzmann line ratios (corpus median
#: T = 8818 K for MW2-112).
DEFAULT_KT_EV = 0.76


def _element_lines(db, el, lo, hi):
    """(stage, wl, strength_per_kT, Ek) arrays for one element in [lo, hi].

    Returns ``None`` for elements the database cannot serve — those with
    no line list, flagged radioactively unstable, or simply unknown — so
    a caller's established-element list can be passed verbatim without
    pre-filtering.
    """
    unstable = getattr(db, "unstable_elements", ())
    if el in getattr(db, "no_lines", ()) or el in unstable \
            or el not in getattr(db, "atom_dict", {el: None}):
        return None
    arr = db.lines(el)
    if arr.size == 0:
        return None
    ion = arr[:, 0].astype(float)
    wl = arr[:, 1].astype(float)
    gA = arr[:, 3].astype(float)
    Ek = arr[:, 5].astype(float)
    keep = (ion <= 2) & (wl >= lo) & (wl <= hi) & (gA > 0)
    if not np.any(keep):
        return None
    return ion[keep], wl[keep], gA[keep], Ek[keep]


def _strength(gA, Ek, kT_ev):
    return gA * np.exp(-Ek / kT_ev)


def _segment_of(wavelength, segment_edges):
    """Detector-segment index of a wavelength (# of edges below it)."""
    return int(np.searchsorted(np.sort(np.asarray(segment_edges,
                                                   dtype=float)),
                               float(wavelength)))


def match_and_scale(peak_array, db, elements, kT_ev=DEFAULT_KT_EV,
                    shift_nm=0.0, tol_nm=0.06, x_range=None,
                    isolation_nm=0.15, min_ref_amp=0.0, segment_edges=()):
    """Per-(element, stage) area scales from clean matched lines.

    Matches each element's database lines to fitted peaks (observed =
    db + shift).  A match is a usable REFERENCE when

    - no same-element database line within ``isolation_nm`` contributes
      more than 10 % of its Boltzmann strength (contamination from the
      element's own multiplets is computable exactly);
    - no OTHER fitted peak sits within ``isolation_nm`` of the matched
      peak (cross-element blends show up as fitted neighbours; an
      unresolved one biases every method equally);
    - the matched peak's amplitude is at least ``min_ref_amp``.

    Returns::

        scales[(el, stage)] = {"scale": median(area/strength),
                               "n_ref": ..., "ref_wl": [...],
                               "spread": MAD of log-ratio}

    together with the set of (db-frame) wavelengths already matched.
    """
    peaks = np.atleast_2d(np.asarray(peak_array, dtype=float))
    if peaks.size == 0:
        return {}, set()
    if x_range is None:
        x_range = (float(np.min(peaks[:, 1])) - 1.0,
                   float(np.max(peaks[:, 1])) + 1.0)
    lo = x_range[0] - float(_shift_at(shift_nm, x_range[0]))
    hi = x_range[1] - float(_shift_at(shift_nm, x_range[1]))

    scales, matched = {}, set()
    for el in elements:
        got = _element_lines(db, el, lo, hi)
        if got is None:
            continue
        ion, wl, gA, Ek = got
        s = _strength(gA, Ek, kT_ev)
        for stage in (1.0, 2.0):
            sel = ion == stage
            if not np.any(sel):
                continue
            wl_s, s_s = wl[sel], s[sel]
            ratios, refs, ref_areas = [], [], []
            for wl_j, s_j in zip(wl_s, s_s):
                near = np.abs(wl_s - wl_j) <= isolation_nm
                if np.sum(s_s[near]) - s_j > 0.1 * s_j:
                    continue  # same-element multiplet contamination
                d = np.abs(peaks[:, 1]
                           - (wl_j + float(_shift_at(shift_nm, wl_j))))
                k = int(np.argmin(d))
                if d[k] > tol_nm or peaks[k, 0] < min_ref_amp:
                    continue
                d_other = np.abs(np.delete(peaks[:, 1], k)
                                 - float(peaks[k, 1]))
                if d_other.size and np.min(d_other) <= isolation_nm:
                    continue  # fitted neighbour: blended reference
                ratios.append(peaks[k, 0] / s_j)
                refs.append(float(wl_j))
                ref_areas.append(float(peaks[k, 0]))
                matched.add(float(wl_j))
            if not ratios:
                continue
            # trim outlier references: a coincidence with a foreign line
            # sits decades off the element's own ratio locus; it must
            # neither bias the scale nor be marked "matched" (dropping
            # it from ``matched`` frees it up as a DEBLEND candidate)
            logr = np.log(ratios)
            med = float(np.median(logr))
            mad = float(np.median(np.abs(logr - med)))
            keep_r = np.abs(logr - med) <= max(3.0 * mad, np.log(2.0))
            for r_wl, ok in zip(refs, keep_r):
                if not ok:
                    matched.discard(r_wl)
            logr = logr[keep_r]
            kept_areas = np.array(ref_areas)[keep_r]
            kept_wl = [r for r, ok in zip(refs, keep_r) if ok]
            scales[(el, int(stage))] = {
                "scale": float(np.exp(np.median(logr))),
                "n_ref": int(np.sum(keep_r)),
                "ref_wl": kept_wl,
                "spread": float(np.median(np.abs(logr - np.median(logr)))),
                "max_ref_area": float(np.max(kept_areas)),
                # detector segments the scale is actually calibrated in;
                # a per-segment response step (uncorrected 620 nm gain)
                # collapses the references onto one segment, and
                # predicting into the other would be wrong by the step
                "ref_segments": {
                    _segment_of(w + float(_shift_at(shift_nm, w)),
                                segment_edges)
                    for w in kept_wl},
            }
    return scales, matched


def seed_minor_lines(x, y, fit_dict, db, elements, kT_ev=DEFAULT_KT_EV,
                     shift_nm=0.0, tol_nm=0.06, min_expected_snr=2.0,
                     accept_snr=2.0, bic_margin=2.0, min_ref_lines=3,
                     max_scale_spread=0.6, min_ref_snr=10.0,
                     consistency_factor=5.0, extrapolation_factor=10.0,
                     segment_edges=None,
                     robust_elements=None,
                     robust_min_ref_lines=6,
                     exclude=()) -> Tuple[dict, List[dict]]:
    """Fit predicted-but-unfitted minor lines of established elements.

    ``elements`` is the established-element list (from the indexer or
    the >=2-strong-line rule); predictions are per (element, ion stage),
    anchored to that stage's clean matched lines (see
    :func:`match_and_scale`).  A stage only PREDICTS when its scale is
    trustworthy: at least ``min_ref_lines`` references (3: with 2, a
    pair of mutually-consistent COINCIDENCE matches can establish an
    absent element) whose log-ratios agree to ``max_scale_spread`` (MAD;
    0.6 = a factor 1.8, loose enough for self-absorbed references) and whose peaks clear ``min_ref_snr`` — measured
    on MW2-112, junk matches otherwise "establish" an absent element and
    inject fake components into real lines (N I inside Li I 670.8).

    Every unmatched database line whose EXPECTED peak (scale x Boltzmann
    strength, at the median instrument width) clears
    ``min_expected_snr`` x local noise is refit locally: a new component
    with its center pinned to within ``tol_nm`` of the predicted
    position, JOINTLY with every existing component in the window (their
    centers/widths confined near their current values).  The new
    component is accepted when its area is ``accept_snr`` x its
    matched-filter uncertainty AND the joint fit improves BIC by
    ``bic_margin`` over the no-new-line refit.

    Returns ``(new_fit_dict, records)``.  Each record carries element,
    stage, predicted/fitted areas, the acceptance statistics, and — for
    windows containing other components — the per-component
    ``interference`` area changes of those established lines.  Records
    with ``action == "missing"`` flag CONFIDENT predictions (expected
    SNR >= 10) that the data refused: response/temperature/blacklist
    problems show up here.

    ``robust_elements`` is a set of elements the caller has independently
    CONFIRMED present (e.g. from the indexer's whole-pattern fit); for a
    confirmed element's stage that also carries at least
    ``robust_min_ref_lines`` references the ``max_scale_spread`` trust gate
    is bypassed, so a line-rich element whose strong references are
    self-absorbed/blended (Fe) can still predict its weaker lines.  The
    reference-count floor keeps the bypass from reaching an ABSENT higher ion
    stage of a confirmed element (whose scale rests on a few coincidence
    matches).  The median scale is robust and every per-line acceptance is
    gated as usual, so this only accounts for a confirmed element where the
    data actually show a >= ``min_expected_snr`` peak — it never manufactures
    lines for an absent element or stage.

    ``exclude`` lists ``(center_nm, half_width_nm)`` zones no candidate
    may seed into — the refinement's asymmetric merges, whose symmetric
    table proxy leaves a core-shaped residual BY DESIGN.  Seeding a
    component there converts the design residual into a phantom
    satellite whose joint refit erodes the merged row's observed-area
    proxy (measured 21-93% area loss on MW2-112/K, Ca, Si archetypes
    before this gate existed).  Same convention as
    :func:`recover_residual_lines`.
    """
    from alibz.peaky_finder import PeakyFinder

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bg = np.asarray(fit_dict.get("background", np.zeros_like(y)), dtype=float)
    y_bgsub = y - bg
    peaks = np.atleast_2d(np.asarray(fit_dict["sorted_parameter_array"],
                                     dtype=float))[:, :4].copy()
    if peaks.size == 0:
        return fit_dict, []

    if segment_edges is None:
        segment_edges = PeakyFinder.DEFAULT_SEGMENT_EDGES
    segment_edges = tuple(np.atleast_1d(np.asarray(segment_edges,
                                                   dtype=float)).ravel())
    segment_indices = (np.searchsorted(x, np.sort(np.asarray(segment_edges)))
                       if len(segment_edges) and x.size else None)
    noise = PeakyFinder._noise_scale_local(y_bgsub,
                                           segment_indices=segment_indices)

    fwhm = _voigt_width(np.maximum(peaks[:, 2], 1e-6),
                        np.maximum(peaks[:, 3], 1e-6))
    wmed = _weighted_median(fwhm, np.maximum(peaks[:, 0], 1e-12))
    narrow = fwhm <= 2.0 * wmed
    sig_med = float(np.median(peaks[narrow, 2])) if np.any(narrow) else 0.05
    gam_med = float(np.median(peaks[narrow, 3])) if np.any(narrow) else 0.02
    sig_med, gam_med = max(sig_med, 1e-3), max(gam_med, 0.0)
    peak_shape0 = _voigt(0.0, sig_med, max(gam_med, 1e-9))
    pitch = float(np.median(np.abs(np.diff(x)))) if x.size > 1 else 0.01

    scales, matched = match_and_scale(
        peaks, db, elements, kT_ev=kT_ev, shift_nm=shift_nm, tol_nm=tol_nm,
        x_range=(float(x[0]), float(x[-1])), segment_edges=segment_edges,
        min_ref_amp=min_ref_snr * float(np.median(noise)) / peak_shape0)

    # ---- enumerate candidates, strongest expected first -----------------
    # elements the caller has independently CONFIRMED present (from the
    # indexer's whole-pattern fit) may bypass the scale-spread trust gate: a
    # line-rich element whose strong references are self-absorbed/blended
    # (Fe: log-ratio spread ~2.2 vs the 0.6 limit) is otherwise silenced,
    # leaving its dozens of genuine weak lines unmodeled.  Its median scale
    # is already robust (3-MAD trimmed), and every INDIVIDUAL acceptance
    # still passes the per-line Boltzmann-SNR, extrapolation, and BIC guards
    # below.  The bypass is keyed per (element, STAGE), not per element: the
    # caller confirms elements, so an ABSENT higher ion stage (Fe II when
    # only Fe I is present) reaches here too, with only a handful of faint
    # COINCIDENCE references whose wide spread is exactly what the gate is
    # for.  A genuinely-present line-rich stage instead has many references,
    # so the reference count separates them — bypass demands
    # ``robust_min_ref_lines`` references, well above the coincidence floor.
    trusted = frozenset(robust_elements or ())
    candidates = []
    for (el, stage), info in scales.items():
        if info["n_ref"] < min_ref_lines:
            continue
        trusted_stage = (el in trusted
                         and info["n_ref"] >= robust_min_ref_lines)
        if info["spread"] > max_scale_spread and not trusted_stage:
            continue
        got = _element_lines(
            db, el, float(x[0]) - float(_shift_at(shift_nm, float(x[0]))),
            float(x[-1]) - float(_shift_at(shift_nm, float(x[-1]))))
        ion, wl, gA, Ek = got
        s = _strength(gA, Ek, kT_ev)
        wl_stage, s_stage = wl[ion == stage], s[ion == stage]
        # Falsification: an element/stage that is actually present must
        # SHOW its strongest in-range line.  A spurious stage (a few
        # faint coincidences of an absent Na II) predicts its own
        # resonance lines at 5e5 counts, and the data are empty there.
        #
        # The test must key on the DATA, not on max_ref_area: the
        # strongest line is also the most self-absorption/blend-prone, so
        # a suppressed-but-present resonance line (Ca I 422.7, Sr I
        # 460.7) drops out of the clean references yet is still strong
        # evidence FOR the element.  So: locate the brightest predicted
        # line; if it would far exceed the strongest reference AND no
        # peak of at least its predicted/extrapolation_factor sits at its
        # position, the stage is falsified.
        if s_stage.size:
            j_str = int(np.argmax(s_stage))
            pred_str = info["scale"] * float(s_stage[j_str])
            if pred_str > extrapolation_factor * info["max_ref_area"]:
                mu_str = float(wl_stage[j_str])
                mu_str += float(_shift_at(shift_nm, mu_str))
                d_str = np.abs(peaks[:, 1] - mu_str)
                k_str = int(np.argmin(d_str))
                observed = (peaks[k_str, 0] if d_str[k_str] <= 2 * tol_nm
                            else 0.0)
                if observed < pred_str / extrapolation_factor:
                    continue  # strongest line predicted big, data empty
        ref_segs = info.get("ref_segments", set())
        for wl_j, s_j in zip(wl_stage, s_stage):
            if float(wl_j) in matched:
                continue
            mu_pred = float(wl_j) + float(_shift_at(shift_nm, wl_j))
            if any(abs(mu_pred - c) <= h for c, h in exclude):
                # asymmetric-merge zone: its core residual is deliberate
                continue
            # don't predict into a detector segment the scale was never
            # calibrated in — across an uncorrected 620 nm response step
            # the gain differs and the prediction is off by the step
            if (len(segment_edges) and ref_segs
                    and _segment_of(mu_pred, segment_edges) not in ref_segs):
                continue
            expected_area = info["scale"] * s_j
            # extrapolation guard: a line-ratio scale is only trustworthy
            # near where it was calibrated.  Predicting a line brighter
            # than ``extrapolation_factor`` x the strongest reference
            # means the scale's few faint anchors were coincidences of a
            # stage that is not actually present (measured: a spurious
            # Na II scale predicting 5e5-count lines that the data
            # refused, injecting a phantom into the Mg II 279/280 pair).
            if expected_area > extrapolation_factor * info["max_ref_area"]:
                continue
            d = np.abs(peaks[:, 1] - mu_pred)
            k_near = int(np.argmin(d))
            if (d[k_near] <= tol_nm
                    and peaks[k_near, 0] <= consistency_factor
                    * expected_area):
                continue  # already fitted at a compatible amplitude
            # a peak within tol that is MUCH stronger than the
            # prediction is a different line: the minor line is buried
            # under it -> deblend candidate
            k = int(np.searchsorted(x, mu_pred).clip(0, x.size - 1))
            exp_snr = expected_area * peak_shape0 / max(noise[k], 1e-12)
            if exp_snr < min_expected_snr:
                continue
            candidates.append({
                "element": el, "stage": int(stage),
                "wavelength_db": float(wl_j), "center_pred": float(mu_pred),
                "expected_area": float(expected_area),
                "expected_snr": float(exp_snr),
                "scale_refs": info["n_ref"],
            })
    candidates.sort(key=lambda c: -c["expected_area"])

    def _fit_with_priors(xw, yw, w, p0, lo, hi, prior):
        """least_squares with soft area priors.

        ``prior`` maps component slot j (area parameter p[4j]) to its
        predicted area; each contributes ((A_j - pred)/(0.5 pred))^2.
        At sub-2-pixel blend separations the flux attribution between a
        minor line and a 10-20x stronger neighbour is nearly degenerate,
        and this is exactly the direction where the Boltzmann prediction
        carries real information; where the data are informative the
        penalty is negligible against thousands of chi2 units.
        """
        from scipy.optimize import least_squares

        def resid(p):
            rows = [(yw - _multi_voigt(xw, np.asarray(p))) * w]
            for j, pred in prior.items():
                rows.append(np.atleast_1d(
                    (p[4 * j] - pred) / (0.5 * max(pred, 1e-12))))
            return np.concatenate(rows)

        try:
            res = least_squares(resid,
                                np.clip(np.asarray(p0, dtype=float), lo, hi),
                                bounds=(lo, hi), x_scale="jac", max_nfev=400)
            return res.x, float(np.sum(res.fun ** 2))
        except (ValueError, RuntimeError):
            return None, np.inf

    records = []
    added_mu = []
    pred_area = np.full(peaks.shape[0], np.nan)
    for cand in candidates:
        mu_pred = cand["center_pred"]
        if any(abs(mu_pred - m) <= max(tol_nm, 0.5 * wmed) for m in added_mu):
            cand.update(action="coincident-skip")
            records.append(cand)
            continue
        span = max(3.0 * wmed, 0.3)
        m = (x >= mu_pred - span) & (x <= mu_pred + span)
        if int(np.sum(m)) < 12:
            cand.update(action="no-data")
            records.append(cand)
            continue
        xw = x[m]
        w = 1.0 / np.maximum(noise[m], 1e-12)
        n = xw.size

        # components centred INSIDE the fit window are refit jointly;
        # ones that merely reach in with a wing are frozen — freeing a
        # parameterised profile whose core lies outside the data window
        # is ill-posed (measured: a 45-count addition at 615.60 halved
        # its 616.13 neighbour through exactly this hole).  Exclude-zone
        # rows (asymmetric merges) are frozen the same way: their
        # observed-area proxy must survive neighbouring refits intact.
        inwin = np.flatnonzero(np.abs(peaks[:, 1] - mu_pred) <= span)
        inwin = np.array([k for k in inwin
                          if not any(abs(peaks[k, 1] - c) <= hw
                                     for c, hw in exclude)], dtype=int)
        others = np.delete(np.arange(peaks.shape[0]), inwin)
        model_others = (_multi_voigt(x, np.ravel(peaks[others, :4]))
                        if others.size else np.zeros_like(x))
        yw = (y_bgsub - model_others)[m]

        # joint refit: existing window components confined near their
        # first-pass values, the new component pinned at the prediction
        p0, lo, hi = [], [], []
        for k in inwin:
            a, mu, sg, gm = peaks[k]
            p0 += [a, mu, max(sg, 1e-4), max(gm, 0.0)]
            lo += [0.0, mu - 0.5 * tol_nm, max(0.7 * sg, 1e-4), 0.7 * gm]
            hi += [4.0 * a + 1e-12, mu + 0.5 * tol_nm,
                   1.3 * sg + 1e-3, 1.3 * gm + 1e-3]

        prior = {j: float(pred_area[k]) for j, k in enumerate(inwin)
                 if np.isfinite(pred_area[k])}

        chi_base = float(np.sum(
            ((yw - _multi_voigt(xw, np.array(p0))) * w) ** 2))
        if inwin.size:
            p_re, chi_re = _fit_with_priors(xw, yw, w, p0, lo, hi, prior)
            if p_re is None:
                p_re, chi_re = np.array(p0), chi_base
        else:
            p_re, chi_re = np.array(p0), chi_base

        # near-fixed-shape template: a minor line carries the instrument
        # profile, and the first-pass medians estimate it to a few
        # percent — letting the shape float at sub-2-pixel blend
        # separations just trades width against the neighbour's area.
        # The CENTER is pinned hard: the db wavelength plus the measured
        # shift locates the line to ~20 pm, and this is flux extraction,
        # not localization — with a loose pin the component slides to
        # its bound and eats wing flux (measured: 45-area satellite
        # fitted 161 at the 60 pm edge).
        pin = min(0.5 * tol_nm, 0.02)
        a_pred = cand["expected_area"]
        p0n = list(p_re) + [a_pred, mu_pred, sig_med, gam_med]
        lon = list(lo) + [0.0, mu_pred - pin, 0.9 * sig_med,
                          0.9 * gam_med]
        hin = list(hi) + [50.0 * a_pred, mu_pred + pin,
                          1.15 * sig_med + 1e-3, 1.15 * gam_med + 1e-3]
        prior_new = dict(prior)
        prior_new[len(p0n) // 4 - 1] = a_pred
        p_new, chi_new = _fit_with_priors(xw, yw, w, p0n, lon, hin,
                                          prior_new)

        rec = dict(cand)
        if p_new is None:
            rec.update(action="fit-failed")
            records.append(rec)
            continue

        # Independent (prior-free) area: with the fitted geometry frozen,
        # the model is LINEAR in the component amplitudes, so a
        # non-negative least-squares of the window against that basis
        # recovers the area the DATA alone support.  The acceptance and
        # consistency decisions use THIS value, not the prior-regularised
        # area_fit — otherwise the soft prior manufactures the very
        # agreement the consistency gate then checks (circular).  At a
        # degenerate tight blend the prior-free area is ill-determined
        # and lands far from the prediction, so the line is (correctly)
        # refused rather than claimed on the prior alone.
        centres = np.concatenate([p_new[1:-4:4], p_new[-3:-2]]) \
            if inwin.size else p_new[-3:-2]
        sigs = np.concatenate([p_new[2:-4:4], p_new[-2:-1]]) \
            if inwin.size else p_new[-2:-1]
        gams = np.concatenate([p_new[3:-4:4], p_new[-1:]]) \
            if inwin.size else p_new[-1:]
        basis = np.stack([_voigt(xw - c, max(sg, 1e-9), max(gm, 1e-9))
                          for c, sg, gm in zip(centres, sigs, gams)], axis=1)
        try:
            from scipy.optimize import nnls
            coef, _ = nnls((basis * w[:, None]), yw * w)
            area_indep = float(coef[-1])
        except (ValueError, RuntimeError):
            area_indep = float(p_new[-4])

        area_fit = float(p_new[-4])
        # matched-filter area uncertainty: sigma_A = noise/sqrt(sum v^2)
        v = _voigt(xw - p_new[-3], max(p_new[-2], 1e-9),
                   max(p_new[-1], 1e-9))
        sigma_area = float(np.median(noise[m])
                           / max(np.sqrt(np.sum(v ** 2) * pitch), 1e-12)
                           * np.sqrt(pitch))
        d_bic = (chi_re - (chi_new + 4.0 * np.log(max(n, 2))))
        rec.update(
            area=area_fit, area_indep=area_indep, area_sigma=sigma_area,
            snr=area_indep / max(sigma_area, 1e-300), delta_bic=float(d_bic),
            center=float(p_new[-3]),
        )
        # the low acceptance threshold is JUSTIFIED by the prediction; a
        # fit contradicting its own prediction by more than
        # ``consistency_factor`` falsifies that prior (wrong scale,
        # wrong stage, response step...) and must not be trusted as a
        # prior-driven detection
        consistent = (area_indep <= consistency_factor * a_pred
                      and consistency_factor * area_indep >= a_pred)
        if not consistent:
            rec["action"] = ("missing" if cand["expected_snr"] >= 10.0
                             and area_indep < a_pred else "inconsistent")
        elif area_indep >= accept_snr * sigma_area and d_bic >= bic_margin:
            rec["action"] = "added"
            interference = []
            for j, k in enumerate(inwin):
                old = float(peaks[k, 0])
                new = float(p_new[4 * j])
                if old > 0 and abs(new - old) / old > 0.005:
                    interference.append({
                        "center": float(peaks[k, 1]),
                        "area_before": old, "area_after": new,
                        "change": (new - old) / old,
                    })
                peaks[k, :4] = p_new[4 * j:4 * j + 4]
            rec["interference"] = interference
            peaks = np.vstack([peaks, p_new[-4:]])
            fwhm = np.append(fwhm, _voigt_width(max(p_new[-2], 1e-6),
                                                max(p_new[-1], 1e-6)))
            pred_area = np.append(pred_area, a_pred)
            added_mu.append(float(p_new[-3]))
        else:
            rec["action"] = ("missing" if cand["expected_snr"] >= 10.0
                             else "rejected")
        records.append(rec)

    if added_mu:
        order = np.argsort(peaks[:, 0])[::-1]
        peaks = peaks[order]
        new_fit = dict(fit_dict)
        new_fit["sorted_parameter_array"] = peaks
        new_fit["profile"] = _multi_voigt(x, np.ravel(peaks[:, :4]))
        new_fit["residual_data"] = y_bgsub - new_fit["profile"]
        for stale in ("peak_dictionary", "spectrum_dictionary"):
            new_fit.pop(stale, None)
        return new_fit, records
    return fit_dict, records


def recover_residual_lines(x, y, fit_dict, snr_min=4.0, accept_snr=4.0,
                           bic_margin=6.0, max_new=40,
                           exclude=(),
                           supported_lines=(),
                           snr_min_supported=None,
                           accept_snr_supported=None,
                           support_tol_nm=0.06,
                           segment_edges=None) -> Tuple[dict, List[dict]]:
    """Element-agnostic recovery of significant positive residual peaks.

    The Boltzmann seeder (:func:`seed_minor_lines`) can only predict lines
    of elements whose per-stage scale passes its trust gate — a line-rich
    element like Fe on real rock routinely FAILS that gate (measured:
    Fe I log-ratio spread 2.2 over 72 references vs the 0.6 limit,
    because strong references are self-absorbed/blended), silencing every
    Fe prediction while hundreds of counts of genuinely present Fe lines
    sit unmodeled in the residual.  This pass needs no prior at all: any
    local maximum of the POSITIVE residual whose PROMINENCE above the
    window's median residual exceeds ``snr_min`` local noise is refit as
    a candidate new Voigt component, jointly with the existing components
    whose centers share its window (out-of-window components are frozen
    and subtracted).  The candidate's center floats only within +-1.5
    samples of the residual maximum and its shape is clamped near the
    instrument profile — this is recovery of a line the data already
    show, not blind detection.

    Guards (each closes a measured failure mode):

    - ``exclude`` — (center_nm, halfwidth_nm) zones where candidates are
      skipped with action ``excluded``.  The caller passes the
      asymmetric-merged self-absorbed lines from ``refine_fit``: their
      symmetric table proxy DELIBERATELY leaves a core-shaped residual
      (see refinement docs), and fitting components to those lobes
      manufactures phantom satellites while gutting the merged row
      (measured: K I 766.49 proxy area 1096 -> 58 with two phantoms
      accepted).
    - prominence, not level: the significance test subtracts the window's
      median residual, and BOTH window fits carry a free constant
      pedestal, so a broad positive background residual (junction ledge,
      wing pedestal) cannot be converted into carpets of narrow phantom
      lines (measured pre-fix: a 5-sigma ledge produced 29 accepted
      phantoms).
    - frozen confinement: existing components are confined about their
      PRE-RECOVERY values, so acceptances in overlapping windows cannot
      compound the +-0.03 nm / 0.7-1.3x limits multiplicatively.

    Acceptance is intentionally stricter than the seeder's (there is no
    physics prior standing behind the candidate): fitted area >=
    ``accept_snr`` x its matched-filter uncertainty AND the window BIC
    must improve by ``bic_margin`` over the no-new-line refit (same
    robust loss on both sides).  Accepted components are appended to the
    table; downstream identification is the indexer's job (their element
    assignments then appear in the detection report through the normal
    support counting).

    Confident-ion coverage (``supported_lines``): a bare low prominence
    bar over the whole spectrum re-admits the chance-coincidence noise
    the high bar exists to reject, but a residual bump sitting on a line
    of an ion already ANCHORED by its intense peaks is very likely a real
    (faint) line of that ion, not noise.  So a candidate within
    ``support_tol_nm`` of any ``supported_lines`` wavelength (pass the
    instrument-frame db lines of the confident ions) is judged at the
    lower ``snr_min_supported`` / ``accept_snr_supported`` bars, while
    everything else keeps the full ``snr_min`` / ``accept_snr``.  Both
    default to ``None`` = no supported set (uniform high bar, the
    original behaviour).  The iterative deepening in
    :func:`alibz.pipeline.analyze_spectrum` sweeps the supported bar down
    from ~3 sigma to the 2-sigma noise floor as more ions are quantified.

    Returns ``(new_fit_dict, records)``; records carry ``action`` in
    ``added`` / ``rejected`` / ``fit-failed`` / ``excluded`` /
    ``absorbed`` (residual at the candidate collapsed after an earlier
    acceptance in the same region) and a ``supported`` bool.
    """
    from alibz.peaky_finder import PeakyFinder

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bg = np.asarray(fit_dict.get("background", np.zeros_like(y)), dtype=float)
    y_bgsub = y - bg
    peaks = np.atleast_2d(np.asarray(fit_dict["sorted_parameter_array"],
                                     dtype=float))[:, :4].copy()
    if peaks.size == 0 or x.size < 16:
        return fit_dict, []

    if segment_edges is None:
        segment_edges = PeakyFinder.DEFAULT_SEGMENT_EDGES
    segment_edges = tuple(np.atleast_1d(np.asarray(segment_edges,
                                                   dtype=float)).ravel())
    segment_indices = (np.searchsorted(x, np.sort(np.asarray(segment_edges)))
                       if len(segment_edges) and x.size else None)
    noise = PeakyFinder._noise_scale_local(y_bgsub,
                                           segment_indices=segment_indices)
    pitch = float(np.median(np.abs(np.diff(x))))

    fwhm = _voigt_width(np.maximum(peaks[:, 2], 1e-6),
                        np.maximum(peaks[:, 3], 1e-6))
    wmed = _weighted_median(fwhm, np.maximum(peaks[:, 0], 1e-12))
    narrow = fwhm <= 2.0 * wmed
    sig_med = float(np.median(peaks[narrow, 2])) if np.any(narrow) else 0.05
    gam_med = float(np.median(peaks[narrow, 3])) if np.any(narrow) else 0.02
    sig_med, gam_med = max(sig_med, 1e-3), max(gam_med, 0.0)

    # confident-ion coverage: a sorted lookup of instrument-frame lines of
    # ions anchored by their intense peaks.  A candidate near one of these
    # is judged at the lower "supported" bar (see docstring).
    sup_wl = np.sort(np.asarray(supported_lines, dtype=float).ravel())
    lo_min = snr_min if snr_min_supported is None else float(snr_min_supported)
    lo_acc = accept_snr if accept_snr_supported is None else float(
        accept_snr_supported)

    def _supported(mu):
        if sup_wl.size == 0:
            return False
        j = int(np.searchsorted(sup_wl, mu))
        for jj in (j - 1, j):
            if 0 <= jj < sup_wl.size and abs(sup_wl[jj] - mu) <= support_tol_nm:
                return True
        return False

    def _bars(mu):
        # (prominence bar, acceptance bar) for a candidate at mu
        return (lo_min, lo_acc) if _supported(mu) else (snr_min, accept_snr)

    # candidate residual maxima on the CURRENT model, strongest first.
    # Significance is PROMINENCE above the window's median residual, not
    # the absolute level: a broad pedestal (junction ledge, wing residue)
    # raises the level of every local maximum riding it without making
    # any of them a line (measured pre-fix: a 5-sigma ledge yielded 29
    # accepted phantoms).
    finder = PeakyFinder.__new__(PeakyFinder)
    resid0 = y_bgsub - _multi_voigt(x, np.ravel(peaks))
    pos = np.where(resid0 > 0, resid0, 0.0)
    span = max(3.0 * wmed, 0.3)
    cand_idx = []
    for k in finder.find_peaks(pos)[0]:
        k = int(k)
        bar_k = _bars(float(x[k]))[0]
        if resid0[k] <= bar_k * noise[k]:
            continue
        mwin = (x >= x[k] - span) & (x <= x[k] + span)
        sub = resid0[mwin]
        # the baseline reference is the QUIET pixels: in a crowded line
        # forest the raw window median is elevated by the neighbouring
        # real lines and would veto them all, while a genuine pedestal
        # has no quiet pixels at all and correctly falls back to its own
        # level (prominence ~ 0)
        quiet = sub[sub < 2.0 * noise[k]]
        base_lvl = (float(np.median(quiet))
                    if quiet.size >= max(4, 0.25 * sub.size)
                    else float(np.median(sub)))
        prominence = resid0[k] - base_lvl
        if prominence > bar_k * noise[k]:
            cand_idx.append(k)
    cand_idx.sort(key=lambda k: -resid0[k])
    cand_idx = cand_idx[: int(max_new)]

    # confinement anchors are FROZEN at the pre-recovery table: deriving
    # bounds from already-mutated rows lets +-0.03 nm / 0.7-1.3x compound
    # across overlapping windows (measured: 0.49x width walk in two
    # acceptances)
    peaks0 = peaks.copy()

    def _robust_chi(res_w):
        # identical metric to _fit's soft_l1 2*cost: the BIC baseline
        # and the candidate fit MUST share one loss, or the plain
        # squared baseline (inflated exactly at the >8-sigma residual
        # peak under test) biases d_bic toward acceptance
        z = (np.asarray(res_w) / _SA_F_SCALE) ** 2
        return float(np.sum(_SA_F_SCALE ** 2
                            * 2.0 * (np.sqrt(1.0 + z) - 1.0)))

    def _model_off(xx, pp):
        # Voigt components plus a free constant pedestal: without it the
        # only way the window fit can explain broad background residue is
        # narrow components, which manufactures lines from ledges
        prof = _multi_voigt(xx, np.asarray(pp[:-1]))
        return prof + pp[-1]

    records: List[dict] = []
    added_mu: List[float] = []
    for k in cand_idx:
        mu0 = float(x[k])
        bar_prom, bar_acc = _bars(mu0)
        rec = dict(center0=mu0, resid0=float(resid0[k]),
                   snr0=float(resid0[k] / max(noise[k], 1e-12)),
                   supported=bool(_supported(mu0)))
        if any(abs(mu0 - c) <= hw for c, hw in exclude):
            # asymmetric-merged self-absorbed line: its symmetric table
            # proxy leaves a core-shaped residual BY DESIGN — fitting it
            # would re-split the merge (see docstring)
            rec.update(action="excluded")
            records.append(rec)
            continue
        if any(abs(mu0 - m) <= max(0.06, 0.5 * wmed) for m in added_mu):
            rec.update(action="absorbed")
            records.append(rec)
            continue
        m = (x >= mu0 - span) & (x <= mu0 + span)
        if int(np.sum(m)) < 12:
            rec.update(action="rejected", reason="no-data")
            records.append(rec)
            continue
        xw = x[m]
        w = 1.0 / np.maximum(noise[m], 1e-12)
        n = xw.size
        nmed = float(np.median(noise[m]))

        # re-check against the CURRENT model: an earlier acceptance in
        # this region may already explain the flux.  Rows inside an
        # exclude zone are dropped from the JOINT refit too (they render
        # through model_others, frozen): the zone only blocks new
        # components, and a floating merged row measurably walked +30 pm
        # and lost 40% of its area to a neighbouring window's refit.
        inwin = np.flatnonzero(np.abs(peaks[:, 1] - mu0) <= span)
        inwin = np.array([j for j in inwin
                          if not any(abs(peaks[j, 1] - c) <= hw
                                     for c, hw in exclude)], dtype=int)
        others = np.delete(np.arange(peaks.shape[0]), inwin)
        model_others = (_multi_voigt(x, np.ravel(peaks[others, :4]))
                        if others.size else np.zeros_like(x))
        yw = (y_bgsub - model_others)[m]
        r_now = yw - (_multi_voigt(xw, np.ravel(peaks[inwin, :4]))
                      if inwin.size else 0.0)
        off0 = float(np.median(r_now))
        k_loc = int(np.argmin(np.abs(xw - mu0)))
        if r_now[k_loc] - off0 < bar_prom * noise[m][k_loc]:
            rec.update(action="absorbed")
            records.append(rec)
            continue

        # joint refit of the window: current values as start, bounds
        # anchored to the FROZEN pre-recovery table
        p0, lo, hi = [], [], []
        for j in inwin:
            if j < peaks0.shape[0]:
                a0, mu_j, sg0, gm0 = peaks0[j]
            else:
                # a row this pass itself added: anchor to its own values
                a0, mu_j, sg0, gm0 = peaks[j]
            a, mu, sg, gm = peaks[j]
            p0 += [a, mu, max(sg, 1e-4), max(gm, 0.0)]
            lo += [0.0, mu_j - 0.03, max(0.7 * sg0, 1e-4), 0.7 * gm0]
            hi += [4.0 * a0 + 1e-12, mu_j + 0.03,
                   1.3 * sg0 + 1e-3, 1.3 * gm0 + 1e-3]
        off_lo = min(0.0, off0) - 3.0 * nmed
        off_hi = max(0.0, off0) + 3.0 * nmed

        p_re, chi_re = _fit(_model_off, xw, yw, w,
                            p0 + [off0], lo + [off_lo], hi + [off_hi])
        if p_re is not None:
            peaks_win = list(p_re[:-1])
        else:
            peaks_win = list(p0)
            chi_re = _robust_chi(
                (yw - _model_off(xw, np.array(p0 + [off0]))) * w)

        # add the candidate: center within +-1.5 samples of the residual
        # maximum, near-instrument shape (recovery, not blind detection)
        amp0 = max(float(r_now[k_loc]) - off0, nmed) / max(
            _voigt(0.0, sig_med, max(gam_med, 1e-9)), 1e-12)
        p0n = peaks_win + [amp0, mu0, sig_med, gam_med, off0]
        lon = lo + [0.0, mu0 - 1.5 * pitch, 0.85 * sig_med,
                    0.85 * gam_med, off_lo]
        hin = hi + [50.0 * amp0 + 1e-12, mu0 + 1.5 * pitch,
                    1.2 * sig_med + 1e-3, 1.2 * gam_med + 1e-3, off_hi]
        p_new, chi_new = _fit(_model_off, xw, yw, w, p0n, lon, hin)
        if p_new is None:
            rec.update(action="fit-failed")
            records.append(rec)
            continue

        new_a, new_mu, new_sg, new_gm, _new_off = (float(v)
                                                   for v in p_new[-5:])
        v = _voigt(xw - new_mu, max(new_sg, 1e-9), max(new_gm, 1e-9))
        sigma_area = float(nmed
                           / max(np.sqrt(np.sum(v ** 2) * pitch), 1e-12)
                           * np.sqrt(pitch))
        # both models carry the pedestal, so the new component still
        # costs exactly 4 extra parameters
        d_bic = chi_re - (chi_new + 4.0 * np.log(max(n, 2)))
        rec.update(area=new_a, area_sigma=sigma_area,
                   snr=new_a / max(sigma_area, 1e-300),
                   delta_bic=float(d_bic), center=new_mu)
        if new_a >= bar_acc * sigma_area and d_bic >= bic_margin:
            rec["action"] = "added"
            for slot, j in enumerate(inwin):
                peaks[j, :4] = p_new[4 * slot: 4 * slot + 4]
            peaks = np.vstack([peaks, [new_a, new_mu, new_sg, new_gm]])
            added_mu.append(new_mu)
        else:
            rec["action"] = "rejected"
        records.append(rec)

    if added_mu:
        order = np.argsort(peaks[:, 0])[::-1]
        peaks = peaks[order]
        new_fit = dict(fit_dict)
        new_fit["sorted_parameter_array"] = peaks
        new_fit["profile"] = _multi_voigt(x, np.ravel(peaks[:, :4]))
        new_fit["residual_data"] = y_bgsub - new_fit["profile"]
        for stale in ("peak_dictionary", "spectrum_dictionary"):
            new_fit.pop(stale, None)
        return new_fit, records
    return fit_dict, records
