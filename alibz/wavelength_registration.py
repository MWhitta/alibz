"""Independent wavelength registration from ambient and sample lines.

The Z300's pixel->nm calibration is performed on a cold instrument (near
25 C, immediately after power-up) while sample spectra are collected minutes
later, once the spectrometer has warmed toward 35-40 C.  A grating/detector
that expands with temperature moves every line by a wavelength-dependent
amount -- a scale change, not a constant offset -- which the mixed-element
residual estimator in :mod:`alibz.utils.wavelength` only captures as a single
per-segment median.  This module re-registers the wavelength axis of an
individual spectrum against lines whose air wavelengths are known exactly:

* :func:`ambient_registration` -- Ar I / O I (and optionally N I, H-alpha),
  present in almost every spectrum through the argon purge and ambient air,
  giving an *independent* calibration that does not depend on the sample
  composition.  These lines live almost entirely in the NIR segment.
* :func:`element_registration` -- the declared sample composition's strong,
  isolated I/II lines, which populate all three detector segments and let the
  registration resolve a *slope* (scale change) within a segment, not just an
  offset.
* :func:`combined_registration` -- the deployment rule: element registration
  where a segment's fit is trustworthy, ambient in the NIR (and as the
  cross-check wherever both exist), a pooled global fallback otherwise, and a
  disagreement flag when the two independent NIR estimates diverge.

Convention throughout: a *shift* is ``observed - database`` in nm (the same
convention as :func:`alibz.utils.wavelength.estimate_wavelength_shift`).  To
move an observed axis into the database frame, subtract the shift; that is what
:func:`apply_registration` does.

The two other estimators are diagnostic / provenance tools:

* :func:`vendor_calibration_shift` converts the handheld's stored pixel->nm
  cubics (``calibration.json`` base vs ``currentwlcalibration.json`` current)
  into a per-segment Delta-lambda(lambda), so the size of a recalibration can
  be compared directly with the observed line offsets.
* :func:`thermal_drift_model` fits per-segment shift versus minutes since
  calibration (and versus temperature or the ambient NIR shift as a thermal
  proxy) across many runs -- the "temperature correction".

Pure functions, numpy only.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

#: Registration format version, recorded in every output for provenance.
REGISTRATION_VERSION = "1.0"

#: Detector segment edges [nm]; the Z300 CCD is split at 365 and 620 nm.
SEGMENT_EDGES = (365.0, 620.0)
SEGMENT_NAMES = ("UV", "VIS", "NIR")

#: Ambient Ar I lines (air nm).  Exact wavelengths are always resolved from
#: ``Database.lines('Ar')``; these are search targets only.
_AR_TARGETS = (696.54, 706.72, 738.40, 750.39, 763.51, 772.38, 794.82,
               801.48, 811.53, 826.45, 842.46, 912.30)
#: Ambient O I features.  777 nm is the unresolved 777.19/777.42/777.54
#: triplet, entered as a single blended feature whose reference is the
#: gA-weighted centroid resolved from the database.
_O_SINGLETS = (844.64, 615.82)
_O_BLENDS = ((777.0, 777.7),)
#: Optional N I multiplets (each entered as a blended feature).
_N_BLENDS = ((742.20, 747.00), (818.30, 824.40))
#: H-alpha (air nm).
_H_TARGET = 656.28

_DEFAULTS = {
    # feature measurement
    "search_window_nm": 0.5,      # +-window around the prior-shifted position
    "instrument_width_px": 1.5,   # nominal line FWHM in native pixels
    "min_snr": 5.0,
    "noise_floor": 1.0,
    "side_inner_nm": 0.30,        # sideband baseline (nm from feature edge)
    "side_outer_nm": 0.90,
    "centroid_tol_nm": 0.5,       # reject a centroid this far from expected
    # matching / unambiguity
    "ambiguity_ratio": 3.0,       # 2nd candidate must be >=ratio x farther
    "prior_shift_nm": 0.0,
    # element line selection
    "kT_eV": 8.617333262e-5 * 9000.0,   # Boltzmann weight temperature (9 kK)
    "isolation_window_nm": 0.30,
    "isolation_fraction": 0.10,
    "element_min_gA": 1.0e5,
    "element_max_ion": 2,
    "element_max_lines": 400,     # cap per composition to bound runtime
    # per-window / per-segment aggregation
    "window_nm": 20.0,
    "window_min_lines": 5,
    "segment_min_lines": 5,
    "slope_min_lines": 4,         # min lines to fit a slope in a segment
    "quad_min_lines": 12,         # min lines before a quadratic is considered
    # ambient species to try
    "use_nitrogen": True,
    "use_hydrogen": True,
    # sub-pixel line-centre method: "gaussian" (instrumental-profile fit) or
    # "parabolic" (three-point interpolation).
    "subpixel": "gaussian",
    # --- vote-mode element estimator (replaces per-line matching) ----------
    # Element registration votes every (observed peak, database line) pairing
    # within +-vote_pair_window_nm into a smoothed offset histogram per sliding
    # window; the mode is the shift.  This is robust to alias locking in dense
    # forests, where per-line matching scatters ~200 pm run-to-run.
    "vote_min_snr": 8.0,          # observed-peak detection SNR
    "vote_prefilter_snr": 5.0,    # coarse pre-filter before the Gaussian fit
    "vote_pair_window_nm": 0.6,   # max |observed - database| for a pairing
    "vote_bin_nm": 0.01,          # offset histogram bin (10 pm)
    "vote_smooth_sigma_nm": 0.06, # Gaussian smoothing of the histogram
    "vote_refine_nm": 0.09,       # mode refinement half-width (strength mean)
    "vote_window_nm": 35.0,       # sliding analysis window
    "vote_window_step_nm": 17.5,  # slide by half a window
    "vote_min_pairings": 40,      # minimum pairings for a usable window
    "vote_n_db_lines": 300,       # strongest database lines per composition
    "vote_dominance_min": 1.2,    # mode/second-max ratio below which bimodal
    "vote_dom_sep_nm": 0.15,      # second maximum must be this far from mode
    "ion2_weight": 0.5,           # ion-II strength weight in the db catalog
    # --- deterministic golden-line element estimator (the primary path) ----
    "golden_top_n": 150,          # candidate pool: strongest lines per segment
    "golden_window_nm": 0.50,     # unambiguity radius
    "golden_competitor_fraction": 0.03,   # veto if a neighbour exceeds 3%
    "golden_search_nm": 0.45,     # +- search for the observed peak
    "golden_min_snr": 6.0,        # observed-peak SNR gate
    "golden_min_lines": 3,        # fewer -> segment reported "unregistered"
    "attach_vote_diagnostic": False,  # opt-in (the pipeline records it once)
}

_kB_eV = 8.617333262e-5


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _config(config: Mapping | None) -> dict:
    out = dict(_DEFAULTS)
    if config is not None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping or None")
        unknown = set(config) - set(_DEFAULTS)
        if unknown:
            raise KeyError(f"unknown config keys: {sorted(unknown)}")
        out.update(config)
    return out


def _validate_spectrum(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 5:
        raise ValueError("x and y must be equal-length 1-D arrays (>=5)")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must be finite")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    return x, y


def _segment_of(wl):
    """Segment index (0=UV,1=VIS,2=NIR) for scalar or array wl."""
    return np.digitize(np.asarray(wl, dtype=float), SEGMENT_EDGES)


def _local_pitch(x, center, span_nm=3.0):
    local = np.abs(x - center) <= span_nm
    d = np.diff(x[local])
    if d.size:
        return float(np.median(d))
    return float(np.median(np.diff(x)))


def _lines_of(db, elements, ion_max=2):
    """Sorted (wl, strength_at_9kK) for the union of ``elements``.

    ``db.lines`` returns string arrays; column 1 is the air wavelength,
    column 3 gA, column 5 E_k (eV).  Strength is the Boltzmann-weighted
    upper-level emissivity used for both isolation and dominance tests.
    """
    kT = _DEFAULTS["kT_eV"]
    wl_all, s_all = [], []
    for el in elements:
        if el not in getattr(db, "elements", (el,)):
            continue
        if el in getattr(db, "no_lines", ()):
            continue
        arr = np.asarray(db.lines(el))
        if arr.size == 0:
            continue
        ion = arr[:, 0].astype(float)
        wl = arr[:, 1].astype(float)
        gA = arr[:, 3].astype(float)
        Ek = arr[:, 5].astype(float)
        keep = (ion <= ion_max) & np.isfinite(gA) & (gA > 0) & np.isfinite(Ek)
        wl_all.append(wl[keep])
        s_all.append(gA[keep] * np.exp(-Ek[keep] / kT))
    if not wl_all:
        return np.empty(0), np.empty(0)
    wl_all = np.concatenate(wl_all)
    s_all = np.concatenate(s_all)
    order = np.argsort(wl_all)
    return wl_all[order], s_all[order]


def _species_lines(db, element):
    """(wl, strength) for one ambient species, ion stage I only where sensible."""
    return _lines_of(db, (element,), ion_max=1)


def _strong_competitors(db, elements, cfg, band_nm=100.0):
    """Wavelengths of lines strong enough to be confusable with a real peak.

    Weak database lines produce no measurable peak, so the unambiguity gate
    must only consider lines that could actually be mistaken for the target:
    those whose Boltzmann strength (at 9 kK) is at least
    ``isolation_fraction`` of the local per-(ion, band) maximum.  Using the
    full dense line list instead would reject almost every line in a forest.
    """
    kT = float(cfg["kT_eV"])
    frac = float(cfg["isolation_fraction"])
    ion_max = int(cfg["element_max_ion"])
    keep_wl = []
    for el in elements:
        if el not in getattr(db, "elements", (el,)) or el in getattr(db, "no_lines", ()):
            continue
        arr = np.asarray(db.lines(el))
        if arr.size == 0:
            continue
        ion = arr[:, 0].astype(float)
        wl = arr[:, 1].astype(float)
        gA = arr[:, 3].astype(float)
        Ek = arr[:, 5].astype(float)
        use = (ion <= ion_max) & np.isfinite(gA) & (gA > 0) & np.isfinite(Ek)
        wl, ion, s = wl[use], ion[use], gA[use] * np.exp(-Ek[use] / kT)
        if wl.size == 0:
            continue
        band = np.floor(wl / band_nm).astype(int)
        keep = np.zeros(wl.size, dtype=bool)
        for st, bd in set(zip(ion, band)):
            part = (ion == st) & (band == bd)
            if np.any(part):
                keep[part] = s[part] >= frac * float(np.max(s[part]))
        keep_wl.append(wl[keep])
    if not keep_wl:
        return np.empty(0)
    out = np.concatenate(keep_wl)
    return np.sort(out)


def _resolve_target(db_wl, db_s, target_nm, tol=0.20):
    """Nearest database line to a target; returns (wl, strength) or None."""
    if db_wl.size == 0:
        return None
    j = int(np.argmin(np.abs(db_wl - target_nm)))
    if abs(db_wl[j] - target_nm) > tol:
        return None
    return float(db_wl[j]), float(db_s[j])


def _resolve_blend(db_wl, db_s, lo, hi):
    """Strength-weighted centroid of database lines within [lo, hi]."""
    sel = (db_wl >= lo) & (db_wl <= hi)
    if not np.any(sel):
        return None
    w = db_s[sel]
    if np.sum(w) <= 0:
        w = np.ones_like(w)
    return float(np.sum(db_wl[sel] * w) / np.sum(w)), float(np.sum(w))


def _measure_feature(x, y, expected_obs, cfg, pitch=None, half_window=None):
    """Sub-pixel centroid + SNR of the local feature near ``expected_obs``.

    Linear sideband baseline (median of two flanks), robust MAD noise, and a
    parabolic sub-pixel peak at the native grid -- the same construction as
    :func:`scripts.fe_plasma_analysis.measure_line`, evaluated over a window
    sized from the local pitch so a narrow NIR line still gets real cells.
    Returns a dict or ``None`` when the window cannot be measured.
    """
    pitch = _local_pitch(x, expected_obs) if pitch is None else pitch
    # peak-search half-window: caller-controlled (tight on the second pass);
    # baseline flanks sit outside it plus one core half-width.
    win = float(cfg["search_window_nm"]) if half_window is None else float(half_window)
    core_half = max(2.0 * cfg["instrument_width_px"] * pitch, 0.12)
    inner = max(float(cfg["side_inner_nm"]), 1.5 * pitch)
    outer = inner + max(float(cfg["side_outer_nm"]) - float(cfg["side_inner_nm"]),
                        6.0 * pitch)
    core = (x >= expected_obs - core_half - win) & (x <= expected_obs + core_half + win)
    left = (x >= expected_obs - core_half - win - outer) & (x <= expected_obs - core_half - win - inner)
    right = (x >= expected_obs + core_half + win + inner) & (x <= expected_obs + core_half + win + outer)
    if np.sum(core) < 3 or np.sum(left) < 2 or np.sum(right) < 2:
        return None
    xl, xr = float(np.median(x[left])), float(np.median(x[right]))
    yl, yr = float(np.median(y[left])), float(np.median(y[right]))
    if xr == xl:
        return None
    slope = (yr - yl) / (xr - xl)
    resid = y - (yl + slope * (x - xl))
    side_res = np.concatenate([resid[left], resid[right]])
    noise = max(1.4826 * float(np.median(np.abs(side_res - np.median(side_res)))),
                float(cfg["noise_floor"]))
    xc = x[core]
    rc = resid[core]
    k = int(np.argmax(rc))
    height = float(rc[k])
    snr = height / noise
    j = int(np.searchsorted(x, xc[k]))
    j = min(max(j, 1), x.size - 2)
    sigma_fit = blend = None
    if str(cfg.get("subpixel", "gaussian")) == "gaussian":
        # Gaussian instrumental-profile fit at the peak sample; falls back to
        # parabolic if the fit fails.  Uses the raw y (not baseline-subtracted)
        # with its own local linear baseline.
        from alibz.utils.peakfit import gaussian_subpixel_center, instrument_sigma_px
        gf = gaussian_subpixel_center(
            x, y, j, sigma_px=instrument_sigma_px(float(xc[k])))
        if gf.get("ok"):
            center = gf["center_nm"]
            snr = gf["snr"]
            sigma_fit = gf["sigma_px"]
            blend = bool(gf["blend"])
        else:
            center = _parabolic_center(resid, x, j, xc[k])
    else:
        center = _parabolic_center(resid, x, j, xc[k])
    if abs(center - expected_obs) > float(cfg["centroid_tol_nm"]):
        return None
    return {"center_nm": float(center), "height": height, "snr": float(snr),
            "noise": float(noise), "pitch_nm": float(pitch),
            "sigma_px": sigma_fit, "blend": blend}


def _parabolic_center(resid, x, j, fallback):
    """Parabolic sub-pixel peak center on the native grid."""
    y1, y2, y3 = resid[j - 1], resid[j], resid[j + 1]
    denom = y1 - 2 * y2 + y3
    if denom < 0:
        return x[j] + 0.5 * (y1 - y3) / denom * 0.5 * (x[j + 1] - x[j - 1])
    return float(fallback)


def _unambiguous(center_db_frame, intended_db, competitors, ratio, window):
    """Intended db line is the closest competitor and unambiguously so.

    ``center_db_frame`` is the measured centroid mapped back to the database
    frame (observed - prior_shift).  ``competitors`` is the sorted array of all
    plausible database lines (composition + ambient) in the neighbourhood.
    Returns (ok, second_distance_nm).
    """
    if competitors.size == 0:
        return True, np.inf
    d = np.abs(competitors - center_db_frame)
    order = np.argsort(d)
    nearest = competitors[order[0]]
    if abs(nearest - intended_db) > 1e-6:
        return False, float(d[order[0]])  # closest line is not the intended one
    if order.size == 1:
        return True, np.inf
    second = float(d[order[1]])
    first = max(float(d[order[0]]), 1e-6)
    return (second >= ratio * first), second


def _robust_mode(values, weights, radius):
    """SNR-weighted mode of a set of candidate shifts.

    In a line forest a wide search grabs the wrong neighbour for many lines,
    so the raw deltas have a huge MAD; but the CORRECT matches still cluster
    tightly at the true shift while the mismatches scatter.  The mode of the
    weighted deltas (the value whose +-``radius`` neighbourhood carries the
    most weight) recovers that cluster.  Returns ``(mode, weight_in_cluster)``
    or ``(None, 0.0)``.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return None, 0.0
    # Select the cluster by COUNT: the correct matches are the plurality, while
    # mismatches scatter.  Weighting the SELECTION by SNR lets a few bright
    # wrong grabs outvote the true cluster, so count decides the cluster and
    # SNR only sets the centroid within it (ties broken by summed weight).
    best_score, best_c = (-1.0, -1.0), None
    for v in values:
        near = np.abs(values - v) <= radius
        score = (int(np.sum(near)), float(np.sum(weights[near])))
        if score > best_score:
            best_score, best_c = score, v
    near = np.abs(values - best_c) <= radius
    w = weights[near]
    center = float(np.sum(values[near] * w) / np.sum(w)) if np.sum(w) > 0 \
        else float(best_c)
    return center, float(best_score[0])


def _match_catalog(x, y, targets, competitor_wl, cfg):
    """Two-stage, unambiguity-gated matching of a target-line catalog.

    ``targets`` is a list of ``(label_dict, db_wl, kind)``.  Pass 1 measures
    the local maximum within the full +-search_window of each target and
    records a raw delta; a per-segment robust mode of those deltas gives a
    coarse prior.  Pass 2 re-measures each target within a TIGHT window around
    ``db_wl + prior`` and accepts it only when the sub-pixel centroid lands
    within ``tight_tol`` of the prior AND is unambiguous against the strong
    competitors.  This is the line-shape-aware estimator the naive
    argmax-within-tolerance approach fails to be (MAD 250-450 pm in these
    forests).  Returns ``(line_records, segment_priors)``.
    """
    win = float(cfg["search_window_nm"])
    ratio = float(cfg["ambiguity_ratio"])
    mode_radius = 0.05    # 50 pm cluster radius for the robust mode
    tight_tol = 0.12      # accept centroids within +-120 pm of the prior
    xmin, xmax = float(x[0]), float(x[-1])

    # ---- pass 1: raw candidate deltas, grouped by segment -----------------
    raw = []  # (seg, delta, snr, db_wl, label, kind, obs, pitch)
    for label, db_wl, kind in targets:
        if db_wl < xmin or db_wl > xmax:
            continue
        m = _measure_feature(x, y, db_wl + float(cfg["prior_shift_nm"]), cfg,
                             half_window=win)
        if m is None or m["snr"] < float(cfg["min_snr"]):
            continue
        seg = int(_segment_of(db_wl))
        raw.append((seg, m["center_nm"] - db_wl, m["snr"], db_wl, label, kind,
                    m["center_nm"], m["pitch_nm"]))

    seg_priors = {}
    for s in range(len(SEGMENT_NAMES)):
        deltas = [r[1] for r in raw if r[0] == s]
        snrs = [r[2] for r in raw if r[0] == s]
        mode, _ = _robust_mode(deltas, snrs, mode_radius)
        seg_priors[SEGMENT_NAMES[s]] = mode
    # global prior for segments with no pass-1 cluster
    all_d = [r[1] for r in raw]
    all_s = [r[2] for r in raw]
    global_prior, _ = _robust_mode(all_d, all_s, mode_radius)
    if global_prior is None:
        global_prior = float(cfg["prior_shift_nm"])

    # ---- pass 2: tight re-match against the per-segment prior -------------
    lines = []
    for label, db_wl, kind in targets:
        rec = dict(label)
        rec.update({"db_nm": float(db_wl),
                    "segment": SEGMENT_NAMES[int(_segment_of(db_wl))],
                    "kind": kind, "matched": False, "reasons": []})
        if db_wl < xmin or db_wl > xmax:
            rec["reasons"].append("out_of_range")
            lines.append(rec)
            continue
        prior = seg_priors.get(rec["segment"])
        if prior is None:
            prior = global_prior
        m = _measure_feature(x, y, db_wl + prior, cfg,
                             half_window=min(win, tight_tol + mode_radius))
        if m is None:
            rec["reasons"].append("no_window")
            lines.append(rec)
            continue
        delta = m["center_nm"] - db_wl
        rec.update({"observed_nm": m["center_nm"], "snr": m["snr"],
                    "shift_nm": delta, "pitch_nm": m["pitch_nm"],
                    "prior_nm": float(prior), "sigma_px": m.get("sigma_px"),
                    "blend": m.get("blend")})
        if m["snr"] < float(cfg["min_snr"]):
            rec["reasons"].append("below_snr")
            lines.append(rec)
            continue
        if abs(delta - prior) > tight_tol:
            rec["reasons"].append("off_prior")
            lines.append(rec)
            continue
        center_db = m["center_nm"] - float(cfg["prior_shift_nm"])
        near = competitor_wl[(competitor_wl >= db_wl - win - 0.15)
                             & (competitor_wl <= db_wl + win + 0.15)]
        if kind == "blend":
            near = np.sort(np.append(near[np.abs(near - db_wl) > 0.25], db_wl))
        ok, second = _unambiguous(center_db, db_wl, near, ratio, win)
        rec["second_candidate_nm"] = None if not np.isfinite(second) else float(second)
        if not ok:
            rec["reasons"].append("ambiguous")
            lines.append(rec)
            continue
        rec["matched"] = True
        lines.append(rec)
    return lines, seg_priors


def _robust_line_fit(lam, delta, cfg, allow_quad=True):
    """Fit delta(lambda) as constant / linear / quadratic, chosen by residual.

    Returns a model dict: type, coeffs (highest power last, np.polyfit order),
    ref_nm (lambda origin), rms_nm, and the per-point residuals.
    """
    lam = np.asarray(lam, dtype=float)
    delta = np.asarray(delta, dtype=float)
    n = lam.size
    ref = float(np.median(lam)) if n else 0.0
    t = lam - ref
    if n == 0:
        return {"type": "none", "coeffs": [0.0], "ref_nm": ref,
                "rms_nm": 0.0, "residuals_nm": []}
    if n < int(cfg["slope_min_lines"]):
        c = [float(np.median(delta))]
        res = delta - c[0]
        return {"type": "const", "coeffs": c, "ref_nm": ref,
                "rms_nm": float(np.sqrt(np.mean(res ** 2))),
                "residuals_nm": [float(v) for v in res]}
    candidates = [("linear", 1)]
    if allow_quad and n >= int(cfg["quad_min_lines"]):
        candidates.append(("quad", 2))
    best = None
    for name, deg in candidates:
        coeffs = np.polyfit(t, delta, deg)
        res = delta - np.polyval(coeffs, t)
        rms = float(np.sqrt(np.mean(res ** 2)))
        # penalise the extra parameter slightly (BIC-like) so a quadratic is
        # only chosen when it genuinely reduces the residual
        score = rms * (1.0 + 0.5 * deg / n)
        if best is None or score < best[0]:
            best = (score, name, coeffs, res, rms)
    _, name, coeffs, res, rms = best
    return {"type": name, "coeffs": [float(v) for v in coeffs], "ref_nm": ref,
            "rms_nm": rms, "residuals_nm": [float(v) for v in res]}


def _eval_model(model, wl):
    wl = np.asarray(wl, dtype=float)
    t = wl - float(model.get("ref_nm", 0.0))
    coeffs = np.asarray(model.get("coeffs", [0.0]), dtype=float)
    out = np.polyval(coeffs, t)
    return out


# ---------------------------------------------------------------------------
# ambient registration
# ---------------------------------------------------------------------------
def ambient_registration(x, y, db, *, config=None) -> dict:
    """Composition-independent wavelength registration from ambient lines.

    Ar I, O I (and optionally N I, H-alpha) are present in almost every Z300
    spectrum via the argon purge and ambient air.  Each feature is measured
    (baseline, SNR gate, sub-pixel centroid), matched to its exact database
    wavelength, and gated for unambiguity against every other database line of
    the declared composition plus the ambient species within the search
    window.  The surviving offsets (observed - database) are grouped by
    detector segment; a segment reports a shift, and a slope when >= 4 lines
    survive.

    Works from a ``shift_nm = 0`` prior with a +-0.5 nm search window.
    """
    x, y = _validate_spectrum(x, y)
    cfg = _config(config)

    ambient_elems = ["Ar", "O"]
    if cfg["use_nitrogen"]:
        ambient_elems.append("N")
    if cfg["use_hydrogen"]:
        ambient_elems.append("H")
    comp = list(config.get("composition", ())) if isinstance(config, Mapping) else []
    competitor_wl = _strong_competitors(db, sorted(set(ambient_elems) | set(comp)),
                                        cfg)

    # build the target list: (label, db_wl, kind)
    targets = []
    ar_wl, ar_s = _species_lines(db, "Ar")
    for t in _AR_TARGETS:
        r = _resolve_target(ar_wl, ar_s, t)
        if r:
            targets.append(({"species": "Ar"}, r[0], "line"))
    o_wl, o_s = _species_lines(db, "O")
    for t in _O_SINGLETS:
        r = _resolve_target(o_wl, o_s, t)
        if r:
            targets.append(({"species": "O"}, r[0], "line"))
    for lo, hi in _O_BLENDS:
        r = _resolve_blend(o_wl, o_s, lo, hi)
        if r:
            targets.append(({"species": "O"}, r[0], "blend"))
    if cfg["use_nitrogen"]:
        n_wl, n_s = _species_lines(db, "N")
        for lo, hi in _N_BLENDS:
            r = _resolve_blend(n_wl, n_s, lo, hi)
            if r:
                targets.append(({"species": "N"}, r[0], "blend"))
    if cfg["use_hydrogen"]:
        h_wl, h_s = _species_lines(db, "H")
        r = _resolve_target(h_wl, h_s, _H_TARGET, tol=0.10)
        if r:
            targets.append(({"species": "H"}, r[0], "line"))

    lines, _ = _match_catalog(x, y, targets, competitor_wl, cfg)
    species_seen = sorted({r.get("species") for r in lines
                           if r.get("matched") and r.get("species")})
    return _aggregate_segments(lines, cfg, source="ambient",
                               species=species_seen)


# ---------------------------------------------------------------------------
# element registration
# ---------------------------------------------------------------------------
def _isolated_strong_lines(db, composition, ambient, cfg):
    """Strong, locally isolated I/II lines of the composition.

    Strength = gA * exp(-E_k / kT9000).  A line is *isolated* when no other
    line (composition + ambient) within +- isolation_window is above
    isolation_fraction of the line's own strength.
    """
    comp_wl, comp_s = _lines_of(db, composition, ion_max=int(cfg["element_max_ion"]))
    if comp_wl.size == 0:
        return np.empty(0), np.empty(0)
    # apply gA floor via strength proxy: keep only lines whose gA passes.
    # Recompute gA-only strength cheaply by re-reading; simpler: gate on the
    # Boltzmann strength being above a small fraction of the local max later.
    all_wl, all_s = _lines_of(db, sorted(set(composition) | set(ambient)),
                              ion_max=int(cfg["element_max_ion"]))
    iso_win = float(cfg["isolation_window_nm"])
    frac = float(cfg["isolation_fraction"])
    keep_wl, keep_s = [], []
    for wl, s in zip(comp_wl, comp_s):
        if s <= 0:
            continue
        lo = np.searchsorted(all_wl, wl - iso_win)
        hi = np.searchsorted(all_wl, wl + iso_win)
        neigh = np.concatenate([all_s[lo:hi]])
        # exclude the line itself (closest match in strength & position)
        others = all_s[lo:hi][np.abs(all_wl[lo:hi] - wl) > 1e-4]
        if others.size and np.max(others) >= frac * s:
            continue
        keep_wl.append(float(wl))
        keep_s.append(float(s))
    keep_wl = np.asarray(keep_wl)
    keep_s = np.asarray(keep_s)
    if keep_wl.size == 0:
        return keep_wl, keep_s
    # rank by strength, cap count
    order = np.argsort(keep_s)[::-1][: int(cfg["element_max_lines"])]
    order = order[np.argsort(keep_wl[order])]
    return keep_wl[order], keep_s[order]


def _detect_peaks(x, y, cfg):
    """Observed peak centres (sub-pixel) with SNR >= ``vote_min_snr``.

    A coarse rolling-baseline SNR pre-filter selects local maxima, each then
    refined by the configured sub-pixel method (Gaussian by default); the peak
    is kept when its own local-baseline fit clears ``vote_min_snr``.
    """
    from scipy.ndimage import percentile_filter
    from alibz.utils.peakfit import gaussian_subpixel_center, instrument_sigma_px
    n = x.size
    size = max(15, int(round(2.0 / float(np.median(np.diff(x))))))  # ~2 nm
    base = percentile_filter(y, 25, size=size)
    resid = y - base
    noise = max(1.4826 * float(np.median(np.abs(resid - np.median(resid)))), 1e-9)
    ismax = np.zeros(n, dtype=bool)
    ismax[1:-1] = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
    cand = np.where(ismax & (resid >= float(cfg["vote_prefilter_snr"]) * noise))[0]
    use_gauss = str(cfg.get("subpixel", "gaussian")) == "gaussian"
    centers, snrs = [], []
    for i in cand:
        if use_gauss:
            r = gaussian_subpixel_center(x, y, int(i),
                                         sigma_px=instrument_sigma_px(float(x[i])))
            if r.get("ok") and r["snr"] >= float(cfg["vote_min_snr"]):
                centers.append(r["center_nm"]); snrs.append(r["snr"])
        else:
            j = min(max(int(i), 1), n - 2)
            c = _parabolic_center(resid, x, j, x[j])
            snr = resid[j] / noise
            if snr >= float(cfg["vote_min_snr"]):
                centers.append(float(c)); snrs.append(float(snr))
    return np.asarray(centers), np.asarray(snrs)


def _db_strong_lines(db, composition, cfg):
    """(wl, strength) of the strongest composition lines; ion II down-weighted."""
    kT = float(cfg["kT_eV"])
    w2 = float(cfg["ion2_weight"])
    wl_all, s_all = [], []
    for el in composition:
        if el not in getattr(db, "elements", (el,)) or el in getattr(db, "no_lines", ()):
            continue
        arr = np.asarray(db.lines(el))
        if arr.size == 0:
            continue
        ion = arr[:, 0].astype(float)
        wl = arr[:, 1].astype(float)
        gA = arr[:, 3].astype(float)
        Ek = arr[:, 5].astype(float)
        keep = (ion <= 2) & np.isfinite(gA) & (gA > 0) & np.isfinite(Ek) & (wl > 180) & (wl < 1000)
        s = gA[keep] * np.exp(-Ek[keep] / kT)
        s = s * np.where(ion[keep] >= 2, w2, 1.0)
        wl_all.append(wl[keep]); s_all.append(s)
    if not wl_all:
        return np.empty(0), np.empty(0)
    wl_all = np.concatenate(wl_all); s_all = np.concatenate(s_all)
    top = np.argsort(s_all)[::-1][: int(cfg["vote_n_db_lines"])]
    wl_all, s_all = wl_all[top], s_all[top]
    order = np.argsort(wl_all)
    return wl_all[order], s_all[order]


# ---------------------------------------------------------------------------
# deterministic golden-line element registration
# ---------------------------------------------------------------------------
_SEGMENT_RANGES = {"UV": (200.0, 365.0), "VIS": (365.0, 620.0), "NIR": (620.0, 1000.0)}


def _predicted_strength_table(db, elements, ion_max=2):
    """(wl, strength, element) for all I/II lines; ion II weighted 0.5."""
    kT = _DEFAULTS["kT_eV"]
    w2 = _DEFAULTS["ion2_weight"]
    wl_all, s_all, el_all = [], [], []
    for el in elements:
        if el not in getattr(db, "elements", (el,)) or el in getattr(db, "no_lines", ()):
            continue
        arr = np.asarray(db.lines(el))
        if arr.size == 0:
            continue
        ion = arr[:, 0].astype(float)
        wl = arr[:, 1].astype(float)
        gA = arr[:, 3].astype(float)
        Ek = arr[:, 5].astype(float)
        keep = (ion <= ion_max) & np.isfinite(gA) & (gA > 0) & np.isfinite(Ek)
        s = gA[keep] * np.exp(-Ek[keep] / kT) * np.where(ion[keep] >= 2, w2, 1.0)
        wl_all.append(wl[keep]); s_all.append(s)
        el_all.append(np.full(int(np.sum(keep)), el))
    if not wl_all:
        return np.empty(0), np.empty(0), np.empty(0, dtype=object)
    return (np.concatenate(wl_all), np.concatenate(s_all),
            np.concatenate(el_all))


def golden_lines(db, composition, segment, *, config=None):
    """Lines unambiguous by construction in one detector segment.

    A golden line is a sample-element (composition) I/II line whose predicted
    strength (gA*exp(-E_k/9000 K), ion II x0.5) is in the top ~150 of the
    segment AND has NO other line of the composition + Ar/N/O/H within
    +-0.50 nm whose predicted strength exceeds 3 % of the candidate's
    (same-element lines counted).  The Ar I NIR set is added as NIR golden
    lines.  Returns a list of ``{wavelength_nm, strength, species,
    n_competitors, competitors}`` sorted by wavelength.
    """
    cfg = _config(config)
    composition = [str(e) for e in composition]
    lo, hi = _SEGMENT_RANGES[segment]
    top_n = int(cfg["golden_top_n"])
    win = float(cfg["golden_window_nm"])
    frac = float(cfg["golden_competitor_fraction"])

    cw, cs, ce = _predicted_strength_table(db, composition)
    comp_elems = sorted(set(composition) | {"Ar", "N", "O", "H"})
    aw, as_, ae = _predicted_strength_table(db, comp_elems)

    golden = []
    if cw.size:
        seg = (cw >= lo) & (cw < hi)
        sw, ss, se = cw[seg], cs[seg], ce[seg]
        if sw.size:
            top = np.argsort(ss)[::-1][:top_n]
            for i in top:
                wl0, s0 = float(sw[i]), float(ss[i])
                near = (np.abs(aw - wl0) <= win) & (np.abs(aw - wl0) > 1e-4)
                jj = np.where(near & (as_ > frac * s0))[0]
                if jj.size == 0:
                    golden.append({"wavelength_nm": wl0, "strength": s0,
                                   "species": str(se[i]), "n_competitors": 0,
                                   "competitors": []})
    if segment == "NIR":
        ar_wl, ar_s = _species_lines(db, "Ar")
        for t in _AR_TARGETS:
            r = _resolve_target(ar_wl, ar_s, t)
            if r is None or r[0] < lo or r[0] >= hi:
                continue
            wl0, s0 = r[0], max(r[1], 1e-9)
            near = (np.abs(aw - wl0) <= win) & (np.abs(aw - wl0) > 1e-4)
            jj = np.where(near & (as_ > frac * s0))[0]
            if jj.size == 0 and not any(abs(g["wavelength_nm"] - wl0) < 0.06
                                        for g in golden):
                golden.append({"wavelength_nm": float(wl0), "strength": float(s0),
                               "species": "Ar", "n_competitors": 0,
                               "competitors": []})
    golden.sort(key=lambda g: g["wavelength_nm"])
    return golden


def _huber_polyfit(t, yv, deg):
    """Robust (Huber) polynomial fit; returns (coeffs high->low, residuals)."""
    from scipy.optimize import least_squares
    t = np.asarray(t, float); yv = np.asarray(yv, float)
    A = np.vander(t, deg + 1)
    p0 = np.linalg.lstsq(A, yv, rcond=None)[0]
    r0 = A @ p0 - yv
    scale = max(1.4826 * float(np.median(np.abs(r0 - np.median(r0)))), 0.01)
    sol = least_squares(lambda p: A @ p - yv, p0, loss="huber", f_scale=scale)
    return sol.x, (A @ sol.x - yv)


def _measure_golden(x, y, resid, noise, db_wl, cfg):
    """Measure one golden line, enforcing OBSERVED isolation.

    Golden lines are unambiguous by database-predicted strength, but the
    observed intensities do not match predictions, so a wide ``argmax`` grabs
    the wrong neighbour.  Require EXACTLY ONE local maximum clearing
    ``golden_min_snr`` within +-``golden_search_nm``; anything else (0, or a
    genuine second observed peak) rejects the line.  The single peak is then
    Gaussian-fit and its blend flag re-checked.
    """
    from alibz.utils.peakfit import gaussian_subpixel_center, instrument_sigma_px
    win = float(cfg["golden_search_nm"])
    lo = int(np.searchsorted(x, db_wl - win))
    hi = int(np.searchsorted(x, db_wl + win))
    if hi - lo < 3:
        return None
    thr = float(cfg["golden_min_snr"]) * noise
    maxima = [j for j in range(max(lo, 1), min(hi, x.size - 1))
              if resid[j] > resid[j - 1] and resid[j] >= resid[j + 1]
              and resid[j] >= thr]
    if not maxima:
        return None
    # nearest clearing peak to the database position (not the brightest -- a
    # distant bright forest neighbour must not win)
    i = min(maxima, key=lambda j: abs(x[j] - db_wl))
    g = gaussian_subpixel_center(x, y, i, sigma_px=instrument_sigma_px(db_wl))
    if not g.get("ok") or g["snr"] < float(cfg["golden_min_snr"]) or g["blend"]:
        return None
    if abs(g["center_nm"] - db_wl) > win:
        return None
    return g


def element_registration(x, y, db, composition, *, config=None) -> dict:
    """Deterministic golden-line wavelength registration.

    For each detector segment, measure the golden lines (unambiguous by
    construction; see :func:`golden_lines`) with a Gaussian instrumental-profile
    fit and robustly fit Delta-lambda(lambda) = a + b(lambda - lambda0) (Huber
    weights; quadratic only with >= 8 lines and a lower residual).  No forest
    votes enter the fit.  A segment with < 3 golden lines is reported
    ``unregistered`` (the combined registration falls back there, never to a
    vote mode).  The vote-mode estimator is attached under ``diagnostic`` only.
    """
    x, y = _validate_spectrum(x, y)
    cfg = _config(config)
    composition = [str(e) for e in composition]
    from scipy.ndimage import percentile_filter
    size = max(15, int(round(2.0 / float(np.median(np.diff(x))))))
    resid = y - percentile_filter(y, 25, size=size)

    segments = {}
    all_shifts = []
    for name, (lo0, hi0) in _SEGMENT_RANGES.items():
        lo, hi = max(lo0, float(x[0])), min(hi0, float(x[-1]))
        seg_mask = (x >= lo) & (x < hi)
        noise = max(1.4826 * float(np.median(np.abs(
            resid[seg_mask] - np.median(resid[seg_mask])))), 1e-9) \
            if np.any(seg_mask) else 1.0
        gl = [g for g in golden_lines(db, composition, name, config=cfg)
              if lo <= g["wavelength_nm"] <= hi]
        lines = []
        for g in gl:
            m = _measure_golden(x, y, resid, noise, g["wavelength_nm"], cfg)
            rec = {"db_nm": g["wavelength_nm"], "species": g["species"],
                   "matched": False, "reasons": []}
            if m is None:
                rec["reasons"].append("no_clean_peak")
            else:
                rec.update({"observed_nm": m["center_nm"], "snr": m["snr"],
                            "shift_nm": m["center_nm"] - g["wavelength_nm"],
                            "sigma_px": m["sigma_px"], "matched": True})
            lines.append(rec)
        matched = [r for r in lines if r["matched"]]
        # light outlier rejection: drop only gross wrong-peak grabs (> 3 MAD
        # from the segment median); the Huber fit handles the rest softly
        if len(matched) >= 5:
            sh = np.array([r["shift_nm"] for r in matched])
            med = float(np.median(sh))
            mad0 = 1.4826 * float(np.median(np.abs(sh - med)))
            if mad0 > 0:
                for r in matched:
                    if abs(r["shift_nm"] - med) > 3.0 * mad0:
                        r["matched"] = False
                        r["reasons"].append("outlier_shift")
                matched = [r for r in matched if r["matched"]]
        if len(matched) < int(cfg["golden_min_lines"]):
            segments[name] = {"shift_nm": None, "mad_nm": None, "sigma_nm": None,
                              "slope_nm_per_nm": None, "n_lines": len(matched),
                              "n_golden": len(gl), "model": None, "windows": [],
                              "quality": "unregistered", "residual_mad_nm": None,
                              "lines": lines}
            continue
        lam = np.array([r["db_nm"] for r in matched])
        dl = np.array([r["shift_nm"] for r in matched])
        lam0 = float(np.median(lam)); t = lam - lam0
        span = float(np.ptp(lam))
        # Robust OFFSET (Huber intercept at the golden-line centroid) is the
        # DEPLOYED correction: the Fe->V transfer test shows the within-segment
        # SLOPE does NOT transfer between elements (Fe VIS -5.8 vs V VIS +1.5
        # pm/nm on the identical axis), i.e. the slope is dominated by per-
        # element database wavelength errors, not a shared instrument scale.
        # So a is applied (a constant per segment); the fitted slope b is kept
        # as a DIAGNOSTIC only and never applied.
        lin, res = _huber_polyfit(t, dl, 1) if len(matched) >= 4 \
            else (np.array([float(np.median(dl))]), dl - float(np.median(dl)))
        a = float(np.polyval(lin, 0.0)) if lin.size > 1 else float(lin[0])
        slope_diag = float(lin[0]) if lin.size > 1 else 0.0
        res = dl - a           # residual about the deployed constant offset
        mad = 1.4826 * float(np.median(np.abs(res - np.median(res))))
        model = {"type": "const", "coeffs": [a], "ref_nm": lam0,
                 "rms_nm": float(np.sqrt(np.mean(res ** 2))),
                 "residuals_nm": [float(v) for v in res],
                 "lambda_span_nm": span,
                 "slope_diagnostic_nm_per_nm": slope_diag,
                 "slope_applied": False}
        for r, rr in zip(matched, res):
            r["residual_nm"] = float(rr)
        shift_seg = a
        slope = slope_diag
        se = 1.2533 * mad / np.sqrt(len(matched))
        segments[name] = {"shift_nm": shift_seg, "mad_nm": mad, "sigma_nm": float(se),
                          "slope_nm_per_nm": slope, "n_lines": len(matched),
                          "n_golden": len(gl), "model": model, "windows": [],
                          "quality": "ok", "residual_mad_nm": mad, "lines": lines}
        all_shifts.append(shift_seg)

    gmed = float(np.median(all_shifts)) if all_shifts else 0.0
    overall = "ok" if all_shifts else "unregistered"
    result = {
        "version": REGISTRATION_VERSION, "source": "element_golden",
        "convention": "observed_minus_database_nm",
        "prior_shift_nm": float(cfg["prior_shift_nm"]),
        "species": list(composition), "species_detected": list(composition),
        "n_lines": int(sum(s["n_lines"] for s in segments.values())),
        "global_shift_nm": gmed, "global_mad_nm": None,
        "segments": segments, "residual_histogram": None, "outliers": [],
        "lines": [], "quality": overall, "estimator": "golden_line",
    }
    if cfg.get("attach_vote_diagnostic", True):
        try:
            result["diagnostic"] = {"vote_mode": vote_mode_diagnostic(
                x, y, db, composition, config=cfg)}
        except Exception:
            result["diagnostic"] = None
    return result


def _vote_mode(offsets, weights, cfg):
    """Smoothed-histogram mode of pairing offsets (the window shift).

    Returns shift, its sigma, the pairing counts (total and in the mode), the
    dominance ratio (mode height / second-highest local maximum >= dom_sep
    away), and the second maximum's location for bimodal resolution.
    """
    half = float(cfg["vote_pair_window_nm"])
    binw = float(cfg["vote_bin_nm"])
    offsets = np.asarray(offsets, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if offsets.size == 0:
        return None
    edges = np.arange(-half, half + binw, binw)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist, _ = np.histogram(offsets, bins=edges, weights=weights)
    # Gaussian smoothing
    sig_bins = float(cfg["vote_smooth_sigma_nm"]) / binw
    half_k = int(np.ceil(3 * sig_bins))
    kx = np.arange(-half_k, half_k + 1)
    kernel = np.exp(-0.5 * (kx / sig_bins) ** 2)
    kernel /= kernel.sum()
    sm = np.convolve(hist, kernel, mode="same")
    mode_i = int(np.argmax(sm))
    mode = float(centers[mode_i])
    height = float(sm[mode_i])
    # refine: strength-weighted mean of offsets within +- refine of the mode
    refine = float(cfg["vote_refine_nm"])
    near = np.abs(offsets - mode) <= refine
    wn = weights[near]
    if wn.sum() > 0:
        shift = float(np.sum(offsets[near] * wn) / wn.sum())
        var = float(np.sum(wn * (offsets[near] - shift) ** 2) / wn.sum())
        neff = float(wn.sum() ** 2 / np.sum(wn ** 2)) if np.sum(wn ** 2) > 0 else 1.0
        sigma = float(np.sqrt(max(var, 0.0) / max(neff, 1.0)))
    else:
        shift, sigma = mode, float("nan")
    # dominance vs the second-highest local maximum >= dom_sep away
    far = np.abs(centers - mode) >= float(cfg["vote_dom_sep_nm"])
    if np.any(far):
        second_i = int(np.where(far)[0][np.argmax(sm[far])])
        second_height = float(sm[second_i])
        second_shift = float(centers[second_i])
    else:
        second_height, second_shift = 0.0, float("nan")
    dominance = height / second_height if second_height > 0 else float("inf")
    return {"shift_nm": shift, "sigma_nm": sigma, "mode_nm": mode,
            "n_pairings": int(offsets.size), "n_pairings_mode": int(np.sum(near)),
            "dominance": dominance, "second_shift_nm": second_shift}


def _pairings(obs, snr, dbwl, log_s, pair):
    """(offset, weight) pairings within +-pair; weight = db log-strength x
    log10(observed SNR).  The SNR factor suppresses the spurious pairings that
    make a dense forest's raw offset histogram nearly flat: a strong database
    line paired with a strong observed peak (likely the true match) outweighs
    the same line paired with weak neighbours."""
    offsets, weights = [], []
    lsnr = np.log10(np.maximum(snr, 1.0) + 1.0)
    for c, w in zip(obs, lsnr):
        near = np.abs(dbwl - c) <= pair
        if np.any(near):
            offsets.append(c - dbwl[near])
            weights.append(log_s[near] * w)
    if not offsets:
        return np.empty(0), np.empty(0)
    return np.concatenate(offsets), np.concatenate(weights)


def _vote_windows(peaks_obs, snrs, dbwl, dbs, seg_lo, seg_hi, cfg):
    """Per-window vote modes across one segment's covered range."""
    win = float(cfg["vote_window_nm"])
    step = float(cfg["vote_window_step_nm"])
    pair = float(cfg["vote_pair_window_nm"])
    log_s = np.log10(np.maximum(dbs, 1.0) + 1.0)
    out = []
    if peaks_obs.size == 0 or dbwl.size == 0 or seg_hi - seg_lo < 5:
        return out
    starts = np.arange(seg_lo, max(seg_lo + step, seg_hi - win) + step, step)
    for w_lo in starts:
        w_hi = min(w_lo + win, seg_hi)
        if w_hi - w_lo < 0.5 * win:
            continue
        om = (peaks_obs >= w_lo) & (peaks_obs < w_hi)
        dsel = (dbwl >= w_lo - pair) & (dbwl <= w_hi + pair)
        if not np.any(om) or not np.any(dsel):
            continue
        offsets, weights = _pairings(peaks_obs[om], snrs[om], dbwl[dsel],
                                     log_s[dsel], pair)
        if offsets.size < int(cfg["vote_min_pairings"]):
            continue
        vm = _vote_mode(offsets, weights, cfg)
        if vm is None:
            continue
        vm["center_nm"] = float(0.5 * (w_lo + w_hi))
        vm["lo_nm"] = float(w_lo); vm["hi_nm"] = float(w_hi)
        vm["bimodal"] = bool(vm["dominance"] < float(cfg["vote_dominance_min"]))
        out.append(vm)
    return out


def _segment_vote(peaks_obs, snrs, dbwl, dbs, seg_lo, seg_hi, cfg):
    """Robust segment-level vote over ALL pairings (the stable anchor shift)."""
    pair = float(cfg["vote_pair_window_nm"])
    log_s = np.log10(np.maximum(dbs, 1.0) + 1.0)
    om = (peaks_obs >= seg_lo) & (peaks_obs < seg_hi)
    dsel = (dbwl >= seg_lo - pair) & (dbwl <= seg_hi + pair)
    if not np.any(om) or not np.any(dsel):
        return None
    offsets, weights = _pairings(peaks_obs[om], snrs[om], dbwl[dsel],
                                 log_s[dsel], pair)
    if offsets.size < int(cfg["vote_min_pairings"]):
        return None
    return _vote_mode(offsets, weights, cfg)


def _fit_windows(windows, cfg):
    """Weighted per-segment polynomial fit of window modes, with continuity.

    Bimodal windows are re-pointed to their second mode when that lies closer
    to the trend of the confident windows.  Returns (model, shift_at_mid,
    residual_mad, slope, resolved_windows) or None.
    """
    if len(windows) < 1:
        return None
    lam = np.array([w["center_nm"] for w in windows])
    prim = np.array([w["shift_nm"] for w in windows])
    sec = np.array([w["second_shift_nm"] for w in windows])
    dom = np.array([w["dominance"] if np.isfinite(w["dominance"]) else 5.0
                    for w in windows])
    npair = np.array([w["n_pairings_mode"] for w in windows], dtype=float)
    weights = np.clip(dom, 0.0, 5.0) * np.log10(npair + 10.0)
    ref = float(np.median(lam))

    # Segment-level cluster selection FIRST: in a dense segment (UV) the window
    # modes split between competing alias clusters, and fitting a line through
    # both gives a meaningless, run-unstable midpoint.  Pick the dominant
    # cluster (weighted mode of the window shifts), re-point each window to
    # whichever of its two modes falls in that cluster, and drop windows with
    # neither.  This locks the segment onto the consistent offset.
    radius = 0.08
    cluster, _ = _robust_mode(prim, weights, radius)
    if cluster is None:
        return None
    chosen = prim.copy()
    keep = np.ones(lam.size, dtype=bool)
    for i in range(lam.size):
        dp = abs(prim[i] - cluster)
        dsec = abs(sec[i] - cluster) if np.isfinite(sec[i]) else np.inf
        if dp <= 1.5 * radius:
            chosen[i] = prim[i]
        elif dsec <= 1.5 * radius:
            chosen[i] = sec[i]
        else:
            keep[i] = False
    if np.sum(keep) < 1:
        return None
    lam, chosen, weights, dom, npair = (lam[keep], chosen[keep], weights[keep],
                                        dom[keep], npair[keep])
    confident = dom >= float(cfg["vote_dominance_min"])
    # neighbour continuity: refine any residual outliers toward the local trend
    if np.sum(confident) >= 2 and np.ptp(lam) > 0:
        c0 = np.polyfit(lam - ref, chosen, 1, w=weights)
        for i in range(lam.size):
            pred = np.polyval(c0, lam[i] - ref)
            if abs(chosen[i] - pred) > 0.10:
                weights[i] *= 0.25   # down-weight a window off the trend
    # final fit: linear, quadratic if it lowers the weighted residual
    def wfit(deg):
        if lam.size <= deg or np.ptp(lam) == 0:
            return None
        c = np.polyfit(lam - ref, chosen, deg, w=weights)
        r = chosen - np.polyval(c, lam - ref)
        rms = float(np.sqrt(np.sum(weights * r ** 2) / np.sum(weights)))
        return c, r, rms
    lin = wfit(1) or (np.array([float(np.average(chosen, weights=weights))]),
                      chosen - float(np.average(chosen, weights=weights)), 0.0)
    best_c, best_r, best_rms, best_type = lin[0], lin[1], lin[2], (
        "linear" if len(lin[0]) == 2 else "const")
    q = wfit(2) if lam.size >= 4 else None
    if q and q[2] < 0.8 * best_rms:
        best_c, best_r, best_rms, best_type = q[0], q[1], q[2], "quad"
    model = {"type": best_type, "coeffs": [float(v) for v in best_c],
             "ref_nm": ref, "rms_nm": best_rms,
             "residuals_nm": [float(v) for v in best_r]}
    mad = 1.4826 * float(np.median(np.abs(best_r - np.median(best_r)))) if best_r.size else 0.0
    slope = None
    if best_type in ("linear", "quad"):
        slope = float(np.polyder(np.asarray(best_c, dtype=float))[-1])
    return model, float(np.median(chosen)), mad, slope, int(np.sum(confident))


def vote_mode_diagnostic(x, y, db, composition, *, config=None) -> dict:
    """DIAGNOSTIC ONLY: vote-mode piecewise estimator (not used for the fit).

    Retained to report the offset-histogram mode and its dominance per segment.
    It is NOT the element registration: a joint offset+slope fit by vote score
    gives OPPOSITE within-segment slopes for Fe and V on the identical axis
    (Fe VIS +1.86, V VIS -1.49 pm/nm, dominance <= 1.23), i.e. statistical
    forest matching measures each element's ALIAS pattern, not the instrument.
    :func:`element_registration` uses deterministic golden lines instead.
    """
    x, y = _validate_spectrum(x, y)
    cfg = _config(config)
    composition = [str(e) for e in composition]
    peaks_obs, snrs = _detect_peaks(x, y, cfg)
    dbwl, dbs = _db_strong_lines(db, composition, cfg)

    seg_bounds = [(200.0, SEGMENT_EDGES[0]), (SEGMENT_EDGES[0], SEGMENT_EDGES[1]),
                  (SEGMENT_EDGES[1], 1000.0)]
    anchor_nm = 0.15   # keep window modes within this of the segment anchor
    segments = {}
    all_shifts = []
    for name, (lo0, hi0) in zip(SEGMENT_NAMES, seg_bounds):
        lo = max(lo0, float(x[0])); hi = min(hi0, float(x[-1]))
        sv = _segment_vote(peaks_obs, snrs, dbwl, dbs, lo, hi, cfg)
        if sv is None:
            segments[name] = {"shift_nm": None, "mad_nm": None, "sigma_nm": None,
                              "slope_nm_per_nm": None, "n_lines": 0, "model": None,
                              "windows": [], "quality": "none",
                              "residual_mad_nm": None, "n_windows": 0,
                              "dominance": None}
            continue
        anchor = sv["shift_nm"]
        wins = _vote_windows(peaks_obs, snrs, dbwl, dbs, lo, hi, cfg)
        # anchor each window to the segment vote (primary or 2nd mode, whichever
        # is within anchor_nm); drop windows anchored to neither
        aw = []
        for w in wins:
            cand = [w["shift_nm"], w["second_shift_nm"]]
            best = min((c for c in cand if np.isfinite(c)),
                       key=lambda c: abs(c - anchor), default=None)
            if best is not None and abs(best - anchor) <= anchor_nm:
                w2 = dict(w); w2["shift_nm"] = best
                aw.append(w2)
        # slope from anchored windows if enough; else constant at the anchor
        model, slope, mad = None, None, None
        if len(aw) >= 3 and np.ptp([w["center_nm"] for w in aw]) > 0:
            lamw = np.array([w["center_nm"] for w in aw])
            shw = np.array([w["shift_nm"] for w in aw])
            ww = np.array([max(w["dominance"], 0.1) if np.isfinite(w["dominance"])
                           else 0.1 for w in aw]) * np.log10(
                np.array([w["n_pairings_mode"] for w in aw], float) + 10.0)
            ref = float(np.median(lamw))
            c = np.polyfit(lamw - ref, shw, 1, w=ww)
            r = shw - np.polyval(c, lamw - ref)
            mad = 1.4826 * float(np.median(np.abs(r - np.median(r))))
            model = {"type": "linear", "coeffs": [float(v) for v in c],
                     "ref_nm": ref, "rms_nm": float(np.sqrt(np.mean(r ** 2))),
                     "residuals_nm": [float(v) for v in r]}
            slope = float(c[0])
        if model is None:
            model = {"type": "const", "coeffs": [float(anchor)], "ref_nm": 0.0,
                     "rms_nm": 0.0, "residuals_nm": []}
        mid = 0.5 * (lo + hi)
        shift_seg = float(_eval_model(model, mid))
        dom = sv["dominance"]
        # A segment vote exists here (>= vote_min_pairings total pairings).
        # Trust it as "ok" only when the mode dominates the second maximum;
        # otherwise "weak" (an aliased forest -- recorded but not applied).
        quality = "ok" if dom >= float(cfg["vote_dominance_min"]) else "weak"
        segments[name] = {
            "shift_nm": shift_seg, "mad_nm": mad, "sigma_nm": sv["sigma_nm"],
            "slope_nm_per_nm": slope, "n_lines": int(sv["n_pairings_mode"]),
            "n_windows": int(len(aw)), "n_confident_windows": int(len(aw)),
            "dominance": float(dom) if np.isfinite(dom) else None,
            "segment_vote_shift_nm": float(anchor),
            "model": model, "windows": wins, "quality": quality,
            "residual_mad_nm": mad}
        if quality != "none":
            all_shifts.append(shift_seg)

    gmed = float(np.median(all_shifts)) if all_shifts else 0.0
    overall = ("ok" if any(s["quality"] == "ok" for s in segments.values())
               else ("weak" if all_shifts else "none"))
    return {
        "version": REGISTRATION_VERSION, "source": "vote_mode_diagnostic",
        "convention": "observed_minus_database_nm",
        "prior_shift_nm": float(cfg["prior_shift_nm"]),
        "species": list(composition), "species_detected": list(composition),
        "n_lines": int(sum(s["n_lines"] for s in segments.values())),
        "n_observed_peaks": int(peaks_obs.size),
        "global_shift_nm": gmed, "global_mad_nm": None,
        "segments": segments, "residual_histogram": None, "outliers": [],
        "lines": [], "quality": overall,
        "estimator": "vote_mode",
    }


# ---------------------------------------------------------------------------
# aggregation shared by ambient + element
# ---------------------------------------------------------------------------
def _aggregate_segments(lines, cfg, *, source, species, windows=False):
    matched = [r for r in lines if r.get("matched")]
    lam = np.asarray([r["db_nm"] for r in matched], dtype=float)
    dl = np.asarray([r["shift_nm"] for r in matched], dtype=float)
    seg_idx = _segment_of(lam) if lam.size else np.empty(0, dtype=int)

    segments = {}
    for s, name in enumerate(SEGMENT_NAMES):
        sel = seg_idx == s if lam.size else np.zeros(0, dtype=bool)
        s_lam, s_dl = lam[sel], dl[sel]
        window_table = []
        if windows and s_lam.size:
            edges = np.arange(np.floor(s_lam.min() / cfg["window_nm"]) * cfg["window_nm"],
                              s_lam.max() + cfg["window_nm"], cfg["window_nm"])
            for lo, hi in zip(edges[:-1], edges[1:]):
                w = (s_lam >= lo) & (s_lam < hi)
                if np.sum(w) >= int(cfg["window_min_lines"]):
                    wv = s_dl[w]
                    med = float(np.median(wv))
                    mad = 1.4826 * float(np.median(np.abs(wv - med)))
                    window_table.append({"lo_nm": float(lo), "hi_nm": float(hi),
                                         "center_nm": float(0.5 * (lo + hi)),
                                         "shift_nm": med, "mad_nm": mad,
                                         "n_lines": int(np.sum(w))})
        if s_dl.size:
            med = float(np.median(s_dl))
            mad = 1.4826 * float(np.median(np.abs(s_dl - med)))
            se = 1.2533 * mad / np.sqrt(s_dl.size) if s_dl.size else np.inf
            model = _robust_line_fit(s_lam, s_dl, cfg,
                                     allow_quad=windows)
            slope = None
            if model["type"] in ("linear", "quad") and s_dl.size >= int(cfg["slope_min_lines"]):
                slope = float(model["coeffs"][-2]) if model["type"] == "linear" \
                    else float(np.polyder(np.asarray(model["coeffs"]))[-1])
            quality = ("ok" if s_dl.size >= int(cfg["segment_min_lines"])
                       else "weak")
            segments[name] = {
                "shift_nm": med, "mad_nm": mad, "sigma_nm": float(se),
                "slope_nm_per_nm": slope, "n_lines": int(s_dl.size),
                "model": model, "windows": window_table, "quality": quality,
                "residual_mad_nm": mad,
            }
        else:
            segments[name] = {"shift_nm": None, "mad_nm": None, "sigma_nm": None,
                              "slope_nm_per_nm": None, "n_lines": 0,
                              "model": None, "windows": window_table,
                              "quality": "none", "residual_mad_nm": None}

    if dl.size:
        gmed = float(np.median(dl))
        gmad = 1.4826 * float(np.median(np.abs(dl - gmed)))
    else:
        gmed, gmad = 0.0, None
    # residual histogram after removing the pooled global shift
    hist = None
    if dl.size:
        counts, edges = np.histogram(dl - gmed, bins=min(20, max(5, dl.size)))
        hist = {"counts": [int(c) for c in counts],
                "edges_nm": [float(e) for e in edges]}
    # outliers: > 3 MAD from the global
    outliers = []
    if dl.size and gmad and gmad > 0:
        for r in matched:
            if abs(r["shift_nm"] - gmed) > 3.0 * gmad:
                outliers.append({"db_nm": r["db_nm"], "shift_nm": r["shift_nm"]})

    overall = ("ok" if any(seg["quality"] == "ok" for seg in segments.values())
               else ("weak" if dl.size else "none"))
    return {
        "version": REGISTRATION_VERSION,
        "source": source,
        "convention": "observed_minus_database_nm",
        "prior_shift_nm": float(cfg["prior_shift_nm"]),
        "species": list(species),
        "species_detected": sorted({r.get("species") for r in matched
                                    if r.get("species")}) or list(species),
        "n_lines": int(dl.size),
        "global_shift_nm": gmed,
        "global_mad_nm": gmad,
        "segments": segments,
        "residual_histogram": hist,
        "outliers": outliers,
        "lines": lines,
        "quality": overall,
    }


# ---------------------------------------------------------------------------
# combined registration + application
# ---------------------------------------------------------------------------
class RegistrationShift:
    """A per-segment Delta-lambda(lambda) callable, drop-in for ``shift_at``.

    ``at(wl)`` returns the shift (observed - database) at wl using the chosen
    per-segment model; ``at_in_frame`` accepts observed/database (the two
    differ by <1 pm at these shift magnitudes).  ``float()`` returns the
    pooled global shift so summary consumers keep working.
    """

    __slots__ = ("edges", "models", "global_shift", "segment_names")

    def __init__(self, edges, models, global_shift, segment_names=SEGMENT_NAMES):
        self.edges = tuple(float(e) for e in edges)
        self.models = tuple(models)
        self.global_shift = float(global_shift)
        self.segment_names = tuple(segment_names)

    def at(self, wl):
        arr = np.asarray(wl, dtype=float)
        idx = np.digitize(arr, self.edges)
        flat = np.atleast_1d(arr).astype(float)
        fidx = np.atleast_1d(idx)
        out = np.empty(flat.shape, dtype=float)
        for s in range(len(self.segment_names)):
            m = self.models[s]
            sel = fidx == s
            if not np.any(sel):
                continue
            if m is None:
                out[sel] = self.global_shift
            else:
                out[sel] = _eval_model(m, flat[sel])
        out = out.reshape(np.asarray(wl, dtype=float).shape)
        return float(out) if arr.ndim == 0 else out

    def at_in_frame(self, wl, *, frame="observed"):
        return self.at(wl)

    def __float__(self):
        return self.global_shift

    def __repr__(self):
        return (f"RegistrationShift(edges={self.edges}, "
                f"global={1000 * self.global_shift:+.1f} pm)")


def combined_registration(ambient: dict, element: dict, *, config=None) -> dict:
    """Deployment rule combining element and ambient registrations.

    Per segment: use the element registration where its segment fit passes
    quality; fall back to ambient (which dominates the NIR); then to a pooled
    global shift.  The NIR ambient and element shifts are cross-checked and a
    disagreement is flagged when they differ by more than 2 sigma.
    """
    cfg = _config(config)
    edges = SEGMENT_EDGES
    seg_choice = {}
    models = []
    # pooled global fallback: prefer element global, else ambient
    pooled = element.get("global_shift_nm")
    if pooled is None or element.get("n_lines", 0) == 0:
        pooled = ambient.get("global_shift_nm", 0.0) or 0.0

    for name in SEGMENT_NAMES:
        e_seg = element.get("segments", {}).get(name, {})
        a_seg = ambient.get("segments", {}).get(name, {})
        chosen, model, src = None, None, "pooled"
        if name == "NIR":
            # Ar is the NIR anchor (stable across runs); element NIR is only a
            # cross-check because the sample's NIR lines are sparse/aliased.
            if a_seg.get("n_lines", 0) >= 3 and a_seg.get("quality") in ("ok", "weak"):
                chosen, model, src = a_seg, a_seg.get("model"), "ambient"
            elif e_seg.get("quality") == "ok":
                chosen, model, src = e_seg, e_seg.get("model"), "element"
            elif a_seg.get("n_lines", 0) > 0:
                chosen, model, src = a_seg, a_seg.get("model"), "ambient"
        else:
            if e_seg.get("quality") == "ok":
                chosen, model, src = e_seg, e_seg.get("model"), "element"
            elif a_seg.get("quality") in ("ok", "weak") and a_seg.get("n_lines", 0) > 0:
                chosen, model, src = a_seg, a_seg.get("model"), "ambient"
            elif e_seg.get("n_lines", 0) > 0:
                chosen, model, src = e_seg, e_seg.get("model"), "element"
        if model is None:
            model = {"type": "const", "coeffs": [float(pooled)], "ref_nm": 0.0,
                     "rms_nm": 0.0, "residuals_nm": []}
            src = "pooled"
        models.append(model)
        seg_choice[name] = {
            "source": src,
            "shift_nm": (chosen.get("shift_nm") if chosen else float(pooled)),
            "sigma_nm": (chosen.get("sigma_nm") if chosen else None),
            "n_lines": (chosen.get("n_lines", 0) if chosen else 0),
            "slope_nm_per_nm": (chosen.get("slope_nm_per_nm") if chosen else None),
            "quality": (chosen.get("quality") if chosen else "pooled"),
        }

    # NIR cross-check
    nir_e = element.get("segments", {}).get("NIR", {})
    nir_a = ambient.get("segments", {}).get("NIR", {})
    disagreement = None
    if (nir_e.get("shift_nm") is not None and nir_a.get("shift_nm") is not None):
        d = float(nir_e["shift_nm"]) - float(nir_a["shift_nm"])
        se = np.hypot(nir_e.get("sigma_nm") or 0.0, nir_a.get("sigma_nm") or 0.0)
        se = max(se, 1e-6)
        disagreement = {
            "element_nir_shift_nm": float(nir_e["shift_nm"]),
            "ambient_nir_shift_nm": float(nir_a["shift_nm"]),
            "difference_nm": d, "combined_sigma_nm": float(se),
            "n_sigma": float(abs(d) / se),
            "flagged": bool(abs(d) > 2.0 * se),
        }

    quality = ("ok" if any(v["source"] in ("element", "ambient")
                           and v["quality"] == "ok"
                           for v in seg_choice.values())
               else ("weak" if any(v["n_lines"] > 0 for v in seg_choice.values())
                     else "none"))
    return {
        "version": REGISTRATION_VERSION,
        "convention": "observed_minus_database_nm",
        "edges_nm": list(edges),
        "segment_names": list(SEGMENT_NAMES),
        "segments": seg_choice,
        "models": [dict(m) for m in models],
        "global_shift_nm": float(pooled),
        "nir_disagreement": disagreement,
        "quality": quality,
        "ambient_quality": ambient.get("quality"),
        "element_quality": element.get("quality"),
    }


def registration_shift(combined: dict) -> RegistrationShift:
    """Build the :class:`RegistrationShift` callable from a combined result."""
    return RegistrationShift(combined["edges_nm"], combined["models"],
                             combined["global_shift_nm"],
                             combined.get("segment_names", SEGMENT_NAMES))


def apply_registration(x, registration) -> np.ndarray:
    """Correct an observed wavelength axis into the database frame.

    ``registration`` may be a combined-registration dict or a
    :class:`RegistrationShift`.  Returns ``x - shift(x)``.
    """
    if isinstance(registration, RegistrationShift):
        shift = registration
    else:
        shift = registration_shift(registration)
    x = np.asarray(x, dtype=float)
    return x - shift.at(x)


def registration_to_json(registration) -> dict:
    """JSON-safe copy of a registration dict (or RegistrationShift)."""
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return [clean(v) for v in obj.tolist()]
        return obj
    if isinstance(registration, RegistrationShift):
        registration = {
            "version": REGISTRATION_VERSION,
            "edges_nm": list(registration.edges),
            "models": [dict(m) if m else None for m in registration.models],
            "global_shift_nm": registration.global_shift,
        }
    return clean(registration)


# ---------------------------------------------------------------------------
# vendor calibration change
# ---------------------------------------------------------------------------
def vendor_calibration_shift(base_coeffs, current_coeffs, knots,
                             *, n_samples=200) -> dict:
    """Per-segment Delta-lambda(lambda) between two pixel->nm cubic sets.

    ``base_coeffs`` / ``current_coeffs`` are sequences of per-segment cubic
    coefficient lists ``[c0, c1, c2, c3]`` with ``lambda(p) = c0 + c1 p +
    c2 p^2 + c3 p^3`` (p = pixel index within the segment; the Z300 pixel axis
    runs so lambda DECREASES with p).  ``knots`` are the nm segment boundaries
    (e.g. ``[180, 365, 620, 960, 961]``).  For each real segment the pixel
    range is recovered from the base cubic over the knot interval, and
    Delta-lambda = lambda_current - lambda_base is tabulated versus lambda and
    fit with a low-order polynomial.  A degenerate final segment (width < 5 nm)
    is skipped.
    """
    knots = [float(k) for k in knots]
    base = [np.asarray(c, dtype=float) for c in base_coeffs]
    cur = [np.asarray(c, dtype=float) for c in current_coeffs]
    n = min(len(base), len(cur), len(knots) - 1)
    # dense pixel grid; find each segment's pixels by nm bounds of base cubic
    p = np.linspace(0.0, 4096.0, 8193)
    segments = {}
    for i in range(n):
        lo_nm, hi_nm = sorted((knots[i], knots[i + 1]))
        if hi_nm - lo_nm < 5.0:
            continue  # dummy segment
        # lambda(p) = c0 + c1 p + c2 p^2 + c3 p^3 (poly with c3 highest)
        def evalc(c, pp):
            return c[0] + c[1] * pp + c[2] * pp ** 2 + c[3] * pp ** 3
        lam_b = evalc(base[i], p)
        inseg = (lam_b >= lo_nm) & (lam_b <= hi_nm)
        if np.sum(inseg) < 10:
            continue
        pp = p[inseg]
        lam_base = evalc(base[i], pp)
        lam_cur = evalc(cur[i], pp)
        dl = lam_cur - lam_base
        order = np.argsort(lam_base)
        lam_base, dl = lam_base[order], dl[order]
        # subsample for the table
        take = np.linspace(0, lam_base.size - 1, min(n_samples, lam_base.size)).astype(int)
        deg = 2 if lam_base.size > 50 else 1
        coeffs = np.polyfit(lam_base - np.median(lam_base), dl, deg)
        name = SEGMENT_NAMES[int(_segment_of(0.5 * (lo_nm + hi_nm)))]
        segments[name] = {
            "nm_range": [float(lo_nm), float(hi_nm)],
            "delta_nm_mean": float(np.mean(dl)),
            "delta_nm_min": float(np.min(dl)),
            "delta_nm_max": float(np.max(dl)),
            "delta_nm_at_lo": float(dl[0]),
            "delta_nm_at_hi": float(dl[-1]),
            "poly_coeffs": [float(v) for v in coeffs],
            "ref_nm": float(np.median(lam_base)),
            "lambda_nm": [float(v) for v in lam_base[take]],
            "delta_nm": [float(v) for v in dl[take]],
        }
    return {"version": REGISTRATION_VERSION,
            "convention": "current_minus_base_nm",
            "segments": segments}


# ---------------------------------------------------------------------------
# thermal drift model
# ---------------------------------------------------------------------------
def _linfit_with_sigma(t, z):
    """OLS slope/intercept with slope standard error; robust to tiny n."""
    t = np.asarray(t, dtype=float)
    z = np.asarray(z, dtype=float)
    n = t.size
    if n < 2 or np.ptp(t) <= 0:
        return None
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    b, a = float(coef[0]), float(coef[1])
    resid = z - (a + b * t)
    dof = max(n - 2, 1)
    s2 = float(np.sum(resid ** 2) / dof)
    sxx = float(np.sum((t - np.mean(t)) ** 2))
    sb = float(np.sqrt(s2 / sxx)) if sxx > 0 else np.inf
    return {"intercept": a, "slope": b, "slope_sigma": sb,
            "rms_nm": float(np.sqrt(np.mean(resid ** 2))), "n": int(n)}


def thermal_drift_model(records, *, config=None) -> dict:
    """Fit per-segment wavelength drift versus warm-up time / temperature.

    ``records`` is a sequence of dicts, one per run, with keys:
      * ``minutes_since_calibration`` (float),
      * ``segment_shifts`` -> {segment_name: shift_nm},
      * optionally ``temperature_c`` (float or None),
      * optionally ``ambient_nir_shift_nm`` (float),
      * optionally ``calibration_id`` (str, used to fit within one epoch).

    For each segment it fits shift = a + b * minutes (and, when available,
    versus temperature and versus the ambient NIR shift as a thermal proxy),
    reporting slope, its sigma, and the 2-sigma detection limit.  When the data
    cannot resolve a drift (|slope| < 2 sigma), :meth:`predict_shift` returns
    0.0 and ``resolved`` is False.
    """
    cfg = _config(config)
    recs = list(records)
    fits = {}
    for name in SEGMENT_NAMES:
        minutes, shifts, temps, proxies = [], [], [], []
        for r in recs:
            ss = r.get("segment_shifts", {})
            v = ss.get(name)
            if v is None or not np.isfinite(v):
                continue
            m = r.get("minutes_since_calibration")
            if m is None or not np.isfinite(m):
                continue
            minutes.append(float(m))
            shifts.append(float(v))
            temps.append(r.get("temperature_c"))
            proxies.append(r.get("ambient_nir_shift_nm"))
        entry = {"n": len(shifts)}
        vs_min = _linfit_with_sigma(minutes, shifts) if len(shifts) >= 2 else None
        entry["vs_minutes"] = vs_min
        # temperature fit if any temperature present
        tt = [(t, s) for t, s in zip(temps, shifts) if t is not None and np.isfinite(t)]
        entry["vs_temperature"] = (_linfit_with_sigma([a for a, _ in tt],
                                                      [b for _, b in tt])
                                   if len(tt) >= 2 else None)
        pp = [(p, s) for p, s in zip(proxies, shifts)
              if p is not None and np.isfinite(p)]
        entry["vs_ambient_nir_shift"] = (_linfit_with_sigma([a for a, _ in pp],
                                                            [b for _, b in pp])
                                         if len(pp) >= 2 else None)
        resolved = bool(vs_min and np.isfinite(vs_min["slope_sigma"])
                        and abs(vs_min["slope"]) > 2.0 * vs_min["slope_sigma"])
        entry["resolved"] = resolved
        if vs_min:
            entry["detection_limit_nm_per_hour"] = float(2.0 * vs_min["slope_sigma"] * 60.0)
            entry["slope_nm_per_hour"] = float(vs_min["slope"] * 60.0)
        else:
            entry["detection_limit_nm_per_hour"] = None
            entry["slope_nm_per_hour"] = None
        fits[name] = entry

    def predict_shift(segment, minutes_since_calibration=None, temperature=None):
        """Predicted shift [nm]; 0.0 when the drift is unresolved."""
        f = fits.get(segment)
        if f is None:
            return 0.0
        if (temperature is not None and f.get("vs_temperature")
                and abs(f["vs_temperature"]["slope"])
                > 2.0 * f["vs_temperature"]["slope_sigma"]):
            m = f["vs_temperature"]
            return float(m["intercept"] + m["slope"] * float(temperature))
        if f["resolved"] and minutes_since_calibration is not None:
            m = f["vs_minutes"]
            return float(m["intercept"] + m["slope"] * float(minutes_since_calibration))
        return 0.0

    return {
        "version": REGISTRATION_VERSION,
        "convention": "observed_minus_database_nm",
        "n_records": len(recs),
        "segments": fits,
        "predict_shift": predict_shift,
        "resolved": {name: fits[name]["resolved"] for name in SEGMENT_NAMES},
    }


__all__ = [
    "REGISTRATION_VERSION", "SEGMENT_EDGES", "SEGMENT_NAMES",
    "ambient_registration", "element_registration", "golden_lines",
    "vote_mode_diagnostic", "combined_registration",
    "RegistrationShift", "registration_shift", "apply_registration",
    "registration_to_json", "vendor_calibration_shift", "thermal_drift_model",
]
