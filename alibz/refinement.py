"""Second-iteration refinement: blends vs asymmetry, decided by physics.

After the first fit, two kinds of ambiguous structure remain:

1. significant RESIDUAL structure next to a fitted peak — a genuine
   overlapping line the window fit missed, or the signature of an
   asymmetric single line (self-absorption) that a symmetric Voigt
   cannot follow;
2. sub-FWHM PAIRS of fitted components — a real blend, or one
   self-absorbed line that the fitter split into two phantom components
   (measured: K I 766.49, flat-topped by an optical depth of ~1, fitted
   as 766.40 + 766.61).

Each candidate feature is refit locally under three models —

- ``S``: a single symmetric Voigt (4 parameters),
- ``A``: a single Voigt attenuated by a cold-layer absorber of the same
  shape, ``I(x) = A V(x-mu) exp(-tau_a V(x-mu-delta)/V(0))`` (6
  parameters; ``delta`` produces red/blue shading, large ``tau_a`` at
  ``delta ~ 0`` produces the flat top / self-reversal),
- ``B``: two symmetric Voigts (8 parameters),

compared by noise-weighted BIC, and the verdict is GATED by
physics/chemistry from the line database: claiming a blend requires at
least two plausible database lines under the feature (matched to the
fitted component centers), while an asymmetry verdict is corroborated
when the primary line is a ground-term resonance line — the class that
self-absorbs.  Statistical preference can override missing database
knowledge only with a large margin.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.special import voigt_profile as _voigt

from alibz.utils.voigt import multi_voigt as _multi_voigt
from alibz.utils.voigt import voigt_width as _voigt_width

#: BIC margin required for a db-supported blend verdict.
BIC_MARGIN = 10.0
#: BIC margin required to claim a blend WITHOUT database support.
BIC_MARGIN_UNSUPPORTED = 30.0
#: Boltzmann temperature used only to rank database-line plausibility.
_RANK_KT_EV = 0.75
#: Absorber optical-depth fit ceiling; a fit AT the ceiling has not
#: converged and is never actionable.
TAU_MAX = 8.0


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def sa_voigt(x, area, mu, sigma, gamma, tau_a, delta):
    """Voigt emission attenuated by a shifted same-shape cold absorber.

    ``area`` is the UNATTENUATED (thin-equivalent) emission area; the
    observed area is smaller by the absorption.  ``tau_a`` is the
    absorber's line-centre optical depth and ``delta`` its centre shift
    relative to the emitter (cooler-layer Doppler/Stark offset).
    """
    sigma = max(float(sigma), 1e-9)
    gamma = max(float(gamma), 1e-9)
    emit = area * _voigt(x - mu, sigma, gamma)
    absorb = _voigt(x - mu - delta, sigma, gamma)
    peak = _voigt(0.0, sigma, gamma)
    return emit * np.exp(-max(float(tau_a), 0.0) * absorb / peak)


def _model_S(x, p):
    return p[0] * _voigt(x - p[1], max(p[2], 1e-9), max(p[3], 1e-9))


def _model_A(x, p):
    return sa_voigt(x, *p)


def _model_B(x, p):
    return _model_S(x, p[:4]) + _model_S(x, p[4:])


#: Robust-loss scale for the S/A/B window fits, in units of the local
#: noise sigma (residuals are noise-weighted before the loss is applied).
#: ``soft_l1`` KEEPS every sample but rolls its penalty from quadratic to
#: linear past this many sigma, so a single-pixel instrument notch on a
#: strong line's flank — the spectrometer's own export ringing, measured
#: ~19 sigma deep on Sr II 407.77 — can no longer lever the fitted optical
#: depth or centre, while genuine line structure at a few sigma is left
#: untouched.  On clean data with no such outliers the fit is identical to
#: ordinary least squares (soft_l1 -> quadratic well inside f_scale), so
#: the synthetic refinement fixtures are unaffected.  8 sigma is chosen to
#: catch the export notches (>10 sigma) without down-weighting the ~5-7
#: sigma line-shape misfit that a Voigt leaves on the brightest lines
#: (that is model inadequacy to be fixed by a real line-spread function,
#: not an outlier to reject — see docs/development_guide.md).
SA_ROBUST_F_SCALE = 8.0


def _fit(model, x, y, w, p0, lo, hi, loss="soft_l1", f_scale=None):
    if f_scale is None:
        f_scale = SA_ROBUST_F_SCALE
    p0 = np.clip(np.asarray(p0, dtype=float), lo, hi)
    try:
        res = least_squares(
            lambda p: (y - model(x, p)) * w, p0, bounds=(lo, hi),
            x_scale="jac", max_nfev=400, loss=loss, f_scale=f_scale,
        )
    except (ValueError, RuntimeError):
        return None, np.inf
    # 2*cost is the robust sum-of-squares (identical to sum(fun**2) for
    # loss="linear"), so an outlier's leverage is removed from the BIC
    # model comparison too, not just from the fitted parameters.
    return res.x, 2.0 * float(res.cost)


def _bic(chi2, n, n_params):
    return chi2 + n_params * np.log(max(n, 2))


# ---------------------------------------------------------------------------
# Database evidence
# ---------------------------------------------------------------------------

def db_lines_in(db, lo, hi, rel_strength=0.03, elements=None):
    """Plausible database lines in [lo, hi]: (wavelength, strength, Ei).

    Strength is the Boltzmann-ranked ``gA exp(-Ek/kT)`` at a nominal LIBS
    temperature; lines below ``rel_strength`` of the window's strongest
    are dropped.  ``elements`` restricts to a detected-element list.
    """
    excluded = getattr(db, "analysis_excluded_elements",
                       getattr(db, "unsupported_elements", ()))
    wl_all, s_all, ei_all = [], [], []
    for el in db.elements:
        if el in db.no_lines or el in excluded:
            continue
        if elements is not None and el not in elements:
            continue
        arr = db.lines(el)
        if arr.size == 0:
            continue
        ion = arr[:, 0].astype(float)
        wl = arr[:, 1].astype(float)
        gA = arr[:, 3].astype(float)
        Ei = arr[:, 4].astype(float)
        Ek = arr[:, 5].astype(float)
        keep = (ion <= 2) & (wl >= lo) & (wl <= hi)
        if not np.any(keep):
            continue
        wl_all.append(wl[keep])
        s_all.append(gA[keep] * np.exp(-Ek[keep] / _RANK_KT_EV))
        ei_all.append(Ei[keep])
    if not wl_all:
        return np.empty(0), np.empty(0), np.empty(0)
    wl_all = np.concatenate(wl_all)
    s_all = np.concatenate(s_all)
    ei_all = np.concatenate(ei_all)
    strong = s_all >= rel_strength * np.max(s_all)
    order = np.argsort(wl_all[strong])
    return wl_all[strong][order], s_all[strong][order], ei_all[strong][order]


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------

def _weighted_median(values, weights):
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    k = min(int(np.searchsorted(cw, 0.5 * cw[-1])), values.size - 1)
    return float(values[order][k])


def _feature_candidates(x, y_bgsub, peaks, model_total, noise,
                        snr_min=4.0, pair_sep_factor=2.0):
    """Ambiguous features: (kind, peak_indices) tuples.

    ``kind`` is ``"residual"`` (significant residual next to one peak)
    or ``"pair"`` (two fitted components separated by less than
    ``pair_sep_factor x`` their mean FWHM).  A phantom split of one
    self-absorbed line leaves the two components 1–1.7x their own fitted
    FWHM apart (deeper absorption and a shifted absorber push them
    further apart while squeezing each narrower than the true line —
    tau=2.5 with a 0.05 nm absorber shift measures 1.64x), so the factor
    must comfortably exceed that.  Over-pairing is safe: pairs are only
    CANDIDATES, and every action needs a verdict from the model
    competition and its physics gates.  Pairs are resolved greedily by
    combined area
    so no peak participates in two features — otherwise a strong line
    flanked by junk peaks gets classified (and dropped/added) twice.

    Components much broader than the area-weighted median width are
    background mops the first pass parked under a strong line's wings,
    not lines (genuine lines share the instrument width): they neither
    pair nor seed residual features.
    """
    fwhm = _voigt_width(np.maximum(peaks[:, 2], 1e-6),
                        np.maximum(peaks[:, 3], 1e-6))
    wmed = _weighted_median(fwhm, np.maximum(peaks[:, 0], 1e-12))
    mop = fwhm > 5.0 * wmed

    order = np.array([i for i in np.argsort(peaks[:, 1]) if not mop[i]],
                     dtype=int)
    pairs = []
    for a, b in zip(order[:-1], order[1:]):
        sep = peaks[b, 1] - peaks[a, 1]
        if sep < pair_sep_factor * 0.5 * (fwhm[a] + fwhm[b]):
            pairs.append((float(peaks[a, 0] + peaks[b, 0]), int(a), int(b)))
    pairs.sort(reverse=True)
    out, used = [], set()
    for _, a, b in pairs:
        if a in used or b in used:
            continue
        used.update((a, b))
        out.append(("pair", (a, b)))

    residual = y_bgsub - model_total
    for k in range(peaks.shape[0]):
        if k in used or mop[k]:
            continue
        span = max(1.5 * fwhm[k], 0.15)
        m = (x >= peaks[k, 1] - span) & (x <= peaks[k, 1] + span)
        if not np.any(m):
            continue
        core = np.abs(x - peaks[k, 1]) < 0.35 * fwhm[k]
        r = np.where(m & ~core, residual, 0.0)
        if np.max(np.abs(r) / np.maximum(noise, 1e-12)) >= snr_min:
            out.append(("residual", (k,)))
    return out


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _feature_window(sub):
    """Amplitude-weighted centre, max component FWHM, and half-span."""
    mu0 = float(np.average(sub[:, 1], weights=np.maximum(sub[:, 0], 1e-12)))
    fwhm0 = float(np.max(_voigt_width(np.maximum(sub[:, 2], 1e-6),
                                      np.maximum(sub[:, 3], 1e-6))))
    return mu0, fwhm0, max(4.0 * fwhm0, 0.4)


def classify_feature(x, y_bgsub, peaks, indices, noise, db=None,
                     elements=None, shift_nm=0.0, tol_nm=0.06,
                     model_others=None, extra_area=0.0):
    """Fit S / A / B models to one feature and decide blend vs asymmetry.

    ``model_others`` is the summed profile of every peak NOT in
    ``indices``; it is subtracted from the window so the competing models
    only have to explain the feature itself.  ``extra_area`` is flux that
    belongs to the feature but sits in other table rows (wing-soaker
    components absorbed by the caller — see :func:`refine_fit`); it
    enlarges the area seeds/bounds.  ``shift_nm`` is the value returned
    by ``estimate_wavelength_shift`` (observed = db + shift).

    Returns a decision dict with the verdict, per-model BIC, fitted
    parameters, and the database evidence used.
    """
    sub = peaks[list(indices)]
    mu0, fwhm0, span = _feature_window(sub)
    m = (x >= mu0 - span) & (x <= mu0 + span)
    xw, yw = x[m], y_bgsub[m]
    n = xw.size
    if n < 15:
        return None
    if model_others is not None:
        # a feature whose window is dominated by a much stronger
        # neighbour cannot be reclassified robustly: the neighbour's own
        # model error dwarfs the feature (typical for the broad junk
        # components the first pass parks on a strong line's wings) —
        # leave it alone
        own_max = float(np.max(_multi_voigt(xw, np.ravel(sub[:, :4]))))
        if float(np.max(model_others[m])) > 3.0 * max(own_max, 1e-12):
            return None
        yw = yw - model_others[m]
    w = 1.0 / np.maximum(noise[m], 1e-12)

    area0 = float(np.sum(sub[:, 0])) + max(float(extra_area), 0.0)
    sig0 = float(np.median(np.maximum(sub[:, 2], 1e-3)))
    gam0 = float(np.median(np.maximum(sub[:, 3], 1e-3)))
    wmax = max(3.0 * fwhm0, 0.2)

    pS, chiS = _fit(_model_S, xw, yw, w,
                    [area0, mu0, sig0, gam0],
                    [0.0, mu0 - fwhm0, 1e-4, 0.0],
                    [20 * area0, mu0 + fwhm0, wmax, wmax])
    # the SA surface is strongly non-convex in tau (weak-absorption and
    # flat-top regimes are separated by a barrier) AND in delta (red- vs
    # blue-shaded absorbers are separate basins): multi-start the grid.
    # |delta| is capped at 0.1 nm — cool-layer Doppler/Stark offsets are
    # tens of pm; a larger "absorber shift" is the model exploiting the
    # window to fit neighbouring structure, which drags the emission
    # centre off the line.
    dmax = min(0.75 * fwhm0, 0.1)
    pA, chiA = None, np.inf
    for tau0 in (0.3, 1.5, 4.0):
        for delta0 in (-0.5 * dmax, 0.0, 0.5 * dmax):
            p, c = _fit(_model_A, xw, yw, w,
                        [area0 * (1.0 + tau0), mu0, sig0, gam0, tau0, delta0],
                        [0.0, mu0 - fwhm0, 1e-4, 0.0, 0.0, -dmax],
                        [50 * area0, mu0 + fwhm0, wmax, wmax, TAU_MAX,
                         dmax])
            if c < chiA:
                pA, chiA = p, c
    if len(indices) == 2:
        b0 = [sub[0, 0], sub[0, 1], max(sub[0, 2], 1e-3), max(sub[0, 3], 1e-3),
              sub[1, 0], sub[1, 1], max(sub[1, 2], 1e-3), max(sub[1, 3], 1e-3)]
    else:
        b0 = [0.7 * area0, mu0 - 0.3 * fwhm0, sig0, gam0,
              0.4 * area0, mu0 + 0.4 * fwhm0, sig0, gam0]
    pB, chiB = _fit(_model_B, xw, yw, w, b0,
                    [0.0, mu0 - span, 1e-4, 0.0] * 2,
                    [20 * area0, mu0 + span, wmax, wmax] * 2)

    # Local windows on real spectra are systematics-dominated (imperfect
    # neighbour subtraction, background residue), so raw chi2 sits far
    # above the photon-noise expectation and raw-chi2 BIC margins would
    # always favour the richest model.  Rescale by the best model's
    # reduced chi2 (floored at 1) so margins measure RELATIVE improvement
    # — the same device as inspection.estimate_peak_uncertainties.
    chis = {"S": chiS, "A": chiA, "B": chiB}
    npar = {"S": 4, "A": 6, "B": 8}
    if not np.isfinite(min(chis.values())):
        return None  # every model failed to fit: nothing to decide
    s2 = max(1.0, min(c / max(n - npar[k], 1) for k, c in chis.items()))
    bic = {k: _bic(c / s2, n, npar[k]) for k, c in chis.items()}

    # --- physics/chemistry evidence ---
    # A blend claim needs two DISTINCT database lines matching the two
    # fitted centers with a consistent separation.  Matching each centre
    # to "any db line within tol" is too weak: a dense window (or a
    # close doublet like Li I 670.776/670.791) coincidentally matches
    # phantom-split components that sit ~0.1 nm apart.
    n_db, resonance_primary = None, None
    blend_supported = db is None
    if db is not None:
        # shift_nm follows estimate_wavelength_shift (observed = db +
        # shift); scalar or per-segment — evaluate at this feature
        from alibz.utils.wavelength import shift_at
        s_loc = float(shift_at(shift_nm, mu0))
        wl_db, s_db, ei_db = db_lines_in(
            db, mu0 - span - s_loc, mu0 + span - s_loc,
            elements=elements)
        wl_db = wl_db + s_loc  # into the observed frame
        n_db = int(wl_db.size)
        if wl_db.size:
            # The primary is the line most plausibly producing the
            # feature: judge resonance capability from the STRONG lines
            # within match range, not the single nearest one — for a
            # self-absorbed line mu0 is displaced by the absorber, and a
            # weak coincidental neighbour (e.g. La I 766.433 at 3 % of
            # K I 766.490) must not veto the identification.
            near = np.abs(wl_db - mu0) <= 2 * tol_nm
            wl_resonance = np.empty(0)
            if np.any(near):
                s_near = s_db[near]
                capable = ((ei_db[near] <= 0.2)
                           & (s_near >= 0.25 * np.max(s_near)))
                wl_resonance = wl_db[near][capable]
                resonance_primary = bool(np.any(capable))
            else:
                resonance_primary = False
            if pB is not None and n_db >= 2:
                c1, c2 = float(pB[1]), float(pB[5])
                j1 = int(np.argmin(np.abs(wl_db - c1)))
                j2 = int(np.argmin(np.abs(wl_db - c2)))
                sep_fit = abs(c2 - c1)
                sep_db = abs(float(wl_db[j2]) - float(wl_db[j1]))
                blend_supported = (
                    j1 != j2
                    and abs(float(wl_db[j1]) - c1) <= tol_nm
                    and abs(float(wl_db[j2]) - c2) <= tol_nm
                    and abs(sep_fit - sep_db) <= tol_nm
                    and sep_fit > tol_nm
                )

    best = min(bic, key=bic.get)
    # "single" is an ACTIONABLE verdict (pairs get merged on it), so it
    # is only issued when S actually wins; any other unresolved
    # competition falls through to "ambiguous" (no action).
    verdict = "single" if best == "S" else "ambiguous"
    if best == "B":
        margin = min(bic["S"], bic["A"]) - bic["B"]
        if blend_supported and margin >= BIC_MARGIN:
            verdict = "blend"
        elif margin >= BIC_MARGIN_UNSUPPORTED:
            verdict = "blend-unassigned"
        elif (not (db is not None and blend_supported)
              and bic["A"] - bic["B"] <= BIC_MARGIN
              and bic["A"] <= bic["S"] - BIC_MARGIN):
            # B is nominally best but A is statistically competitive
            # (within the decision margin) — an SA model must never be
            # claimed when the two-Voigt model DECISIVELY beats it, or
            # every unlisted genuine blend gets merged into a fictitious
            # self-absorbed line.  When the database SUPPORTS the blend
            # (two DISTINCT lines match the two centers with a consistent
            # separation) this escape is barred entirely: near-degenerate
            # statistics must not override line-list evidence (measured:
            # Fe I 248.814/249.064 on MW2-112 merged into a fictitious
            # tau=2.7 line at 248.99, whose exclusion zone then blocked
            # recovery of the real 248.73/249.10 residuals).
            verdict = "asymmetric"
    elif best == "A" and bic["S"] - bic["A"] >= BIC_MARGIN:
        verdict = "asymmetric"

    # An asymmetric verdict is only ACTIONABLE (merged by the driver)
    # when the fit and the physics agree; otherwise it is recorded under
    # a qualified name and left alone.
    if verdict == "asymmetric":
        if pA is not None and pA[4] >= 0.98 * TAU_MAX:
            # optical depth pinned at the fit ceiling: not converged
            verdict = "asymmetric-saturated"
        elif db is not None and resonance_primary is not True:
            # No POSITIVE resonance evidence: optical depths deep enough
            # to distort a profile need a ground-state lower level
            # (metastable lower levels — Ca II 854, O I 777, Ar I 811 —
            # also qualify but are not yet flagged in the db, so they
            # land here too).  A window with zero db lines
            # (resonance_primary None) is treated the same as a negative.
            verdict = "asymmetric-nonresonant"
        elif (db is not None and pA is not None
              and float(np.min(np.abs(wl_resonance - pA[1]))) > tol_nm):
            # the fitted emission centre walked off the resonance line it
            # claims to be — the model is fitting neighbouring structure
            # (measured: Li I 670.78 dragged to 670.96 by a +235 pm
            # "absorber" before |delta| was capped)
            verdict = "asymmetric-displaced"

    return {
        "indices": tuple(int(i) for i in indices),
        "center": mu0,
        "verdict": verdict,
        "bic": bic,
        "params_single": pS,
        "params_asym": pA,
        "params_blend": pB,
        "n_db_lines": n_db,
        "resonance_primary": resonance_primary,
        "noise_rescale": float(s2),
        "window": (float(xw[0]), float(xw[-1])),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def refine_fit(x, y, fit_dict, db=None, elements=None, shift_nm=0.0,
               snr_min=4.0, segment_edges=None,
               s2_action_max=50.0,
               asymmetric="apply") -> Tuple[dict, List[dict]]:
    """Second-iteration refinement of a fitted spectrum.

    Enumerates ambiguous features (significant residual structure next to
    a peak; sub-FWHM component pairs), classifies each (see
    :func:`classify_feature`), and applies the verdicts:

    - ``blend`` / ``blend-unassigned`` on a residual feature: the second
      fitted component is ADDED to the peak table;
    - ``blend`` on an existing pair: both components are replaced by the
      two-Voigt refit (cleaner joint parameters);
    - ``asymmetric``: the feature is replaced by ONE peak whose table row
      carries the OBSERVED area (attenuated flux, consistent with every
      other row); the unattenuated emission area, ``tau_a`` and ``delta``
      are reported in the decision record (do NOT also apply a downstream
      self-absorption correction to rows merged this way);
    - ``asymmetric-nonresonant``: recorded but NOT applied — without a
      resonance-capable lower level the "absorber" is more likely an
      unresolved blend or instrumental, and merging would destroy it;
    - ``single`` on a PAIR: the symmetric single Voigt beats both
      richer models even after the BIC penalty, so the split was
      redundant — the pair is merged into the single-Voigt refit;
    - ``single`` on a residual feature / ``ambiguous``: unchanged.

    Wing-soaker absorption: the first pass parks broad low-amplitude
    components on a strong line's wings (its own unmodelled flux).  Any
    component overlapping the feature window that is broader than
    1.5x the area-weighted median width AND smaller than 25 % of the
    feature's area is treated as part of the feature's flux — NOT
    subtracted from the classification target, and dropped alongside the
    feature when an action fires (its indices appear in the decision's
    ``absorbed``).  Genuine minor lines keep the instrument width and are
    never absorbed.

    Actions additionally require the window to be statistically usable:
    a decision whose ``noise_rescale`` exceeds ``s2_action_max`` is
    recorded but NOT applied (the best model explains nothing at the
    stated noise level, so model preference is not trustworthy).

    ``asymmetric`` stages the DATA evidence apart from the PHYSICS
    adjudication (the resonance-capability gates behind an asymmetric
    merge are the only verdicts that depend on which elements are
    actually present — with ``elements=None`` they run against the whole
    periodic table):

    - ``"apply"`` (default): single-pass behaviour, everything above;
    - ``"defer"``: blend splits and single-merges (pure model evidence)
      are applied, but the whole asymmetric family is recorded with
      ``action="deferred"`` and left untouched — run this BEFORE any
      element identification;
    - ``"only"``: only asymmetric verdicts act (blend/single verdicts
      are recorded with ``action="none"``); run this AFTER an element
      posterior exists, passing it as ``elements`` so the resonance
      gates are conditioned on species plausibly in the plasma instead
      of on any line in the database.

    ``segment_edges`` (same convention and default as
    ``PeakyFinder.fit_spectrum``) keeps the local noise estimate from
    bleeding across detector-segment junctions; pass ``()`` to disable.

    The returned ``profile``/``residual_data`` represent every row as a
    symmetric Voigt of its tabulated (observed) area, so a merged
    self-absorbed line shows a core-shaped residual there — the decision
    record's ``params_asym`` holds its true attenuated shape.

    Returns ``(new_fit_dict, decisions)``.
    """
    from alibz.peaky_finder import PeakyFinder

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bg = np.asarray(fit_dict.get("background", np.zeros_like(y)), dtype=float)
    y_bgsub = y - bg
    peaks = np.atleast_2d(np.asarray(fit_dict["sorted_parameter_array"],
                                     dtype=float)).copy()
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
    model_total = _multi_voigt(x, np.ravel(peaks[:, :4]))
    fwhm_all = _voigt_width(np.maximum(peaks[:, 2], 1e-6),
                            np.maximum(peaks[:, 3], 1e-6))
    wmed = _weighted_median(fwhm_all, np.maximum(peaks[:, 0], 1e-12))

    decisions = []
    drop, add = set(), []
    for kind, indices in _feature_candidates(
            x, y_bgsub, peaks, model_total, noise, snr_min=snr_min):
        if any(int(i) in drop for i in indices):
            continue  # consumed by an earlier feature's absorption
        sub = peaks[list(indices)]
        mu0, _, span = _feature_window(sub)
        feat_area = float(np.sum(sub[:, 0]))
        absorb = ((np.abs(peaks[:, 1] - mu0) <= span + fwhm_all)
                  & (fwhm_all > 1.5 * wmed)
                  & (peaks[:, 0] <= 0.25 * feat_area))
        absorb[list(indices)] = False
        absorb[[k for k in np.flatnonzero(absorb) if int(k) in drop]] = False
        absorbed = [int(k) for k in np.flatnonzero(absorb)]
        own_idx = list(indices) + absorbed
        own = _multi_voigt(x, np.ravel(peaks[own_idx, :4]))
        dec = classify_feature(x, y_bgsub, peaks, indices, noise,
                               db=db, elements=elements, shift_nm=shift_nm,
                               model_others=model_total - own,
                               extra_area=float(peaks[absorbed, 0].sum()))
        if dec is None:
            continue
        dec["kind"] = kind
        dec["action"] = "none"
        dec["absorbed"] = tuple(absorbed)
        v = dec["verdict"]
        if dec["noise_rescale"] > s2_action_max:
            # the window fit explains nothing at the stated noise level:
            # model preference is not trustworthy enough to act on
            decisions.append(dec)
            continue
        if v in ("blend", "blend-unassigned") and dec["params_blend"] is not None:
            if asymmetric == "only":
                decisions.append(dec)   # data-only actions ran in the
                continue                # deferred pass; record, don't act
            dec["action"] = "split"
            pB = dec["params_blend"]
            for i in list(indices) + absorbed:
                drop.add(int(i))
            add.append(pB[:4])
            add.append(pB[4:])
        elif v.startswith("asymmetric") and dec["params_asym"] is not None:
            pA = dec["params_asym"]
            pS = dec["params_single"]
            dec["emission_area"] = float(pA[0])
            dec["tau_a"] = float(pA[4])
            dec["delta_nm"] = float(pA[5])
            # observed (attenuated) area = the faithful symmetric fit's
            # area; set for every asymmetric record (deferred / tagged /
            # non-resonant) so consumers can read it uniformly
            dec["observed_area"] = (float(pS[0]) if pS is not None
                                    else float("nan"))
            if asymmetric == "defer":
                dec["action"] = "deferred"
            elif v == "asymmetric" and pS is not None:
                # NON-DESTRUCTIVE self-absorption TAG.  Keep the
                # data-faithful symmetric fit of the (attenuated) profile
                # — the same single-Voigt fit the `single` merge stores —
                # and record the SA metadata (unattenuated emission area,
                # tau) for the downstream growth-curve amplitude recovery.
                # Storing the SA-MODEL proxy instead (its narrow absorbed
                # core width carrying the wing-inflated area) overshoots
                # the core by up to +19 sigma on real data and drops
                # neighbours that resurface as 40 sigma residuals; the
                # symmetric envelope fits the observed line and leaves at
                # most a shallow (negative) core dip where the true line is
                # genuinely flat-topped.  The stored area IS the observed
                # (attenuated) emission, so the recovery factor
                # emission_area/observed_area acts on the right baseline.
                dec["action"] = "sa-tag"
                dec["observed_area"] = float(pS[0])
                for i in list(indices) + absorbed:
                    drop.add(int(i))
                add.append(list(pS[:4]))
        elif (v == "single" and kind == "pair"
              and dec["params_single"] is not None):
            if asymmetric == "only":
                decisions.append(dec)
                continue
            dec["action"] = "merge"
            pS = dec["params_single"]
            for i in list(indices) + absorbed:
                drop.add(int(i))
            add.append(list(pS[:4]))
        decisions.append(dec)

    if drop or add:
        keep = np.array([k not in drop for k in range(peaks.shape[0])])
        peaks = np.vstack([peaks[keep][:, :4], np.array(add)]) if add else peaks[keep][:, :4]
        peaks = peaks[np.argsort(peaks[:, 0])[::-1]]
        new_fit = dict(fit_dict)
        new_fit["sorted_parameter_array"] = peaks
        new_fit["profile"] = _multi_voigt(x, np.ravel(peaks[:, :4]))
        new_fit["residual_data"] = y_bgsub - new_fit["profile"]
        # first-pass bookkeeping keyed to the OLD component list would
        # silently resurrect dropped phantoms if rebuilt from — remove
        for stale in ("peak_dictionary", "spectrum_dictionary"):
            new_fit.pop(stale, None)
        return new_fit, decisions
    return fit_dict, decisions
