"""Recovery of the spectrometer's native (pre-resampling) wavelength grid.

The vendor CSV export (SciAps Z-series) is a UNIFORM 1/30 nm grid (180.0 to
961.0 nm, 23431 points) produced by resampling three independent detector
segments -- UV (180-365 nm), VIS (365-620 nm), NIR (620-961 nm) -- from
their native, coarser, unevenly pitched pixel grids with an interpolating
not-a-knot cubic spline.  The export is therefore a LINEAR map
``y_export = S @ y_native`` of the native samples (``export_kernel_matrix``),
and with the knots known the native samples follow from one sparse
least-squares solve, exactly (relative residual ~1e-11, the solver floor).

THE NATIVE GRID IS HARDWARE.  Each segment's pixel wavelengths follow a
smooth cubic in pixel index, ``INSTRUMENT_CALIBRATION`` (the exact
solution of the full pitch/phase search on REE_01.csv: ~1,950 / 1,970 /
1,820 pixels, pitch 0.096 -> 0.079, 0.142 -> 0.113, 0.198 -> 0.156 nm for
UV / VIS / NIR).  From one acquisition to the next only the vendor's
wavelength calibration changes: a shift of a few tenths of a native pixel
plus a stretch of tens to hundreds of ppm, with measurable curvature in the
NIR.  ``recover_native_grid`` (default ``mode="calibrated"``) therefore
measures, per spectrum and per segment, a cubic correction in pixel index
to the calibration -- each of 13 short windows yields the local shift
(coarse scan over one pitch + bounded polish; the shift is only defined
modulo the pitch, so the windows are unwrapped), a robust polynomial is fit
through them, and the coefficients are polished jointly on the window
residuals with Levenberg-Marquardt -- and then verifies exactness with one
full-segment solve.  It is deterministic (no random starts) and takes
seconds.  On the four bundled samples every segment reaches the solver
floor (see ``tests/test_native_grid.py``).

``mode="search"`` (or a calibrated segment that fails the exact bar with
``fallback=True``) runs the full per-window pitch/phase search that
produced the calibration (minutes per segment; see
``_phase_search_segment``); ``calibrate_instrument`` packages that search
into a calibration table for another instrument.

Edges: the vendor spline is off-model within a few nm of a zero-padded
dead edge (UV: on-model 192-361 nm of a 187.7-363.9 live range), so the
exactness check (``info["segments"][name]["relres"]``) excludes the
calibrated ``edge_exclude_nm`` margins, widened automatically while the
binned residual at an edge stays high; ``relres_full`` includes them.  The
returned native grid still covers the whole live range.

Kernel identification evidence (interpolating cubic spline vs PCHIP, Akima,
Keys cubic convolution: 2.4e-15 vs 1.3e-2 on an exact-knot synthetic; 200x
on real NIR data) and the derivation history are in
reports/2026-09-20-native-grid-recovery.md and
reports/2026-09-20-alibz-review.md.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, minimize, minimize_scalar
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

#: Name of the interpolation kernel identified for the vendor resampling.
#: See the module docstring; PCHIP, Akima, and Keys cubic convolution were
#: all tested and rejected (2-10x worse reconstruction residual, or worse
#: even with the exact correct native knots -- see the report).
KERNEL_NAME = "cubic_notaknot"

#: Relative reconstruction residual below which a segment's recovery is
#: considered ``exact`` for the purposes of ``info["segments"][name]["exact"]``.
EXACT_RELRES_THRESHOLD = 1e-6

#: One export pixel, in nm.
_PX = 1.0 / 30.0

#: Per-segment (lo, hi, pitch_lo_px, pitch_hi_px). Pitch bounds are the
#: physically-plausible range from the corpus power-spectrum estimate /
#: d4-bracket evidence (report §2); the phase/pitch search never leaves
#: this range (an UNBOUNDED Nelder-Mead polish was observed to wander to a
#: degenerate ~0.3 px pitch -- report + orchestrator validation).
_SEGMENT_RANGE = {
    "UV": dict(lo=180.0, hi=365.0, p_lo=2.1, p_hi=3.2),
    "VIS": dict(lo=365.0, hi=620.0, p_lo=3.2, p_hi=4.8),
    "NIR": dict(lo=620.0, hi=961.0, p_lo=4.6, p_hi=6.2),
}

#: Window search parameters (nm unless noted). Matches the orchestrator's
#: validated reference implementation (3 nm step, 50% overlap with the
#: 6 nm window) after a follow-up verification run showed the previous
#: 6 nm non-overlapping step (half the window density) was one of several
#: shortcuts costing real accuracy -- see the report.
_WIN_NM = 6.0
_STEP_NM = 3.0
_EDGE_MARGIN_NM = 1.0
_EDGE_EXCLUDE_NM = 2.0  # excluded from the fit/relres objective (module docstring)

#: Coarse-then-polish grid for the FIRST window of a segment (no warm
#: start available yet); subsequent windows also try a smaller
#: warm-started local grid (see ``_LOCAL_*`` below) but ALWAYS run this
#: full-range grid too and keep whichever is better -- see the module
#: docstring for why a local-only search was found unsafe.
_COARSE_DP_PX = 0.02
_COARSE_DPHI_PX = 0.1
_LOCAL_P_HALFWIDTH_PX = 0.10
_LOCAL_DP_PX = 0.02
_LOCAL_PHI_HALFWIDTH_PX = 0.35
_LOCAL_DPHI_PX = 0.07

_WINDOW_POLISH_MAXITER = 40
#: Matches the orchestrator's validated reference implementation's global
#: refine budget exactly (maxiter=400, xatol=1e-9, fatol=1e-12 below) --
#: reduced budgets (100-180) were found to leave several (file, segment)
#: pairs 4-8 orders of magnitude short of the reference's ~1e-12.
_GLOBAL_REFINE_MAXITER = 400


def export_kernel_matrix(x_native, x_export, threshold=1e-10):
    """Sparse forward operator ``S`` such that ``y_export ~= S @ y_native``.

    Builds the not-a-knot interpolating cubic spline through ``x_native``
    (the identified vendor resampling kernel, see the module docstring) and
    evaluates it at ``x_export``.  ``S`` is linear in the native ordinate
    values, so a forward model fit on the native grid can be pushed through
    the SAME kernel with ``S @ y_native`` to compare against vendor-exported
    data.

    The interpolating cubic spline is technically dense (a global
    tridiagonal solve couples every knot), but its weights decay
    geometrically away from each evaluation point (>90% of entries are
    below 1e-6 for representative native/export grids); entries smaller
    than ``threshold`` are dropped so the returned matrix is genuinely
    sparse without materially changing ``S @ y_native``.

    Parameters
    ----------
    x_native : array_like, shape (n_native,)
        Strictly increasing native (pre-resampling) wavelengths [nm].
    x_export : array_like, shape (n_export,)
        Wavelengths [nm] to evaluate the interpolant at.  Must lie within
        ``[x_native[0], x_native[-1]]``.
    threshold : float, optional
        Absolute weight below which an ``S`` entry is zeroed before sparse
        conversion.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n_export, n_native)
    """
    x_native = np.asarray(x_native, dtype=float)
    x_export = np.asarray(x_export, dtype=float)
    if x_native.ndim != 1 or x_native.size < 4:
        raise ValueError("x_native must be 1-D with at least 4 points")
    if np.any(np.diff(x_native) <= 0):
        raise ValueError("x_native must be strictly increasing")
    if x_export.size and (x_export.min() < x_native[0] or x_export.max() > x_native[-1]):
        raise ValueError(
            "x_export must lie within [x_native[0], x_native[-1]] "
            f"({x_native[0]!r}, {x_native[-1]!r}); got range "
            f"[{x_export.min()!r}, {x_export.max()!r}]"
        )
    identity = np.eye(x_native.size)
    dense = CubicSpline(x_native, identity, axis=0, bc_type="not-a-knot")(x_export)
    dense[np.abs(dense) < threshold] = 0.0
    return csr_matrix(dense)


def _relres_dense(xk, xe, ye):
    """Dense lstsq relres for a SMALL knot set (one search window)."""
    M = CubicSpline(xk, np.eye(xk.size), axis=0)(xe)
    sol, *_ = np.linalg.lstsq(M, ye, rcond=None)
    r = ye - M @ sol
    denom = np.linalg.norm(ye)
    return float(np.linalg.norm(r) / denom) if denom > 0 else 0.0


def _resvec_dense(xk, xe, ye):
    """Normalised residual VECTOR of the dense window inverse (for LM)."""
    M = CubicSpline(xk, np.eye(xk.size), axis=0)(xe)
    sol, *_ = np.linalg.lstsq(M, ye, rcond=None)
    denom = np.linalg.norm(ye)
    return (ye - M @ sol) / denom if denom > 0 else np.zeros_like(ye)


def _relres_sparse(xk, xe, ye, threshold=1e-10):
    """Sparse lsqr relres for a LARGE knot set (a full segment)."""
    S = export_kernel_matrix(xk, xe, threshold=threshold)
    sol = lsqr(S, ye, atol=1e-12, btol=1e-12, iter_lim=8000)[0]
    denom = np.linalg.norm(ye)
    return float(np.linalg.norm(ye - S @ sol) / denom) if denom > 0 else 0.0


def _knots_for(x0, phi, p, span):
    k = np.arange(-2, int(np.ceil((span - phi) / p)) + 3)
    return x0 + phi + k * p


def _window_search(xe, ye, p_lo_px, p_hi_px, seed=None):
    """Best (phase, pitch) for one window; returns (knots, pitch_nm, phase_nm, relres).

    ``seed`` is ``(phase_nm, pitch_nm)`` from the previous window, or
    ``None`` for the first window of a segment.  With a seed, only a small
    neighbourhood around it is grid-searched (the dominant speedup vs. a
    from-scratch full-range search every window); without one, the full
    ``[p_lo_px, p_hi_px]`` range is coarsely scanned once.
    """
    x0 = xe[0]
    span = xe[-1] - xe[0]

    def relres_for(phi, p):
        return _relres_dense(_knots_for(x0, phi, p, span), xe, ye)

    def full_range_best():
        best = (None, None, np.inf)
        for p_px in np.arange(p_lo_px, p_hi_px + 1e-9, _COARSE_DP_PX):
            p = p_px * _PX
            for phi_px in np.arange(0.0, p_px, _COARSE_DPHI_PX):
                phi = phi_px * _PX
                rr = relres_for(phi, p)
                if rr < best[2]:
                    best = (phi, p, rr)
        return best

    def local_best(seed_phi_nm, seed_p_nm):
        best = (None, None, np.inf)
        p_center = np.clip(seed_p_nm / _PX, p_lo_px, p_hi_px)
        p_vals = np.clip(
            np.arange(p_center - _LOCAL_P_HALFWIDTH_PX,
                      p_center + _LOCAL_P_HALFWIDTH_PX + 1e-9, _LOCAL_DP_PX),
            p_lo_px, p_hi_px,
        )
        phi_center_px = seed_phi_nm / _PX
        for p_px in np.unique(p_vals):
            p = p_px * _PX
            phi_center_mod = phi_center_px % p_px
            phi_px_vals = np.arange(
                phi_center_mod - _LOCAL_PHI_HALFWIDTH_PX,
                phi_center_mod + _LOCAL_PHI_HALFWIDTH_PX + 1e-9, _LOCAL_DPHI_PX,
            ) % p_px
            for phi_px in phi_px_vals:
                phi = phi_px * _PX
                rr = relres_for(phi, p)
                if rr < best[2]:
                    best = (phi, p, rr)
        return best

    def polish(phi0, p0):
        def obj(q):
            phi, p = q
            if p <= 0 or p < p_lo_px * _PX or p > p_hi_px * _PX:
                return 1e9
            return relres_for(phi, p)

        result = minimize(
            obj, [phi0, p0], method="Nelder-Mead",
            options=dict(xatol=1e-6, fatol=1e-9, maxiter=_WINDOW_POLISH_MAXITER),
        )
        phi, p = result.x
        p = float(np.clip(p, p_lo_px * _PX, p_hi_px * _PX))
        return phi, p, float(result.fun)

    # ALWAYS run the full-range coarse grid: on weak-signal windows (UV
    # especially) a warm-started LOCAL-only grid was found to alias to a
    # wrong (half- or double-pitch) basin often enough to corrupt the
    # whole chain (13/29 UV windows usable, rms 5.7 px) even with a
    # railing-based fallback check -- not safe. The seed is still useful
    # (below) as an extra, finer candidate near the expected continuation,
    # but never REPLACES the full-range safety net; this costs the full
    # coarse-grid pass every window (no window-count-based speedup versus
    # a from-scratch search) but stays correct, which matters more than
    # the runtime target (module docstring; see the report for the
    # runtime actually achieved).
    phi0, p0, r0 = full_range_best()
    if seed is not None:
        phi0l, p0l, r0l = local_best(*seed)
        if r0l < r0:
            phi0, p0 = phi0l, p0l
    phi, p, rr = polish(phi0, p0)

    xk = _knots_for(x0, phi, p, span)
    inside = (xk >= xe[0]) & (xk <= xe[-1])
    return xk[inside], p, phi, rr


def _trim_dead_edges(xs, ys):
    """Trim leading/trailing constant runs (zero-padded detector edges)."""
    a = 0
    while a < ys.size - 1 and ys[a + 1] == ys[0]:
        a += 1
    b = ys.size - 1
    while b > 0 and ys[b - 1] == ys[-1]:
        b -= 1
    if a > 3 or b < ys.size - 4:
        return xs[a + 1:b], ys[a + 1:b]
    return xs, ys


def _phase_search_segment(x_seg, y_seg, seg_lo, seg_hi, p_lo_px, p_hi_px):
    """Full per-segment local-pitch/phase-search dispersion recovery.

    Returns a dict with ``coeffs``, ``center`` (0.0 -- the polynomial is
    fit directly in absolute native-index units here, unlike the older
    bracket/affine code path), ``relres_inner`` (edge-excluded, module
    docstring), ``relres_full``, ``n_windows_used``, ``n_knots``,
    ``polyfit_rms_px``, ``pitch_start``, ``pitch_end`` [nm].
    """
    margin_mask = (x_seg >= seg_lo + _EDGE_MARGIN_NM) & (x_seg < seg_hi - _EDGE_MARGIN_NM)
    xs, ys = x_seg[margin_mask], y_seg[margin_mask]
    xs, ys = _trim_dead_edges(xs, ys)

    starts = np.arange(xs[0], xs[-1] - _WIN_NM, _STEP_NM)
    knots_all, pitches, centers = [], [], []
    seed = None
    for s in starts:
        w = (xs >= s) & (xs < s + _WIN_NM)
        xe, ye = xs[w], ys[w]
        if xe.size < 10 or np.all(ye == ye[0]):
            knots_all.append(np.array([])); pitches.append(np.nan); centers.append(s + 0.5 * _WIN_NM)
            continue
        xk, p, phi, _ = _window_search(xe, ye, p_lo_px, p_hi_px, seed=seed)
        knots_all.append(xk); pitches.append(p / _PX); centers.append(s + 0.5 * _WIN_NM)
        seed = (phi, p)

    centers = np.array(centers)
    pit = np.array(pitches, dtype=float)
    good = np.isfinite(pit) & (pit > p_lo_px + 0.02) & (pit < p_hi_px - 0.02)
    if good.sum() < 3:
        raise ValueError("phase search: too few usable windows to fit a dispersion")
    for _ in range(4):
        cp = np.polyfit(centers[good], pit[good], 2)
        trend = np.polyval(cp, centers)
        dev = np.abs(pit - trend) / np.where(trend != 0, trend, 1.0)
        new_good = np.isfinite(pit) & (dev < 0.03)
        if new_good.sum() < 3:
            break
        good = new_good
    n_windows_used = int(good.sum())

    def local_pitch_px(pos):
        return float(np.interp(pos, centers[good], pit[good]))

    kept_knots = [k for k, g in zip(knots_all, good) if g and k.size]
    if not kept_knots:
        raise ValueError("phase search: no surviving window knots to chain")
    allk = np.sort(np.concatenate(kept_knots))
    merged = [allk[0]]
    for v in allk[1:]:
        if (v - merged[-1]) / _PX < 0.5 * local_pitch_px(v):
            merged[-1] = 0.5 * (merged[-1] + v)
        else:
            merged.append(v)
    merged = np.array(merged)

    gaps = np.diff(merged) / _PX
    idx = [0.0]
    for g, pos in zip(gaps, merged[1:]):
        idx.append(idx[-1] + max(1, int(round(g / local_pitch_px(pos)))))
    idx = np.array(idx)

    keep = np.ones(idx.size, dtype=bool)
    coeffs = np.polyfit(idx, merged, 3)
    for _ in range(6):
        coeffs = np.polyfit(idx[keep], merged[keep], 3)
        resid_px = (merged - np.polyval(coeffs, idx)) / _PX
        mad = np.median(np.abs(resid_px[keep])) if keep.any() else 0.0
        new_keep = np.abs(resid_px) < max(3.0 * 1.4826 * mad, 0.5)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    polyfit_rms_px = float(np.std(resid_px[keep])) if keep.any() else float("nan")

    # Generous padding beyond the chained-knot index range so the native
    # grid brackets xs (incl. the un-excluded edges) for every candidate
    # polynomial tried during refinement, not just the final one.
    n_ext = np.arange(-20, idx[-1] + 21)
    inner_mask = (xs >= xs[0] + _EDGE_EXCLUDE_NM) & (xs <= xs[-1] - _EDGE_EXCLUDE_NM)
    xs_inner, ys_inner = xs[inner_mask], ys[inner_mask]
    # NO subsampling: an earlier fixed-stride subsample made this objective
    # UNDERDETERMINED for UV (more native knots than subsampled export
    # points), giving a degenerate near-perfect fit (8e-13) that did not
    # generalize (7.7% on the full data). A cap that kept the objective
    # determined fixed that specific failure, but a follow-up orchestrator
    # verification run (independently reproducing this module's own
    # per-file numbers almost exactly) showed the reference (unsubsampled)
    # implementation still reaches 4-8 orders of magnitude better relres
    # (~1e-12) than this capped-stride version (~1e-4 to 1e-5) on several
    # (file, segment) pairs -- i.e. subsampling AT ALL, even safely capped,
    # was still costing real accuracy, not just the degenerate case. Full
    # resolution is used here; see the module docstring for the resulting
    # runtime (~10 min/spectrum, accepted -- speed is a follow-up).

    def _clip_to_domain(xk, xe, ye):
        m = (xe >= xk[0]) & (xe <= xk[-1])
        return xe[m], ye[m]

    def objective(cc):
        xk = np.polyval(cc, n_ext)
        if np.any(np.diff(xk) <= 0):
            return 1e9
        xe, ye = _clip_to_domain(xk, xs_inner, ys_inner)
        if xe.size < 10:
            return 1e9
        return _relres_sparse(xk, xe, ye)

    scale = np.abs(coeffs) + 1e-12
    result = minimize(
        lambda q: objective(q * scale), coeffs / scale, method="Nelder-Mead",
        options=dict(xatol=1e-9, fatol=1e-12, maxiter=_GLOBAL_REFINE_MAXITER),
    )
    refined_coeffs = result.x * scale
    xn_refined = np.polyval(refined_coeffs, n_ext)
    if np.any(np.diff(xn_refined) <= 0):
        refined_coeffs = coeffs  # refine diverged; fall back to the polyfit

    xn_final = np.polyval(refined_coeffs, n_ext)
    xs_inner_c, ys_inner_c = _clip_to_domain(xn_final, xs_inner, ys_inner)
    xs_full_c, ys_full_c = _clip_to_domain(xn_final, xs, ys)
    relres_inner = _relres_sparse(xn_final, xs_inner_c, ys_inner_c)
    relres_full = _relres_sparse(xn_final, xs_full_c, ys_full_c)
    der = np.polyder(refined_coeffs)

    return dict(
        coeffs=tuple(refined_coeffs),
        center=0.0,
        n_ext=n_ext,
        relres_inner=float(relres_inner),
        relres_full=float(relres_full),
        n_windows_used=n_windows_used,
        n_windows_total=int(len(starts)),
        n_knots=int(merged.size),
        polyfit_rms_px=polyfit_rms_px,
        pitch_start=float(np.polyval(der, n_ext[0])),
        pitch_end=float(np.polyval(der, n_ext[-1])),
    )



#: Instrument dispersion calibration (SciAps Z-series export, three
#: detector segments).  ``wavelength(n) = polyval(coeffs, n)`` for native
#: pixel index ``n`` in ``[n0, n1]`` -- the EXACT solution of the full
#: pitch/phase search on REE_01.csv (inversion residual 6e-10 / 1.4e-12 /
#: 4.5e-12 for UV / VIS / NIR).  The pixel grid is hardware: on every other
#: spectrum the same polynomial reproduces the export to the solver floor
#: after a per-spectrum AFFINE wavelength correction (a shift of a few
#: tenths of a native pixel plus a stretch of tens to hundreds of ppm --
#: the vendor's per-acquisition wavelength calibration), which is what
#: :func:`recover_native_grid` solves for in its default ``"calibrated"``
#: mode.  ``edge_exclude_nm`` is the (short, long) wavelength margin inside
#: the live data that is OFF-model in the vendor export (the spline does
#: not follow the knot model within a few nm of a dead pad; measured
#: 192-361 nm on-model for the UV segment whose live range is 187.7-363.9).
#: Re-derive for another instrument with :func:`calibrate_instrument`.
INSTRUMENT_CALIBRATION = {
    "UV": dict(coeffs=(-4.94282461475568e-10, -3.006465856248076e-06,
                       0.09594915874617632, 191.0634922573234),
               n0=-6, n1=1942, edge_exclude_nm=(5.0, 3.5)),
    "VIS": dict(coeffs=(-6.529979968625204e-10, -5.380453878473395e-06,
                        0.14161441698946808, 366.0046301078284),
                n0=-6, n1=1965, edge_exclude_nm=(2.0, 2.0)),
    "NIR": dict(coeffs=(-9.01320397050826e-10, -9.093365510156847e-06,
                        0.19819231318324787, 621.1984435584822),
                n0=-6, n1=1817, edge_exclude_nm=(2.0, 2.0)),
}

#: Calibrated mode: number of 6 nm windows used to measure the affine
#: correction, and the coarse shift-scan step as a fraction of the pitch.
_AFFINE_N_WINDOWS = 13
_AFFINE_SCAN_STEP = 0.02
#: Degree of the per-spectrum correction polynomial (in native pixel index).
_CORRECTION_DEGREE = 3
#: Levenberg-Marquardt evaluation budget for the joint polish of the correction coefficients on
#: the window residuals (0 disables the polish; the robust window fit alone
#: is then used).
_CORRECTION_POLISH_MAXITER = 200
#: A window whose shift deviates from the robust fit by more than this
#: (native px) is dropped (strong saturated lines, dead pixels).
_AFFINE_OUTLIER_PX = 0.15


def _calibrated_knots(calib):
    n = np.arange(calib["n0"], calib["n1"] + 1, dtype=float)
    return np.polyval(calib["coeffs"], n)


def _corrected_knots(knots0, coef, n_ref):
    """Calibration knots plus a per-spectrum polynomial correction in native
    pixel index: k'_n = k_n + sum_j coef[j] (n - n_ref)^j.  A cubic here
    re-parametrises the cubic dispersion exactly, so any per-acquisition
    recalibration of the vendor polynomial is representable."""
    n = np.arange(knots0.size, dtype=float) - n_ref
    return knots0 + np.polyval(coef[::-1], n)


def _affine_from_windows(xs, ys, knots0, n_win=_AFFINE_N_WINDOWS, win_nm=_WIN_NM,
                         degree=_CORRECTION_DEGREE):
    """Measure the per-spectrum correction of the calibration dispersion.

    Each window's shift is found on its own by a coarse scan over one
    pitch and a bounded polish (the residual is periodic in the shift, so
    a window determines it only modulo the local pitch); the windows are
    unwrapped for continuity, a robust polynomial of ``degree`` in native
    pixel index is fit through (window index, shift) with outliers dropped,
    and its coefficients are polished jointly on the sum of the window
    residuals.  Deterministic: no random starts.
    """
    pitch_local = np.diff(knots0)
    k_mid = 0.5 * (knots0[1:] + knots0[:-1])
    n_of_x = lambda xv: np.interp(xv, knots0, np.arange(knots0.size, dtype=float))

    centers = np.linspace(xs[0] + win_nm, xs[-1] - win_nm, n_win)
    wins = []
    for cw in centers:
        w = (xs >= cw - win_nm / 2) & (xs < cw + win_nm / 2)
        if np.count_nonzero(w) >= 20 and not np.all(ys[w] == ys[w][0]):
            wins.append((xs[w], ys[w], float(cw)))
    if len(wins) < degree + 2:
        raise ValueError("too few usable windows for the calibration correction search")
    n_ref = float(n_of_x(np.mean([w[2] for w in wins])))

    def win_res(xe, ye, coef):
        k = _corrected_knots(knots0, coef, n_ref)
        p = float(np.interp(xe[0], k_mid, pitch_local))
        kk = k[(k > xe[0] - 3 * p) & (k < xe[-1] + 3 * p)]
        if kk.size < 6:
            return 1.0
        return _relres_dense(kk, xe, ye)

    deltas, resids = [], []
    for xe, ye, cw in wins:
        p = float(np.interp(cw, k_mid, pitch_local))
        grid = np.arange(-0.5 * p, 0.5 * p, _AFFINE_SCAN_STEP * p)
        rr = np.array([win_res(xe, ye, [d]) for d in grid])
        d0 = float(grid[int(np.argmin(rr))])
        res = minimize_scalar(lambda d: win_res(xe, ye, [d]), bounds=(d0 - 0.1 * p, d0 + 0.1 * p),
                              method="bounded", options=dict(xatol=1e-10))
        deltas.append(float(res.x))
        resids.append(float(res.fun))
    deltas = np.array(deltas)
    cw = np.array([w[2] for w in wins])
    for i in range(1, deltas.size):           # unwrap modulo the local pitch
        p = float(np.interp(cw[i], k_mid, pitch_local))
        while deltas[i] - deltas[i - 1] > 0.5 * p:
            deltas[i] -= p
        while deltas[i] - deltas[i - 1] < -0.5 * p:
            deltas[i] += p
    u = n_of_x(cw) - n_ref
    A = np.vstack([u ** j for j in range(degree + 1)]).T
    keep = np.ones(cw.size, dtype=bool)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[keep], deltas[keep], rcond=None)
        dev = np.abs(deltas - A @ coef)
        p_px = np.interp(cw, k_mid, pitch_local) / _PX
        new_keep = dev / _PX < _AFFINE_OUTLIER_PX * (p_px / p_px.mean())
        if new_keep.sum() < degree + 2 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    coef, *_ = np.linalg.lstsq(A[keep], deltas[keep], rcond=None)
    used = [w for w, k in zip(wins, keep) if k]

    def total(q):
        return sum(win_res(xe, ye, q) for xe, ye, _ in used)

    if _CORRECTION_POLISH_MAXITER > 0:
        # Joint polish on the concatenated window residual vectors with
        # Levenberg-Marquardt: the residual is smooth and near-quadratic in
        # the coefficients, so this converges in ~10 iterations where
        # Nelder-Mead on the scalar sum needed ~2500 evaluations.
        def resvec(q):
            k = _corrected_knots(knots0, q * scale, n_ref)
            parts = []
            for xe, ye, _ in used:
                p = float(np.interp(xe[0], k_mid, pitch_local))
                kk = k[(k > xe[0] - 3 * p) & (k < xe[-1] + 3 * p)]
                parts.append(_resvec_dense(kk, xe, ye) if kk.size >= 6 else np.ones(xe.size))
            return np.concatenate(parts)

        scale = np.array([max(abs(c), 1e-9 * _PX ** j) for j, c in enumerate(coef)])
        res = least_squares(resvec, coef / scale, method="lm", xtol=1e-14, ftol=1e-14,
                            gtol=1e-14, max_nfev=_CORRECTION_POLISH_MAXITER)
        coef = res.x * scale
    return dict(coef=coef, n_ref=n_ref,
                window_centers=cw, window_deltas=deltas, window_relres=np.array(resids),
                windows_used=int(keep.sum()), windows_total=int(cw.size))


def _on_model_mask(xn, xs, ys, edge_exclude_nm, threshold=1e-10):
    """Mask of export samples inside the on-model range.

    Starts from the calibrated ``edge_exclude_nm`` margins and widens them
    while the 1 nm-binned residual of the full-segment inverse at an edge
    exceeds 1e-4 of the segment's rms (the vendor spline is off-model near
    dead pads).
    """
    lo_ex, hi_ex = edge_exclude_nm
    xs_c, ys_c = xs, ys
    dom = (xs_c >= xn[0]) & (xs_c <= xn[-1])
    xs_c, ys_c = xs_c[dom], ys_c[dom]
    S = export_kernel_matrix(xn, xs_c, threshold=threshold)
    sol = lsqr(S, ys_c, atol=1e-12, btol=1e-12, iter_lim=8000)[0]
    r = np.abs(ys_c - S @ sol)
    scale = max(float(np.sqrt(np.mean(ys_c ** 2))), 1e-300)
    edges = np.arange(np.floor(xs_c[0]), np.ceil(xs_c[-1]) + 1.0, 1.0)
    bad = np.array([np.any(r[(xs_c >= a) & (xs_c < b)] > 1e-4 * scale)
                    if np.any((xs_c >= a) & (xs_c < b)) else False
                    for a, b in zip(edges[:-1], edges[1:])])
    lo = xs_c[0] + lo_ex
    i = 0
    while i < bad.size and (edges[i + 1] <= lo or bad[i]):
        lo = max(lo, edges[i + 1]); i += 1
    hi = xs_c[-1] - hi_ex
    j = bad.size - 1
    while j >= 0 and (edges[j] >= hi or bad[j]):
        hi = min(hi, edges[j]); j -= 1
    return (xs >= lo) & (xs <= hi), (float(lo), float(hi))


def _calibrated_segment(x_seg, y_seg, name, calib, threshold=1e-10):
    """Fast path: fixed dispersion + per-spectrum affine correction."""
    rng = _SEGMENT_RANGE[name]
    margin_mask = (x_seg >= rng["lo"] + _EDGE_MARGIN_NM) & (x_seg < rng["hi"] - _EDGE_MARGIN_NM)
    xs, ys = _trim_dead_edges(x_seg[margin_mask], y_seg[margin_mask])
    knots0 = _calibrated_knots(calib)
    aff = _affine_from_windows(xs, ys, knots0)
    xn = _corrected_knots(knots0, aff["coef"], aff["n_ref"])
    if np.any(np.diff(xn) <= 0):
        raise ValueError(f"{name}: affine-corrected calibration is non-monotone")
    inner, on_model = _on_model_mask(xn, xs, ys, calib["edge_exclude_nm"], threshold)
    xs_i, ys_i = xs[inner], ys[inner]
    relres_inner = _relres_sparse(xn, xs_i, ys_i, threshold) if xs_i.size else float("nan")
    dom = (xs >= xn[0]) & (xs <= xn[-1])
    relres_full = _relres_sparse(xn, xs[dom], ys[dom], threshold)
    der = np.polyder(np.asarray(calib["coeffs"], dtype=float))
    n_lo, n_hi = calib["n0"], calib["n1"]
    coef = np.asarray(aff["coef"], dtype=float)
    pitch0 = float(np.median(np.diff(knots0)))
    return dict(
        mode="calibrated",
        x_native=xn,
        correction_coef=tuple(coef.tolist()),
        correction_n_ref=aff["n_ref"],
        delta_px=coef[0] / _PX,
        stretch_ppm=(coef[1] / pitch0) * 1e6 if coef.size > 1 else 0.0,
        window_centers=aff["window_centers"],
        window_deltas_px=aff["window_deltas"] / _PX,
        window_relres=aff["window_relres"],
        windows_used=aff["windows_used"],
        windows_total=aff["windows_total"],
        on_model_range=on_model,
        relres_inner=float(relres_inner),
        relres_full=float(relres_full),
        pitch_start=float(xn[1] - xn[0]),
        pitch_end=float(xn[-1] - xn[-2]),
        n_knots=int(xn.size),
    )


def calibrate_instrument(x, y, segment_edges=(365.0, 620.0)):
    """Derive an ``INSTRUMENT_CALIBRATION``-style table from one spectrum.

    Runs the full per-window pitch/phase search (minutes per segment) and
    returns ``{segment: dict(coeffs, n0, n1, edge_exclude_nm, relres)}``.
    Use once per instrument; ``recover_native_grid`` then needs only the
    fast affine correction per spectrum.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = tuple(float(e) for e in segment_edges)
    bounds = [("UV", x.min(), edges[0]), ("VIS", edges[0], edges[1]), ("NIR", edges[1], x.max())]
    out = {}
    for name, lo, hi in bounds:
        m = (x >= lo) & (x <= hi)
        if np.count_nonzero(m) < 8 or hi - lo < 2 * _WIN_NM:
            continue
        rng = _SEGMENT_RANGE[name]
        fit = _phase_search_segment(x[m], y[m], rng["lo"], rng["hi"], rng["p_lo"], rng["p_hi"])
        out[name] = dict(coeffs=tuple(float(c) for c in fit["coeffs"]),
                         n0=int(fit["n_ext"][0]), n1=int(fit["n_ext"][-1]),
                         edge_exclude_nm=INSTRUMENT_CALIBRATION.get(name, {}).get("edge_exclude_nm", (2.0, 2.0)),
                         relres=float(fit["relres_inner"]))
    return out


def recover_native_grid(x, y, segment_edges=(365.0, 620.0), threshold=1e-10,
                        mode="calibrated", calibration=None, fallback=True):
    """Recover the instrument's native wavelength/intensity grid.

    Splits ``(x, y)`` into three detector segments at ``segment_edges`` and
    derives EACH segment's full dispersion polynomial FRESH from this
    spectrum via the local pitch/phase search described in the module
    docstring -- there is no fixed per-instrument calibration table; the
    vendor shifts the native dispersion per acquisition.  Each segment then
    solves the linear least-squares inverse of the not-a-knot cubic-spline
    resampling kernel for the native intensities.

    Parameters
    ----------
    x : array_like
        Vendor-exported wavelengths [nm], uniform 1/30 nm pitch, spanning
        (a subset of) 180.0-961.0 nm.
    y : array_like
        Vendor-exported intensities, same shape as ``x``.
    segment_edges : tuple of float, optional
        The two UV/VIS and VIS/NIR detector-segment boundaries [nm].  Only
        the default ``(365.0, 620.0)`` is currently supported (see
        :data:`_SEGMENT_RANGE`); other values raise ``ValueError``.
    threshold : float, optional
        Passed through to :func:`export_kernel_matrix`.

    Returns
    -------
    x_native : ndarray
        Recovered native wavelengths [nm], ascending, concatenated across
        segments.
    y_native : ndarray
        Recovered native intensities, same shape as ``x_native``.
    info : dict
        ``{"kernel": KERNEL_NAME, "exact": bool, "segments": {name:
        {"kernel", "coeffs", "n_native", "n_knots", "n_windows_used",
        "polyfit_rms_px", "pitch_start", "pitch_end", "relres_inner",
        "relres_full", "relres", "exact"}, ...}}`` -- one sub-dict per
        detector segment actually covered by ``x``.  ``relres`` is an
        alias for ``relres_inner`` (the edge-excluded residual used for the
        ``exact`` gate); ``exact`` is ``True`` iff ``relres_inner <
        EXACT_RELRES_THRESHOLD`` (1e-6). Top-level ``info["exact"]`` is the
        AND of all covered segments.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x, y must be 1-D with at least 2 points")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")

    edges = tuple(sorted(float(e) for e in segment_edges))
    if edges != (365.0, 620.0):
        raise ValueError(
            "recover_native_grid only supports segment_edges (365.0, 620.0); "
            f"got {edges}. See _SEGMENT_RANGE / the module docstring."
        )
    bounds = [
        ("UV", x.min(), min(edges[0], x.max())),
        ("VIS", max(edges[0], x.min()), min(edges[1], x.max())),
        ("NIR", max(edges[1], x.min()), x.max()),
    ]
    segments = {
        name: (x[(x >= lo) & (x <= hi)], y[(x >= lo) & (x <= hi)])
        for name, lo, hi in bounds if hi - lo >= 2 * _WIN_NM
    }
    segments = {name: (xs, ys) for name, (xs, ys) in segments.items() if xs.size >= 8}
    if not segments:
        raise ValueError("no calibrated detector segment overlaps the given x range")

    if mode not in ("calibrated", "search"):
        raise ValueError("mode must be 'calibrated' or 'search'")
    calibration = INSTRUMENT_CALIBRATION if calibration is None else calibration

    x_native_parts, y_native_parts = [], []
    seg_info = {}
    for name, (x_seg, y_seg) in segments.items():
        rng = _SEGMENT_RANGE[name]
        fit = None
        used_fallback = False
        if mode == "calibrated" and name in calibration:
            fit = _calibrated_segment(x_seg, y_seg, name, calibration[name], threshold)
            if fit["relres_inner"] >= EXACT_RELRES_THRESHOLD and fallback:
                fit = None
                used_fallback = True
        if fit is None:
            sfit = _phase_search_segment(x_seg, y_seg, rng["lo"], rng["hi"], rng["p_lo"], rng["p_hi"])
            n_lo = sfit["n_ext"][0] - 3
            n_hi = sfit["n_ext"][-1] + 3
            n_idx = np.arange(n_lo, n_hi + 1, dtype=float)
            xn = np.polyval(sfit["coeffs"], n_idx)
            fit = dict(mode="search", x_native=xn, coeffs=sfit["coeffs"],
                       relres_inner=sfit["relres_inner"], relres_full=sfit["relres_full"],
                       n_knots=sfit["n_knots"], windows_used=sfit["n_windows_used"],
                       windows_total=sfit["n_windows_total"], polyfit_rms_px=sfit["polyfit_rms_px"],
                       pitch_start=sfit["pitch_start"], pitch_end=sfit["pitch_end"],
                       delta_px=None, stretch_ppm=None, on_model_range=None)
            fit["_poly"] = (np.asarray(sfit["coeffs"], dtype=float), n_idx)
        x_native = np.asarray(fit["x_native"], dtype=float)
        if np.any(np.diff(x_native) <= 0):
            raise ValueError(f"{name}: recovered dispersion is non-monotone")
        # extend to fully bracket this spectrum's actual segment data
        if fit["mode"] == "search":
            coeffs, n_idx = fit["_poly"]
            while x_native[0] > x_seg[0]:
                x_native = np.concatenate(([np.polyval(coeffs, n_idx[0] - 1)], x_native))
                n_idx = np.concatenate(([n_idx[0] - 1], n_idx))
            while x_native[-1] < x_seg[-1]:
                x_native = np.concatenate((x_native, [np.polyval(coeffs, n_idx[-1] + 1)]))
                n_idx = np.concatenate((n_idx, [n_idx[-1] + 1]))
        else:
            p0 = x_native[1] - x_native[0]
            p1 = x_native[-1] - x_native[-2]
            while x_native[0] > x_seg[0]:
                x_native = np.concatenate(([x_native[0] - p0], x_native))
            while x_native[-1] < x_seg[-1]:
                x_native = np.concatenate((x_native, [x_native[-1] + p1]))
        fit["used_fallback"] = used_fallback

        S = export_kernel_matrix(x_native, x_seg, threshold=threshold)
        y_native = lsqr(S, y_seg, atol=1e-12, btol=1e-12, iter_lim=8000)[0]
        x_native_parts.append(x_native)
        y_native_parts.append(y_native)

        seg_info[name] = dict(
            kernel=KERNEL_NAME,
            mode=fit["mode"],
            used_fallback=bool(used_fallback),
            n_native=int(x_native.size),
            n_knots=fit["n_knots"],
            windows_used=fit["windows_used"],
            windows_total=fit["windows_total"],
            delta_px=fit.get("delta_px"),
            stretch_ppm=fit.get("stretch_ppm"),
            correction_coef=fit.get("correction_coef"),
            correction_n_ref=fit.get("correction_n_ref"),
            window_centers=fit.get("window_centers"),
            window_deltas_px=fit.get("window_deltas_px"),
            window_relres=fit.get("window_relres"),
            on_model_range=fit.get("on_model_range"),
            coeffs=fit.get("coeffs"),
            polyfit_rms_px=fit.get("polyfit_rms_px"),
            pitch_start=fit["pitch_start"],
            pitch_end=fit["pitch_end"],
            x_native_range=(float(x_native[0]), float(x_native[-1])),
            relres_inner=fit["relres_inner"],
            relres_full=fit["relres_full"],
            relres=fit["relres_inner"],
            exact=bool(fit["relres_inner"] < EXACT_RELRES_THRESHOLD),
        )

    x_native = np.concatenate(x_native_parts)
    y_native = np.concatenate(y_native_parts)
    order = np.argsort(x_native)
    x_native, y_native = x_native[order], y_native[order]

    info = dict(
        kernel=KERNEL_NAME,
        mode=mode,
        exact=all(s["exact"] for s in seg_info.values()),
        segments=seg_info,
    )
    return x_native, y_native, info
