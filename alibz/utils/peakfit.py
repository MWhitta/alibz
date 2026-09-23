"""Gaussian instrumental-profile sub-pixel peak fitting.

A line centre estimated by parabolic interpolation of the three samples around
the peak is biased when the line is under-sampled (the Z300 native FWHM is
~1.3-1.8 px) and gives no principled uncertainty.  This module fits the local
peak as a single Gaussian at the instrument width plus a linear baseline by
weighted nonlinear least squares, returning the centre with a covariance-based
1-sigma, the fitted width, chi-squared/nu, an SNR, and a blend flag (with a
two-Gaussian fallback for the nearest pair).

Widths are carried in *pixels* (a detector property, roughly segment-constant)
and converted to nm through the LOCAL pitch, because the native grid pitch
varies within a segment.  Deterministic; numpy + scipy only; never raises on
bad input (returns ``{"ok": False, "reason": ...}``).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

SEGMENT_EDGES = (365.0, 620.0)
SEGMENT_NAMES = ("UV", "VIS", "NIR")

#: Default instrument Gaussian sigma per segment [pixels], from the measured
#: FWHM lower envelope: UV/VIS ~1.3-1.5 px, Ar I NIR ~1.5-1.8 px; sigma =
#: FWHM / (2*sqrt(2 ln2)) = FWHM / 2.35482.
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
DEFAULT_PROFILE_PX = {"UV": 1.4 * _FWHM_TO_SIGMA,
                      "VIS": 1.4 * _FWHM_TO_SIGMA,
                      "NIR": 1.65 * _FWHM_TO_SIGMA}

#: A pitch jump larger than this multiple of the local median marks a detector
#: segment boundary / the NIR gap; fitting windows are clipped there.
_GAP_FACTOR = 2.5


def _segment_name(wavelength):
    return SEGMENT_NAMES[int(np.digitize(float(wavelength), SEGMENT_EDGES))]


def instrument_sigma_px(segment_or_wavelength, profile=None):
    """Instrument Gaussian sigma [px] for a segment name or a wavelength.

    ``profile`` overrides the default per-segment dict (keys UV/VIS/NIR).
    """
    prof = dict(DEFAULT_PROFILE_PX)
    if profile:
        prof.update({k: float(v) for k, v in profile.items()})
    if isinstance(segment_or_wavelength, str):
        name = segment_or_wavelength
    else:
        name = _segment_name(segment_or_wavelength)
    return float(prof[name])


def _local_pitch(x, index, span=6):
    lo = max(0, index - span)
    hi = min(len(x) - 1, index + span)
    d = np.diff(x[lo:hi + 1])
    d = d[d > 0]
    return float(np.median(d)) if d.size else float(np.median(np.diff(x)))


def _window(x, index, half_px, pitch):
    """Index window [lo, hi] around ``index`` clipped at a pitch jump."""
    n = len(x)
    lo, hi = index, index
    for _ in range(half_px):
        if lo - 1 >= 0 and (x[lo] - x[lo - 1]) <= _GAP_FACTOR * pitch:
            lo -= 1
        if hi + 1 < n and (x[hi + 1] - x[hi]) <= _GAP_FACTOR * pitch:
            hi += 1
    return lo, hi


def _robust_noise(res):
    res = np.asarray(res, dtype=float)
    if res.size == 0:
        return 1.0
    return max(1.4826 * float(np.median(np.abs(res - np.median(res)))), 1e-9)


def _gauss(mu, sigma, x):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_subpixel_center(x, y, index, *, sigma_px, baseline="linear",
                             half_window_px=None, sigma_bounds=(0.6, 2.0),
                             max_iter=50, noise=None):
    """Weighted single-Gaussian + linear-baseline fit near sample ``index``.

    Returns a dict with ``ok``; on success: ``center_nm``, ``center_sigma_nm``
    (covariance 1-sigma), ``amplitude``, ``sigma_px``/``sigma_nm`` (fitted),
    ``baseline`` (value under the peak), ``chi2_nu``, ``snr``, ``blend`` (bool)
    and, when blended, ``two_gaussian`` (the shared-sigma pair fit).  With
    ``sigma_bounds=None`` the width is fixed at the instrument value.
    """
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = x.size
        index = int(index)
        if n < 5 or y.size != n or index < 0 or index >= n:
            return {"ok": False, "reason": "bad_index_or_shape"}
        if not np.all(np.isfinite(x[max(0, index - 12):index + 13])):
            return {"ok": False, "reason": "nonfinite_neighbourhood"}
        pitch = _local_pitch(x, index)
        sigma_px = float(sigma_px)
        half_px = int(half_window_px if half_window_px is not None
                      else max(3, int(np.ceil(2.5 * sigma_px))))
        lo, hi = _window(x, index, half_px, pitch)
        xs = x[lo:hi + 1]
        ys = y[lo:hi + 1]
        need = 5 if baseline == "linear" else 4
        if xs.size < need:
            return {"ok": False, "reason": "window_too_small"}
        mu0 = float(x[index])
        xspan = xs - mu0
        b_edge = 0.5 * (np.median(ys[:2]) + np.median(ys[-2:]))
        A0 = float(ys[index - lo] - b_edge) if 0 <= index - lo < ys.size else float(ys.max() - b_edge)
        A0 = A0 if abs(A0) > 1e-9 else float(ys.max() - b_edge + 1e-6)
        sig0 = sigma_px * pitch

        fit_sigma = sigma_bounds is not None
        # parameter vector: [A, mu, (sigma), b0, (b1)]
        def unpack(p):
            i = 0
            A = p[i]; i += 1
            mu = p[i]; i += 1
            if fit_sigma:
                sig = p[i]; i += 1
            else:
                sig = sig0
            b0 = p[i]; i += 1
            b1 = p[i] if baseline == "linear" else 0.0
            return A, mu, sig, b0, b1

        def model(p):
            A, mu, sig, b0, b1 = unpack(p)
            return A * _gauss(mu, sig, xs) + b0 + b1 * xspan

        def resid(p):
            return model(p) - ys

        p0 = [A0, mu0]
        lb = [-np.inf, xs[0]]
        ub = [np.inf, xs[-1]]
        if fit_sigma:
            p0.append(sig0)
            lb.append(sigma_bounds[0] * sig0)
            ub.append(sigma_bounds[1] * sig0)
        p0.append(float(b_edge))
        lb.append(-np.inf); ub.append(np.inf)
        if baseline == "linear":
            p0.append(0.0); lb.append(-np.inf); ub.append(np.inf)

        sol = least_squares(resid, p0, bounds=(lb, ub), max_nfev=max_iter * len(p0),
                            method="trf")
        A, mu, sig, b0, b1 = unpack(sol.x)
        r = resid(sol.x)
        m, k = xs.size, len(p0)
        dof = max(m - k, 1)
        s2 = float(np.sum(r ** 2) / dof)
        noise_val = float(noise) if noise is not None else _robust_noise(
            np.concatenate([ys[:2], ys[-2:]]) - b_edge)
        # covariance from the Jacobian at the solution
        J = sol.jac
        try:
            cov = np.linalg.inv(J.T @ J) * s2
            center_sigma = float(np.sqrt(max(cov[1, 1], 0.0)))
        except np.linalg.LinAlgError:
            center_sigma = float("nan")
        chi2_nu = float(np.sum(r ** 2) / (noise_val ** 2 * dof))
        snr = float(abs(A) / noise_val)
        sig_px_fit = sig / pitch

        # ---- blend detection --------------------------------------------
        reasons = []
        ssr_single = float(np.sum(r ** 2))
        if fit_sigma and sig >= 0.995 * sigma_bounds[1] * sig0:
            reasons.append("sigma_at_upper_bound")
        # a second local maximum in the baseline-subtracted window
        bsub = ys - (b0 + b1 * xspan)
        loc_max = [i for i in range(1, bsub.size - 1)
                   if bsub[i] > bsub[i - 1] and bsub[i] > bsub[i + 1]
                   and bsub[i] > 0.25 * max(A, 1e-9)]
        if len(loc_max) >= 2:
            reasons.append("second_local_maximum")
        # systematic residual: same-sign, large residuals across the core
        core = np.abs(xs - mu) <= sig
        if np.sum(core) >= 4:
            signs = np.sign(r[core])
            if np.all(signs == signs[0]) and np.max(np.abs(r[core])) > 3 * noise_val:
                reasons.append("systematic_core_residual")

        # A close pair 1-2 px apart merges into one broadened bump with no
        # second maximum and only a few-percent width excess -- width alone
        # cannot resolve it.  When the single fit is even mildly broadened,
        # fit a shared-sigma pair (seeded split around the centre) and flag a
        # blend when it MEANINGFULLY reduces the residual and the recovered
        # separation is real.
        two = None
        if snr >= 8 and (sig_px_fit >= 1.2 * sigma_px or chi2_nu > 5.0
                         or "second_local_maximum" in reasons):
            two = _two_gaussian(xs, ys, xspan, float(mu), 2.0 * sig0, baseline,
                                noise_val, seed_sep=2.0 * sig0)
            if (two.get("ok") and two["ssr"] < 0.4 * ssr_single
                    and 1.2 <= two["separation_nm"] / pitch <= (hi - lo)
                    and min(two["amplitudes"]) > 0.30 * max(two["amplitudes"])):
                reasons.append("resolved_pair")
        blend = bool(reasons)

        out = {"ok": True, "center_nm": float(mu), "center_sigma_nm": center_sigma,
               "amplitude": float(A), "sigma_px": float(sig_px_fit),
               "sigma_nm": float(sig), "baseline": float(b0 + 0.0),
               "chi2_nu": chi2_nu, "snr": snr, "blend": blend,
               "blend_reasons": reasons, "pitch_nm": float(pitch),
               "window": [int(lo), int(hi)], "n_samples": int(m),
               "fixed_sigma": (not fit_sigma)}
        if blend and two is not None and two.get("ok"):
            out["two_gaussian"] = two
        elif blend:
            out["two_gaussian"] = _two_gaussian(xs, ys, xspan, float(mu), sig0,
                                                baseline, noise_val)
        return out
    except Exception as exc:  # never raise
        return {"ok": False, "reason": f"exception:{type(exc).__name__}"}


def _two_gaussian(xs, ys, xspan, mu0, sig0, baseline, noise_val, seed_sep=None):
    """Shared-sigma two-Gaussian fit near ``mu0``.

    Centres are seeded either at the two strongest local maxima, or (when the
    bump has merged into one) split by ``seed_sep`` about ``mu0``.
    """
    try:
        b_edge = 0.5 * (np.median(ys[:2]) + np.median(ys[-2:]))
        bsub = ys - b_edge
        maxima = [i for i in range(1, xs.size - 1)
                  if bsub[i] > bsub[i - 1] and bsub[i] > bsub[i + 1]]
        maxima.sort(key=lambda i: bsub[i], reverse=True)
        if len(maxima) >= 2:
            m1, m2 = sorted(maxima[:2])
            seed1, seed2 = float(xs[m1]), float(xs[m2])
            a1, a2 = max(bsub[m1], 1e-6), max(bsub[m2], 1e-6)
        else:
            sep = seed_sep if seed_sep is not None else sig0
            seed1, seed2 = mu0 - 0.5 * sep, mu0 + 0.5 * sep
            a0 = max(float(np.max(bsub)), 1e-6)
            a1 = a2 = 0.6 * a0

        def unpack(p):
            A1, mu1, A2, mu2, sig, b0 = p[:6]
            b1 = p[6] if baseline == "linear" else 0.0
            return A1, mu1, A2, mu2, sig, b0, b1

        def resid(p):
            A1, mu1, A2, mu2, sig, b0, b1 = unpack(p)
            return (A1 * _gauss(mu1, sig, xs) + A2 * _gauss(mu2, sig, xs)
                    + b0 + b1 * xspan) - ys

        p0 = [a1, seed1, a2, seed2, sig0, b_edge]
        lb = [0.0, xs[0], 0.0, xs[0], 0.5 * sig0, -np.inf]
        ub = [np.inf, xs[-1], np.inf, xs[-1], 1.8 * sig0, np.inf]
        if baseline == "linear":
            p0.append(0.0); lb.append(-np.inf); ub.append(np.inf)
        sol = least_squares(resid, p0, bounds=(lb, ub), method="trf")
        A1, mu1, A2, mu2, sig, b0, b1 = unpack(sol.x)
        r = resid(sol.x)
        dof = max(xs.size - len(p0), 1)
        order = [0, 1] if mu1 <= mu2 else [1, 0]
        amps = [float(A1), float(A2)]
        centres = [float(mu1), float(mu2)]
        return {"ok": True, "centers_nm": [centres[order[0]], centres[order[1]]],
                "amplitudes": [amps[order[0]], amps[order[1]]],
                "sigma_nm": float(sig),
                "ssr": float(np.sum(r ** 2)),
                "chi2_nu": float(np.sum(r ** 2) / (noise_val ** 2 * dof)),
                "separation_nm": float(abs(mu1 - mu2))}
    except Exception as exc:
        return {"ok": False, "reason": f"exception:{type(exc).__name__}"}


def estimate_instrument_profile(x, y, candidates, *, profile=None,
                                sigma_bounds=(0.5, 2.5)):
    """Robust per-segment instrument sigma [px] from narrow isolated lines.

    ``candidates`` is a sequence of sample indices (isolated, high-SNR peaks).
    Fits each, keeps the narrowest well-fit widths per segment (lower
    envelope), and returns ``{segment: {"sigma_px", "n", "mad_px"}}``.
    """
    x = np.asarray(x, dtype=float)
    per_seg = {name: [] for name in SEGMENT_NAMES}
    for idx in candidates:
        seg = _segment_name(x[int(idx)])
        r = gaussian_subpixel_center(x, y, int(idx),
                                     sigma_px=instrument_sigma_px(seg, profile),
                                     sigma_bounds=sigma_bounds)
        if r.get("ok") and r["snr"] >= 8 and not r["blend"] and r["chi2_nu"] < 10:
            per_seg[seg].append(r["sigma_px"])
    out = {}
    for name, vals in per_seg.items():
        if not vals:
            continue
        v = np.asarray(vals)
        # lower envelope: the narrowest quartile is the instrument limit
        thresh = np.percentile(v, 40)
        narrow = v[v <= thresh]
        med = float(np.median(narrow)) if narrow.size else float(np.median(v))
        out[name] = {"sigma_px": med, "n": int(v.size),
                     "mad_px": float(1.4826 * np.median(np.abs(v - np.median(v))))}
    return out


def refine_peaks(x, y, indices, *, profile=None, baseline="linear",
                 sigma_bounds=(0.6, 2.0)):
    """Gaussian-refine a list of peak sample indices for a whole spectrum.

    Returns a list of records (the :func:`gaussian_subpixel_center` dict with
    an added ``index``), one per input index, ``ok=False`` for failures.
    """
    x = np.asarray(x, dtype=float)
    out = []
    for idx in indices:
        idx = int(idx)
        seg = _segment_name(x[idx]) if 0 <= idx < x.size else "VIS"
        r = gaussian_subpixel_center(x, y, idx,
                                     sigma_px=instrument_sigma_px(seg, profile),
                                     baseline=baseline, sigma_bounds=sigma_bounds)
        r["index"] = idx
        out.append(r)
    return out


__all__ = ["instrument_sigma_px", "gaussian_subpixel_center",
           "estimate_instrument_profile", "refine_peaks",
           "SEGMENT_EDGES", "SEGMENT_NAMES", "DEFAULT_PROFILE_PX"]
