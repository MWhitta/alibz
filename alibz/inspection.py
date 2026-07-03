"""Visual inspection of spectra, fits, and peak parameters.

Plotting helpers for raw spectra, backgrounds, fitted models, peak
locations, and residuals, plus a per-peak parameter table with
uncertainties.

Uncertainties are generalised-least-squares estimates computed post hoc:
peaks are clustered into blend groups (overlapping fit spans), a joint
Jacobian of the multi-Voigt model is evaluated over each group's pixels,
and the covariance is ``(J^T W J)^+`` with ``W = diag(1/noise_i^2)`` from
the blockwise local noise scale.  Blended peaks therefore carry honestly
inflated uncertainties; parameters pinned at a bound (e.g. ``gamma = 0``)
are flagged rather than assigned a symmetric error.
"""

from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import voigt_profile as _voigt

from alibz.utils.voigt import multi_voigt as _multi_voigt, voigt_width as _voigt_width


# ---------------------------------------------------------------------------
# Uncertainties
# ---------------------------------------------------------------------------

def _blend_groups(peak_array: np.ndarray, span_factor: float = 3.0) -> List[np.ndarray]:
    """Cluster peak indices whose fit spans overlap (sorted by center)."""
    if peak_array.size == 0:
        return []
    order = np.argsort(peak_array[:, 1])
    centers = peak_array[order, 1]
    fwhm = _voigt_width(
        np.maximum(peak_array[order, 2], 1e-6), np.maximum(peak_array[order, 3], 1e-6)
    )
    span = np.maximum(span_factor * fwhm, 0.15)
    groups, current = [], [order[0]]
    for k in range(1, centers.size):
        if centers[k] - span[k] <= centers[k - 1] + span[k - 1]:
            current.append(order[k])
        else:
            groups.append(np.asarray(current))
            current = [order[k]]
    groups.append(np.asarray(current))
    return groups


def estimate_peak_uncertainties(
    x: np.ndarray,
    y_bgsub: np.ndarray,
    peak_array: np.ndarray,
    noise: Optional[np.ndarray] = None,
    span_factor: float = 3.0,
) -> np.ndarray:
    """Per-peak 1-sigma uncertainties for [area, center, sigma, gamma].

    Joint GLS covariance per blend group (see module docstring), scaled
    by the group's reduced chi-square (never below 1).  Returns an
    ``(n, 4)`` array aligned with ``peak_array``; entries are ``nan``
    where a parameter sits at its zero bound (a one-sided constraint has
    no symmetric error) or where the group covariance is degenerate.

    Calibration status (measured on synthetic fixtures): blended peaks
    report honestly inflated uncertainties (~10% where isolated lines
    report ~0.001%), but for STRONG ISOLATED lines the reported sigma is
    a statistical lower bound — the pipeline's empirical fit scatter
    exceeds it several-fold (pull RMS ~5) through procedure effects
    (grid-railed centers, window boundaries).  Treat small sigmas as
    "statistics-limited"; the floor of real accuracy is set by the
    finder, not the photon noise.
    """
    x = np.asarray(x, dtype=float)
    y_bgsub = np.asarray(y_bgsub, dtype=float)
    peak_array = np.atleast_2d(np.asarray(peak_array, dtype=float))
    n = peak_array.shape[0]
    out = np.full((n, 4), np.nan)
    if n == 0 or x.size == 0:
        return out

    if noise is None:
        from alibz.peaky_finder import PeakyFinder
        noise = PeakyFinder._noise_scale_local(y_bgsub)
    noise = np.maximum(np.asarray(noise, dtype=float), 1e-12)

    for group in _blend_groups(peak_array, span_factor):
        params = peak_array[group][:, :4]
        fwhm = _voigt_width(
            np.maximum(params[:, 2], 1e-6), np.maximum(params[:, 3], 1e-6)
        )
        lo = np.min(params[:, 1] - np.maximum(span_factor * fwhm, 0.15))
        hi = np.max(params[:, 1] + np.maximum(span_factor * fwhm, 0.15))
        mask = (x >= lo) & (x <= hi)
        if np.count_nonzero(mask) < 4 * len(group) + 2:
            continue
        xw = x[mask]
        w = 1.0 / noise[mask]

        # free parameters: skip zero-pinned sigma/gamma (bound-constrained)
        free = []
        for gi, p in enumerate(params):
            for pi in range(4):
                if pi in (2, 3) and p[pi] <= 0.0:
                    continue
                free.append((gi, pi))
        if not free:
            continue

        def model(pmat):
            return _multi_voigt(
                xw,
                np.ravel(
                    np.column_stack([
                        pmat[:, 0], pmat[:, 1],
                        np.maximum(pmat[:, 2], 1e-9),
                        np.maximum(pmat[:, 3], 1e-9),
                    ])
                ),
            )

        base = model(params)
        J = np.empty((xw.size, len(free)))
        for col, (gi, pi) in enumerate(free):
            # Parameter-appropriate steps: the center's scale is the LINE
            # WIDTH, not its wavelength value — a relative step there is
            # several FWHM wide and corrupts the joint covariance.
            if pi == 1:
                step = 1e-3
            elif pi == 0:
                step = max(1e-3 * abs(params[gi, 0]), 1e-9)
            else:
                step = max(1e-3 * abs(params[gi, pi]), 1e-6)
            pp = params.copy()
            pp[gi, pi] += step
            J[:, col] = (model(pp) - base) / step
        Jw = J * w[:, None]
        try:
            cov = np.linalg.pinv(Jw.T @ Jw)
        except np.linalg.LinAlgError:
            continue
        # Scale by the group's reduced chi-square: when the local model
        # misfits (blends, shape mismatch, procedure effects) the pure
        # GLS covariance is optimistic; chi2-scaling is the standard
        # correction.  Never scale DOWN (chi2_red < 1 just means an
        # overestimated noise floor).
        resid_w = (y_bgsub[mask] - base) * w
        dof = max(xw.size - len(free), 1)
        chi2_red = float(np.sum(resid_w ** 2)) / dof
        cov = cov * max(chi2_red, 1.0)
        sig = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        for col, (gi, pi) in enumerate(free):
            out[group[gi], pi] = sig[col]
    return out


# ---------------------------------------------------------------------------
# Parameter table
# ---------------------------------------------------------------------------

def peak_table(
    x: np.ndarray,
    y: np.ndarray,
    fit_dict: dict,
    max_rows: Optional[int] = None,
    sort_by: str = "amplitude",
) -> List[dict]:
    """Per-peak fit parameters with uncertainties, as a list of dicts.

    Columns: center, area, sigma, gamma (each with 1-sigma uncertainty),
    Voigt FWHM, implied height, and height signal-to-noise against the
    local noise scale.  ``sort_by`` is ``"amplitude"`` (default) or
    ``"center"``.
    """
    peaks = np.atleast_2d(np.asarray(fit_dict["sorted_parameter_array"], dtype=float))
    if peaks.size == 0:
        return []
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bg = np.asarray(fit_dict.get("background", np.zeros_like(y)), dtype=float)
    y_bgsub = y - bg

    from alibz.peaky_finder import PeakyFinder
    noise = PeakyFinder._noise_scale_local(y_bgsub)
    sigmas = estimate_peak_uncertainties(x, y_bgsub, peaks, noise=noise)

    rows = []
    for k, (amp, mu, sg, gm) in enumerate(peaks[:, :4]):
        peak_value = _voigt(0.0, max(sg, 1e-6), max(gm, 1e-6))
        height = amp * peak_value
        j = int(np.clip(np.searchsorted(x, mu), 0, x.size - 1))
        rows.append({
            "center_nm": mu, "center_err": sigmas[k, 1],
            "area": amp, "area_err": sigmas[k, 0],
            "sigma_nm": sg, "sigma_err": sigmas[k, 2],
            "gamma_nm": gm, "gamma_err": sigmas[k, 3],
            "fwhm_nm": float(_voigt_width(max(sg, 1e-9), max(gm, 1e-9))),
            "height": height,
            "snr": height / max(float(noise[j]), 1e-12),
        })
    key = "area" if sort_by == "amplitude" else "center_nm"
    rows.sort(key=lambda r: -r[key] if sort_by == "amplitude" else r[key])
    return rows[:max_rows] if max_rows else rows


def format_peak_table(rows: List[dict], max_rows: Optional[int] = 25) -> str:
    """Human-readable text table (uncertainties as +/-; nan -> 'pinned')."""
    def pm(v, e, fmt):
        if not np.isfinite(e):
            return f"{v:{fmt}} (pinned)" if v <= 0 else f"{v:{fmt}} +/- ?"
        return f"{v:{fmt}} +/- {e:{fmt}}"

    lines = [
        f"{'center [nm]':>24s} {'area':>22s} {'sigma [nm]':>22s} "
        f"{'gamma [nm]':>22s} {'FWHM':>7s} {'height':>9s} {'S/N':>7s}"
    ]
    for r in rows[:max_rows] if max_rows else rows:
        lines.append(
            f"{pm(r['center_nm'], r['center_err'], '9.4f'):>24s} "
            f"{pm(r['area'], r['area_err'], '8.1f'):>22s} "
            f"{pm(r['sigma_nm'], r['sigma_err'], '7.4f'):>22s} "
            f"{pm(r['gamma_nm'], r['gamma_err'], '7.4f'):>22s} "
            f"{r['fwhm_nm']:7.3f} {r['height']:9.1f} {r['snr']:7.1f}"
        )
    if max_rows and len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more peaks)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_spectrum_overview(
    x: np.ndarray,
    y: np.ndarray,
    fit_dict: dict,
    xlim: Optional[tuple] = None,
    figsize: tuple = (16, 9),
    log_scale: bool = False,
):
    """Three-panel overview: raw + background, fit + peak locations, residual.

    The residual panel shows a +/-2-sigma local-noise band, so structure
    outside the band is real misfit rather than noise.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bg = np.asarray(fit_dict.get("background", np.zeros_like(y)), dtype=float)
    y_bgsub = y - bg
    peaks = np.atleast_2d(np.asarray(fit_dict["sorted_parameter_array"], dtype=float))
    profile = fit_dict.get("profile")
    if profile is None and peaks.size:
        profile = _multi_voigt(x, np.ravel(peaks[:, :4]))
    if profile is None:
        profile = np.zeros_like(y)

    from alibz.peaky_finder import PeakyFinder
    noise = PeakyFinder._noise_scale_local(y_bgsub)

    fig, axs = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [2, 3, 1.3]},
    )
    axs[0].plot(x, y, "k-", lw=0.6, label="raw spectrum")
    axs[0].plot(x, bg, "b-", lw=1.2, label="background")
    axs[0].set_ylabel("counts")
    axs[0].legend(loc="upper right", fontsize=9)

    axs[1].plot(x, y_bgsub, "k-", lw=0.6, label="background-subtracted")
    axs[1].plot(x, profile, "r-", lw=0.9, alpha=0.85, label="fitted model")
    if peaks.size:
        heights = peaks[:, 0] * _voigt(
            0.0, np.maximum(peaks[:, 2], 1e-6), np.maximum(peaks[:, 3], 1e-6)
        )
        axs[1].scatter(peaks[:, 1], heights, color="r", s=14, zorder=3,
                       label=f"{peaks.shape[0]} peaks")
    axs[1].set_ylabel("counts")
    axs[1].legend(loc="upper right", fontsize=9)

    residual = y_bgsub - profile
    axs[2].plot(x, residual, "k-", lw=0.5, label="residual")
    axs[2].fill_between(x, -2 * noise, 2 * noise, color="g", alpha=0.15,
                        label=r"$\pm 2\sigma$ local noise")
    axs[2].axhline(0.0, color="0.5", lw=0.7)
    axs[2].set_xlabel("wavelength [nm]")
    axs[2].set_ylabel("counts")
    axs[2].legend(loc="upper right", fontsize=9)

    if log_scale:
        axs[0].set_yscale("log")
        axs[1].set_yscale("log")
    if xlim:
        axs[0].set_xlim(*xlim)
        for ax in axs[1:]:
            ax.set_xlim(*xlim)
        m = (x >= xlim[0]) & (x <= xlim[1])
        if np.any(m) and not log_scale:
            axs[0].set_ylim(min(0, float(np.min(y[m]))), 1.1 * float(np.max(y[m])))
            axs[1].set_ylim(min(0, float(np.min(y_bgsub[m]))),
                            1.15 * float(np.max(y_bgsub[m])))
            axs[2].set_ylim(1.2 * float(np.min(residual[m])),
                            1.2 * float(np.max(residual[m])))
    fig.align_ylabels(axs)
    fig.tight_layout()
    return fig, axs


def plot_peak_zoom(
    x: np.ndarray,
    y: np.ndarray,
    fit_dict: dict,
    center_nm: float,
    span_nm: float = 1.5,
    figsize: tuple = (8, 5.5),
):
    """Zoom on one fitted peak: data, its own Voigt component, the other
    components' contribution, the total model, and the residual with the
    fitted parameters (and uncertainties) annotated."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bg = np.asarray(fit_dict.get("background", np.zeros_like(y)), dtype=float)
    y_bgsub = y - bg
    peaks = np.atleast_2d(np.asarray(fit_dict["sorted_parameter_array"], dtype=float))
    if peaks.size == 0:
        raise ValueError("fit_dict contains no peaks")
    k = int(np.argmin(np.abs(peaks[:, 1] - center_nm)))
    amp, mu, sg, gm = peaks[k, :4]

    m = (x >= mu - span_nm / 2) & (x <= mu + span_nm / 2)
    own = amp * _voigt(x[m] - mu, max(sg, 1e-9), max(gm, 1e-9))
    total = _multi_voigt(x[m], np.ravel(peaks[:, :4]))
    rest = total - own

    from alibz.peaky_finder import PeakyFinder
    noise = PeakyFinder._noise_scale_local(y_bgsub)
    errs = estimate_peak_uncertainties(x, y_bgsub, peaks, noise=noise)[k]

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax.plot(x[m], y_bgsub[m], "k.-", ms=4, lw=0.8, label="data (bg-subtracted)")
    ax.plot(x[m], total, "r-", lw=1.3, label="total model")
    ax.plot(x[m], own, "b-", lw=1.3, label="this peak")
    ax.fill_between(x[m], own, alpha=0.2, color="b")
    ax.plot(x[m], rest, color="0.5", ls="--", lw=1.0, label="other peaks")
    ax.set_ylabel("counts")

    def fmt(v, e):
        return f"{v:.4f}±{e:.4f}" if np.isfinite(e) else f"{v:.4f} (pinned)"

    ax.set_title(
        f"λ = {fmt(mu, errs[1])} nm   area = {amp:.1f}±"
        f"{errs[0]:.1f}\nσ = {fmt(sg, errs[2])}   γ = {fmt(gm, errs[3])} nm",
        fontsize=10,
    )
    ax.legend(fontsize=8)

    axr.plot(x[m], y_bgsub[m] - total, "k-", lw=0.8)
    axr.fill_between(x[m], -2 * noise[m], 2 * noise[m], color="g", alpha=0.15)
    axr.axhline(0.0, color="0.5", lw=0.7)
    axr.set_xlabel("wavelength [nm]")
    axr.set_ylabel("residual")
    fig.tight_layout()
    return fig, (ax, axr)
