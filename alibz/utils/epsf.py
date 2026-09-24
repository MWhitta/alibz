"""Empirical pixel-integrated line-spread function (ePSF) and fitter.

The SciAps Z300 native grid samples lines at ~1-2 px per FWHM, so the line
shape is critically sampled or undersampled.  A point-sampled or
pixel-integrated *Gaussian* model is then biased in centre (pixel locking or
repulsion, depending on the true shape), in width and in area.  Following
Anderson & King (2000, PASP 112, 1360) for undersampled HST images, we model a
sample directly as

    y_k = A * Psi(u_k - c) + b0 + b1 * (u_k - c)

where ``u_k`` is the sample's PIXEL coordinate (consecutive native pixels
within one detector segment), ``c`` is the line centre in pixels, ``A`` the
integrated line area in counts * px, and ``Psi`` the *effective* LSF: the
instrument line-spread function already convolved with the pixel response
(box of width 1 px).  Because ``Psi`` is pixel-integrated, the identity
``sum_k Psi(k - c) = integral(Psi) = 1`` holds at every sub-pixel phase, so
``A`` is a phase-independent area.

``Psi`` is a cubic B-spline with knot spacing ``h`` (default 1/8 px) with
compact support [-R, R] px (default R = 3; the fit window, default +-4.5 px,
must extend beyond R so the background is set by samples where Psi = 0), normalised to unit integral and centred (zero first moment
inside |d| <= ``center_half``).  It is built self-consistently from many
lines at random sub-pixel phases (:func:`build_epsf`): fit centres with the
current shape, pool the normalised samples on the super-sampled grid, solve a
second-difference-regularised least-squares problem for the shape under the
partition-of-unity constraint sum_k Psi(k + x) = 1 (a property of any
pixel-integrated LSF), re-centre, apply the uniform-phase warp, repeat.
Both constraints are required: without them the (shape, centres, areas)
problem has exact or near-exact degeneracies (see :func:`build_epsf`).

Genuine physical line-shape differences are NOT folded into ``Psi``.  They
are modelled as a physical kernel convolved with the instrument ePSF
(:func:`physical_kernel`): ``lorentz`` (Lorentzian half-width ``gamma`` px,
Stark), and ``selfabs`` (emergent profile ``1 - exp(-tau * phi)`` with
``phi`` a unit-peak Lorentzian of half-width ``gamma`` px, a homogeneous
self-absorbed slab; its FWHM grows with ``tau`` and its core flattens).  The
reference ePSF already contains the reference class's own (small) physical
width, so class kernels are EXCESS kernels relative to the reference class.

Fallback: :func:`fit_line` never raises.  If the ePSF fit fails, hits a
centre bound, or returns a non-positive area, it falls back to a
pixel-integrated Gaussian fit (flag ``fallback='gauss'``); if that also fails
it returns ``ok=False`` with a reason.

numpy + scipy only; deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.special import erf

__all__ = [
    "EPSF", "pixint_gauss", "gauss_epsf", "fit_line", "fit_pixint_gauss",
    "build_epsf", "physical_kernel", "convolve_kernel", "phase_of",
    "LineData", "phase_warp", "dither_warp", "dither_structure",
]


# ---------------------------------------------------------------------------
# Shape object
# ---------------------------------------------------------------------------
def _knots(h: float, R: float) -> np.ndarray:
    n = int(round(2 * R / h))
    return -R + h * np.arange(-3, n + 4)


class EPSF:
    """Cubic B-spline ePSF on [-R, R] px; zero outside.

    Parameters
    ----------
    coef : B-spline coefficients (len = n_knots - 4).
    h, R : knot spacing and half-range in pixels.
    meta : free-form provenance / diagnostics.
    """

    def __init__(self, coef, h: float = 0.125, R: float = 3.0, meta=None):
        self.h = float(h)
        self.R = float(R)
        self.t = _knots(self.h, self.R)
        coef = np.asarray(coef, dtype=float)
        if coef.size != self.t.size - 4:
            raise ValueError("coef length does not match knots")
        self.coef = coef
        self.meta = dict(meta or {})
        self._spl = BSpline(self.t, self.coef, 3, extrapolate=False)
        self._dspl = self._spl.derivative()

    # -- evaluation -------------------------------------------------------
    def __call__(self, d) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        v = self._spl(d)
        return np.where(np.isfinite(v), v, 0.0)

    def deriv(self, d) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        v = self._dspl(d)
        return np.where(np.isfinite(v), v, 0.0)

    # -- diagnostics ------------------------------------------------------
    def grid(self, step: float = 1.0 / 64):
        d = np.arange(-self.R, self.R + step / 2, step)
        return d, self(d)

    def integral(self) -> float:
        d, v = self.grid()
        return float(np.trapezoid(v, d))

    def centroid(self, half: float = 2.0) -> float:
        d, v = self.grid()
        m = np.abs(d) <= half
        return float(np.sum(d[m] * v[m]) / np.sum(v[m]))

    def fwhm(self) -> float:
        """Full width at half maximum of the pixel-integrated shape [px]."""
        d, v = self.grid(1.0 / 256)
        i = int(np.argmax(v)); half = v[i] / 2
        l = i
        while l > 0 and v[l] > half:
            l -= 1
        r = i
        while r < len(v) - 1 and v[r] > half:
            r += 1
        dl = d[l] + (half - v[l]) * (d[l + 1] - d[l]) / (v[l + 1] - v[l])
        dr = d[r - 1] + (half - v[r - 1]) * (d[r] - d[r - 1]) / (v[r] - v[r - 1])
        return float(dr - dl)

    def peak(self) -> float:
        d, v = self.grid()
        return float(v.max())

    # -- (de)serialisation ------------------------------------------------
    def to_dict(self) -> dict:
        d, v = self.grid(self.h)
        return dict(kind="bspline3", h=self.h, R=self.R,
                    coef=[float(x) for x in self.coef],
                    samples_d=[float(x) for x in d],
                    samples=[float(x) for x in v],
                    fwhm_px=self.fwhm(), meta=self.meta)

    @classmethod
    def from_dict(cls, dct) -> "EPSF":
        return cls(dct["coef"], dct["h"], dct["R"], dct.get("meta"))

    @classmethod
    def from_function(cls, f, h: float = 0.125, R: float = 3.0, meta=None,
                      normalise: bool = True) -> "EPSF":
        """Least-squares B-spline projection of a callable ``f(d)``."""
        t = _knots(h, R)
        d = np.linspace(-R, R, int(round(2 * R / h)) * 16 + 1)
        B = BSpline.design_matrix(d, t, 3).toarray()
        n = B.shape[1]
        free = np.arange(3, n - 3)          # compact support, see _solve_shape
        coef = np.zeros(n)
        coef[free] = np.linalg.lstsq(B[:, free], f(d), rcond=None)[0]
        e = cls(coef, h, R, meta)
        if normalise:
            e = e.scaled(1.0 / e.integral())
        return e

    def scaled(self, s: float) -> "EPSF":
        return EPSF(self.coef * s, self.h, self.R, self.meta)

    def shifted(self, s: float) -> "EPSF":
        """Return Psi(d + s) (i.e. the shape moved by -s)."""
        return EPSF.from_function(lambda d: self(d + s), self.h, self.R,
                                  self.meta, normalise=False)


def pixint_gauss(d, sigma):
    """Unit-area Gaussian integrated over a 1-px box, as a function of d."""
    d = np.asarray(d, dtype=float)
    s = sigma * np.sqrt(2.0)
    return 0.5 * (erf((d + 0.5) / s) - erf((d - 0.5) / s))


def gauss_epsf(sigma: float = 0.55, h: float = 0.125, R: float = 3.0) -> EPSF:
    """Pixel-integrated Gaussian as an :class:`EPSF` (start shape / fallback)."""
    return EPSF.from_function(lambda d: pixint_gauss(d, sigma), h, R,
                              meta=dict(kind="pixint_gauss", sigma_px=sigma))


def phase_of(c) -> np.ndarray:
    """Sub-pixel phase in [-0.5, 0.5): c - round(c)."""
    c = np.asarray(c, dtype=float)
    return (c + 0.5) % 1.0 - 0.5


# ---------------------------------------------------------------------------
# Physical kernels
# ---------------------------------------------------------------------------
_FINE = 1.0 / 32


def physical_kernel(kind: str, params: Sequence[float], half: float = 6.0):
    """Physical line profile on the fine grid (unit sum).  Units: px.

    kind='lorentz'  params=(gamma,)        Lorentzian HWHM gamma.
    kind='selfabs'  params=(tau, gamma)    1 - exp(-tau*phi), phi a unit-peak
                                           Lorentzian of HWHM gamma (slab).
    Returns (x, k) with k summing to 1; a delta when the width is ~0.
    """
    x = np.arange(-half, half + _FINE / 2, _FINE)
    if kind == "lorentz":
        g = max(float(params[0]), 1e-4)
        k = g / (x ** 2 + g ** 2)
    elif kind == "selfabs":
        tau = max(float(params[0]), 1e-6)
        g = max(float(params[1]), 1e-4)
        phi = g ** 2 / (x ** 2 + g ** 2)
        k = -np.expm1(-tau * phi)
    else:
        raise ValueError(f"unknown kernel {kind!r}")
    s = k.sum()
    if not np.isfinite(s) or s <= 0:
        k = np.zeros_like(x); k[len(x) // 2] = 1.0
        return x, k
    return x, k / s


def convolve_kernel(epsf: EPSF, kind: str, params: Sequence[float]):
    """Return a callable Psi_phys(d) = (Psi (x) K)(d), unit integral."""
    xk, k = physical_kernel(kind, params)
    d = np.arange(-epsf.R - xk[-1], epsf.R + xk[-1] + _FINE / 2, _FINE)
    v = epsf(d)
    conv = np.convolve(v, k, mode="same")
    conv /= np.trapezoid(conv, d)

    def f(q):
        return np.interp(np.asarray(q, dtype=float), d, conv, left=0.0, right=0.0)
    return f


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
@dataclass
class LineData:
    """One line's samples in pixel coordinates (consecutive native pixels)."""
    u: np.ndarray          # pixel coordinate of each sample
    y: np.ndarray          # counts
    sigma: np.ndarray      # 1-sigma per sample
    c0: float              # initial centre [px]
    key: object = None     # identifier (run, segment, db line ...)
    info: dict = field(default_factory=dict)


def _window(u, c0, half):
    return np.abs(u - c0) <= half


def fit_pixint_gauss(u, y, sigma=None, c0=None, half=4.5, sig_bounds=(0.2, 2.5)):
    """Pixel-integrated Gaussian + linear background (free width).

    Returns dict(ok, area, center, sigma_px, b0, b1, chi2nu, reason).
    """
    u = np.asarray(u, float); y = np.asarray(y, float)
    sg = np.ones_like(y) if sigma is None else np.asarray(sigma, float)
    if c0 is None:
        c0 = float(u[np.argmax(y)])
    m = _window(u, c0, half)
    if m.sum() < 5:
        return dict(ok=False, reason="window<5")
    uu, yy, ss = u[m], y[m], sg[m]
    bg0 = float(np.median(np.r_[yy[:2], yy[-2:]]))
    a0 = max(float(np.sum(yy - bg0)), 1.0)

    def model(p):
        return p[0] * pixint_gauss(uu - p[1], p[2]) + p[3] + p[4] * (uu - p[1])

    lo = [0.0, c0 - 1.0, sig_bounds[0], -np.inf, -np.inf]
    hi = [np.inf, c0 + 1.0, sig_bounds[1], np.inf, np.inf]
    p0 = [a0, c0, 0.6, bg0, 0.0]
    try:
        r = least_squares(lambda p: (model(p) - yy) / ss, p0, bounds=(lo, hi),
                          x_scale="jac")
    except Exception as exc:  # pragma: no cover - defensive
        return dict(ok=False, reason=f"lsq:{exc}")
    dof = max(1, m.sum() - 5)
    return dict(ok=bool(r.success), area=float(r.x[0]), center=float(r.x[1]),
                sigma_px=float(r.x[2]), b0=float(r.x[3]), b1=float(r.x[4]),
                chi2nu=float(np.sum(r.fun ** 2) / dof), reason="")


def fit_line(epsf: EPSF, u, y, sigma=None, c0=None, half: float = 4.5,
             kernel: str | None = None, kernel_p0=None, kernel_bounds=None,
             fallback: bool = True, center_bound: float = 1.0,
             neighbors: Sequence[float] | None = None,
             neighbor_bound: float = 0.7):
    """Fit A, c, b0, b1 (+ optional physical-kernel params) with the ePSF.

    Parameters
    ----------
    u, y, sigma : pixel coordinates, counts, 1-sigma (None -> unit weights).
    c0 : initial centre (default: argmax of y).
    half : fit window half-width in px about c0.
    kernel : None, 'lorentz' (params gamma) or 'selfabs' (params tau, gamma),
        applied to the target line only.
    center_bound : |c - c0| bound [px].
    neighbors : initial centres of neighbouring lines to fit simultaneously
        (instrument ePSF, own area and centre within +-``neighbor_bound``);
        those whose centre lies outside the window +- R are ignored.

    Returns dict(ok, area, center, b0, b1, kernel_params, neighbors
    [(area, centre)], chi2nu, resid, u_fit, neighbor_model, fallback,
    reason, center_err, area_err).  ``area`` is in counts * px (multiply by
    the local pitch for counts * nm).  Never raises.
    """
    u = np.asarray(u, float); y = np.asarray(y, float)
    sg = np.ones_like(y) if sigma is None else np.asarray(sigma, float)
    good = np.isfinite(sg) & (sg > 0)
    sg = np.where(good, sg, np.nanmedian(sg[good]) if np.any(good) else 1.0)
    if u.size < 5 or not np.all(np.isfinite(y)):
        return dict(ok=False, reason="bad input")
    if c0 is None:
        c0 = float(u[np.argmax(y)])
    m = _window(u, c0, half)
    if m.sum() < 5:
        return dict(ok=False, reason="window<5")
    uu, yy, ss = u[m], y[m], sg[m]
    nbr = [float(c) for c in (neighbors or [])
           if abs(c - c0) > 1.0 and abs(c - c0) < half + epsf.R]
    nn = len(nbr)
    nk = 0
    kp0, kb = [], ([], [])
    if kernel is not None:
        kp0 = list(kernel_p0) if kernel_p0 is not None else (
            [0.2] if kernel == "lorentz" else [1.0, 0.1])
        kb = kernel_bounds or (([1e-3], [3.0]) if kernel == "lorentz"
                               else ([1e-3, 0.02], [200.0, 1.5]))
        nk = len(kp0)
    bg0 = float(np.median(np.r_[yy[:2], yy[-2:]]))
    a0 = max(float(np.sum(yy - bg0)), 1.0)
    i_n = 4 + nk                       # first neighbour parameter

    def nmodel(p, uv):
        out = np.zeros_like(uv)
        for j in range(nn):
            out += p[i_n + 2 * j] * epsf(uv - p[i_n + 2 * j + 1])
        return out

    def model(p):
        if nk:
            f = convolve_kernel(epsf, kernel, p[4:4 + nk])
        else:
            f = epsf
        return p[0] * f(uu - p[1]) + p[2] + p[3] * (uu - p[1]) + nmodel(p, uu)

    def jac(p):
        d = uu - p[1]
        J = np.zeros((uu.size, 4 + 2 * nn))
        J[:, 0] = epsf(d)
        J[:, 1] = -p[0] * epsf.deriv(d) - p[3]
        J[:, 2] = 1.0
        J[:, 3] = d
        for j in range(nn):
            dj = uu - p[i_n + 2 * j + 1]
            J[:, 4 + 2 * j] = epsf(dj)
            J[:, 5 + 2 * j] = -p[i_n + 2 * j] * epsf.deriv(dj)
        return J / ss[:, None]

    p0 = [a0, c0, bg0, 0.0] + kp0
    lo = [0.0, c0 - center_bound, -np.inf, -np.inf] + list(kb[0])
    hi = [np.inf, c0 + center_bound, np.inf, np.inf] + list(kb[1])
    for c in nbr:
        k = int(np.argmin(np.abs(u - c)))
        p0 += [max(float(y[k] - bg0), 1.0), c]
        lo += [0.0, c - neighbor_bound]
        hi += [np.inf, c + neighbor_bound]
    p0 = list(np.clip(p0, np.array(lo, float) + 1e-9, np.array(hi, float) - 1e-9))
    jf = jac if nk == 0 else "2-point"
    try:
        r = least_squares(lambda p: (model(p) - yy) / ss, p0, jac=jf,
                          bounds=(lo, hi), x_scale="jac")
        ok = bool(r.success)
        p = r.x
        reason = ""
        if abs(p[1] - c0) > center_bound - 1e-3:
            ok, reason = False, "center at bound"
        if p[0] <= 0:
            ok, reason = False, "area<=0"
    except Exception as exc:
        ok, reason, r = False, f"lsq:{exc}", None
    if not ok:
        if fallback:
            g = fit_pixint_gauss(u, y, sg, c0=c0, half=half)
            if g.get("ok"):
                g.update(fallback="gauss", reason=reason or "epsf failed",
                         kernel_params=None, neighbors=[])
                return g
        return dict(ok=False, reason=reason or "fit failed")
    dof = max(1, m.sum() - len(p))
    chi2 = float(np.sum(r.fun ** 2))
    try:
        J = r.jac
        cov = np.linalg.pinv(J.T @ J) * max(1.0, chi2 / dof)
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:  # pragma: no cover
        err = np.full(len(p), np.nan)
    return dict(ok=True, area=float(p[0]), center=float(p[1]),
                b0=float(p[2]), b1=float(p[3]),
                kernel=kernel,
                kernel_params=[float(x) for x in p[4:4 + nk]] if nk else None,
                neighbors=[(float(p[i_n + 2 * j]), float(p[i_n + 2 * j + 1])) for j in range(nn)],
                neighbor_model=nmodel(p, uu),
                chi2nu=chi2 / dof, resid=(yy - model(p)), u_fit=uu,
                center_err=float(err[1]), area_err=float(err[0]),
                kernel_err=[float(x) for x in err[4:4 + nk]] if nk else None,
                fallback=None, reason="")


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------
def _second_diff(n):
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i:i + 3] = (1.0, -2.0, 1.0)
    return D


def _solve_shape(d, z, w, h, R, lam, partition_of_unity=True):
    """Regularised LSQ for the B-spline coefficients.

    The 3 outermost coefficients on each side are fixed at 0 so Psi, Psi' and
    Psi'' vanish at +-R: the shape has compact support, which removes the
    degeneracy between a constant pedestal in the shape and the local
    background (the fit window must extend beyond R).
    """
    t = _knots(h, R)
    inside = np.abs(d) < R
    d, z, w = d[inside], z[inside], w[inside]
    B = BSpline.design_matrix(d, t, 3).toarray()
    n = B.shape[1]
    free = np.arange(3, n - 3)
    Bf = B[:, free]
    D = _second_diff(n)[:, free]
    BtW = Bf.T * w
    M = BtW @ Bf + lam * (D.T @ D)
    coef = np.zeros(n)
    if not partition_of_unity:
        coef[free] = np.linalg.solve(M + 1e-10 * np.eye(free.size), BtW @ z)
        return coef
    C = _pou_matrix(h, R)[:, free]
    # null-space elimination of the equality constraints C x = 1 (stable,
    # SPD reduced system): x = x0 + N z
    x0 = np.linalg.lstsq(C, np.ones(C.shape[0]), rcond=None)[0]
    U, sv, Vt = np.linalg.svd(C)
    rank = int(np.sum(sv > sv[0] * 1e-10))
    N = Vt[rank:].T
    Mr = N.T @ M @ N
    rr = N.T @ (BtW @ z - M @ x0)
    Mr = Mr + 1e-10 * np.trace(Mr) / Mr.shape[0] * np.eye(Mr.shape[0])
    zz = np.linalg.solve(Mr, rr)
    coef[free] = x0 + N @ zz
    return coef


def _pou_matrix(h, R, n_x=None):
    """Rows: sum_k B(k + x) for x on a grid in [0, 1) -> partition of unity.

    A pixel-integrated LSF satisfies sum_k Psi(k + x) = 1 for every x.
    Imposing it removes the per-phase amplitude degeneracy (a shape
    Psi(d)*(1 + a(frac d)) with fitted areas rescaled by phase fits the samples
    equally well) and fixes the normalisation.
    """
    t = _knots(h, R)
    n_x = n_x or int(round(1.0 / h))
    xs = (np.arange(n_x) + 0.5) / n_x - 0.5
    ks = np.arange(-int(np.ceil(R)) - 1, int(np.ceil(R)) + 2)
    rows = []
    for x in xs:
        d = ks + x
        d = d[np.abs(d) < R]
        rows.append(BSpline.design_matrix(d, t, 3).toarray().sum(0))
    return np.array(rows)


def phase_warp(c, n_harm: int = 3):
    """Smooth periodic warp g(phi) that maps fitted phases to a uniform CDF.

    With many lines whose TRUE sub-pixel phases are uniform, the rank of each
    fitted phase gives its uniform-quantile target; g is the least-squares
    Fourier series (``n_harm`` harmonics, zero mean) of target - phase.
    Returns a callable g(phi) [px].
    """
    ph = phase_of(c)
    ph = ph[np.isfinite(ph)]
    n = ph.size
    if n < 4 * n_harm + 2:
        return lambda p: np.zeros_like(np.asarray(p, float))
    order = np.argsort(ph)
    target = np.empty(n)
    target[order] = (np.arange(n) + 0.5) / n - 0.5
    resid = target - ph
    resid -= resid.mean()

    def basis(p):
        p = np.asarray(p, float)
        cols = []
        for m in range(1, n_harm + 1):
            cols += [np.sin(2 * np.pi * m * p), np.cos(2 * np.pi * m * p)]
        return np.stack(cols, axis=-1)
    coef = np.linalg.lstsq(basis(ph), resid, rcond=None)[0]

    def g(p):
        return basis(p) @ coef
    return g


def dither_warp(c, groups, runs, upos, n_harm: int = 3, weights=None,
                clip: float = 4.0, n_clip: int = 3, prior_sd: float = 0.05,
                run_slope: bool = False, min_per_group: int = 3,
                min_per_run: int = 3):
    """Phase warp from DITHER consistency (recurring lines across runs).

    The same physical line (``groups``) observed in several runs (``runs``)
    sits at different sub-pixel phases because each run's registration
    drifts.  With a correct shape, c_{L,r} = kappa_L + a_r (+ b_r * u_L if
    ``run_slope``): line constant + per-run registration.  Groups seen in
    fewer than ``min_per_group`` runs and runs holding fewer than
    ``min_per_run`` of the remaining lines are dropped (they only add
    nuisance parameters).  The residual scale is dof-corrected, so the
    returned significance is honest when the model is not over-parametrised.  A phase bias breaks this.
    We solve the linear least-squares problem

        c_i + g(phi_i) = kappa_{L(i)} + a_{r(i)} + b_{r(i)} * u_i

    for the zero-mean Fourier warp g (``n_harm`` harmonics) jointly with the
    per-line constants and per-run linear registration, with iterative
    ``clip``-sigma rejection and a Gaussian prior (sd ``prior_sd`` px) on
    each Fourier coefficient, which keeps g bounded at phases the drift does
    not sample.  This is Anderson & King's dither constraint
    with the run-to-run drift as the (unknown, fitted) dither.  Only groups
    seen in >= 2 runs contribute.  Returns (g, info).
    """
    c = np.asarray(c, float); upos = np.asarray(upos, float)
    groups = np.asarray(groups); runs = np.asarray(runs)
    use = np.isfinite(c)
    for _ in range(5):
        gl, gcount = np.unique(groups[use], return_counts=True)
        multi = set(gl[gcount >= min_per_group])
        use = use & np.array([g in multi for g in groups])
        rl, rcount = np.unique(runs[use], return_counts=True)
        rok = set(rl[rcount >= min_per_run])
        use = use & np.array([r in rok for r in runs])
    if use.sum() < 4 * n_harm + 10:
        return (lambda p: np.zeros_like(np.asarray(p, float))), dict(n=int(use.sum()), ok=False)
    ci, gi, ri, ui = c[use], groups[use], runs[use], upos[use]
    w = np.ones(ci.size) if weights is None else np.asarray(weights, float)[use]
    Lk, Li = np.unique(gi, return_inverse=True)
    Rk, Ri = np.unique(ri, return_inverse=True)
    um = ui.mean(); us = max(ui.std(), 1.0)
    ph = phase_of(ci)
    nL, nR = Lk.size, Rk.size
    ns = nR if run_slope else 0
    cols = nL + (nR - 1) + ns + 2 * n_harm
    X = np.zeros((ci.size, cols))
    X[np.arange(ci.size), Li] = 1.0
    rr = Ri > 0
    X[np.where(rr)[0], nL + Ri[rr] - 1] = 1.0
    if run_slope:
        X[np.arange(ci.size), nL + nR - 1 + Ri] = (ui - um) / us
    k0 = nL + nR - 1 + ns
    for m in range(1, n_harm + 1):
        X[:, k0 + 2 * (m - 1)] = -np.sin(2 * np.pi * m * ph)
        X[:, k0 + 2 * (m - 1) + 1] = -np.cos(2 * np.pi * m * ph)
    keep = np.ones(ci.size, bool)
    s = 0.02
    nf = 2 * n_harm
    for _ in range(n_clip + 1):
        sw = np.sqrt(w[keep] / np.mean(w[keep]))
        P = np.zeros((nf, cols))
        P[np.arange(nf), k0 + np.arange(nf)] = max(s, 1e-3) / prior_sd
        A = np.vstack([X[keep] * sw[:, None], P])
        bvec = np.r_[ci[keep] * sw, np.zeros(nf)]
        coef = np.linalg.lstsq(A, bvec, rcond=None)[0]
        res = ci - X @ coef
        dof_frac = max(keep.sum() - cols, 1) / keep.sum()
        s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) / np.sqrt(dof_frac)
        new = np.abs(res) <= clip * max(s, 1e-4)
        if np.array_equal(new, keep):
            break
        keep = new
    fc = coef[k0:]
    # coefficient covariance (robust residual scale) -> significance
    try:
        sw = np.sqrt(w[keep] / np.mean(w[keep]))
        A = np.vstack([X[keep] * sw[:, None], P])
        cov = np.linalg.pinv(A.T @ A) * s ** 2
        cf = cov[k0:, k0:]
        chi2_sig = float(fc @ np.linalg.pinv(cf) @ fc)
        coef_sd = [float(x) for x in np.sqrt(np.clip(np.diag(cf), 0, None))]
    except Exception:  # pragma: no cover
        chi2_sig, coef_sd = np.nan, None

    def g(p):
        p = np.asarray(p, float)
        out = np.zeros_like(p)
        for m in range(1, n_harm + 1):
            out = out + fc[2 * (m - 1)] * np.sin(2 * np.pi * m * p) \
                + fc[2 * (m - 1) + 1] * np.cos(2 * np.pi * m * p)
        return out
    return g, dict(n=int(keep.sum()), n_lines=int(nL), n_runs=int(nR), n_par=int(cols),
                   resid_mad=float(s), ok=True, coef=[float(x) for x in fc],
                   coef_sd=coef_sd, chi2_sig=chi2_sig, dof=nf)


def dither_structure(c, groups, runs, upos=None, weights=None, run_slope=False,
                     clip: float = 4.0, min_per_group: int = 3, min_per_run: int = 3):
    """Robust LSQ of c_i = kappa_{L(i)} + a_{r(i)} (+ b_r * u_i).

    Returns (c_struct, used_mask, info): the structured centres for windows
    that belong to a group seen in >= ``min_per_group`` runs and a run with
    >= ``min_per_run`` such windows; NaN elsewhere.
    """
    c = np.asarray(c, float)
    groups = np.asarray(groups); runs = np.asarray(runs)
    upos = np.zeros_like(c) if upos is None else np.asarray(upos, float)
    use = np.isfinite(c)
    for _ in range(5):
        gl, gcount = np.unique(groups[use], return_counts=True)
        multi = set(gl[gcount >= min_per_group])
        use = use & np.array([g in multi for g in groups])
        rl, rcount = np.unique(runs[use], return_counts=True)
        rok = set(rl[rcount >= min_per_run])
        use = use & np.array([r in rok for r in runs])
    out = np.full(c.size, np.nan)
    if use.sum() < 10:
        return out, use, dict(ok=False, n=int(use.sum()))
    idx = np.where(use)[0]
    ci, gi, ri, ui = c[idx], groups[idx], runs[idx], upos[idx]
    w = np.ones(ci.size) if weights is None else np.asarray(weights, float)[idx]
    Lk, Li = np.unique(gi, return_inverse=True)
    Rk, Ri = np.unique(ri, return_inverse=True)
    nL, nR = Lk.size, Rk.size
    ns = nR if run_slope else 0
    cols = nL + nR - 1 + ns
    X = np.zeros((ci.size, cols))
    X[np.arange(ci.size), Li] = 1.0
    rr = Ri > 0
    X[np.where(rr)[0], nL + Ri[rr] - 1] = 1.0
    if run_slope:
        um = ui.mean(); us = max(ui.std(), 1.0)
        X[np.arange(ci.size), nL + nR - 1 + Ri] = (ui - um) / us
    keep = np.ones(ci.size, bool)
    for _ in range(4):
        sw = np.sqrt(w[keep] / np.mean(w[keep]))
        coef = np.linalg.lstsq(X[keep] * sw[:, None], ci[keep] * sw, rcond=None)[0]
        res = ci - X @ coef
        s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        new = np.abs(res) <= clip * max(s, 1e-4)
        if np.array_equal(new, keep):
            break
        keep = new
    fitted = X @ coef
    fitted[~keep] = np.nan          # outliers keep their free centre
    out[idx] = fitted
    return out, use, dict(ok=True, n=int(keep.sum()), n_lines=int(nL), n_runs=int(nR),
                          resid_mad=float(s), n_par=int(cols))


def _refit_linear(ep, u, y, sigma, c, extra):
    """A, b0, b1 by weighted linear LSQ at FIXED centre c (extra = neighbour model)."""
    d = u - c
    X = np.stack([ep(d), np.ones_like(d), d], axis=1) / sigma[:, None]
    sol = np.linalg.lstsq(X, (y - extra) / sigma, rcond=None)[0]
    return sol


def build_epsf(lines: Sequence[LineData], init: EPSF | None = None,
               h: float = 0.125, R: float = 3.0, half: float = 4.5,
               lam: float = 0.03, n_iter: int = 30, tol: float = 1e-3,
               center_half: float = 2.0, snr_cap: float = 300.0,
               clip: float = 5.0, equalize_phase: bool = True,
               warp_mode: str = "uniform", run_slope: bool = True,
               n_harm: int = 3, damping: float = 1.0,
               partition_of_unity: bool = True, verbose: bool = False):
    """Self-consistent ePSF from many lines (Anderson & King iteration).

    Each iteration: fit (A, c, b0, b1) of every line with the current shape;
    map samples to d = u - c, z = (y - b0 - b1 d) / A with weight
    (A / sigma)^2 (per-line total weight capped at ``snr_cap`` ^ 2 so the
    brightest lines do not dominate); solve the regularised B-spline
    least-squares problem; normalise to unit integral; re-centre so the
    centroid within |d| <= ``center_half`` is 0.  Samples with normalised
    residual > ``clip`` sigma are down-weighted to zero (robustness to
    neighbours).  Stops when max |delta c| < ``tol`` px and (if
    ``equalize_phase``) the phase warp amplitude < ``tol``.

    Why ``equalize_phase``: for 1-D undersampled lines at UNKNOWN phases the
    joint (shape, centres) problem is degenerate up to a periodic phase warp
    -- a slightly wider shape plus centres pulled by g(phi) fits the samples
    as well as the truth (verified on synthetic data: chi2/nu 0.9367 vs
    0.9358).  Plain alternation stalls on such a warped solution.  True line
    positions are uniform in sub-pixel phase, so each iteration applies the
    smooth warp (:func:`phase_warp`) that makes the fitted phases uniform
    before rebuilding the shape (the empirical pixel-phase correction used in
    undersampled astrometry).  The build set is therefore flat by
    construction; unbiasedness must be validated on HELD-OUT lines.

    ``warp_mode='dither_struct'`` (recommended for recurring lines) goes
    further: each iteration replaces the free centres of recurring lines by
    the structured centres kappa_L + a_r (:func:`dither_structure`), refits
    area/background at those fixed centres, and builds the shape from them.
    A phase bias cannot survive because, within a line, the relative
    positions across runs are set by the registration a_r, which is
    estimated from all lines at once.  No Fourier warp and no
    significance gate are needed.

    ``warp_mode='dither'`` replaces the uniformity prior by the dither
    constraint of :func:`dither_warp` (needs ``LineData.info`` keys
    ``group`` = physical line id, ``run`` = exposure id, ``upos`` = pixel
    position).  Use it when the lines are a few distinct transitions
    recurring over many runs, whose true phases are NOT a fair uniform
    sample.

    Returns (epsf, fits, history) where fits is a list of per-line fit dicts.
    """
    ep = init if init is not None else gauss_epsf(0.55, h, R)
    if ep.h != h or ep.R != R:
        ep = EPSF.from_function(ep, h, R)
    prev = None
    history = []
    fits = []
    for it in range(n_iter):
        fits = [fit_line(ep, L.u, L.y, L.sigma, c0=L.c0, half=half,
                         fallback=False, neighbors=L.info.get("neighbors"))
                for L in lines]
        craw = np.array([f["center"] if f.get("ok") else np.nan for f in fits])
        winfo = {}
        if equalize_phase and warp_mode == "dither_struct":
            cst, used, winfo = dither_structure(
                craw, [L.info.get("group") for L in lines],
                [L.info.get("run") for L in lines],
                [L.info.get("upos", L.c0) for L in lines], run_slope=run_slope,
                min_per_run=5 if run_slope else 3)
            ccorr = np.where(np.isfinite(cst), cst, craw)
            dd = ccorr - craw
            warp_amp = float(np.nanmax(np.abs(dd))) if np.any(np.isfinite(dd)) else 0.0
            # phase-binned mean correction = the warp this step removes
            ph = phase_of(np.nan_to_num(craw))
            ok_ = np.isfinite(cst)
            if ok_.sum() > 20:
                bins = np.linspace(-0.5, 0.5, 9)
                mb = [np.nanmean(dd[ok_ & (ph >= a) & (ph < b)]) for a, b in zip(bins[:-1], bins[1:])]
                warp_amp = float(np.nanmax(mb) - np.nanmin(mb))
            winfo["frac_structured"] = float(np.mean(ok_))
        elif equalize_phase and warp_mode == "dither":
            g, winfo = dither_warp(craw, [L.info.get("group") for L in lines],
                                   [L.info.get("run") for L in lines],
                                   [L.info.get("upos", L.c0) for L in lines], n_harm,
                                   run_slope=run_slope, min_per_run=5 if run_slope else 3)
            grid = np.linspace(-0.5, 0.5, 201)
            # apply only a SIGNIFICANT warp (95 %): below that the dither
            # data cannot distinguish it from zero and applying it only
            # injects noise into the shape
            from scipy.stats import chi2 as _chi2
            sig = winfo.get("chi2_sig", np.inf)
            if not np.isfinite(sig) or sig < _chi2.ppf(0.95, 2 * n_harm):
                winfo["applied"] = False
                g = (lambda p: np.zeros_like(np.asarray(p, float)))
            else:
                winfo["applied"] = True
            warp_amp = float(np.max(np.abs(g(grid))))
            ccorr = craw + damping * np.where(np.isfinite(craw), g(phase_of(np.nan_to_num(craw))), 0.0)
        elif equalize_phase:
            g = phase_warp(craw[np.isfinite(craw)], n_harm)
            grid = np.linspace(-0.5, 0.5, 201)
            warp_amp = float(np.max(np.abs(g(grid))))
            ccorr = craw + damping * np.where(np.isfinite(craw), g(phase_of(np.nan_to_num(craw))), 0.0)
        else:
            warp_amp = 0.0
            ccorr = craw
        D, Z, W = [], [], []
        cs = []
        for L, f, cc in zip(lines, fits, ccorr):
            if not f.get("ok"):
                cs.append(np.nan)
                continue
            cs.append(f["center"])
            m = _window(L.u, L.c0, half)
            d = L.u[m] - cc
            A, b0, b1 = f["area"], f["b0"], f["b1"]
            if warp_mode == "dither_struct" and cc != f["center"]:
                A, b0, b1 = _refit_linear(ep, L.u[m], L.y[m], L.sigma[m], cc, f["neighbor_model"])
                if not (np.isfinite(A) and A > 0.5 * f["area"]):
                    continue
            if not (np.isfinite(A) and A > 0 and A > 3 * f.get("area_err", 0.0)):
                continue
            z = (L.y[m] - f["neighbor_model"] - b0 - b1 * d) / A
            if not np.all(np.isfinite(z)):
                continue
            w = (A / L.sigma[m]) ** 2
            tot = w.sum()
            cap = snr_cap ** 2 * m.sum()
            if tot > cap:
                w = w * cap / tot
            dfit = L.u[m] - f["center"]
            rn = f["resid"] / L.sigma[m]
            chi = np.sqrt(max(f["chi2nu"], 1.0))
            w = np.where(np.abs(rn) > clip * chi, 0.0, w)
            D.append(d); Z.append(z); W.append(w)
        D = np.concatenate(D); Z = np.concatenate(Z); W = np.concatenate(W)
        # weights normalised so the data weight per B-spline coefficient is 1:
        # ``lam`` is then the smoothness penalty relative to the data
        ncoef = int(round(2 * R / h)) - 3
        Wn = W * ncoef / np.sum(W)
        coef = _solve_shape(D, Z, Wn, h, R, lam, partition_of_unity)
        new = EPSF(coef, h, R)
        integ = new.integral()
        if not np.isfinite(integ) or integ <= 0 or not np.all(np.isfinite(coef)):
            history.append(dict(iter=it, n_ok=int(np.isfinite(craw).sum()), max_dc=np.nan,
                                fwhm=ep.fwhm(), shift=0.0, warp_amp=np.nan,
                                failed="degenerate shape solve; previous shape kept"))
            break
        new = new.scaled(1.0 / integ)
        s = new.centroid(center_half)
        if abs(s) > 1e-6:
            new = new.shifted(s)
            new = new.scaled(1.0 / new.integral())
        cs = np.array(cs)
        dc = np.nan if prev is None else float(np.nanmax(np.abs(cs - prev)))
        history.append(dict(iter=it, n_ok=int(np.isfinite(cs).sum()),
                            max_dc=dc, fwhm=new.fwhm(), shift=s,
                            warp_amp=warp_amp,
                            warp_chi2=winfo.get("chi2_sig"), warp_coef=winfo.get("coef"),
                            warp_coef_sd=winfo.get("coef_sd"),
                            dither_resid_mad=winfo.get("resid_mad")))
        if verbose:
            print(history[-1])
        ep = new
        if prev is not None and dc < tol and warp_amp < tol:
            break
        prev = cs
    fits = [fit_line(ep, L.u, L.y, L.sigma, c0=L.c0, half=half, fallback=False,
                     neighbors=L.info.get("neighbors")) for L in lines]
    ep.meta.update(dict(n_lines=int(sum(f.get("ok", False) for f in fits)),
                        iterations=len(history), lam=lam, h=h, R=R,
                        half=half, center_half=center_half,
                        equalize_phase=equalize_phase, warp_mode=warp_mode, n_harm=n_harm,
                        final_warp_amp=history[-1]["warp_amp"] if history else None))
    return ep, fits, history
