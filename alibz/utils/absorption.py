"""Self-absorption (optical depth) corrections for emission lines.

The optically thin assumption fails on ground-state resonance lines of
major elements: measured on a Li-rich mineral, the K I 766.5/769.9
doublet area ratio was 1.30 against the T-independent optically-thin
value of 2.02 (shared upper levels), and Na D gave 1.51 vs 2.00 — a
~35% compression on exactly the lines that identify the alkalis.  A
linear (thin) design matrix has no degree of freedom to absorb this.

Model: homogeneous-slab escape factor.  The integrated absorption of a
line is

    integral k dnu = n_l * (gA)_upper * lambda^2 / (8 pi Z_s) ,

so the line-centre optical depth is proportional to the element density
times a per-line RELATIVE strength

    kappa_j ~ lambda_j^2 * gA_j * exp(-E_lower,j / kT) / Z_s(T) * f_stage,

(the lower-level population, not the upper), and the emitted line area
is compressed by the escape factor

    f_SA(tau) = (1 - exp(-tau)) / tau .

A single global scale s_tau maps concentration * kappa to tau; it
absorbs the column length and absolute density scale, and is either set
externally or fitted as an outer parameter.
"""

import numpy as np


def escape_factor(tau):
    """Homogeneous-slab escape factor ``(1 - exp(-tau)) / tau``.

    -> 1 as tau -> 0 (optically thin), ~ 1/tau for tau >> 1 (saturated).
    Safe at tau = 0; negative tau is clipped to 0.
    """
    tau = np.clip(np.asarray(tau, dtype=float), 0.0, None)
    small = tau < 1e-6
    safe = np.where(small, 1.0, tau)
    out = (1.0 - np.exp(-safe)) / safe
    # second-order expansion keeps the thin limit exact
    return np.where(small, 1.0 - tau / 2.0, out)


def doublet_ratio(tau_weak, strength_ratio=2.0):
    """Predicted area ratio of a resonance doublet under self-absorption.

    The stronger member has ``tau = strength_ratio * tau_weak`` (shared
    lower level, gA ratio = thin intensity ratio).  Returns
    ``I_strong / I_weak``; -> ``strength_ratio`` as tau -> 0 and -> 1 as
    tau -> infinity.  Useful to invert a measured doublet ratio into an
    optical-depth estimate (K I 766.5/769.9 at 1.30 implies
    tau_weak ~ 1.25).
    """
    tau_weak = np.asarray(tau_weak, dtype=float)
    num = strength_ratio * escape_factor(strength_ratio * tau_weak)
    den = escape_factor(tau_weak)
    return num / den


def invert_doublet_tau(measured_ratio, thin_ratio, strength_ratio=None,
                       tau_max=50.0):
    """Per-element optical depth from a measured resonance-doublet ratio.

    Doublet members share (nearly) the same lower level, so their optical
    depths are locked in the known ratio of their absorption strengths:
    given the measured area ratio ``R`` and the optically thin emissivity
    ratio ``R0``, solve

        R = R0 * f(strength_ratio * tau_weak) / f(tau_weak)

    for ``tau_weak`` (``f`` the escape factor).  This anchors an element's
    self-absorption DIRECTLY to its own measured lines — no global scale
    can misallocate it (measured failure of the global model: fitted K
    doublet ratio 1.97 against 1.37 observed).

    ``strength_ratio`` defaults to ``thin_ratio`` (shared upper-level
    structure makes emission and absorption ratios equal).  Returns 0
    when ``R >= R0`` (no measurable compression) and ``tau_max`` when the
    doublet is compressed beyond the model's saturated limit.
    """
    from scipy.optimize import brentq

    R = float(measured_ratio)
    R0 = float(thin_ratio)
    sr = float(strength_ratio) if strength_ratio is not None else R0
    if not np.isfinite(R) or not np.isfinite(R0) or R0 <= 1.0 or sr <= 1.0:
        return 0.0
    if R >= R0:
        return 0.0

    def g(t):
        return R0 * float(escape_factor(sr * t)) / float(escape_factor(t)) - R

    if g(tau_max) > 0.0:
        # even at tau_max the model ratio stays above the measurement:
        # saturated beyond the model's range
        return float(tau_max)
    return float(brentq(g, 1e-9, tau_max))
