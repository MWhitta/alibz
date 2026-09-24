"""Synthetic-truth tests for alibz.utils.epsf (ePSF build + fitter)."""
import numpy as np
import pytest

from alibz.utils.epsf import (EPSF, LineData, build_epsf, convolve_kernel,
                              fit_line, fit_pixint_gauss, gauss_epsf,
                              phase_of, physical_kernel, pixint_gauss)


def true_lsf_px(d, sigma=0.45, slit=0.9):
    """Flat-topped instrument LSF (Gaussian (x) slit box), NOT pixel-integrated."""
    from scipy.special import erf
    s = sigma * np.sqrt(2)
    return 0.5 * (erf((d + slit / 2) / s) - erf((d - slit / 2) / s)) / slit


def pixel_integrate(f, d, n=64):
    off = (np.arange(n) + 0.5) / n - 0.5
    return np.mean([f(d + o) for o in off], axis=0)


def truth_epsf(sigma=0.45, slit=0.9):
    return EPSF.from_function(lambda d: pixel_integrate(lambda x: true_lsf_px(x, sigma, slit), d))


def make_line(rng, shape, center, area=1000.0, bg=(20.0, 0.5), noise=1.0, n=21,
              ushift=0):
    u = np.arange(n, dtype=float) + ushift
    y = area * shape(u - center) + bg[0] + bg[1] * (u - center)
    y = y + rng.normal(0, noise, n)
    return u, y, np.full(n, noise)


def test_normalisation_and_partition_of_unity():
    ep = truth_epsf()
    assert abs(ep.integral() - 1) < 1e-3
    for c in np.linspace(-0.5, 0.5, 11):
        s = ep(np.arange(-4, 5) - c).sum()
        assert abs(s - 1) < 5e-3


def test_phase_helper():
    assert np.allclose(phase_of([0.2, 0.7, -0.6, 3.5]), [0.2, -0.3, 0.4, -0.5])


@pytest.mark.parametrize("snr", [30, 300])
def test_fit_unbiased_with_true_shape(snr):
    rng = np.random.default_rng(1)
    ep = truth_epsf()
    errs, aerr = [], []
    for c in np.linspace(9.5, 10.5, 41):
        u, y, s = make_line(rng, ep, c, area=1000.0, noise=1000.0 * ep.peak() / snr)
        f = fit_line(ep, u, y, s, c0=float(np.round(c)))
        assert f["ok"] and f["fallback"] is None
        errs.append(f["center"] - c); aerr.append(f["area"] / 1000 - 1)
    errs = np.array(errs)
    assert abs(errs.mean()) < 3.0 / snr
    assert np.std(errs) < 10.0 / snr
    assert abs(np.mean(aerr)) < 5.0 / snr


def test_gaussian_model_is_phase_biased_on_flat_top_but_epsf_is_not():
    """The motivating failure: pixel-int Gaussian on a flat-topped LSF."""
    ep = truth_epsf(0.3, 1.1)
    cs = np.linspace(9.5, 10.5, 21)
    g_err, e_err = [], []
    for c in cs:
        u = np.arange(21.0)
        y = 1000 * ep(u - c) + 5.0
        g = fit_pixint_gauss(u, y, c0=float(np.round(c)))
        e = fit_line(ep, u, y, c0=float(np.round(c)))
        g_err.append(g["center"] - c); e_err.append(e["center"] - c)
    assert np.ptp(e_err) < 2e-3
    assert np.ptp(g_err) > 5 * np.ptp(e_err)


def _synthetic_build_set(truth, n, rng, stratified=False):
    lines, true_c = [], []
    phs = ((np.arange(n) + 0.5) / n - 0.5) if stratified else rng.uniform(-0.5, 0.5, n)
    rng.shuffle(phs)
    for ph in phs:
        c = 10 + ph
        u, y, s = make_line(rng, truth, c, area=rng.uniform(500, 5000), noise=5.0,
                            bg=(rng.uniform(0, 50), rng.uniform(-1, 1)))
        lines.append(LineData(u, y, s, c0=float(np.round(c))))
        true_c.append(c)
    return lines, np.array(true_c)


def _phase_bias(cent, true_c, nb=8):
    err = cent - true_c
    err = err - np.median(err)          # centring convention = global constant
    ph = phase_of(true_c)
    bins = np.linspace(-0.5, 0.5, nb + 1)
    return np.array([np.median(err[(ph >= a) & (ph < b)]) for a, b in zip(bins[:-1], bins[1:])])


@pytest.mark.parametrize("sig,slit,init", [(0.35, 1.0, 0.6), (0.25, 1.4, 0.4), (0.35, 0.01, 0.8)])
def test_build_recovers_shape_and_flat_phase(sig, slit, init):
    rng = np.random.default_rng(7)
    truth = truth_epsf(sig, slit)
    lines, true_c = _synthetic_build_set(truth, 300, rng, stratified=True)
    ep, fits, hist = build_epsf(lines, init=gauss_epsf(init), lam=0.03, n_iter=40)
    med = _phase_bias(np.array([f["center"] for f in fits]), true_c)
    assert np.ptp(med) < 0.006, med
    assert abs(ep.fwhm() - truth.fwhm()) < 0.02 * truth.fwhm()
    g = [fit_pixint_gauss(L.u, L.y, L.sigma, c0=L.c0)["center"] for L in lines]
    assert np.ptp(_phase_bias(np.array(g), true_c)) > 2 * np.ptp(med)


def test_build_random_phases_generalises_to_heldout():
    rng = np.random.default_rng(5)
    truth = truth_epsf(0.35, 1.0)
    lines, true_c = _synthetic_build_set(truth, 600, rng)
    test, test_c = _synthetic_build_set(truth, 400, np.random.default_rng(99))
    ep, _, _ = build_epsf(lines, init=gauss_epsf(0.6), lam=0.03, n_iter=40)
    cen = np.array([fit_line(ep, L.u, L.y, L.sigma, c0=L.c0)["center"] for L in test])
    # statistical floor: a random sample of N true phases is itself only
    # uniform to ~1/sqrt(N); equalising it leaves a first-harmonic warp of
    # ~0.02-0.06 px p-p for N = 300-600 (measured over seeds), vs >0.1 px for
    # an unconstrained build.
    assert np.ptp(_phase_bias(cen, test_c)) < 0.05


def test_phase_warp_degeneracy_needs_uniform_phase_prior():
    """Without phase equalisation the alternation stalls on a warped solution."""
    rng = np.random.default_rng(7)
    truth = truth_epsf(0.35, 0.01)
    lines, true_c = _synthetic_build_set(truth, 300, rng, stratified=True)
    ep0, f0, _ = build_epsf(lines, init=gauss_epsf(0.8), lam=0.03, n_iter=40,
                            equalize_phase=False)
    ep1, f1, _ = build_epsf(lines, init=gauss_epsf(0.8), lam=0.03, n_iter=40)
    b0 = np.ptp(_phase_bias(np.array([f["center"] for f in f0]), true_c))
    b1 = np.ptp(_phase_bias(np.array([f["center"] for f in f1]), true_c))
    assert b0 > 0.05 and b1 < 0.006


def _recurring_set(truth, nL, nR, rng, drift=0.3):
    posL = rng.uniform(50, 1900, nL); a = rng.uniform(-drift, drift, nR); b = rng.normal(0, 0.1, nR)
    lines, tc = [], []
    for r in range(nR):
        for l in range(nL):
            c = posL[l] + a[r] + b[r] * (posL[l] - 1000) / 1000
            k = int(round(c)); u = np.arange(k - 10, k + 11, dtype=float)
            y = rng.uniform(500, 5000) * truth(u - c) + rng.uniform(0, 50) + rng.normal(0, 5.0, u.size)
            lines.append(LineData(u, y, np.full(u.size, 5.0), c0=float(k),
                                  info=dict(group=l, run=r, upos=posL[l])))
            tc.append(c)
    return lines, np.array(tc)


def test_dither_warp_beats_uniform_prior_for_recurring_lines():
    """40 transitions x 15 runs (drift +-0.3 px): the true phases are a lumpy
    40-point sample, so the uniform prior is noisy; the dither constraint is not."""
    truth = truth_epsf(0.35, 0.01)
    L, _ = _recurring_set(truth, 40, 15, np.random.default_rng(1))
    test, test_c = _synthetic_build_set(truth, 400, np.random.default_rng(99), stratified=True)
    res = {}
    for mode in ("uniform", "dither", "dither_struct"):
        ep, _, _ = build_epsf(L, init=gauss_epsf(0.8), lam=0.03, n_iter=40, warp_mode=mode)
        cen = np.array([fit_line(ep, T.u, T.y, T.sigma, c0=T.c0)["center"] for T in test])
        res[mode] = np.ptp(_phase_bias(cen, test_c))
        if mode != "uniform":
            assert abs(ep.fwhm() - truth.fwhm()) < 0.01 * truth.fwhm()
    assert res["dither"] < 0.006 and res["dither_struct"] < 0.006
    assert res["uniform"] > 3 * max(res["dither"], res["dither_struct"])


def test_blend_two_components_fit_separately_with_window():
    """A neighbour 3 px away biases a wide window; a narrow window limits it."""
    rng = np.random.default_rng(3)
    ep = truth_epsf()
    u = np.arange(30.0)
    y = 1000 * ep(u - 12.3) + 300 * ep(u - 15.3) + 10 + rng.normal(0, 1, u.size)
    f = fit_line(ep, u, y, np.ones_like(u), c0=12, half=2.0)
    assert f["ok"] and abs(f["center"] - 12.3) < 0.05


def test_neighbor_component_removes_blend_bias():
    rng = np.random.default_rng(4)
    ep = truth_epsf()
    errs_n, errs_0 = [], []
    for c in np.linspace(11.5, 12.5, 11):
        u = np.arange(30.0)
        y = 1000 * ep(u - c) + 800 * ep(u - c - 3.2) + 10 + rng.normal(0, 1, u.size)
        f0 = fit_line(ep, u, y, np.ones_like(u), c0=round(c), half=4.5)
        fn = fit_line(ep, u, y, np.ones_like(u), c0=round(c), half=4.5,
                      neighbors=[round(c + 3.2)])
        errs_0.append(f0["center"] - c); errs_n.append(fn["center"] - c)
        assert abs(fn["area"] / 1000 - 1) < 0.01
    assert np.max(np.abs(errs_n)) < 0.005
    assert np.max(np.abs(errs_0)) > 10 * np.max(np.abs(errs_n))


def test_kernels_unit_area_and_width_growth():
    x, k = physical_kernel("lorentz", [0.3])
    assert abs(k.sum() - 1) < 1e-9
    ep = truth_epsf()
    w = []
    for tau in (0.1, 3, 30):
        f = convolve_kernel(ep, "selfabs", [tau, 0.1])
        d = np.linspace(-5, 5, 2001); v = f(d)
        assert abs(np.trapezoid(v, d) - 1) < 2e-2   # Lorentz wings beyond +-5 px
        above = d[v >= v.max() / 2]; w.append(above[-1] - above[0])
    assert w[0] < w[1] < w[2]


def test_thick_line_recovered_with_selfabs_kernel():
    rng = np.random.default_rng(11)
    ep = truth_epsf()
    f_true = convolve_kernel(ep, "selfabs", [20.0, 0.15])
    u = np.arange(25.0)
    y = 5000 * f_true(u - 12.2) + 10 + rng.normal(0, 2, u.size)
    thin = fit_line(ep, u, y, np.full(u.size, 2.0), c0=12, half=4)
    thick = fit_line(ep, u, y, np.full(u.size, 2.0), c0=12, half=4,
                     kernel="selfabs", kernel_p0=[3.0, 0.1])
    assert thick["ok"]
    assert thick["chi2nu"] < 0.2 * thin["chi2nu"]
    assert abs(thick["center"] - 12.2) < 0.02
    assert abs(thick["area"] / 5000 - 1) < 0.03


def test_fallback_and_never_raises():
    ep = truth_epsf()
    u = np.arange(21.0)
    y = np.zeros(21); y[3] = 1e4
    f = fit_line(ep, u, y, c0=15.0, center_bound=0.3)
    assert "ok" in f
    assert fit_line(ep, u[:3], y[:3])["ok"] is False
    assert fit_line(ep, u, np.full(21, np.nan))["ok"] is False


def test_roundtrip_dict():
    ep = truth_epsf()
    e2 = EPSF.from_dict(ep.to_dict())
    d = np.linspace(-3, 3, 50)
    assert np.allclose(ep(d), e2(d))
