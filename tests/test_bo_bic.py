"""Opt-in GP-BO settings (bo_settings) and BIC effective-n (bic_effective_n).

Both default OFF; the OFF paths must be unchanged.
"""
import numpy as np
import pytest

from alibz.peaky_indexer_v3 import PeakyIndexerV3
from alibz.refinement import _bic, effective_n, resolve_bic_rho


def _toy_search(bo_settings, n_calls=12, seed=3):
    """Run the indexer's gp_minimize wiring on a quadratic toy objective."""
    from skopt import gp_minimize
    from skopt.space import Real

    truth = np.array([12000.0, 16.5, 0.08, 0.05])
    lo = np.array([4000.0, 14.0, 0.01, 0.01])
    hi = np.array([25000.0, 19.0, 0.30, 0.30])
    dims = [Real(l, h, name=n) for l, h, n in zip(lo, hi, "T ne s g".split())]
    calls = []

    def f(p):
        # sigma/gamma enter in log space (the BO space is log-uniform there
        # since their bounds span > 1 decade); T and log n_e linearly.
        # (On a toy that is quadratic in LINEAR sigma/gamma the log prior
        # costs ~5x in final cost at 12 calls -- see the report.)
        q = np.asarray(p, dtype=float).copy()
        t_ = truth.copy()
        q[2:] = np.log(q[2:])
        t_[2:] = np.log(t_[2:])
        z = (q - t_) / np.where(np.arange(4) >= 2, np.log(hi / lo), hi - lo)
        v = float(np.sum(z * z))
        calls.append(v)
        return v

    # grid seeds exactly as in fit(search="grid"): 15 x 7 over (T, ne)
    gx, gy = [], []
    for Tg in np.linspace(lo[0], hi[0], 15):
        for ng in np.linspace(lo[1], hi[1], 7):
            p = [float(Tg), float(ng), 0.1, 0.1]
            gx.append(p)
            gy.append(f(p))
    order = np.argsort(gy)[:10]
    x0, y0 = [gx[i] for i in order], [gy[i] for i in order]
    del calls[:]
    kw = {}
    if bo_settings:
        obj = object.__new__(PeakyIndexerV3)
        dims, kw = obj._bo_matern52(dims, seed, n_seeded=len(y0))
    res = gp_minimize(f, dims, n_calls=n_calls, n_initial_points=1, x0=x0,
                      y0=y0, random_state=seed, verbose=False, **kw)
    return res, x0, y0, truth, lo, hi


def test_matern52_finds_truth_like_default():
    for seed in (3, 4, 5):
        base, *_ = _toy_search(None, seed=seed)
        mat, x0, y0, truth, lo, hi = _toy_search("matern52", seed=seed)
        # grid seeds retained; n_calls=12 new evaluations on top of them
        assert len(mat.func_vals) >= len(y0) + 6
        assert mat.fun < min(y0)
        assert mat.fun <= base.fun * 1.05 + 1e-3, (seed, mat.fun, base.fun)
        zn = np.abs((np.asarray(mat.x) - truth) / (hi - lo))
        assert zn[0] < 0.15 and zn[1] < 0.15, (seed, zn)


def test_matern52_early_stop_not_before_six_calls():
    mat, x0, y0, *_ = _toy_search("matern52", n_calls=12)
    assert len(mat.func_vals) - len(y0) >= 6      # seeds are not calls
    stop = PeakyIndexerV3._bo_stopper()

    class R:  # a perfectly flat trace must still not stop before 6 calls
        func_vals = np.ones(5)
    assert stop(R) is False
    R.func_vals = np.ones(6)
    assert stop(R) is True            # 6 flat calls: improvement 0 < 0.5 %
    R.func_vals = np.array([10, 9, 8, 7, 6, 5.0])
    assert stop(R) is False           # still improving by 50 %


def test_invalid_bo_settings_rejected():
    with pytest.raises(ValueError):
        PeakyIndexerV3(np.zeros((1, 6)), dbpath="db", bo_settings="rbf")


def test_bic_penalty_uses_effective_n():
    n, k = 200, 4
    rho = 0.92
    n_eff = n * (1 - rho) / (1 + rho)
    assert effective_n(n, 0.0) == n and abs(effective_n(n, rho) - n_eff) < 1e-12
    # one extra parameter costs ln(n) off, ln(n_eff) on
    d_off = _bic(0.0, n, k + 1) - _bic(0.0, n, k)
    d_on = _bic(0.0, n, k + 1, rho) - _bic(0.0, n, k, rho)
    assert abs(d_off - np.log(n)) < 1e-12
    assert abs(d_on - np.log(n_eff)) < 1e-12
    assert d_on < d_off - 2.0
    assert _bic(5.0, n, k) == _bic(5.0, n, k, 0.0)


def test_auto_rho_by_grid():
    export = np.arange(300.0, 400.0, 1.0 / 30.0)
    native = np.cumsum(np.full(3000, 0.12))
    native = 300.0 + native + 0.01 * np.sin(np.arange(3000))     # nonuniform
    assert resolve_bic_rho(export, "auto") == 0.92
    assert resolve_bic_rho(native, "auto") == 0.0
    assert resolve_bic_rho(np.arange(300.0, 400.0, 0.1), "auto") == 0.0
    assert resolve_bic_rho(export, None) == 0.0
    assert resolve_bic_rho(export, 0.5) == 0.5
    with pytest.raises(ValueError):
        resolve_bic_rho(export, 1.0)
    with pytest.raises(ValueError):
        resolve_bic_rho(export, "bad")


def test_options_off_equal_default():
    from bench.synth_cases import CASES, render_case
    from alibz.pipeline import analyze_spectrum

    spec = dict(CASES["binary_ca_mg"])
    comp = spec.pop("composition")
    x, y, _ = render_case(comp, seed=11, **spec)
    kw = dict(n_calls=8, draws=2)
    a = analyze_spectrum(x, y, "db", **kw)
    b = analyze_spectrum(x, y, "db", bo_settings=None, bic_effective_n=None, **kw)
    ra, rb = a["result"], b["result"]
    assert dict(ra.element_fractions) == dict(rb.element_fractions)
    assert ra.temperature == rb.temperature
    key = lambda d: sorted((k, repr(v)) for k, v in d.items()
                           if isinstance(v, (str, int, float, bool, type(None))))
    assert [key(d) for d in a["decisions"]] == [key(d) for d in b["decisions"]]
    assert "bo_n_calls_used" not in ra.convergence_info
