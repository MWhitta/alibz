"""Tests for :mod:`alibz.wavelength_registration`.

Synthetic spectra with known per-segment offsets and slopes, planted in real
database line forests, exercise the two-stage estimator; separate tests cover
the ambient (argon/no-argon) path, the combined disagreement flag, the vendor
calibration change, and the thermal drift model.
"""

import numpy as np
import pytest

from alibz.utils.database import Database
from alibz import wavelength_registration as wr


@pytest.fixture(scope="module")
def db():
    return Database("db")


# --- synthetic spectrum builder -------------------------------------------
def _native_grid():
    segs = [(186.0, 365.0, 0.089), (365.0, 620.0, 0.129), (620.0, 948.0, 0.179)]
    return np.concatenate([np.arange(a, b, p) for a, b, p in segs])


def _pitch(wl):
    return 0.089 if wl < 365 else (0.129 if wl < 620 else 0.179)


def _plant(x, lines, shift_fn, *, amp_scale=200.0, seed=0, noise=3.0):
    """Plant bright Gaussian lines at db + shift_fn(db) on a noisy baseline.

    The strongest lines get near-equal high amplitude (many high-SNR observed
    peaks, as in real spectra) so the vote-mode estimator has enough pairings.
    """
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, noise, x.size)
    svals = [s for _, s in lines if s > 0]
    if not svals:
        return y
    smax = max(svals)
    floor = 1e-4 * smax
    for wl, s in lines:
        if wl < x[0] or wl > x[-1] or s <= floor:
            continue
        amp = amp_scale * (0.5 + 0.5 * min(1.0, np.log10(s / floor) / 4.0))
        obs = wl + float(shift_fn(wl))
        y += amp * np.exp(-0.5 * ((x - obs) / (0.6 * _pitch(wl))) ** 2)
    return y


# --- vendor calibration ----------------------------------------------------
def test_vendor_calibration_shift_matches_handheld():
    base = [[3.67951969E2, -7.78662478E-2, -5.99133112E-6, 4.94396034E-10],
            [6.25593006E2, -1.11792094E-1, -9.35107472E-6, 6.53116663E-10],
            [9.48170506E2, -1.55936742E-1, -1.40437163E-5, 9.01628935E-10],
            [961, -0.0004, 1E-12, 1E-12]]
    cur = [[368.0025857351851, -0.0778498710316756, -5.990071027925716E-6, 4.942920530128833E-10],
           [625.4823570464079, -0.11177738525661182, -9.349844377553242E-6, 6.530307309358004E-10],
           [947.9356738154337, -0.1559247671299144, -1.404263783910583E-5, 9.0155969609438E-10],
           [961.0, -4.0E-4, 1.0E-12, 1.0E-12]]
    out = wr.vendor_calibration_shift(base, cur, [180, 365, 620, 960, 961])
    segs = out["segments"]
    assert set(segs) == {"UV", "VIS", "NIR"}  # dummy 960-961 segment dropped
    uv = segs["UV"]
    # brief: UV recalibration is +50..+85 pm across the segment
    assert 0.045 < uv["delta_nm_at_hi"] < 0.095   # near 365 nm
    assert 0.045 < uv["delta_nm_at_lo"] < 0.10     # near 180 nm
    assert uv["delta_nm_mean"] > 0
    # VIS and NIR move the other way and are larger
    assert segs["VIS"]["delta_nm_mean"] < 0
    assert segs["NIR"]["delta_nm_mean"] < -0.15


# --- element registration --------------------------------------------------
def test_vote_mode_recovers_linear_shift_with_aliases():
    """The vote-mode core recovers a known linear-in-lambda shift despite
    decoy observed peaks planted at +-0.35 nm aliases of true lines."""
    rng = np.random.default_rng(101)
    cfg = wr._config(None)
    # synthetic VIS database: 150 lines, log-uniform strengths
    dbwl = np.sort(rng.uniform(420.0, 600.0, 150))
    dbs = 10.0 ** rng.uniform(2.0, 8.0, dbwl.size)
    ref = 510.0
    true = lambda w: -0.12 + 0.0006 * (w - ref)   # linear, ~ -0.12 at centre
    obs = dbwl + np.array([true(w) for w in dbwl]) + rng.normal(0, 0.005, dbwl.size)
    snr = np.full(obs.size, 30.0)
    # decoy observed peaks at +-0.35 nm aliases of a third of the lines
    didx = rng.choice(dbwl.size, dbwl.size // 3, replace=False)
    dec = dbwl[didx] + rng.choice([-0.35, 0.35], didx.size)
    obs = np.concatenate([obs, dec]); snr = np.concatenate([snr, np.full(dec.size, 20.0)])
    order = np.argsort(obs); obs, snr = obs[order], snr[order]
    sv = wr._segment_vote(obs, snr, dbwl, dbs, 420.0, 600.0, cfg)
    assert sv is not None
    assert abs(1000 * (sv["shift_nm"] - true(510.0))) < 30    # within 30 pm
    assert sv["dominance"] > 1.1
    wins = wr._vote_windows(obs, snr, dbwl, dbs, 420.0, 600.0, cfg)
    assert len(wins) >= 3
    lam = np.array([w["center_nm"] for w in wins])
    sh = np.array([w["shift_nm"] for w in wins])
    keep = np.abs(sh - np.array([true(c) for c in lam])) < 0.1
    slope = np.polyfit(lam[keep], sh[keep], 1)[0]
    assert abs(slope - 0.0006) < 0.0004   # recovers the linear-in-lambda trend


def test_golden_lines_unambiguity_rule(db):
    """Every returned golden line must satisfy the construction rule: no other
    composition+Ar/N/O/H line within +-0.5 nm above 3 % of its strength."""
    cfg = wr._config(None)
    aw, as_, _ = wr._predicted_strength_table(db, ["Fe", "Ar", "N", "O", "H"])
    for seg in ("UV", "VIS"):
        gl = wr.golden_lines(db, ["Fe"], seg)
        assert 1 <= len(gl) <= 40
        for g in gl:
            wl0, s0 = g["wavelength_nm"], g["strength"]
            near = (np.abs(aw - wl0) <= cfg["golden_window_nm"]) & (np.abs(aw - wl0) > 1e-4)
            assert not np.any(as_[near] > cfg["golden_competitor_fraction"] * s0)


def test_golden_registration_recovers_linear_with_decoys(db):
    """Golden-line registration recovers a planted a + b(lambda-lambda0) shift
    on the real VIS grid, with decoy peaks at +-0.35 nm of some golden lines."""
    x = _native_grid()
    ref = 490.0
    a, b = -0.12, 0.0008
    shift_fn = lambda wl: a + b * (wl - ref)
    gl = wr.golden_lines(db, ["Fe"], "VIS")
    rng = np.random.default_rng(2)
    y = rng.normal(0.0, 1.0, x.size)
    sig = 0.6 * 0.129
    for g in gl:                       # plant the golden lines at db + shift
        c = g["wavelength_nm"] + shift_fn(g["wavelength_nm"])
        y += 120 * np.exp(-0.5 * ((x - c) / sig) ** 2)
    for g in gl[::2]:                  # decoys at +-0.35 nm (weaker)
        c = g["wavelength_nm"] + 0.35 * (1 if rng.random() > 0.5 else -1)
        y += 40 * np.exp(-0.5 * ((x - c) / sig) ** 2)
    reg = wr.element_registration(x, y, db, ["Fe"])
    seg = reg["segments"]["VIS"]
    assert reg["estimator"] == "golden_line"
    assert seg["quality"] == "ok" and seg["n_lines"] >= 3
    pred = float(wr._eval_model(seg["model"], ref))
    assert abs(pred - a) < 0.05, (pred, a)


def test_fe_to_v_transfer_same_function(db):
    """Fit a shift function from one line set and apply it to another planted
    with the SAME function: the transferred residual must be small."""
    ref = 490.0
    fn = lambda wl: -0.10 + 0.0006 * (wl - ref)
    fe = wr.golden_lines(db, ["Fe"], "VIS")
    v = wr.golden_lines(db, ["V"], "VIS")
    fl = np.array([g["wavelength_nm"] for g in fe])
    fs = np.array([fn(w) for w in fl])
    coeffs, _ = wr._huber_polyfit(fl - ref, fs, 1)
    vl = np.array([g["wavelength_nm"] for g in v])
    vres = 1000.0 * (np.array([fn(w) for w in vl]) - np.polyval(coeffs, vl - ref))
    assert np.median(np.abs(vres)) < 5.0   # same function -> ~0 residual


def test_element_registration_unregistered_on_noise(db):
    """Pure noise yields no clean golden peaks -> segments 'unregistered'."""
    x = _native_grid()
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 2.0, x.size)
    reg = wr.element_registration(x, y, db, ["Fe"])
    assert all(s["quality"] != "ok" for s in reg["segments"].values())


# --- ambient registration --------------------------------------------------
def test_ambient_registration_detects_argon(db):
    x = _native_grid()
    shift_fn = lambda wl: -0.15
    lines = list(zip(*wr._lines_of(db, ["Ar"], 1)))
    y = _plant(x, lines, shift_fn, amp_scale=400.0, seed=3, noise=2.0)
    reg = wr.ambient_registration(x, y, db)
    assert "Ar" in reg["species_detected"]
    nir = reg["segments"]["NIR"]
    assert nir["n_lines"] >= 3
    assert abs(nir["shift_nm"] - (-0.15)) < 0.05


def test_ambient_registration_no_argon_is_quiet(db):
    x = _native_grid()
    rng = np.random.default_rng(4)
    y = rng.normal(0.0, 2.0, x.size)  # pure noise, no lines
    reg = wr.ambient_registration(x, y, db)
    # pure noise yields at most a rare spurious weak match, never a segment
    # that clears the quality gate
    assert reg["n_lines"] <= 2
    assert reg["quality"] in ("none", "weak")
    assert all(s["quality"] != "ok" for s in reg["segments"].values())


# --- combined registration -------------------------------------------------
def _fake_seg(shift, sigma, n, quality="ok", ref=780.0):
    return {"shift_nm": shift, "mad_nm": sigma, "sigma_nm": sigma,
            "slope_nm_per_nm": None, "n_lines": n,
            "model": {"type": "const", "coeffs": [shift], "ref_nm": ref,
                      "rms_nm": sigma, "residuals_nm": []},
            "windows": [], "quality": quality, "residual_mad_nm": sigma}


def _fake_reg(source, segs, gshift):
    return {"version": wr.REGISTRATION_VERSION, "source": source,
            "convention": "observed_minus_database_nm", "prior_shift_nm": 0.0,
            "species": [], "species_detected": [], "n_lines": 10,
            "global_shift_nm": gshift, "global_mad_nm": 0.02,
            "segments": segs, "residual_histogram": None, "outliers": [],
            "lines": [], "quality": "ok"}


def test_combined_flags_nir_disagreement():
    amb = _fake_reg("ambient", {"UV": _fake_seg(None, None, 0, "none"),
                                "VIS": _fake_seg(None, None, 0, "none"),
                                "NIR": _fake_seg(-0.15, 0.01, 6)}, -0.15)
    ele = _fake_reg("element", {"UV": _fake_seg(0.05, 0.01, 8),
                                "VIS": _fake_seg(-0.09, 0.01, 10),
                                "NIR": _fake_seg(-0.30, 0.01, 7)}, -0.10)
    comb = wr.combined_registration(amb, ele)
    assert comb["nir_disagreement"]["flagged"] is True
    assert comb["nir_disagreement"]["n_sigma"] > 2.0
    # element chosen where its segment passes; NIR present from both
    assert comb["segments"]["UV"]["source"] == "element"


def test_combined_agrees_when_close():
    amb = _fake_reg("ambient", {"UV": _fake_seg(None, None, 0, "none"),
                                "VIS": _fake_seg(None, None, 0, "none"),
                                "NIR": _fake_seg(-0.20, 0.01, 6)}, -0.20)
    ele = _fake_reg("element", {"UV": _fake_seg(0.05, 0.01, 8),
                                "VIS": _fake_seg(-0.09, 0.01, 10),
                                "NIR": _fake_seg(-0.205, 0.01, 7)}, -0.10)
    comb = wr.combined_registration(amb, ele)
    assert comb["nir_disagreement"]["flagged"] is False


# --- apply / json ----------------------------------------------------------
def test_apply_registration_moves_axis_into_db_frame():
    edges = list(wr.SEGMENT_EDGES)
    models = [{"type": "const", "coeffs": [0.05], "ref_nm": 0.0},
              {"type": "const", "coeffs": [-0.09], "ref_nm": 0.0},
              {"type": "const", "coeffs": [-0.20], "ref_nm": 0.0}]
    comb = {"edges_nm": edges, "models": models, "global_shift_nm": -0.09,
            "segment_names": list(wr.SEGMENT_NAMES)}
    x = np.array([250.0, 500.0, 800.0])
    xc = wr.apply_registration(x, comb)
    np.testing.assert_allclose(xc, x - np.array([0.05, -0.09, -0.20]))


def test_registration_to_json_is_serializable(db):
    import json
    x = _native_grid()
    y = _plant(x, list(zip(*wr._lines_of(db, ["Fe"], 2))),
               lambda wl: -0.1, seed=5)
    reg = wr.combined_registration(wr.ambient_registration(x, y, db),
                                   wr.element_registration(x, y, db, ["Fe"]))
    s = json.dumps(wr.registration_to_json(reg))
    assert isinstance(s, str) and len(s) > 10


def test_config_rejects_unknown_keys():
    with pytest.raises(KeyError):
        wr._config({"not_a_key": 1})


# --- thermal drift model ---------------------------------------------------
def test_thermal_drift_resolved_slope():
    rng = np.random.default_rng(7)
    recs = []
    for m in np.linspace(0, 180, 24):
        recs.append({"minutes_since_calibration": float(m),
                     "segment_shifts": {"UV": 0.02 + 0.0005 * m + rng.normal(0, 0.003),
                                        "VIS": -0.09 + rng.normal(0, 0.003),
                                        "NIR": -0.2 + rng.normal(0, 0.003)},
                     "temperature_c": 25 + 0.05 * m,
                     "ambient_nir_shift_nm": -0.2})
    model = thermal = wr.thermal_drift_model(recs)
    uv = model["segments"]["UV"]
    assert uv["resolved"] is True
    assert abs(uv["slope_nm_per_hour"] - 0.03) < 0.01  # 0.0005 nm/min * 60
    pred = model["predict_shift"]("UV", minutes_since_calibration=100)
    assert abs(pred - (0.02 + 0.05)) < 0.02
    # a segment with no drift is unresolved and predicts 0.0
    assert model["segments"]["VIS"]["resolved"] is False
    assert model["predict_shift"]("VIS", minutes_since_calibration=100) == 0.0


def test_pipeline_registration_on_off(db):
    # The pipeline integration (AnalysisConfig.wavelength_registration) is committed
    # separately from this module; skip until it is present.
    from alibz.pipeline import AnalysisConfig
    if not hasattr(AnalysisConfig, "wavelength_registration"):
        pytest.skip("pipeline wavelength_registration integration not present")
    """analyze_spectrum records the registration and honours the switch."""
    from alibz.pipeline import analyze_spectrum
    x = _native_grid()
    y = _plant(x, list(zip(*wr._lines_of(db, ["Fe"], 2))),
               lambda wl: -0.12, seed=9)
    off = analyze_spectrum(x, y, "db", n_calls=6, wavelength_registration="off")
    assert off["wavelength_registration"] is None
    rep = analyze_spectrum(x, y, "db", n_calls=6, wavelength_registration="report")
    wlr = rep["wavelength_registration"]
    assert wlr is not None and wlr["mode"] == "report"
    assert wlr["applied"] is False   # report mode never changes the shift
    assert wlr["applied_segments"] == []
    # ambient, golden element, combined, and the vote-mode diagnostic recorded
    assert all(k in wlr for k in ("combined", "ambient", "element", "diagnostic"))
    # default mode is "ambient": element UV/VIS is diagnostic, never applied
    dflt = analyze_spectrum(x, y, "db", n_calls=6)
    assert dflt["wavelength_registration"]["mode"] == "ambient"
    assert all(s not in ("UV", "VIS")
               for s in dflt["wavelength_registration"]["applied_segments"])
    # per-peak Gaussian diagnostics are attached additively
    assert rep["peak_refinement"] is not None
    assert rep["peak_refinement"]["method"] == "gaussian"


def test_indexer_wavelength_registration_report(db):
    from alibz.peaky_indexer_v3 import PeakyIndexerV3
    peaks = np.array([[100.0, 396.15, 0.05, 0.0],
                      [80.0, 438.35, 0.05, 0.0],
                      [60.0, 649.9, 0.05, 0.0]])
    reg = {"global_shift_nm": -0.15, "segments": {"NIR": {"shift_nm": -0.2}},
           "edges_nm": [365.0, 620.0], "models": [], "segment_names": ["UV", "VIS", "NIR"]}
    idx = PeakyIndexerV3(peaks, db=db, wavelength_registration=reg)
    rep = idx.wavelength_registration_report()
    assert rep is not None
    assert rep["external_global_shift_nm"] == -0.15
    assert "disagreement" in rep and isinstance(rep["disagreement"], bool)
    # no behaviour change / no report when none supplied
    idx2 = PeakyIndexerV3(peaks, db=db)
    assert idx2.wavelength_registration_report() is None


def test_thermal_drift_unresolved_returns_zero_with_bound():
    # deterministic alternating residuals: zero net slope, non-trivial scatter
    wobble = [+0.03, -0.03] * 5
    recs = [{"minutes_since_calibration": float(m),
             "segment_shifts": {"NIR": -0.2 + w}}
            for m, w in zip(np.linspace(0, 60, 10), wobble)]
    model = wr.thermal_drift_model(recs)
    nir = model["segments"]["NIR"]
    assert nir["resolved"] is False
    assert nir["detection_limit_nm_per_hour"] is not None
    assert model["predict_shift"]("NIR", minutes_since_calibration=30) == 0.0
