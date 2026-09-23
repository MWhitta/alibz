"""Tests for :mod:`alibz.utils.peakfit` (Gaussian sub-pixel peak fitting).

Synthetic Gaussians at random sub-pixel offsets on the real native grids of
each detector segment, at SNR 10 / 20 / 100, check the recovered centre error
and phase-bias, compare against parabolic interpolation, and exercise blend
detection, segment-edge / gap handling and the fixed-sigma variant.
"""

import numpy as np
import pytest

from alibz.utils import peakfit as pf

# native pitches per segment (nm/px), from the fitted native grids
PITCH = {"UV": 0.089, "VIS": 0.129, "NIR": 0.179}
CENTER = {"UV": 250.0, "VIS": 500.0, "NIR": 800.0}


def _grid(seg, n=41):
    p = PITCH[seg]
    return np.arange(CENTER[seg] - n // 2 * p, CENTER[seg] + (n // 2 + 1) * p, p)[:n]


def _parabolic(x, y, j):
    j = min(max(j, 1), len(x) - 2)
    d = y[j - 1] - 2 * y[j] + y[j + 1]
    if d >= 0:
        return x[j]
    return x[j] + 0.5 * (y[j - 1] - y[j + 1]) / d * 0.5 * (x[j + 1] - x[j - 1])


@pytest.mark.parametrize("seg", ["UV", "VIS", "NIR"])
@pytest.mark.parametrize("snr,tol_p50,tol_mean", [(10, 0.15, 0.02),
                                                  (20, 0.05, 0.01),
                                                  (100, 0.02, 0.01)])
def test_center_accuracy_and_phase_bias(seg, snr, tol_p50, tol_mean):
    pitch = PITCH[seg]
    sig_px = pf.instrument_sigma_px(seg)
    sig = sig_px * pitch
    x = _grid(seg)
    rng = np.random.default_rng(hash((seg, snr)) % 2**32)
    g_err, p_err = [], []
    mid = len(x) // 2
    for _ in range(400):
        off = rng.uniform(-0.5, 0.5) * pitch
        mu = x[mid] + off
        y = snr * np.exp(-0.5 * ((x - mu) / sig) ** 2) + rng.normal(0, 1.0, x.size)
        j = int(np.argmin(np.abs(x - mu)))
        r = pf.gaussian_subpixel_center(x, y, j, sigma_px=sig_px)
        if r["ok"]:
            g_err.append((r["center_nm"] - mu) / pitch)
        p_err.append((_parabolic(x, y, j) - mu) / pitch)
    g = np.abs(np.asarray(g_err))
    assert np.median(g) < tol_p50, (seg, snr, np.median(g))
    assert abs(np.mean(g_err)) < tol_mean, (seg, snr, np.mean(g_err))
    # Gaussian must be at least as accurate as parabolic (report the ratio)
    assert np.median(g) <= np.median(np.abs(p_err)) + 1e-9


def test_gaussian_beats_parabolic_reported(capsys):
    seg = "VIS"; pitch = PITCH[seg]; sig_px = pf.instrument_sigma_px(seg)
    sig = sig_px * pitch; x = _grid(seg); rng = np.random.default_rng(11)
    g, p = [], []
    mid = len(x) // 2
    for _ in range(500):
        off = rng.uniform(-0.5, 0.5) * pitch
        mu = x[mid] + off
        y = 20 * np.exp(-0.5 * ((x - mu) / sig) ** 2) + rng.normal(0, 1.0, x.size)
        j = int(np.argmin(np.abs(x - mu)))
        r = pf.gaussian_subpixel_center(x, y, j, sigma_px=sig_px)
        if r["ok"]:
            g.append(abs((r["center_nm"] - mu) / pitch))
            p.append(abs((_parabolic(x, y, j) - mu) / pitch))
    print(f"SNR20 VIS median |err| px: gaussian {np.median(g):.4f} "
          f"parabolic {np.median(p):.4f}")
    assert np.median(g) < np.median(p)


def test_blend_detection_and_specificity():
    seg = "VIS"; pitch = PITCH[seg]; sig_px = pf.instrument_sigma_px(seg)
    sig = sig_px * pitch; x = _grid(seg); rng = np.random.default_rng(21)
    mid = len(x) // 2

    def rate(sep_px, n=120, amp=100.0, amp2=90.0):
        hit = 0
        for _ in range(n):
            mu1 = x[mid] + rng.uniform(-0.3, 0.3) * pitch
            mu2 = mu1 + sep_px * pitch
            y = (amp * np.exp(-0.5 * ((x - mu1) / sig) ** 2)
                 + amp2 * np.exp(-0.5 * ((x - mu2) / sig) ** 2)
                 + rng.normal(0, 1.0, x.size))
            j = int(np.argmin(np.abs(x - 0.5 * (mu1 + mu2))))
            r = pf.gaussian_subpixel_center(x, y, j, sigma_px=sig_px)
            if r["ok"] and r["blend"]:
                hit += 1
        return hit / n

    # 1.5 px (~2.5 sigma) reliably resolved; 1.2 px (~2 sigma) is at the
    # resolution limit -- detected a non-trivial fraction of the time, traded
    # against the low clean false-positive rate asserted below.
    assert rate(1.5) >= 0.9
    assert rate(1.2) >= 0.15
    # clean single lines rarely misflag
    fp = 0
    N = 400
    for _ in range(N):
        off = rng.uniform(-0.5, 0.5) * pitch
        snr = rng.choice([10, 20, 100])
        mu = x[mid] + off
        y = snr * np.exp(-0.5 * ((x - mu) / sig) ** 2) + rng.normal(0, 1.0, x.size)
        j = int(np.argmin(np.abs(x - mu)))
        r = pf.gaussian_subpixel_center(x, y, j, sigma_px=sig_px)
        if r["ok"] and r["blend"]:
            fp += 1
    assert fp / N <= 0.05


def test_blend_returns_two_gaussian_with_separation():
    seg = "VIS"; pitch = PITCH[seg]; sig_px = pf.instrument_sigma_px(seg)
    sig = sig_px * pitch; x = _grid(seg); rng = np.random.default_rng(31)
    mu1 = x[len(x) // 2]; mu2 = mu1 + 1.8 * pitch
    y = (100 * np.exp(-0.5 * ((x - mu1) / sig) ** 2)
         + 100 * np.exp(-0.5 * ((x - mu2) / sig) ** 2) + rng.normal(0, 1.0, x.size))
    j = int(np.argmin(np.abs(x - 0.5 * (mu1 + mu2))))
    r = pf.gaussian_subpixel_center(x, y, j, sigma_px=sig_px)
    assert r["ok"] and r["blend"]
    tg = r["two_gaussian"]
    assert tg["ok"]
    assert abs(tg["separation_nm"] / pitch - 1.8) < 0.4


def test_fixed_sigma_variant():
    seg = "VIS"; pitch = PITCH[seg]; sig_px = pf.instrument_sigma_px(seg)
    sig = sig_px * pitch; x = _grid(seg); rng = np.random.default_rng(41)
    mu = x[len(x) // 2] + 0.3 * pitch
    y = 50 * np.exp(-0.5 * ((x - mu) / sig) ** 2) + rng.normal(0, 1.0, x.size)
    j = int(np.argmin(np.abs(x - mu)))
    r_free = pf.gaussian_subpixel_center(x, y, j, sigma_px=sig_px)
    r_fix = pf.gaussian_subpixel_center(x, y, j, sigma_px=sig_px, sigma_bounds=None)
    assert r_free["ok"] and r_fix["ok"]
    assert r_fix["fixed_sigma"] is True
    assert abs(r_fix["sigma_px"] - sig_px) < 1e-6
    assert abs(r_fix["center_nm"] - mu) < 0.1 * pitch


def test_segment_edge_and_gap_do_not_raise():
    x = _grid("VIS")
    sig_px = pf.instrument_sigma_px("VIS")
    y = np.ones_like(x)
    # peak at the very first/last sample: must not raise
    r0 = pf.gaussian_subpixel_center(x, y, 0, sigma_px=sig_px)
    rN = pf.gaussian_subpixel_center(x, y, x.size - 1, sigma_px=sig_px)
    assert "ok" in r0 and "ok" in rN
    # a native gap (segment boundary) near the peak: window is clipped
    xg = np.concatenate([np.arange(618.0, 620.0, 0.129),
                         np.arange(620.0, 622.0, 0.179)])
    sig = 0.7 * 0.179
    mu = 620.0 + 0.2 * 0.179
    yg = 50 * np.exp(-0.5 * ((xg - mu) / sig) ** 2)
    j = int(np.argmin(np.abs(xg - mu)))
    r = pf.gaussian_subpixel_center(xg, yg, j, sigma_px=0.7)
    assert "ok" in r


def test_bad_input_returns_not_ok():
    x = _grid("VIS")
    y = np.ones_like(x)
    assert pf.gaussian_subpixel_center(x, y, -3, sigma_px=0.6)["ok"] is False
    assert pf.gaussian_subpixel_center(x, y, 10_000, sigma_px=0.6)["ok"] is False
    short = np.arange(3.0)
    assert pf.gaussian_subpixel_center(short, short, 1, sigma_px=0.6)["ok"] is False


def test_instrument_sigma_px_and_profile_override():
    assert pf.instrument_sigma_px("NIR") > pf.instrument_sigma_px("VIS")
    assert pf.instrument_sigma_px(800.0) == pf.instrument_sigma_px("NIR")
    over = pf.instrument_sigma_px("VIS", profile={"VIS": 1.11})
    assert abs(over - 1.11) < 1e-9


def test_estimate_instrument_profile_and_refine_peaks():
    seg = "VIS"; pitch = PITCH[seg]; sig_px = pf.instrument_sigma_px(seg)
    sig = sig_px * pitch
    x = np.arange(480.0, 520.0, pitch)
    rng = np.random.default_rng(51)
    centers = np.linspace(485, 515, 8)
    y = rng.normal(0, 1.0, x.size)
    for c in centers:
        y += 80 * np.exp(-0.5 * ((x - c) / sig) ** 2)
    idx = [int(np.argmin(np.abs(x - c))) for c in centers]
    prof = pf.estimate_instrument_profile(x, y, idx)
    assert "VIS" in prof and prof["VIS"]["n"] >= 4
    assert abs(prof["VIS"]["sigma_px"] - sig_px) < 0.25
    recs = pf.refine_peaks(x, y, idx)
    assert len(recs) == len(idx)
    good = [r for r in recs if r["ok"]]
    assert len(good) >= 6
    for r, c in zip(recs, centers):
        if r["ok"]:
            assert abs(r["center_nm"] - c) < 0.2 * pitch
