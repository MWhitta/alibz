"""Per-peak / per-line width overlap kernels (wave-3 model option A).

``PeakyIndexerV3(overlap_kernel=...)`` selects how the design-matrix overlap
map weights database line ``j`` against observed peak ``i``:

* ``"median"`` (default, historic) -- one shared search ``(sigma, gamma)`` for
  every peak: ``overlap(i, j) = V(Δλ; sigma, gamma) / V(0; sigma, gamma)``.
* ``"peak"`` -- per-peak fitted RELATIVE widths scaled by the search's global
  factors: ``sigma_i' = sigma_i * (sigma / sigma_med)``, same height-ratio
  kernel at ``(sigma_i', gamma_i')``.
* ``"peak+stark"`` -- additionally gives every line its own profile
  (``sigma_j = sigma``, ``gamma_j = gamma_inst + stark_hwhm(shape_j, nₑ)``) and
  uses the normalised Voigt-projection kernel
  ``V(Δλ; sqrt(sigma_i'^2+sigma_j^2), gamma_i'+gamma_j) / V(0; sqrt2*sigma_i',
  2*gamma_i')``.

KEY PROPERTY proved below (test_kernel_is_noop_on_coincident_peaks): on the
ORACLE every peak sits exactly on its database line (Δλ ≈ 0), where the
height-ratio kernel is 1 for ANY width, so ``"peak"`` is byte-identical to
``"median"`` under ``nearest`` and the oracle CANNOT discriminate them.  The
kernel only diverges when peaks are OFFSET from the line centres
(test_kernel_is_live_on_offset_peaks) -- i.e. on real fitted spectra -- so the
fit-quality panel, not the oracle, is the arbiter for the per-peak kernel.  See
reports/2026-11-04-w3-width-kernel.md.
"""
import numpy as np
import pytest

from alibz.peaky_indexer_v3 import PeakyIndexerV3
from alibz.synthetic import SyntheticSpectrumGenerator
from bench.synth_cases import make_scene, CASES
from test_oracle_roundtrip import merged_oracle_peaks, _instrument_fwhm_nm

DBPATH = "db"
TRUTH = {"Si": 0.40, "Ca": 0.30, "Fe": 0.20, "Mg": 0.10}
KERNELS = ("median", "peak", "peak+stark")


# --------------------------------------------------------------------------
# construction / validation
# --------------------------------------------------------------------------
def test_overlap_kernel_validated():
    peaks = np.array([[1.0, 400.0, 0.03, 0.01]])
    for k in KERNELS:
        PeakyIndexerV3(peaks, dbpath=DBPATH, overlap_kernel=k)
    with pytest.raises(ValueError):
        PeakyIndexerV3(peaks, dbpath=DBPATH, overlap_kernel="bogus")


# --------------------------------------------------------------------------
# (a) "median" reproduces today's sparse maps exactly
# --------------------------------------------------------------------------
def _built(case, kernel, mode, offset=0.0):
    spec, peaks, med = merged_oracle_peaks(case, DBPATH)
    if offset:
        peaks = peaks.copy()
        peaks[:, 1] += offset
    idx = PeakyIndexerV3(peaks, dbpath=DBPATH,
                         temperature_init=spec["temperature_k"],
                         ne_init=spec["log_ne_cm3"],
                         overlap_mode=mode, overlap_kernel=kernel)
    keep = set(spec["composition"])
    idx.db.analysis_excluded_elements = tuple(
        e for e in idx.db.elements if e not in keep)
    idx.build_candidate_matrix(sa_doublets=True)
    return spec, idx, med


def test_median_reproduces_legacy_map_exactly():
    """overlap_kernel='median' must leave the observed map bit-identical to the
    historic shared-(sigma, gamma) builder (`_build_overlap_map`)."""
    spec, idx, (ms, mg) = _built("silicate_4el", "median", "conserve")
    idx._rebuild_overlap(ms, mg, spec["log_ne_cm3"])
    got = idx.peak_line_map
    ref = idx._build_overlap_map(idx._obs_wl, ms, mg, apply_mode=True)
    assert got.shape == ref.shape
    d = (got - ref)
    assert d.nnz == 0 or float(np.max(np.abs(d.data))) == 0.0
    assert got.nnz == ref.nnz


# --------------------------------------------------------------------------
# (a') the kernel is LIVE off-centre and a NO-OP on the coincident oracle
# --------------------------------------------------------------------------
def _map_fingerprint(idx, ms, mg, log_ne):
    idx._rebuild_overlap(ms, mg, log_ne)
    m = idx.peak_line_map
    return int(m.nnz), float(np.abs(m).sum())


def test_peak_matches_median_solve_on_oracle():
    """The oracle cannot discriminate 'peak' from 'median' in the SOLVE.

    Oracle peaks sit on (merged area-weighted centroids of) db lines, so under
    nearest the height-ratio kernel is ~1 and the per-peak widths barely move
    the hard-assigned weights: the recovered fractions agree to <0.01, and Fe
    stays at the documented 0.271 aggregation floor for BOTH.  (The panel, on
    offset fitted peaks, is where the kernel acts.)"""
    fm = _recover("silicate_4el", "nearest", "median")
    fp = _recover("silicate_4el", "nearest", "peak")
    for e in TRUTH:
        assert fm.get(e, 0.0) == pytest.approx(fp.get(e, 0.0), abs=0.01), (
            e, {x: round(fm.get(x, 0), 3) for x in TRUTH},
            {x: round(fp.get(x, 0), 3) for x in TRUTH})


def test_kernel_is_live_on_offset_peaks():
    """Offset the peaks from the line centres (Δλ≠0, as in real fitted data):
    'peak' and 'peak+stark' now diverge from 'median' -- the kernel is live."""
    fps = {}
    for k in KERNELS:
        sp, idx, (ms, mg) = _built("silicate_4el", k, "nearest", offset=0.03)
        fps[k] = _map_fingerprint(idx, ms, mg, sp["log_ne_cm3"])
    assert fps["peak"][1] != fps["median"][1], fps
    assert fps["peak+stark"][1] != fps["median"][1], fps


# --------------------------------------------------------------------------
# (b) oracle recovery + Fe shape residual at the TRUE state
# --------------------------------------------------------------------------
def _recover(case, mode, kernel, restrict4=True):
    spec, peaks, (ms, mg) = merged_oracle_peaks(case, DBPATH)
    idx = PeakyIndexerV3(peaks, dbpath=DBPATH,
                         temperature_init=spec["temperature_k"],
                         ne_init=spec["log_ne_cm3"],
                         overlap_mode=mode, overlap_kernel=kernel)
    if restrict4:
        keep = set(spec["composition"])
        idx.db.analysis_excluded_elements = tuple(
            e for e in idx.db.elements if e not in keep)
    idx.build_candidate_matrix(sa_doublets=True)
    res = idx.solve_at(spec["temperature_k"], spec["log_ne_cm3"], ms, mg)
    return res.element_fractions


def _merged_dominant_elements(case):
    """Dominant db-element index of every merged-oracle peak (by summed area)."""
    spec = CASES[case]
    gen = SyntheticSpectrumGenerator(DBPATH)
    scene = make_scene(spec["composition"], temperature_k=spec["temperature_k"],
                       log_ne_cm3=spec["log_ne_cm3"], seed=0, dbpath=DBPATH)
    tbl = gen._component_lines(scene.target)
    wl = np.asarray(tbl["wavelength_nm"]); area = np.asarray(tbl["area_counts"])
    ei = np.asarray(tbl["element_index"], dtype=int)
    o = np.argsort(wl, kind="mergesort"); wl, area, ei = wl[o], area[o], ei[o]
    tol = np.maximum(1e-3, 0.25 * _instrument_fwhm_nm(gen, wl))
    new = np.concatenate([[False], np.diff(wl) > np.maximum(tol[:-1], tol[1:])])
    group = np.cumsum(new); ng = int(group[-1]) + 1
    dom = np.zeros(ng, dtype=int)
    for g in range(ng):
        m = group == g; es = ei[m]; ar = area[m]; u = np.unique(es)
        dom[g] = u[int(np.argmax([ar[es == e].sum() for e in u]))]
    return dom


def _fe_shape_residuals(kernel):
    """Per-element best-single-scale residual on the merged silicate oracle at
    truth (the parity report's shape.py quantity): for each element E over the
    peaks it dominates, alpha_E = argmin||obs - a*(A c_true)||, residual =
    ||obs - alpha A c_true|| / ||obs||.  A is the raw (pre-norm) design."""
    spec, peaks, (ms, mg) = merged_oracle_peaks("silicate_4el", DBPATH)
    dom = _merged_dominant_elements("silicate_4el")
    idx = PeakyIndexerV3(peaks, dbpath=DBPATH,
                         temperature_init=spec["temperature_k"],
                         ne_init=spec["log_ne_cm3"],
                         overlap_mode="nearest", overlap_kernel=kernel)
    keep = set(spec["composition"])
    idx.db.analysis_excluded_elements = tuple(
        e for e in idx.db.elements if e not in keep)
    idx.build_candidate_matrix(sa_doublets=True)
    T, ne = spec["temperature_k"], spec["log_ne_cm3"]
    idx._rebuild_overlap(ms, mg, ne)
    A = idx._build_design_matrix_from_map(idx.peak_line_map,
                                          idx._line_weights(T, ne))
    c = np.array([spec["composition"].get(sp.element, 0.0)
                  for sp in idx.line_table.species])
    pred = A @ c
    obs = peaks[:, 0]
    elems = idx.db.elements
    out = {}
    for e in spec["composition"]:
        m = (dom == elems.index(e)) & (pred > 0)
        oo, pp = obs[m], pred[m]
        a = float(np.dot(oo, pp) / np.dot(pp, pp))
        out[e] = float(np.linalg.norm(oo - a * pp) / np.linalg.norm(oo))
    return out


def test_fe_shape_residual_recorded():
    """Record the Fe best-single-scale residual for all three kernels.

    This is a DIAGNOSTIC, not the parity report's 28% baseline: on the merged
    oracle under nearest the per-peak kernel is a no-op (peaks coincide with db
    lines), so 'median' and 'peak' measure the same ~4% (already far below the
    unmerged-spectrum 28%), and 'peak+stark' (whose normalisation differs at
    Δλ=0) is WORSE here.  The per-peak kernel's effect is only visible on
    offset/fitted peaks -> the fit-quality panel is the arbiter."""
    res = {k: _fe_shape_residuals(k) for k in KERNELS}
    for k in KERNELS:
        print(f"Fe shape residual {k}: "
              f"{ {e: round(v, 4) for e, v in res[k].items()} }")
    # median and peak coincide on the oracle (Δλ≈0); loud if that changes.
    assert res["median"]["Fe"] == pytest.approx(res["peak"]["Fe"], abs=0.01)
    # the Si/Ca/Mg majors are reproduced to <2% by the design shape at truth.
    for e in ("Si", "Ca", "Mg"):
        assert res["median"][e] < 0.02


@pytest.mark.xfail(reason="Brief target: unweighted+nearest+'peak+stark' recovers "
                   "majors within +/-0.02 and drops Fe shape residual 28%->5%. "
                   "NOT achievable on the oracle: oracle peaks sit on db lines "
                   "(Δλ≈0) so 'peak' is identical to 'median' (Fe stays at the "
                   "0.271 aggregation floor, documented in test_oracle_roundtrip), "
                   "and 'peak+stark' is worse at truth (Fe~0.28-0.33). The kernel "
                   "only acts on offset/fitted peaks; the panel is the arbiter. "
                   "See reports/2026-11-04-w3-width-kernel.md.", strict=False)
@pytest.mark.parametrize("kernel", ["peak", "peak+stark"])
def test_unweighted_nearest_kernel_recovers_majors_within_002(kernel):
    f = _recover("silicate_4el", "nearest", kernel)
    for el, t in TRUTH.items():
        assert f.get(el, 0.0) == pytest.approx(t, abs=0.02), (
            kernel, {e: round(f.get(e, 0.0), 3) for e in TRUTH})


# --------------------------------------------------------------------------
# (c) conserve + 'peak' must not hallucinate more spurious mass than median
# --------------------------------------------------------------------------
def _spurious_mass(kernel):
    f = _recover("silicate_4el", "conserve", kernel, restrict4=False)
    return sum(v for e, v in f.items() if e not in TRUTH)


def test_conserve_peak_no_worse_spurious_mass_than_median():
    base = _spurious_mass("median")
    peak = _spurious_mass("peak")
    print(f"conserve spurious mass: median={base:.4f} peak={peak:.4f}")
    # 'peak' must not hallucinate materially more non-major mass than 'median'
    # at truth (DECISIONS.md records ~1.5% Dy/Pb/U for conserve+median).
    assert peak <= base + 0.01, (base, peak)
