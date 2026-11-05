"""Forward<->design composition parity on the merged oracle at truth.

Context (reports/2026-11-04-forward-design-parity.md): at the exact true state
the design matrix reproduces the merged-oracle areas of Si/Ca/Mg to <0.3 %
(pure global scale), but Fe's dense spectrum (~9000 lines coalesced to
area-weighted centroids, per-line Stark widths) carries a ~28 % *shape* misfit
against the design's single-median-(sigma, gamma) overlap kernel.  This is NOT
an aggregation bug (see reports/2026-11-04-aggregation-floor.md) and NOT a
forward/design line-set, Saha, or partition mismatch -- the Fe I:II emissivity
ratio matches (0.441 vs 0.440).  The 1.23x Fe area "gap" is pure per-column
scale, which column normalisation (``A/col_max``) makes invisible to the
recovered fraction.

The composition consequence is entirely a solver-weighting effect: the
UNWEIGHTED bare-oracle least squares lets Fe's thousands of lines absorb the
shape misfit as extra concentration (Fe 0.27 vs truth 0.20), exactly the
line-rich composition collapse the weighted WNNLS was built to stop.  The
production mechanism -- ``weighted_solve=True`` (per-peak area sigma, chi-squared
so a line counts by significance not area) -- recovers every major to <1 %
*under* ``overlap_mode="nearest"``.  It does NOT rescue ``legacy``/``conserve``:
those over/under-book blended flux, the bright-line whitening runs away, and the
solve collapses onto Fe.  So the achievable, meaningful parity invariant is
specifically weighted + nearest; this file pins it and records the two
neighbouring regressions so the next person sees why the pair matters.

Coordinator decision 2026-11-04: close the parity item as CONFIGURATION, not a
bug; score weighted+nearest on the fit-quality panel separately (series 3).
"""
import numpy as np
import pytest

from alibz.peaky_indexer_v3 import PeakyIndexerV3
from test_oracle_roundtrip import merged_oracle_peaks

DBPATH = "db"
TRUTH = {
    "silicate_4el": {"Si": 0.40, "Ca": 0.30, "Fe": 0.20, "Mg": 0.10},
    "binary_ca_mg": {"Ca": 0.60, "Mg": 0.40},
}


def _amp_sigma(peaks):
    """Production-shaped per-peak area sigma: sqrt(area) Poisson, floored at 1.

    The absolute scale is irrelevant to the recovered fractions (the solve
    whitens by it); what matters is that bright lines no longer dominate the
    unweighted sum-of-squares.
    """
    return np.sqrt(np.maximum(peaks[:, 0], 1.0))


def _recover(case, mode, weighted):
    spec, peaks, (ms, mg) = merged_oracle_peaks(case, DBPATH)
    kw = (dict(amp_sigma=_amp_sigma(peaks), weighted_solve=True)
          if weighted else {})
    idx = PeakyIndexerV3(peaks, dbpath=DBPATH,
                         temperature_init=spec["temperature_k"],
                         ne_init=spec["log_ne_cm3"], overlap_mode=mode, **kw)
    keep = set(spec["composition"])
    idx.db.analysis_excluded_elements = tuple(
        e for e in idx.db.elements if e not in keep)
    idx.build_candidate_matrix(sa_doublets=True)
    res = idx.solve_at(spec["temperature_k"], spec["log_ne_cm3"], ms, mg)
    return res.element_fractions


# --- THE INVARIANT: weighted + nearest recovers every major within +/-0.01 ----
@pytest.mark.parametrize("case", ["silicate_4el", "binary_ca_mg"])
def test_weighted_nearest_recovers_majors(case):
    truth = TRUTH[case]
    f = _recover(case, "nearest", weighted=True)
    got = {e: round(f.get(e, 0.0), 3) for e in truth}
    print(f"{case} weighted+nearest:", got)
    for el, t in truth.items():
        assert f.get(el, 0.0) == pytest.approx(t, abs=0.01), (el, got)


def test_weighted_nearest_silicate_measured_values():
    """Pin the documented recovery so a drift is visible, not just the band."""
    f = _recover("silicate_4el", "nearest", weighted=True)
    # Measured 2026-11-04 (reports/2026-11-04-forward-design-parity.md):
    # Si .401 / Ca .300 / Fe .198 / Mg .100, maxdev 0.002 from truth.
    assert f.get("Si", 0.0) == pytest.approx(0.401, abs=0.01)
    assert f.get("Ca", 0.0) == pytest.approx(0.300, abs=0.01)
    assert f.get("Fe", 0.0) == pytest.approx(0.198, abs=0.01)
    assert f.get("Mg", 0.0) == pytest.approx(0.100, abs=0.01)


# --- DOCUMENTED REGRESSIONS (why the weighted+nearest pair is load-bearing) ----
def test_unweighted_nearest_fe_over_recovers_documented():
    """UNWEIGHTED nearest: Fe absorbs the 28% dense-spectrum shape misfit.

    This is the bias the weighted solve removes -- recorded, not failed, so the
    contrast is explicit.  Fe ~0.27 vs truth 0.20; the other majors stay near
    truth (the misfit is Fe-specific).
    """
    f = _recover("silicate_4el", "nearest", weighted=False)
    print("silicate_4el UNWEIGHTED+nearest (documented bias):",
          {e: round(f.get(e, 0.0), 3) for e in TRUTH["silicate_4el"]})
    assert f.get("Fe", 0.0) > 0.24, "if Fe no longer over-recovers unweighted, revisit the note"
    assert f.get("Fe", 0.0) < 0.31
    # Si/Ca/Mg remain near truth even unweighted -- the bias is Fe-only.
    assert f.get("Si", 0.0) == pytest.approx(0.40, abs=0.06)
    assert f.get("Mg", 0.0) == pytest.approx(0.10, abs=0.03)


def test_weighted_legacy_collapses_documented():
    """WEIGHTED legacy/conserve collapse onto Fe -- why 'nearest' is required.

    legacy over-books the dense Fe forest; under chi-squared whitening the
    bright-line misfit runs away and the solve dumps the whole composition into
    Fe.  Recorded so nobody 'fixes' parity by turning on weighting globally.
    """
    f = _recover("silicate_4el", "legacy", weighted=True)
    print("silicate_4el WEIGHTED+legacy (documented collapse):",
          {e: round(f.get(e, 0.0), 3) for e in TRUTH["silicate_4el"]})
    assert f.get("Fe", 0.0) > 0.9, "weighted-legacy no longer collapses; revisit the note"
