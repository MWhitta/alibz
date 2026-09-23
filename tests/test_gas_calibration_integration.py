"""Independent gas residuals replace, never increment, baseline corrections."""
import copy
import csv
import json
import pickle
from unittest.mock import patch

import numpy as np
import pytest

from alibz.utils.wavelength import SegmentShift, shift_at
from alibz.wavelength_calibration import apply_gas_calibrations


def record(offset, uncertainty=.01, span=(700., 850.), n=3):
    segment = dict(status="calibrated", offset_nm=offset,
                   uncertainty_nm=uncertainty, supported_range_nm=list(span),
                   n_inliers=n)
    return dict(segment, segments=[dict(segment)])


def baseline():
    return SegmentShift((365., 620.), [-.03, -.04, -.05], -.04, [5, 6, 7])


def test_absolute_replacement_both_coordinate_frames_and_no_extrapolation():
    base = baseline()
    snapshot = base.shifts.copy()
    gases = {"Ar": record(-.22)}
    original = copy.deepcopy(gases)
    effective, report = apply_gas_calibrations(base, gases)
    db = np.array([500., 700., 775., 850., 900.])
    # Only the bracketed interior receives the full gas correction; endpoints
    # join continuously to baseline, preserving a unique wavelength mapping.
    expected = np.array([-.04, -.05, -.22, -.05, -.05])
    obs = db + expected
    np.testing.assert_allclose(shift_at(effective, db, frame="database"), expected)
    np.testing.assert_allclose(shift_at(effective, obs), expected)
    np.testing.assert_allclose(obs - shift_at(effective, obs), db)
    np.testing.assert_array_equal(base.shifts, snapshot)
    assert gases == original
    assert float(effective) == float(base)
    assert report["applied"]
    assert not report["input_instrument_calibration_modified"]
    assert report["baseline"]["source"] == "mixed_element_residual_estimator"
    json.dumps(report, allow_nan=False)
    restored = pickle.loads(pickle.dumps(effective))
    np.testing.assert_allclose(restored.at(obs), expected)


def test_disagreement_retains_baseline_only_in_overlap():
    gases = {"Ar": record(.20), "O": record(-.10, span=(777., 900.), n=2)}
    effective, report = apply_gas_calibrations(baseline(), gases)
    np.testing.assert_allclose(shift_at(effective, [710., 800., 880.], frame="database"),
                               [.20, -.05, -.10])
    assert len(report["conflicts"]) == 1
    assert report["conflicts"][0]["supported_range_nm"] == [777., 850.]


def test_agreement_uses_precise_reference_without_merging_gas_estimates():
    gases = {"Ar": record(.17), "O": record(.16, uncertainty=.03, n=2)}
    effective, report = apply_gas_calibrations(baseline(), gases)
    assert effective.at(800.) == pytest.approx(.17)
    assert report["gases"]["O"]["offset_nm"] == .16
    assert report["applied_regions"][0]["corroborating_sources"] == ["Ar", "O"]


def test_adjacent_references_have_monotonic_invertible_mapping_at_boundaries():
    gases = {"Ar": record(.2, span=(700., 800.)),
             "O": record(-.2, span=(800.1, 900.), n=2)}
    effective, report = apply_gas_calibrations(baseline(), gases)
    db = np.linspace(698., 902., 10001)
    obs = db + shift_at(effective, db, frame="database")
    assert np.all(np.diff(obs) > 0)
    recovered = obs - shift_at(effective, obs)
    np.testing.assert_allclose(recovered, db, atol=1e-10)
    assert len(report["applied_regions"]) == 2
    assert not report["conflicts"]
    for region in report["applied_regions"]:
        lo, hi = region["supported_range_nm"]
        core_lo, core_hi = region["full_offset_range_nm"]
        assert lo < core_lo < core_hi < hi


def test_no_single_anchor_segment_or_tentative_calibration_is_applied():
    gases = {"Ar": record(.2)}
    gases["Ar"]["segments"][0]["n_inliers"] = 1
    base = baseline()
    effective, report = apply_gas_calibrations(base, gases)
    assert effective is base
    assert not report["applied"]
    gases = {"O": dict(record(.2, n=2), status="tentative")}
    assert apply_gas_calibrations(base, gases)[0] is base


def test_report_and_off_modes_do_not_change_wavelengths():
    base = baseline()
    for mode in ("report", "off"):
        effective, report = apply_gas_calibrations(base, {"Ar": record(.2)}, mode=mode)
        assert effective is base
        assert not report["applied"]
        assert bool(report["proposed_regions"]) == (mode == "report")
    with pytest.raises(ValueError, match="gas_wavelength_calibration"):
        apply_gas_calibrations(base, {}, mode="unknown")


def test_summary_cli_and_notebook_export_separate_calibrations(tmp_path):
    from alibz.cli import main
    from alibz.pipeline import build_inspection_notebook
    _, calibration = apply_gas_calibrations(baseline(), {"Ar": record(.17),
                                                           "O": record(.16, n=2)})
    row = dict(file="s.csv", sample="s", status="ok", fractions={},
               uncertainties={}, wavelength_calibration=calibration,
               Ar_calibration_status="calibrated", Ar_calibration_shift_pm=170.,
               O_calibration_status="calibrated", O_calibration_shift_pm=160.)
    with patch("alibz.cli.analyze_directory", return_value=[row]) as analyze:
        assert main([str(tmp_path), "--db", "db", "--no-notebook",
                     "--gas-wavelength-calibration", "report"]) == 0
    assert analyze.call_args.kwargs["gas_wavelength_calibration"] == "report"
    saved = json.loads((tmp_path / "wavelength_calibration.json").read_text())
    assert saved["s.csv"]["gases"]["Ar"]["offset_nm"] == .17
    with (tmp_path / "summary.csv").open() as handle:
        summary = next(csv.DictReader(handle))
    assert float(summary["Ar_calibration_shift_pm"]) == 170.
    assert float(summary["O_calibration_shift_pm"]) == 160.
    nb = build_inspection_notebook(str(tmp_path), "db", gas_wavelength_calibration="report")
    source = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "gas_wavelength_calibration='report'" in source


def test_pipeline_estimates_gases_before_any_baseline_correction():
    """A deliberately different baseline cannot become a gas-search prior."""
    import sys
    from types import ModuleType
    from alibz.pipeline import analyze_spectrum
    x = np.linspace(695., 900., 1000)
    y = np.ones_like(x)
    peaks = np.array([[100., 706.89, .04, .02], [90., 750.56, .04, .02]])
    calls = []
    fake = ModuleType("alibz.gas_calibration")
    def independent(xx, yy, db, *, peak_array):
        np.testing.assert_array_equal(xx, x)
        np.testing.assert_array_equal(yy, y)
        np.testing.assert_array_equal(peak_array, peaks)
        calls.append("gas")
        return {"Ar": record(.17)}
    fake.calibrate_background_gases = independent
    def mixed(peak_array, db):
        assert calls == ["gas"]
        np.testing.assert_array_equal(peak_array, peaks)
        calls.append("baseline")
        return -.15, 5
    def stop(*args, shift_nm, **kwargs):
        # Baseline refinement remains reproducible independently of gas mode.
        assert shift_nm == -.15
        raise RuntimeError("test stops after independent calibration")
    with patch.dict(sys.modules, {"alibz.gas_calibration": fake}), \
         patch("alibz.pipeline._get_db", return_value=object()), \
         patch("alibz.peaky_finder.PeakyFinder.fit_spectrum",
               return_value={"sorted_parameter_array": peaks}), \
         patch("alibz.utils.wavelength.estimate_wavelength_shift", side_effect=mixed), \
         patch("alibz.refine_fit", side_effect=stop):
        with pytest.raises(RuntimeError, match="test stops"):
            analyze_spectrum(x, y, "db")
    assert calls == ["gas", "baseline"]
