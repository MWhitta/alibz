import json
from dataclasses import replace

import numpy as np
import pytest

from alibz.triage import TriageConfig, triage_candidates


def _line(
    ion, wavelength, gA=1e7, Ei=1.0, Ek=3.0,
    upper=("4p2", "3P", "2"),
):
    row = ["0"] * 14
    row[0] = str(ion)
    row[1] = str(wavelength)
    row[3] = str(gA)
    row[4] = str(Ei)
    row[5] = str(Ek)
    row[6:9] = ["lower", "term", "1"]
    row[9:12] = list(upper)
    row[12:14] = ["3", "5"]
    return row


class FakeDatabase:
    def __init__(self, lines, *, no_lines=(), excluded=(), uncertain=(), observed=None):
        self.elements = tuple(lines)
        self._lines = {key: np.asarray(value, dtype=str) for key, value in lines.items()}
        self.no_lines = set(no_lines)
        self.analysis_excluded_elements = set(excluded)
        self.strength_uncertain_elements = set(uncertain)
        self._observed = observed

    def lines(self, element):
        return self._lines[element]

    def observed_lines(self, element):
        if self._observed is None:
            return []
        return list(self._observed.get(element, ()))


def _peaks(*rows):
    return np.asarray(rows, dtype=float).reshape((-1, 4))


def test_unresolved_doublet_counts_once_and_dense_chain_does_not_merge():
    db = FakeDatabase({
        "X": [
            _line(1, 500.00), _line(1, 500.06),
            _line(1, 510.00), _line(1, 510.08), _line(1, 510.16),
        ]
    })
    result = triage_candidates(
        _peaks([20, 500.03, .02, .01], [10, 510.04, .02, .01], [9, 510.16, .02, .01]),
        db,
        wavelength_range=(499, 511),
    )
    evidence = result.element_evidence["X"]
    assert evidence.matched_feature_count == 3
    assert evidence.species[0].candidate_feature_count == 3


def test_temperature_envelope_selects_lines_strong_at_opposite_extremes():
    db = FakeDatabase({"X": [
        _line(1, 400.0, gA=1e5, Ek=1.0),
        _line(1, 600.0, gA=1e9, Ek=8.0),
        _line(1, 500.0, gA=1e3, Ek=3.0),
    ]})
    config = replace(TriageConfig(), strongest_per_stage=1)
    result = triage_candidates(
        _peaks([5, 400, .02, .01], [5, 600, .02, .01]), db,
        config=config, wavelength_range=(350, 650),
    )
    species = result.element_evidence["X"].species[0]
    assert species.candidate_feature_count == 2
    assert species.matched_feature_count == 2


def test_vuv_lines_do_not_displace_observable_top_lines():
    lines = [_line(1, 100 + i, gA=1e12) for i in range(25)]
    lines.append(_line(1, 500, gA=1e4))
    db = FakeDatabase({"X": lines})
    result = triage_candidates(
        _peaks([10, 500, .02, .01]), db, wavelength_range=(490, 510)
    )
    assert result.element_evidence["X"].matched_feature_count == 1


def test_stage_three_only_is_ranked_and_never_hard_rejected():
    db = FakeDatabase({"X": [_line(3, 500.0)]})
    result = triage_candidates(
        _peaks([12, 500, .02, .01]), db, wavelength_range=(490, 510)
    )
    evidence = result.element_evidence["X"]
    assert evidence.decision == "keep"
    assert "high_ion_stage_without_lower_support_flag_only" in evidence.reasons
    assert result.rejected_elements == ()


def test_multiple_matched_ion_stages_add_support_without_saha_gate():
    db = FakeDatabase({"X": [_line(1, 500), _line(2, 510)]})
    result = triage_candidates(
        _peaks([12, 500, .02, .01], [8, 510, .02, .01]), db,
        wavelength_range=(490, 520),
    )
    evidence = result.element_evidence["X"]
    assert evidence.matched_stage_count == 2
    assert "multiple_ion_stages" in evidence.reasons
    assert evidence.decision == "keep"


def test_same_peak_claimed_by_two_elements_is_ambiguity_downweighted():
    db = FakeDatabase({"X": [_line(1, 500)], "Y": [_line(1, 500.02)]})
    result = triage_candidates(
        _peaks([12, 500.01, .02, .01]), db, wavelength_range=(490, 510)
    )
    assert result.element_evidence["X"].species[0].ambiguity_weighted_support == 0.5
    assert result.element_evidence["Y"].species[0].ambiguity_weighted_support == 0.5


def test_weak_cross_stage_coincidences_are_retained_at_low_priority():
    db = FakeDatabase({"X": [
        _line(1, 500, gA=1), _line(1, 510, gA=1e9),
        _line(2, 520, gA=1), _line(2, 530, gA=1e9),
    ]})
    config = replace(TriageConfig(), strongest_per_stage=1)
    result = triage_candidates(
        _peaks([2, 500.11, .02, .01], [2, 520.11, .02, .01]), db,
        config=config, wavelength_range=(490, 540),
    )
    evidence = result.element_evidence["X"]
    assert evidence.matched_feature_count == 0
    assert evidence.weak_wavelength_match_count == 2
    assert evidence.decision == "keep"
    assert "weak_wavelength_support_low_priority" in evidence.reasons


def test_coverage_hole_is_not_missing_companion_evidence():
    db = FakeDatabase({"X": [
        _line(1, 500, gA=1e8, Ei=2, Ek=3),
        _line(1, 505, gA=1e8, Ei=2, Ek=3),
        _line(1, 510, gA=1e8, Ei=2, Ek=3),
    ]})
    config = replace(TriageConfig(), allow_companion_rejection=True)
    result = triage_candidates(
        _peaks([100, 500, .02, .01]), db, config=config,
        wavelength_range=(495, 515), coverage_intervals=((495, 502),),
        peak_sigma=np.array([1.0]), local_noise=lambda wavelength: np.ones_like(wavelength),
    )
    assert result.element_evidence["X"].decision == "keep"


def test_one_line_narrow_band_is_retained():
    db = FakeDatabase({"X": [_line(1, 500)]})
    result = triage_candidates(
        _peaks([3, 500.05, .02, .01]), db, wavelength_range=(499.5, 500.5)
    )
    assert result.keep_elements == ("X",)
    assert result.rejected_elements == ()


def test_protected_and_strength_uncertain_elements_are_conservative():
    db = FakeDatabase(
        {"Se": [_line(1, 500)], "Pm": [_line(1, 510)]},
        no_lines={"Pm"}, uncertain={"Se"},
    )
    config = replace(TriageConfig(), allow_companion_rejection=True)
    result = triage_candidates(
        _peaks([100, 500, .02, .01]), db, config=config,
        wavelength_range=(490, 520), peak_sigma=np.array([1.0]),
        local_noise=([490, 520], [1, 1]), protected_elements=("Pm",),
    )
    assert set(result.keep_elements) == {"Se", "Pm"}
    assert result.element_evidence["Pm"].reasons == ("protected",)
    assert result.rejected_elements == ()


def test_observed_only_catalog_match_retains_without_invented_strength():
    db = FakeDatabase(
        {"X": []}, no_lines={"X"},
        observed={"X": [{"ion_stage": 1, "wavelength_nm": 500.04,
                          "quantitative_ready": False}]},
    )
    result = triage_candidates(
        _peaks([5, 500, .02, .01]), db, wavelength_range=(490, 510)
    )
    evidence = result.element_evidence["X"]
    assert evidence.decision == "keep"
    assert evidence.observed_wavelength_match_count == 1
    assert evidence.score == 0.0
    assert evidence.reasons == ("observed_catalog_wavelength_support_nonquantitative",)


def test_shifted_or_wholly_missing_pattern_is_deferred_not_rejected():
    db = FakeDatabase({"X": [_line(1, 500), _line(1, 505), _line(1, 510)]})
    result = triage_candidates(
        _peaks([10, 501, .02, .01], [10, 506, .02, .01]), db,
        wavelength_range=(495, 515),
    )
    assert result.element_evidence["X"].decision == "defer"
    assert result.deferred_elements == ("X",)
    assert result.keep_elements == ("X",)
    assert result.rejected_elements == ()


def test_reversal_is_one_tentative_feature_and_ratio_abstains_for_low_ei():
    shared = ("4p2", "3P", "2")
    db = FakeDatabase({"X": [
        _line(1, 500, Ei=0.0, Ek=3.0, upper=shared),
        _line(1, 505, Ei=0.0, Ek=3.0, upper=shared),
    ]})
    result = triage_candidates(
        _peaks(
            [100, 499.85, .02, .01], [90, 500.15, .02, .01],
            [1000, 505.0, .02, .01],
        ),
        db, wavelength_range=(495, 510),
    )
    species = result.element_evidence["X"].species[0]
    assert species.reversal_feature_count == 1
    assert species.matched_feature_count == 2
    assert species.ratio_diagnostics == ()  # reversal amplitudes are never ratio inputs


def test_unselected_low_ei_line_still_guards_against_deferral():
    db = FakeDatabase({"X": [
        _line(1, 500, gA=1, Ei=0.0, Ek=3),
        _line(1, 510, gA=1e12, Ei=2.0, Ek=3),
    ]})
    result = triage_candidates(
        _peaks([10, 499.85, .02, .01], [9, 500.15, .02, .01]), db,
        config=replace(TriageConfig(), strongest_per_stage=1),
        wavelength_range=(490, 520),
    )
    evidence = result.element_evidence["X"]
    assert evidence.decision == "keep"
    assert evidence.reversal_guard_count == 1
    assert "possible_self_reversal_guard" in evidence.reasons


def test_low_ei_wrong_shared_upper_ratio_is_diagnostic_abstention():
    shared = ("4p2", "3P", "2")
    db = FakeDatabase({"X": [
        _line(1, 500, gA=1e8, Ei=0.0, Ek=3.0, upper=shared),
        _line(1, 505, gA=1e8, Ei=0.0, Ek=3.0, upper=shared),
    ]})
    result = triage_candidates(
        _peaks([1000, 500, .02, .01], [1, 505, .02, .01]), db,
        wavelength_range=(495, 510),
    )
    diagnostic = result.element_evidence["X"].species[0].ratio_diagnostics[0]
    assert diagnostic.status == "abstain"
    assert diagnostic.reason == "low_excitation_or_blended"


def test_nominal_upper_label_without_energy_identity_is_not_branching_pair():
    shared = ("4p2", "3P", "2")
    db = FakeDatabase({"X": [
        _line(1, 500, Ei=2, Ek=3.0, upper=shared),
        _line(1, 505, Ei=2, Ek=3.01, upper=shared),
    ]})
    result = triage_candidates(
        _peaks([10, 500, .02, .01], [20, 505, .02, .01]), db,
        wavelength_range=(495, 510),
    )
    diagnostic = result.element_evidence["X"].species[0].ratio_diagnostics[0]
    assert diagnostic.kind == "local_temperature_envelope"


def test_companion_absence_is_diagnostic_by_default_and_opt_in_rejection():
    db = FakeDatabase({"X": [
        _line(1, 500, gA=1e8, Ei=2, Ek=3),
        _line(1, 505, gA=1e8, Ei=2, Ek=3),
        _line(1, 510, gA=1e8, Ei=2, Ek=3),
    ]})
    kwargs = dict(
        wavelength_range=(495, 515), peak_sigma=np.array([1.0]),
        local_noise=([495, 515], [1, 1]),
    )
    report = triage_candidates(_peaks([100, 500, .02, .01]), db, **kwargs)
    evidence = report.element_evidence["X"]
    assert evidence.decision == "keep"
    assert evidence.proposed_missing_companions_nm == (505.0, 510.0)
    assert "missing_companion_diagnostic_only" in evidence.reasons

    enabled = triage_candidates(
        _peaks([100, 500, .02, .01]), db,
        config=replace(TriageConfig(), allow_companion_rejection=True), **kwargs,
    )
    assert enabled.rejected_elements == ("X",)


@pytest.mark.parametrize("peak_array", [
    np.array([1, 500, .1, .1]),
    np.ones((2, 3)),
    np.array([[1, np.nan, .1, .1]]),
    np.array([[-1, 500, .1, .1]]),
    np.array([[1, 500, -.1, .1]]),
])
def test_invalid_peak_arrays_are_rejected(peak_array):
    with pytest.raises(ValueError):
        triage_candidates(peak_array, FakeDatabase({"X": [_line(1, 500)]}))


def test_invalid_ranges_noise_and_protection_are_rejected():
    db = FakeDatabase({"X": [_line(1, 500)]})
    peaks = _peaks([1, 500, .1, .1])
    with pytest.raises(ValueError):
        triage_candidates(peaks, db, wavelength_range=(510, 500))
    with pytest.raises(ValueError):
        triage_candidates(peaks, db, wavelength_range=(490, 510), local_noise=([1], [1]))
    with pytest.raises(ValueError):
        triage_candidates(peaks, db, protected_elements=("Q",))


def test_result_to_dict_is_json_safe_and_labels_scores_correctly():
    result = triage_candidates(
        _peaks([1, 500, .1, .1]), FakeDatabase({"X": [_line(1, 500)]}),
        wavelength_range=(490, 510),
    )
    payload = result.to_dict()
    json.dumps(payload)
    assert "not a probability" in payload["score_interpretation"]
    assert "supplied peak list" in payload["rejection_scope"]
