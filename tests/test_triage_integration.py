"""Keep experimental early filtering reversible and visible to callers."""
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from alibz.peaky_indexer_v3 import PeakyIndexerV3
from alibz.utils.database import Database
from alibz.utils.sahaboltzmann import SahaBoltzmann


@pytest.fixture(scope="module")
def atomic_data():
    return Database("db"), SahaBoltzmann("db")


def build(atomic_data, mode, rejected, deferred=(), reason="no_wavelength_support",
          amp_sigma=None):
    db, sb = atomic_data
    peaks = np.array([[100., 438.354439, .04, .03],
                      [50., 588.995093, .04, .03]])
    idx = PeakyIndexerV3(peaks, db=db, sb=sb, amp_sigma=amp_sigma)
    result = SimpleNamespace(rejected_elements=tuple(rejected),
                             deferred_elements=tuple(deferred),
                             element_evidence={el: SimpleNamespace(reasons=(reason,))
                                               for el in deferred},
                             to_dict=lambda: {"rejected_elements": list(rejected)})
    # Isolate the new gate from existing physical pruning. The real line
    # table and overlap construction still run, so index remapping is tested.
    with ExitStack() as stack:
        for name in ("_prefilter_species_by_initial_strength",
                     "_prefilter_species_by_line_evidence",
                     "_select_pseudo_wavelengths", "_freeze_stark_assignments",
                     "_freeze_sa_reference", "_freeze_doublet_taus"):
            stack.enter_context(patch.object(idx, name))
        call = stack.enter_context(patch("alibz.triage.triage_candidates",
                                         return_value=result))
        idx.build_candidate_matrix(physical_triage=mode,
                                   triage_protected_elements=("Ar", "O"))
    return idx, call


def test_report_preserves_candidates_and_prune_remaps_overlap(atomic_data):
    off, off_call = build(atomic_data, "off", ("Fe",))
    report, report_call = build(atomic_data, "report", ("Fe",))
    pruned, _ = build(atomic_data, "prune", ("Fe",))
    assert not off_call.called
    assert report_call.call_args.kwargs["protected_elements"] == ("Ar", "O")
    np.testing.assert_array_equal(off.line_table.wavelengths,
                                  report.line_table.wavelengths)
    np.testing.assert_array_equal(off.peak_line_map.toarray(),
                                  report.peak_line_map.toarray())
    assert any(s.element == "Fe" for s in report.line_table.species)
    assert all(s.element != "Fe" for s in pruned.line_table.species)
    assert pruned.peak_line_map.shape[1] == pruned.line_table.n_lines
    assert pruned._triage_report["species_before"] > pruned._triage_report["species_after"]


def test_all_rejected_falls_back_and_invalid_mode_fails(atomic_data):
    idx, _ = build(atomic_data, "prune", atomic_data[0].elements)
    assert idx._triage_report["all_rejected_fallback"]
    assert idx.line_table.n_species > 0
    with pytest.raises(ValueError, match="physical_triage"):
        idx.build_candidate_matrix(physical_triage="silently-enable")


def test_configuration_records_opt_in_mode():
    from dataclasses import asdict
    from alibz.pipeline import AnalysisConfig
    assert asdict(AnalysisConfig("db"))["physical_triage"] == "off"
    assert asdict(AnalysisConfig("db", physical_triage="report"))["physical_triage"] == "report"


def test_unobservable_deferral_is_provisional_but_unknown_coverage_abstains(atomic_data):
    pruned, _ = build(atomic_data, "prune", (), deferred=("Fe",))
    unknown, _ = build(atomic_data, "prune", (), deferred=("Fe",),
                       reason="no_covered_wavelengths")
    assert "Fe" in pruned._triage_report["applied_exclusions"]
    assert all(s.element != "Fe" for s in pruned.line_table.species)
    assert any(s.element == "Fe" for s in unknown.line_table.species)


def test_unavailable_fitted_errors_disable_noise_evidence(atomic_data):
    idx, call = build(atomic_data, "report", (), amp_sigma=np.array([0., np.nan]))
    assert call.call_args.kwargs["peak_sigma"] is None
    assert not idx._triage_report["peak_uncertainty_available"]


def test_pipeline_smoke_dry_run_enforces_source_hash(tmp_path):
    import hashlib
    import json
    from scripts.validate_physical_triage_pipeline import main
    source = tmp_path / "s.csv"
    source.write_text("wavelength,intensity\n500,1\n501,2\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"datasets": [{"samples": [{
        "path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()
    }]}]}))
    output = tmp_path / "result.json"
    args = ["--manifest", str(manifest), "--out", str(output), "--dry-run"]
    assert main(args) == 0
    assert not output.exists()
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="pinned source"):
        main(args)


def test_pipeline_smoke_can_replay_pinned_repository_archive(tmp_path):
    from scripts.validate_physical_triage_pipeline import main
    output = tmp_path / "result.json"
    assert main(["--replay-archive",
                 "provenance/physical-triage-real-data-20260922.npz",
                 "--out", str(output), "--dry-run"]) == 0
    assert not output.exists()
