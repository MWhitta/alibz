"""Unit tests for the production directory pipeline (alibz.pipeline).

The heavy per-spectrum chain is exercised end-to-end by the CLI on real
data; these tests pin the pure-python plumbing: CSV parsing, sample-name
extraction, db resolution, summary-table formatting, notebook generation,
and the amplitude-resampling uncertainty (against a stub indexer).
"""

import csv
import json
import os
import tempfile
import unittest

import numpy as np

from alibz.pipeline import (
    ELEMENTS_BY_ATOMIC_NUMBER,
    ELEMENT_COLORS,
    ELEMENT_PERIODIC_BLOCK,
    PERIODIC_BLOCK_COLORS,
    build_inspection_notebook,
    element_block_color,
    element_color,
    element_periodic_block,
    element_sort_key,
    element_uncertainties,
    load_spectrum_csv,
    resolve_dbpath,
    sample_name,
    write_summary_csv,
)


class TestLoading(unittest.TestCase):
    def test_load_spectrum_csv_skips_header_and_sorts(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            fh.write("wavelength,intensity\n500.0,2.0\n180.0,1.0\njunk,row\n")
            path = fh.name
        try:
            x, y = load_spectrum_csv(path)
            np.testing.assert_allclose(x, [180.0, 500.0])
            np.testing.assert_allclose(y, [1.0, 2.0])
        finally:
            os.unlink(path)

    def test_load_spectrum_csv_empty_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            fh.write("wavelength,intensity\n")
            path = fh.name
        try:
            with self.assertRaises(ValueError):
                load_spectrum_csv(path)
        finally:
            os.unlink(path)

    def test_sample_name_strips_export_suffix(self):
        self.assertEqual(
            sample_name("/d/MDD011-15-1 #2 BT_20260614_042717_PM"
                        "_AverageSpectrum.csv"),
            "MDD011-15-1 #2 BT")
        # names without the suffix pass through
        self.assertEqual(sample_name("/d/simple.csv"), "simple")

    def test_resolve_dbpath_explicit_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(resolve_dbpath(d), os.path.abspath(d))
        with self.assertRaises(FileNotFoundError):
            resolve_dbpath("/nonexistent/db/path/xyzzy")


class TestSummaryCsv(unittest.TestCase):
    def _row(self, name, fractions, uncertainties, status="ok"):
        return dict(file=f"{name}.csv", sample=name, status=status,
                    n_peaks=10, shift_pm=-15.0, T_K=8300.0, log_ne=17.1,
                    r_squared=0.9, sa_converged=True, flags="",
                    fractions=fractions, uncertainties=uncertainties)

    def test_wide_table_ordering_and_blanks(self):
        rows = [
            self._row("a", {"K": 0.4, "Li": 0.01}, {"K": 0.02, "Li": 0.001}),
            self._row("b", {"K": 0.5}, {"K": 0.03}),
            self._row("c", {}, {}, status="error: boom"),
        ]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "summary.csv")
            elements = write_summary_csv(rows, path)
            self.assertEqual(elements, ["Li", "K"])  # atomic number order
            with open(path) as fh:
                got = list(csv.DictReader(fh))
        self.assertEqual(len(got), 3)
        self.assertEqual(got[0]["K"], "0.40000")
        self.assertEqual(got[0]["K_unc"], "0.02000")
        self.assertEqual(got[1]["Li"], "")       # absent element blank
        self.assertEqual(got[2]["status"], "error: boom")
        self.assertEqual(got[2]["K"], "")

    def test_nan_uncertainty_left_blank(self):
        rows = [self._row("a", {"K": 0.4}, {"K": float("nan")})]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "summary.csv")
            write_summary_csv(rows, path)
            with open(path) as fh:
                got = list(csv.DictReader(fh))
        self.assertEqual(got[0]["K_unc"], "")


class TestElementOrderingAndColors(unittest.TestCase):
    @staticmethod
    def _color_distance(a, b):
        def rgb(color):
            color = color.lstrip("#")
            return tuple(int(color[i:i + 2], 16) / 255.0
                         for i in (0, 2, 4))
        return float(np.linalg.norm(np.subtract(rgb(a), rgb(b))))

    def test_periodic_sort_key_orders_by_atomic_number(self):
        self.assertLess(element_sort_key("Li"), element_sort_key("K"))
        self.assertLess(element_sort_key("K"), element_sort_key("Rb"))
        self.assertLess(element_sort_key("U"), element_sort_key("Xx"))

    def test_periodic_block_assignments_cover_database_elements(self):
        expected = {
            "K": "group 1",
            "Ca": "group 2",
            "Fe": "3d-block",
            "Pd": "4d-block",
            "Au": "5d-block",
            "Ce": "4f-block",
            "U": "5f-block",
            "Si": "metalloid",
            "Al": "post-transition metal",
            "Ne": "noble gas",
            "Cl": "halogen",
            "O": "reactive nonmetal",
        }
        for element, block in expected.items():
            self.assertEqual(element_periodic_block(element), block)
        self.assertEqual(element_periodic_block("Xx"), "other")
        self.assertTrue(
            all(element_periodic_block(el) != "other"
                for el in ELEMENTS_BY_ATOMIC_NUMBER)
        )
        self.assertEqual(len(ELEMENT_PERIODIC_BLOCK),
                         len(ELEMENTS_BY_ATOMIC_NUMBER))

    def test_periodic_block_colors_are_unique(self):
        self.assertEqual(len(set(PERIODIC_BLOCK_COLORS.values())),
                         len(PERIODIC_BLOCK_COLORS))
        self.assertEqual(element_block_color("Fe"),
                         PERIODIC_BLOCK_COLORS["3d-block"])
        self.assertEqual(len(set(ELEMENT_COLORS.values())),
                         len(ELEMENT_COLORS))
        self.assertEqual(len(ELEMENT_COLORS), len(ELEMENTS_BY_ATOMIC_NUMBER))
        self.assertEqual(element_color("Fe"), ELEMENT_COLORS["Fe"])
        self.assertNotEqual(element_color("Li"), element_color("Na"))
        self.assertNotEqual(element_color("Fe"), element_color("Mn"))
        jchristensen_elements = [
            "Li", "Be", "Na", "Mg", "Al", "Si", "K", "Ca", "Ti",
            "Mn", "Fe", "As", "Rb", "Sr", "Mo", "Pd", "Ba", "Eu",
        ]
        distances = [
            self._color_distance(element_color(a), element_color(b))
            for i, a in enumerate(jchristensen_elements)
            for b in jchristensen_elements[i + 1:]
        ]
        self.assertGreater(min(distances), 0.12)


class TestNotebook(unittest.TestCase):
    def test_notebook_is_valid_nbformat(self):
        nb = build_inspection_notebook("/data/dir", "/db/path")
        try:
            import nbformat
        except ImportError:
            self.skipTest("nbformat not installed")
        nbformat.validate(nbformat.from_dict(nb))
        # round-trips through json
        json.dumps(nb)
        src = "".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("'/data/dir'", src)
        self.assertIn("'/db/path'", src)
        self.assertIn("plot_spectrum_overview", src)
        self.assertIn("element_sort_key", src)
        self.assertIn("element_color", src)
        self.assertIn("periodic block", src)


class _StubIndexer:
    """Mimics the PeakyIndexerV3 surface element_uncertainties touches."""

    def __init__(self, amps):
        self._obs_amp = np.asarray(amps, dtype=float)
        self._last_A = np.eye(len(amps))

    def _solve_concentrations(self, T, ne):
        return self._obs_amp.copy(), 0.0

    def _aggregate_elements(self, c, A):
        total = max(float(np.sum(c)), 1e-300)
        fracs = {"K": float(c[0]) / total, "Si": float(c[1]) / total}
        return dict(fracs), fracs, {}


class _StubResult:
    temperature = 8000.0
    ne = 17.0
    element_fractions = {"K": 0.6, "Si": 0.4}
    stage_disagreement = {"K": 0.1, "Si": float("nan")}


class TestStimulatedEmission(unittest.TestCase):
    def test_factor_values_and_wavelength_dependence(self):
        from alibz.peaky_indexer_v3 import stimulated_emission_factor as f
        # canonical value: Na D at 10 kK
        self.assertAlmostEqual(float(f(589.0, 10_000.0)), 0.913, delta=0.002)
        # monotone: redder lines carry a smaller factor (more induced emission)
        vals = np.asarray(f(np.array([250.0, 589.0, 770.0]), 8300.0))
        self.assertTrue(np.all(np.diff(vals) < 0))
        self.assertGreater(float(vals[0]), 0.99)   # UV ~ unity
        self.assertLess(float(vals[2]), 0.91)      # K I 770 suppressed
        # bounded in (0, 1]
        self.assertTrue(np.all(vals > 0) and np.all(vals <= 1))

    def test_kappa_gated_by_flag(self):
        """_kappa_raw with the flag differs from without by exactly the factor."""
        from alibz.peaky_indexer_v3 import (PeakyIndexerV3,
                                            stimulated_emission_factor)
        peaks = np.array([[100.0, 589.0, 0.05, 0.02],
                          [50.0, 766.5, 0.05, 0.02]])
        idx = PeakyIndexerV3(peaks, dbpath="db")
        idx.build_candidate_matrix(sa_doublets=False)
        k_off = idx._kappa_raw(9000.0, 17.0)
        idx._sa_stim = True
        k_on = idx._kappa_raw(9000.0, 17.0)
        expect = stimulated_emission_factor(idx.line_table.wavelengths, 9000.0)
        np.testing.assert_allclose(k_on, k_off * expect, rtol=1e-12)


class TestClassifyDetections(unittest.TestCase):
    def _detections(self, fractions, stats, support, stage=None):
        from alibz.pipeline import classify_detections

        class R:
            element_fractions = fractions
            stage_disagreement = stage or {}
        return {d["element"]: d for d in classify_detections(R(), stats, support)}

    def test_status_ladder(self):
        det = self._detections(
            fractions={"K": 0.5, "Hg": 0.02, "Mo": 0.01, "Eu": 0.004,
                       "Cd": 0.008},
            stats={"K": {"mean": 0.5, "std": 0.01},
                   "Hg": {"mean": 0.02, "std": 0.005},   # z=4, 1 line
                   "Mo": {"mean": 0.01, "std": 0.004},   # z=2.5
                   "Eu": {"mean": 0.004, "std": 0.004},  # z=1
                   "Cd": {"mean": 0.008, "std": 0.001},  # z=8, 0 lines
                   "Pd": {"mean": 0.001, "std": 0.002}}, # zeroed -> upper limit
            support={"K": [(9.0, 766.5, 10.0), (5.0, 404.4, 6.0)],
                     "Hg": [(4.0, 253.65, 5.0)],
                     "Mo": [(2.0, 379.8, 3.0), (1.0, 386.4, 2.0)]},
        )
        self.assertEqual(det["K"]["status"], "detected")
        self.assertEqual(det["Hg"]["status"], "single-line")
        self.assertEqual(det["Hg"]["n_lines"], 1)
        self.assertEqual(det["Hg"]["strongest_peak_nm"], 253.65)
        self.assertEqual(det["Mo"]["status"], "marginal")
        self.assertEqual(det["Eu"]["status"], "weak")
        self.assertEqual(det["Cd"]["status"], "blended-only")
        self.assertEqual(det["Pd"]["status"], "upper-limit")
        self.assertAlmostEqual(det["Pd"]["upper_limit"], 0.005)

    def test_zeroed_element_without_stats_is_omitted(self):
        det = self._detections(fractions={"K": 1.0},
                               stats={"K": {"mean": 1.0, "std": 0.01}},
                               support={})
        self.assertNotIn("Zz", det)

    def test_detections_csv_round_trip(self):
        from alibz.pipeline import write_detections_csv
        rows = [dict(sample="s1", detections=[
            dict(element="Hg", status="single-line", fraction=0.02,
                 unc=0.005, z=4.0, n_lines=1, strongest_peak_nm=253.652,
                 strongest_obs=120.0, upper_limit=None,
                 stage_disagreement=None),
            dict(element="Pd", status="upper-limit", fraction=0.0,
                 unc=0.002, z=0.0, n_lines=0, strongest_peak_nm=None,
                 strongest_obs=None, upper_limit=0.005,
                 stage_disagreement=None),
        ])]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "detections.csv")
            n = write_detections_csv(rows, path)
            with open(path) as fh:
                got = list(csv.DictReader(fh))
        self.assertEqual(n, 2)
        self.assertEqual(got[0]["element"], "Hg")
        self.assertEqual(got[0]["status"], "single-line")
        self.assertEqual(got[1]["upper_limit"], "0.005")
        self.assertEqual(got[1]["fraction"], "")


class TestElementUncertainties(unittest.TestCase):
    def test_resampling_spread_and_amp_restore(self):
        idx = _StubIndexer([60.0, 40.0])
        amp_before = idx._obs_amp.copy()
        unc = element_uncertainties(idx, _StubResult(), np.array([6.0, 4.0]),
                                    draws=64, seed=1)
        np.testing.assert_array_equal(idx._obs_amp, amp_before)  # restored
        for el in ("K", "Si"):
            self.assertGreater(unc[el], 0.005)
            self.assertLess(unc[el], 0.2)

    def test_nan_sigma_falls_back_to_ten_percent(self):
        idx = _StubIndexer([100.0, 100.0])
        unc = element_uncertainties(
            idx, _StubResult(), np.array([np.nan, np.nan]), draws=64, seed=2)
        # 10% amplitude noise -> a few percent fraction spread, not zero
        self.assertGreater(unc["K"], 0.005)

    def test_deterministic_for_fixed_seed(self):
        idx = _StubIndexer([60.0, 40.0])
        u1 = element_uncertainties(idx, _StubResult(), np.array([6.0, 4.0]),
                                   draws=16, seed=7)
        u2 = element_uncertainties(idx, _StubResult(), np.array([6.0, 4.0]),
                                   draws=16, seed=7)
        self.assertEqual(u1, u2)


class TestContestedSupport(unittest.TestCase):
    """True-negative rival test: the Mn/Mg 279.5 nm archetype in miniature.

    Peaks: [0] the contested resonance-region peak, [1] the rival's
    independent non-resonance line, [2] the claimer's independent line.
    """

    def _run(self, rival_other_obs):
        from alibz.detections import contested_support, merge_contests
        # species columns: 0 = Mn-like claimer, 1 = Mg-like rival
        A = np.array([[10.0, 8.0],     # both respond at the blend peak
                      [0.0, 5.0],      # rival's independent line
                      [6.0, 0.0]])     # claimer's independent line
        obs = np.array([900.0, rival_other_obs, 300.0])
        sig = np.array([10.0, 10.0, 10.0])
        cols = {"Mn": [0], "Mg": [1]}
        sup = {"Mn": [0, 2]}
        per = [contested_support(A, cols, ["Mg", "Mn"], obs, sig, sup,
                                  A_nonres=A * np.array([[0, 0],
                                                         [0, 1],
                                                         [1, 0]]))]
        return merge_contests(sup, obs, per)["Mn"]

    def test_blend_peak_contested_when_rival_cap_allows(self):
        # rival's independent line is bright: its cap covers the blend
        out = self._run(rival_other_obs=800.0)
        self.assertEqual(out["clear_lines"], 1)       # peak 2 stays clear
        self.assertGreater(out["contested_share"], 0.5)
        self.assertEqual(out["confounder"], "Mg")

    def test_blend_peak_clear_when_rival_true_negatives_forbid(self):
        # rival's independent line is nearly absent: cap collapses
        out = self._run(rival_other_obs=5.0)
        self.assertEqual(out["clear_lines"], 2)
        self.assertEqual(out["contested_share"], 0.0)
        self.assertIsNone(out["confounder"])

    def test_merge_is_existential_over_states(self):
        from alibz.detections import merge_contests
        sup = {"X": [0, 1]}
        obs = np.array([100.0, 50.0])
        per = [
            {"X": dict(contested={0}, rivals={"Y": 100.0})},   # state A
            {"X": dict(contested=set(), rivals={})},           # state B
        ]
        out = merge_contests(sup, obs, per)["X"]
        self.assertEqual(out["clear_lines"], 1)   # peak 0 contested SOMEWHERE
        self.assertEqual(out["confounder"], "Y")

    def test_confounder_catalog_counts_pairs(self):
        from alibz.detections import confounder_catalog
        det_by_sample = [
            [dict(element="Mn", confounder="Mg"),
             dict(element="Li", confounder=None)],
            [dict(element="Mn", confounder="Mg"),
             dict(element="Be", confounder="Fe")],
        ]
        cat = confounder_catalog(det_by_sample)          # list-of-lists
        self.assertEqual(cat[("Mn", "Mg")], 2)
        self.assertEqual(cat[("Be", "Fe")], 1)
        self.assertNotIn(("Li", None), cat)
        # also accepts a flat iterable of detection dicts
        flat = confounder_catalog([d for s in det_by_sample for d in s])
        self.assertEqual(flat[("Mn", "Mg")], 2)


class TestRecoverResidualLines(unittest.TestCase):
    def _make(self, include_third):
        from alibz.utils.voigt import multi_voigt
        rng = np.random.default_rng(3)
        x = np.arange(300.0, 320.0, 1.0 / 30.0)
        comps = [[500.0, 305.0, 0.05, 0.02],
                 [300.0, 312.0, 0.05, 0.02],
                 [80.0, 308.0, 0.05, 0.02]]   # the "missed" line
        y = multi_voigt(x, np.ravel(comps)) + rng.normal(0.0, 1.5, x.size)
        table = comps if include_third else comps[:2]
        fit = dict(sorted_parameter_array=np.array(table, dtype=float),
                   background=np.zeros_like(y))
        return x, y, fit

    def test_recovers_missing_line(self):
        from alibz.minor_lines import recover_residual_lines
        x, y, fit = self._make(include_third=False)
        new_fit, recs = recover_residual_lines(x, y, fit, segment_edges=())
        added = [r for r in recs if r["action"] == "added"]
        self.assertEqual(len(added), 1)
        self.assertAlmostEqual(added[0]["center"], 308.0, delta=0.05)
        self.assertAlmostEqual(added[0]["area"] / 80.0, 1.0, delta=0.25)
        self.assertEqual(new_fit["sorted_parameter_array"].shape[0], 3)

    def test_no_false_positives_when_fully_modeled(self):
        from alibz.minor_lines import recover_residual_lines
        x, y, fit = self._make(include_third=True)
        new_fit, recs = recover_residual_lines(x, y, fit, segment_edges=())
        self.assertEqual([r for r in recs if r["action"] == "added"], [])
        self.assertIs(new_fit, fit)   # untouched fit dict returned

    def test_broad_ledge_produces_no_phantom_carpet(self):
        """A multi-nm positive pedestal is background residue, not lines
        (review repro: a 5-sigma ledge yielded 29 accepted phantoms
        pre-fix)."""
        from alibz.minor_lines import recover_residual_lines
        x, y, fit = self._make(include_third=True)
        y = y + np.where(x > 310.0, 10.0, 0.0)   # 5-sigma ledge onward
        _new_fit, recs = recover_residual_lines(x, y, fit,
                                                segment_edges=())
        added = [r for r in recs if r["action"] == "added"]
        self.assertLessEqual(len(added), 1)   # ledge edge at worst

    def test_exclusion_zone_blocks_sa_lobe_resplit(self):
        """Candidates inside a caller-declared self-absorption zone are
        skipped with action='excluded' (review repro: K I 766.49 merged
        row collapsed 1368 -> 0 with two phantoms accepted pre-fix)."""
        from alibz.minor_lines import recover_residual_lines
        x, y, fit = self._make(include_third=False)   # residual at 308
        new_fit, recs = recover_residual_lines(
            x, y, fit, exclude=((308.0, 0.3),), segment_edges=())
        self.assertEqual([r for r in recs if r["action"] == "added"], [])
        self.assertTrue(any(r["action"] == "excluded" for r in recs))
        self.assertIs(new_fit, fit)


class TestCli(unittest.TestCase):
    def test_help_exits_zero(self):
        from alibz.cli import main
        with self.assertRaises(SystemExit) as ctx:
            main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_missing_dir_errors(self):
        from alibz.cli import main
        with self.assertRaises(SystemExit) as ctx:
            main(["/nonexistent/dir/xyzzy"])
        self.assertNotEqual(ctx.exception.code, 0)


class TestDriverRobustness(unittest.TestCase):
    def test_analyze_file_never_raises_on_bad_csv(self):
        from alibz.pipeline import _analyze_file
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            fh.write("wavelength,intensity\nnot,numbers\n")
            path = fh.name
        try:
            row = _analyze_file((path, "db", 4, 4, 0, False))
            self.assertTrue(row["status"].startswith("error:"))
            self.assertEqual(row["n_peaks"], 0)
        finally:
            os.unlink(path)

    def test_execute_notebook_missing_dep_returns_message(self):
        from alibz.pipeline import execute_notebook
        with tempfile.NamedTemporaryFile("w", suffix=".ipynb",
                                         delete=False) as fh:
            fh.write("{}")
            path = fh.name
        try:
            ok, msg = execute_notebook(path)
            self.assertIsInstance(ok, bool)
            self.assertIsInstance(msg, str)
            self.assertTrue(msg)  # always a reason
        finally:
            os.unlink(path)

    def test_duplicate_basenames_do_not_collide(self):
        # rows are keyed by full path, so same basename in two subdirs both survive
        from alibz.pipeline import analyze_directory
        with tempfile.TemporaryDirectory() as root:
            db = os.path.join(root, "db")
            os.makedirs(db)
            for sub in ("a", "b"):
                d = os.path.join(root, sub)
                os.makedirs(d)
                with open(os.path.join(d, "s.csv"), "w") as fh:
                    fh.write("wavelength,intensity\nx,y\n")  # unparseable -> error rows
            rows = analyze_directory(root, pattern="*/s.csv", dbpath=db,
                                     workers=1, progress=lambda *_: None)
            self.assertEqual(len(rows), 2)  # both, not collapsed to one


if __name__ == "__main__":
    unittest.main()
