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
    build_inspection_notebook,
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
            self.assertEqual(elements[0], "K")  # highest median first
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
            row = _analyze_file((path, "db", 4, 4, 0))
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
