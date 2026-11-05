"""Fast tests for scripts/fit_quality_panel.py (analyze/render monkeypatched)."""
import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np

from scripts import fit_quality_panel as fq


class _Res:
    def __init__(self, fr, T=10_000.0, ne=17.0, r2=0.9, sd=None):
        self.element_fractions = fr
        self.temperature, self.ne, self.r_squared = T, ne, r2
        self.stage_disagreement = sd or {}


def _fake_analyze(x, y, db, **kw):
    return {"result": _Res({"Ca": 0.7, "Mg": 0.25, "Be": 0.05}, T=9000.0, ne=17.0,
                           sd={"Ca": 0.1, "Mg": 0.3}),
            "ne_bounds": (14.0, 17.0), "ne_bounds_source": "prior",
            "basin_info": {"basin_ambiguous": False}}


class TestScoring(unittest.TestCase):
    def setUp(self):
        import alibz.pipeline as pl
        import bench.synth_cases as sc
        self.pl, self.sc = pl, sc
        self.saved = (pl.analyze_spectrum, sc.render_case)
        pl.analyze_spectrum = _fake_analyze

        def fake_render(comp, seed=0, dbpath="db", **spec):
            scene = types.SimpleNamespace(metadata={
                "truth_temperature_k": 10_000.0, "truth_log_ne_cm3": 17.0})
            return np.arange(5.0), np.ones(5), scene
        sc.render_case = fake_render

    def tearDown(self):
        self.pl.analyze_spectrum, self.sc.render_case = self.saved

    def test_synthetic_scoring(self):
        m = fq.score_synthetic("binary_ca_mg", "db", {}, 11)
        self.assertEqual(m["majors_missed"], [])
        self.assertAlmostEqual(m["T_rel_err"], -0.1)
        self.assertAlmostEqual(m["abs_T_rel_err"], 0.1)
        self.assertAlmostEqual(m["spurious_mass"], 0.05)
        self.assertEqual(m["dominant_correct"], 1.0)
        self.assertEqual(m["ne_at_bound"], 1.0)  # 17.0 == upper bound
        self.assertEqual(m["log_ne_truth"], 17.0)

    def test_execute_error_row_and_schema(self):
        items = [{"panel": "synthetic", "id": "binary_ca_mg"},
                 {"panel": "synthetic", "id": "no_such_case"}]
        rows = fq.execute(items, "db", {}, 11, 1, 0)
        self.assertEqual([r["status"] for r in rows], ["ok", "error"])
        self.assertIn("KeyError", rows[1]["error"])
        for r in rows:
            self.assertEqual(set(r), {"panel", "id", "status", "metrics", "error"})
        agg = fq.aggregate(rows)
        self.assertEqual(agg["synthetic"]["n_error"], 1)
        self.assertAlmostEqual(agg["synthetic"]["spurious_mass"]["mean"], 0.05)

    def test_fe_masks_dead_channel(self):
        wl, y = fq.load_fe(fq.fe_run_ids()[0])
        self.assertTrue(len(y) < 7914 and y[-1] != 0)

    def test_fe_scoring(self):
        m = fq.score_fe(fq.fe_run_ids()[0], "db", {})
        self.assertAlmostEqual(m["non_fe_mass"], 1.0)  # fake has no Fe
        self.assertEqual(m["fe_fraction"], 0.0)
        self.assertEqual(m["n_non_fe"], 3)


def _doc(variant, vals):
    items = [{"panel": "synthetic", "id": "c", "status": "ok", "error": None,
              "metrics": vals}]
    return {"variant": variant, "config": {}, "items": items,
            "aggregate": fq.aggregate(items)}


class TestCompare(unittest.TestCase):
    def test_sign_conventions(self):
        b = _doc("a", {"median_rel_err": 0.2, "spurious_mass": 0.1,
                       "abs_T_rel_err": 0.1, "n_majors_missed": 1,
                       "dominant_correct": 0.0})
        v = _doc("b", {"median_rel_err": 0.1, "spurious_mass": 0.2,
                       "abs_T_rel_err": 0.1, "n_majors_missed": 0,
                       "dominant_correct": 1.0})
        md = fq.compare_docs(b, v)
        row = lambda m: [l for l in md.splitlines() if f"| {m} |" in l and "synthetic" in l][0]
        self.assertTrue(row("median_rel_err").endswith("better |"))
        self.assertTrue(row("spurious_mass").endswith("worse |"))
        self.assertTrue(row("abs_T_rel_err").endswith("same |"))
        self.assertTrue(row("n_majors_missed").endswith("better |"))
        self.assertTrue(row("dominant_correct").endswith("better |"))
        self.assertIn("**synthetic**", md)

    def test_missing_items_listed_and_fe_real_signs(self):
        b = _doc("a", {"median_rel_err": 0.2})
        v = _doc("b", {"median_rel_err": 0.2})
        v["items"][0]["id"] = "d"
        md = fq.compare_docs(b, v)
        self.assertIn("only in baseline: synthetic/c", md)
        self.assertIn("only in variant: synthetic/d", md)
        self.assertEqual(fq._verdict_one("r_squared", 0.5, 0.6), "better")
        self.assertEqual(fq._verdict_one("fe_fraction", 0.9, 0.8), "worse")
        self.assertEqual(fq._verdict_one("non_fe_mass", 0.1, 0.05), "better")
        self.assertEqual(fq._verdict_one("median_stage_disagreement", 0.1, 0.3), "worse")


if __name__ == "__main__":
    unittest.main()
