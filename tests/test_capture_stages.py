"""Opt-in per-stage capture of the production chain (manuscript figures).

``analyze_spectrum(..., capture_stages=True)`` adds a ``"stages"`` list that
snapshots the real chain stage by stage; ``capture_stages=False`` (the
default) must leave the returned dict byte-for-byte identical.  These tests
prove the no-op guarantee, the snapshot schema/order, and that a snapshot is
decoupled from later in-place mutation.

The end-to-end run is the expensive part, so both runs (False and True) are
done once in ``setUpClass`` on a cropped synthetic case (370-530 nm of the
suite-S ``binary_ca_mg`` scene — Ca + Mg are both present there) with the
deterministic grid search, a fixed GP seed, ``n_calls=2`` and ``draws=1``.
The whole file runs in well under 90 s.

Run:  ALIBZ_SKIP_ROUNDTRIP=1 .venv/bin/python -m pytest \
          tests/test_capture_stages.py -q
"""

import math
import os
import unittest

import numpy as np

# single-threaded BLAS for determinism, as the fit-quality panel does
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from bench.synth_cases import CASES, render_case          # noqa: E402
from alibz.pipeline import AnalysisConfig, analyze_spectrum  # noqa: E402
from alibz.peaky_indexer_v3 import FitResult              # noqa: E402

#: documented pipeline order of the stage machine keys.
EXPECTED_ORDER = [
    "blind_fit", "registration", "refine_data", "pass1", "refine_physics",
    "seed", "recover", "deblend", "pass2", "deepen", "deepen_final",
    "sa_recovery", "detections",
]

RUN_KW = dict(n_calls=2, draws=1, gp_seed=11, search="grid")


def _scalar_equal(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def _equal(a, b, path="root"):
    """Recursive deep equality tailored to the analysis dict.

    Returns ``(ok, path)`` where ``path`` points at the first mismatch.
    FitResults compare by (temperature, ne, element_fractions) per the
    task brief; numpy arrays by element (NaN == NaN); SegmentShift and
    other slotted/attribute objects by their attributes.
    """
    if isinstance(a, FitResult) or isinstance(b, FitResult):
        if not (isinstance(a, FitResult) and isinstance(b, FitResult)):
            return False, f"{path}:FitResult-type"
        for attr in ("temperature", "ne"):
            if not _scalar_equal(float(getattr(a, attr)),
                                 float(getattr(b, attr))):
                return False, f"{path}.{attr}"
        return _equal(dict(a.element_fractions), dict(b.element_fractions),
                      f"{path}.element_fractions")
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        aa, bb = np.asarray(a), np.asarray(b)
        if aa.shape != bb.shape:
            return False, f"{path}:shape"
        try:
            ok = bool(np.array_equal(aa, bb, equal_nan=True))
        except TypeError:                 # non-float dtype: no equal_nan
            ok = bool(np.array_equal(aa, bb))
        return ok, path
    if isinstance(a, np.generic):
        a = a.item()
    if isinstance(b, np.generic):
        b = b.item()
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return False, f"{path}:dict-type"
        if set(a) != set(b):
            return False, f"{path}:keys {set(a) ^ set(b)}"
        for k in a:
            ok, p = _equal(a[k], b[k], f"{path}[{k!r}]")
            if not ok:
                return False, p
        return True, path
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            return False, f"{path}:seq-type"
        if len(a) != len(b):
            return False, f"{path}:len {len(a)}!={len(b)}"
        for i, (ai, bi) in enumerate(zip(a, b)):
            ok, p = _equal(ai, bi, f"{path}[{i}]")
            if not ok:
                return False, p
        return True, path
    if isinstance(a, (str, bytes, bool, int, float, type(None))):
        return _scalar_equal(a, b), path
    # fall-through: slotted / attribute objects such as SegmentShift
    if type(a) is not type(b):
        return False, f"{path}:type {type(a)}!={type(b)}"
    slots = getattr(type(a), "__slots__", None)
    if slots:
        for s in slots:
            ok, p = _equal(getattr(a, s), getattr(b, s), f"{path}.{s}")
            if not ok:
                return False, p
        return True, path
    if hasattr(a, "__dict__"):
        return _equal(vars(a), vars(b), f"{path}.__dict__")
    return _scalar_equal(a, b), path


def _render_cropped():
    spec = dict(CASES["binary_ca_mg"])
    comp = spec.pop("composition")
    x, y, _scene = render_case(comp, seed=11, **spec)
    mask = (x >= 370.0) & (x <= 530.0)
    return x[mask], y[mask]


class TestCaptureStages(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        x, y = _render_cropped()
        cls.off = analyze_spectrum(x, y, "db", capture_stages=False, **RUN_KW)
        cls.on = analyze_spectrum(x, y, "db", capture_stages=True, **RUN_KW)

    # --- test 1: no-op proof --------------------------------------------
    def test_false_has_no_stages_key(self):
        self.assertNotIn("stages", self.off)

    def test_true_minus_stages_equals_false(self):
        self.assertIn("stages", self.on)
        on_wo = {k: v for k, v in self.on.items() if k != "stages"}
        self.assertEqual(set(on_wo), set(self.off),
                         msg="top-level key sets differ")
        ok, path = _equal(on_wo, self.off)
        self.assertTrue(ok, msg=f"first difference at {path}")

    # --- test 2: schema --------------------------------------------------
    def test_stage_keys_in_documented_order(self):
        keys = [s["stage"] for s in self.on["stages"]]
        # collapse the variable number of 'deepen' bars to one for the
        # order check; their relative order is preserved by a stable sort
        collapsed, prev = [], None
        for k in keys:
            if k == "deepen" and prev == "deepen":
                continue
            collapsed.append(k)
            prev = k
        self.assertEqual(collapsed, EXPECTED_ORDER, msg=f"got {keys}")

    def test_every_snapshot_has_core_fields(self):
        for s in self.on["stages"]:
            for f in ("stage", "label", "n_peaks"):
                self.assertIn(f, s, msg=f"{s.get('stage')} missing {f}")
            self.assertIsInstance(s["n_peaks"], int)

    def test_solve_snapshots_have_state_and_increasing_index(self):
        last = -1
        n_solves = 0
        for s in self.on["stages"]:
            if "solve_index" not in s:
                continue
            n_solves += 1
            for f in ("T", "log_ne", "element_fractions"):
                self.assertIn(f, s, msg=f"{s['stage']} missing {f}")
            self.assertGreater(s["solve_index"], last,
                               msg=f"solve_index not increasing at {s['stage']}")
            last = s["solve_index"]
        # pass1, pass2, >=1 deepen bar, deepen_final
        self.assertGreaterEqual(n_solves, 3)

    def test_snapshot_fit_decoupled_from_final(self):
        snaps = [s for s in self.on["stages"] if "fit" in s]
        self.assertTrue(snaps)
        snap = snaps[-1]                       # deepen_final carries `final`
        final_arr = np.array(self.on["final"]["sorted_parameter_array"],
                             copy=True)
        snap["fit"]["sorted_parameter_array"][...] += 1.0
        self.assertTrue(
            np.array_equal(final_arr,
                           self.on["final"]["sorted_parameter_array"]),
            msg="mutating a snapshot fit changed analysis['final']")

    # --- test 3: config default -----------------------------------------
    def test_config_default_is_false(self):
        self.assertIs(AnalysisConfig(dbpath="db").capture_stages, False)


if __name__ == "__main__":
    unittest.main()
