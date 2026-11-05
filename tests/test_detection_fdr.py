"""BH / BY false-discovery-rate control of the element detection gate."""
import csv

import numpy as np

from alibz.peaky_indexer_v3 import (ELEMENT_DETECT_Z_MIN, PeakyIndexerV3,
                                    fdr_select)

PEAKS = np.array([[120.0, 393.37, 0.05, 0.02],
                  [80.0, 396.85, 0.05, 0.02],
                  [40.0, 422.67, 0.06, 0.02],
                  [25.0, 279.55, 0.05, 0.02],
                  [18.0, 285.21, 0.05, 0.02]])


def _zvec():
    # NB m = 303 not 30: with 3 true elements BH's step-up threshold at
    # rank 3-6 is already ~z 2.3-2.6 for m = 30, so BH *legitimately*
    # admits z ~ 3 noise there; m = 300 puts it near z 3.3.
    rng = np.random.default_rng(3)
    z = {"T1": 100.0, "T2": 25.0, "T3": 6.0}
    noise = np.clip(rng.normal(size=297), -1.5, 2.2)
    for i, v in enumerate(noise):
        z[f"N{i:03d}"] = float(v)
    for k, v in zip(("Na", "Nb", "Nc"), (2.5, 3.05, 3.2)):
        z[k] = v
    return z


def test_bh_admits_true_rejects_noise():
    z = _zvec()
    plain = {e for e, v in z.items() if v >= ELEMENT_DETECT_Z_MIN}
    assert any(e.startswith("N") for e in plain)
    keep, info = fdr_select(z, 0.05, "bh")
    assert keep == {"T1", "T2", "T3"}
    assert info["m"] == 303
    assert info["threshold_z"] is not None and info["threshold_z"] >= 6.0 - 1e-9
    assert all(0.0 <= q <= 1.0 for q in info["q_values"].values())
    assert all(info["q_values"][e] <= 0.05 for e in keep)


def test_by_at_least_as_conservative():
    z = _zvec()
    for q in (0.01, 0.05, 0.2, 0.5):
        kb, _ = fdr_select(z, q, "bh")
        ky, _ = fdr_select(z, q, "by")
        assert ky <= kb
    # a borderline element separates them
    zb = {"A": 2.054, "B": 0.0}
    assert fdr_select(zb, 0.05, "bh")[0] == {"A"}
    assert fdr_select(zb, 0.05, "by")[0] == set()


def test_floor_never_admits_below():
    keep, _ = fdr_select({"A": 1.9, "B": 1.0, "C": 0.5}, 0.5, "bh")
    assert keep == set()
    keep, _ = fdr_select({"A": 1.9, "B": 8.0}, 0.5, "bh", z_floor=2.0)
    assert keep == {"B"}


def test_off_matches_hard_gate():
    sig = np.full(len(PEAKS), 2.0)
    idx = PeakyIndexerV3(PEAKS, dbpath="db", amp_sigma=sig)
    r = idx.run(n_calls=6, verbose=False)
    info = r.convergence_info
    assert "detection_fdr_q" not in info and "detection_q_value" not in info
    dz = info["detection_z"]
    assert dz
    assert set(r.element_fractions) <= {e for e, v in dz.items()
                                        if v >= ELEMENT_DETECT_Z_MIN}
    # explicit None is the same code path as unset
    idx2 = PeakyIndexerV3(PEAKS, dbpath="db", amp_sigma=sig,
                          detection_fdr=None)
    r2 = idx2.run(n_calls=6, verbose=False)
    assert r2.element_fractions == r.element_fractions
    assert r2.convergence_info["detection_z"] == dz


def test_on_records_convergence_info():
    sig = np.full(len(PEAKS), 2.0)
    idx = PeakyIndexerV3(PEAKS, dbpath="db", amp_sigma=sig,
                         detection_fdr=0.05)
    info = idx.run(n_calls=6, verbose=False).convergence_info
    assert info["detection_fdr_q"] == 0.05
    assert info["detection_m"] == len(info["detection_z"])
    assert set(info["detection_q_value"]) == set(info["detection_z"])


def test_csv_carries_q_value(tmp_path):
    from alibz.pipeline import write_detections_csv
    base = dict(element="Fe", status="detected", fraction=0.5, unc=0.01,
                z=50.0, n_lines=3)
    rows = [dict(sample="s", status="ok", detections=[dict(base, q_value=0.001)])]
    p = tmp_path / "d.csv"
    write_detections_csv(rows, str(p))
    r = list(csv.DictReader(open(p)))
    assert r[0]["q_value"] == "0.001"
    rows = [dict(sample="s", status="ok", detections=[dict(base)])]
    write_detections_csv(rows, str(p))
    assert "q_value" not in open(p).readline()


def test_classify_status_ladder():
    from alibz.detections import classify_detections

    class R:
        element_fractions = {"A": 0.5, "B": 0.3, "C": 0.15, "D": 0.05}
        stage_disagreement = {}
        convergence_info = dict(
            detection_fdr_q=0.05, detection_fdr_threshold_z=6.0,
            detection_z_floor=2.0,
            detection_z={"A": 50.0, "B": 7.0, "C": 3.0, "D": 1.0},
            detection_q_value={"A": 0.0, "B": 0.01, "C": 0.2, "D": 0.8})
    stats = {e: dict(std=0.001, mean=R.element_fractions[e], p95=0.0)
             for e in "ABCD"}
    sup = {"A": [(1, 300, 9), (1, 301, 9)], "B": [(1, 310, 9)],
           "C": [(1, 320, 9), (1, 321, 9)], "D": [(1, 330, 9), (1, 331, 9)]}
    d = {x["element"]: x for x in classify_detections(R, stats, sup)}
    assert d["A"]["status"] == "detected" and d["A"]["q_value"] == 0.0
    assert d["B"]["status"] == "single-line"
    assert d["C"]["status"] == "marginal"
    assert d["D"]["status"] == "weak"
