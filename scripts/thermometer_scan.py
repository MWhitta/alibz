"""Scan the outer objective's components against the plasma state.

Runs ``analyze_spectrum`` on one spectrum (a CSV path, or ``case:<name>``
for a synthetic scene from ``bench.synth_cases`` whose truth is known),
rebuilds the final indexer at the pipeline's final peak table, and tabulates
the data misfit, the stage-consistency (stage-tie) cost and the composition
over a (T, log ne) grid at the fitted kernel widths.  Shows whether the
thermometer has a minimum, where it is, and how it compares with the
amplitude objective's own (nearly flat) profile.

    python scripts/thermometer_scan.py data/remote_samples/user_spec1.csv
    python scripts/thermometer_scan.py case:binary_ca_mg --weight 0
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alibz.pipeline import (POISSON_GAIN_COUNTS, _get_db, _get_sb,  # noqa: E402
                            _warm_start_temperature, analyze_spectrum,
                            load_spectrum_csv, resolve_dbpath)
from alibz.peaky_indexer_v3 import PeakyIndexerV3  # noqa: E402


def load(spec, seed=11):
    if spec.startswith("case:"):
        from bench.synth_cases import CASES, render_case
        case = dict(CASES[spec[5:]])
        comp = case.pop("composition")
        x, y, scene = render_case(comp, seed=seed, **case)
        truth = dict(scene.metadata)
        truth["truth_composition"] = comp
        return x, y, truth
    x, y = load_spectrum_csv(spec)
    return x, y, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spectrum")
    ap.add_argument("--weight", type=float, default=None,
                    help="stage_consistency_weight for the pipeline run "
                         "(default: the package default)")
    ap.add_argument("--n-calls", type=int, default=None)
    ap.add_argument("--n-t", type=int, default=15)
    ap.add_argument("--n-ne", type=int, default=6)
    ap.add_argument("--save", default=None,
                    help="save the surfaces and per-node compositions (.npz)")
    args = ap.parse_args()

    x, y, truth = load(args.spectrum)
    dbpath = resolve_dbpath()
    kw = {}
    if args.weight is not None:
        kw["stage_consistency_weight"] = args.weight
    if args.n_calls is not None:
        kw["n_calls"] = args.n_calls
    elif truth is not None:
        kw.update(n_calls=8, draws=2)
    t0 = time.time()
    an = analyze_spectrum(x, y, dbpath, **kw)
    res = an["result"]
    info = res.convergence_info or {}
    fr = sorted(res.element_fractions.items(), key=lambda kv: -kv[1])[:6]
    print(f"pipeline {time.time() - t0:.0f}s: T={res.temperature:.0f} "
          f"ne={res.ne:.2f} sig={res.sigma:.3f} gam={res.gamma:.3f} "
          f"r2={res.r_squared:.3f} tie={info.get('stage_tie_cost', 0):.3e} "
          f"top={[(e, round(v, 3)) for e, v in fr]}")
    print(f"  stage_disagreement={ {k: round(v, 2) for k, v in res.stage_disagreement.items()} }")
    print(f"  tie_by_element={ {k: f'{v:.2e}' for k, v in info.get('stage_tie_by_element', {}).items()} }")
    print(f"  segment_response={an.get('segment_response')} "
          f"meta={an.get('segment_response_metadata')}")
    if truth is not None:
        print(f"  TRUTH T={truth['truth_temperature_k']:.0f} "
              f"ne={truth['truth_log_ne_cm3']:.2f} comp={truth['truth_composition']}")

    # rebuild the final indexer on the final peak table (pipeline frame)
    from alibz.detector import correct_segment_response
    from alibz.inspection import estimate_peak_uncertainties
    from alibz.utils.wavelength import shift_at
    db, sb = _get_db(dbpath), _get_sb(dbpath)
    shift, seg_response = an["shift"], an["segment_response"]
    bg0 = np.asarray(an["fit"].get("background", np.zeros_like(y)), float)
    _resid = y - bg0
    noise = 1.4826 * float(np.median(np.abs(_resid - np.median(_resid))))
    peaks = an["final"]["sorted_parameter_array"]
    out = peaks.copy()
    out[:, 1] -= shift_at(shift, out[:, 1])
    out = correct_segment_response(out, seg_response, edges=(620.0,))
    sig = estimate_peak_uncertainties(x, y - bg0, peaks)[:, 0]
    seg = np.searchsorted(np.asarray([620.0]), peaks[:, 1])
    sig = sig / seg_response[seg]
    ne_init = an.get("ne_init")
    ne_prior = ((float(ne_init), 0.15) if ne_init is not None else (17.0, 0.5))
    idx = PeakyIndexerV3(out, dbpath=dbpath, db=db, sb=sb, amp_sigma=sig,
                         amp_sigma_floor=noise,
                         amp_sigma_poisson_gain=POISSON_GAIN_COUNTS,
                         ne_prior=ne_prior,
                         stage_consistency_weight=1.0,
                         temperature_init=_warm_start_temperature(res.temperature),
                         ne_init=res.ne)
    idx.build_candidate_matrix(sa_doublets=True)
    idx._rebuild_overlap(res.sigma, res.gamma)

    Ts = np.linspace(4000.0, 25000.0, args.n_t)
    nes = np.linspace(14.0, 19.0, args.n_ne)
    data = np.zeros((Ts.size, nes.size))
    tie = np.zeros_like(data)
    comp = {}
    for i, T in enumerate(Ts):
        for j, ne in enumerate(nes):
            c, cost = idx._solve_concentrations(float(T), float(ne))
            data[i, j] = cost
            tie[i, j], _per = idx._stage_tie_cost(c, idx._last_A, A_all=idx._last_A_all)
            _ec, fr, dis = idx._aggregate_elements(c, idx._last_A, amp_sigma=sig)
            comp[(i, j)] = (fr, dis)
    tot = data + tie
    if args.save:
        import json
        np.savez(args.save, Ts=Ts, nes=nes, data=data, tie=tie,
                 comp=json.dumps({f"{i},{j}": [fr, dis]
                                  for (i, j), (fr, dis) in comp.items()}),
                 truth=json.dumps(truth), final=json.dumps(dict(
                     T=res.temperature, ne=res.ne, r2=res.r_squared,
                     fractions=res.element_fractions)))
    elems = sorted({e for (fr, _d) in comp.values() for e in fr},
                   key=lambda e: -max(comp[k][0].get(e, 0) for k in comp))[:5]

    def argmin(m):
        i, j = np.unravel_index(np.argmin(m), m.shape)
        return Ts[i], nes[j]

    print(f"\nmin data-only  at T={argmin(data)[0]:.0f} ne={argmin(data)[1]:.2f}"
          f"   (data range {data.min():.3e} .. {data.max():.3e})")
    print(f"min tie-only   at T={argmin(tie)[0]:.0f} ne={argmin(tie)[1]:.2f}"
          f"   (tie range {tie.min():.3e} .. {tie.max():.3e})")
    print(f"min data+tie   at T={argmin(tot)[0]:.0f} ne={argmin(tot)[1]:.2f}")
    j_best = int(np.argmin(np.min(tot, axis=0)))
    print(f"\nprofile at ne={nes[j_best]:.2f} (columns: T, data, tie, total, "
          f"{elems}, max stage disagreement)")
    for i, T in enumerate(Ts):
        fr, dis = comp[(i, j_best)]
        fs = " ".join(f"{fr.get(e, 0):.2f}" for e in elems)
        md = max(dis.values(), default=0.0)
        print(f"  {T:6.0f}  {data[i, j_best]:.4e}  {tie[i, j_best]:.4e}  "
              f"{tot[i, j_best]:.4e}  {fs}  {md:.2f}")
    # full surfaces (rows T, columns ne) and the truth node when known
    def table(name, m):
        print(f"\n{name} surface (rows T, columns ne={[round(v, 2) for v in nes]})")
        for i, T in enumerate(Ts):
            print(f"  {T:6.0f}  " + "  ".join(f"{v:.3e}" for v in m[i]))
    table("data", data)
    table("tie", tie)
    if truth is not None:
        i = int(np.argmin(np.abs(Ts - truth["truth_temperature_k"])))
        j = int(np.argmin(np.abs(nes - truth["truth_log_ne_cm3"])))
        fr, dis = comp[(i, j)]
        print(f"\nnearest node to truth: T={Ts[i]:.0f} ne={nes[j]:.2f} "
              f"data={data[i, j]:.4e} tie={tie[i, j]:.4e} "
              f"fractions={ {k: round(v, 3) for k, v in fr.items()} } "
              f"disagreement={ {k: round(v, 2) for k, v in dis.items()} }")
    i_best = int(np.argmin(np.min(tot, axis=1)))
    print(f"\nprofile at T={Ts[i_best]:.0f} (columns: ne, data, tie, total)")
    for j, ne in enumerate(nes):
        print(f"  {ne:5.2f}  {data[i_best, j]:.4e}  {tie[i_best, j]:.4e}  "
              f"{tot[i_best, j]:.4e}")


if __name__ == "__main__":
    main()
