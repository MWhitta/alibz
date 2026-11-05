"""Acceptance statistics of the outer polish on the synthetic CASES."""
import json, sys
from multiprocessing import Pool
from bench.synth_cases import CASES, render_case


def one(a):
    name, method = a
    from alibz.pipeline import analyze_spectrum
    spec = dict(CASES[name]); comp = spec.pop("composition")
    x, y, _ = render_case(comp, seed=11, **spec)
    a = analyze_spectrum(x, y, "db", n_calls=8, draws=2, outer_polish=method)
    return name, method, a["result"].temperature, (a.get("basin_info") or {}).get("outer_polish")


if __name__ == "__main__":
    jobs = [(n, m) for n in CASES for m in ("powell", "nelder-mead")]
    with Pool(8) as p:
        out = p.map(one, jobs)
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
