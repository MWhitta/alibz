"""Reproduce an end-to-end real-spectrum triage/gas pipeline smoke check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(
        "provenance/physical-triage-real-data-20260922.json"))
    parser.add_argument("--out", type=Path, default=Path(
        "reports/physical-triage-pipeline-smoke-20260922.json"))
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--replay-archive", type=Path,
                        help="Use the pinned repository archive instead of scratch CSVs")
    parser.add_argument("--n-calls", type=int, default=3)
    parser.add_argument("--gas-wavelength-calibration", choices=("off", "report", "apply"),
                        default="apply")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    sample = manifest["datasets"][0]["samples"][args.sample]
    if args.replay_archive:
        from scripts.benchmark_physical_triage import load_replay_archive
        records, digest = load_replay_archive(
            args.replay_archive,
            manifest["datasets"][0]["repo_local_archive"]["sha256"])
        _, x, y = next(record for record in records
                       if record[0]["run_id"] == sample["run_id"])
        data = np.column_stack((x, y))
        source = args.replay_archive
    else:
        source = Path(sample["path"])
        import hashlib
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != sample["sha256"]:
            raise ValueError("benchmark input differs from the pinned source")
        data = None
    print(json.dumps({"input": str(source), "sha256": digest,
                      "output": str(args.out), "dry_run": args.dry_run}))
    if args.dry_run:
        return 0
    from alibz.pipeline import analyze_spectrum
    if data is None:
        data = np.loadtxt(source, delimiter=",", skiprows=1)
    # The pinned native-API Fe export has an anomalous dense fourth-channel
    # tail after a 12-nm gap. Match the audited benchmark's usable coverage.
    data = data[data[:, 0] < 950]
    start = time.perf_counter()
    analysis = analyze_spectrum(
        data[:, 0], data[:, 1], "db", n_calls=args.n_calls, draws=1,
        seed_minor=False, physical_triage="report",
        gas_wavelength_calibration=args.gas_wavelength_calibration)
    result = analysis["result"]
    output = dict(input=str(source), input_sha256=digest,
                  run_id=sample.get("run_id"),
                  seconds=time.perf_counter() - start, n_calls=args.n_calls,
                  physical_triage=analysis["physical_triage"],
                  background_gases=analysis["background_gases"],
                  wavelength_calibration=analysis["wavelength_calibration"],
                  fractions=result.element_fractions, r_squared=result.r_squared,
                  scope="Pipeline wiring smoke check; small optimizer budget is not a composition validation.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"seconds": output["seconds"],
                      "gas_status": {k: v["status"] for k, v in output["background_gases"].items()},
                      "r_squared": output["r_squared"], "fractions": output["fractions"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
