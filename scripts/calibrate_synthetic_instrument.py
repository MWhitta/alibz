#!/usr/bin/env python
"""Calibrate provisional synthetic-observation statistics from individual shots."""

import argparse
from pathlib import Path

import numpy as np

from alibz.synthetic_calibration import calibrate_individual_shots


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Directory containing individual-shot CSV files")
    parser.add_argument("--pattern", default="*.csv", help="Recursive glob pattern")
    parser.add_argument("--max-spectra", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output JSON artifact")
    args = parser.parse_args(argv)

    paths = sorted(Path(args.data_dir).rglob(args.pattern))
    if args.max_spectra is not None and len(paths) > args.max_spectra:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(paths), args.max_spectra, replace=False))
        paths = [paths[i] for i in keep]
    calibration = calibrate_individual_shots(paths)
    calibration.save(args.out)
    print(f"calibrated {calibration.n_spectra} individual shots")
    print(f"artifact: {Path(args.out).resolve()}")
    for label, stats in calibration.segments.items():
        print(
            f"{label}: baseline={stats['baseline_count_median']:.3g}, "
            f"noise={stats['local_noise_median']:.3g}, "
            f"negative={100 * stats['negative_fraction_median']:.3g}%"
        )


if __name__ == "__main__":
    main()
