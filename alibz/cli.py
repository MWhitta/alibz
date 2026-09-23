"""Command-line entry point: ``alibz-analyze <data_dir>``.

Analyzes every spectrum in a directory and writes ``summary.csv`` (element
abundances + uncertainties), ``detections.csv`` (long-format detection
status and upper limits), and ``fit_inspection.ipynb`` into it.  See
:mod:`alibz.pipeline` for the analysis chain and uncertainty semantics.
"""

import argparse
import json
import os
import sys

from alibz.pipeline import (
    DEFAULT_DRAWS,
    DEFAULT_GP_SEED,
    DEFAULT_N_CALLS,
    DEFAULT_PATTERN,
    DEFAULT_SEARCH,
    DEFAULT_STIMULATED_EMISSION,
    STAGE_CONSISTENCY_WEIGHT,
    DEFAULT_TIMEOUT_S,
    DETECTIONS_NAME,
    analyze_directory,
    build_inspection_notebook,
    execute_notebook,
    resolve_dbpath,
    write_detections_csv,
    write_inspection_notebook,
    write_summary_csv,
)


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="alibz-analyze",
        description="Quantitative CF-LIBS analysis of a directory of "
                    "spectra: writes summary.csv (element abundance + "
                    "1-sigma statistical uncertainty), detections.csv "
                    "(long-format detection status + upper limits), and "
                    "fit_inspection.ipynb into the directory.",
    )
    p.add_argument("data_dir", help="directory of wavelength,intensity CSVs")
    p.add_argument("--pattern", default=DEFAULT_PATTERN,
                   help=f"spectrum filename glob (default {DEFAULT_PATTERN!r})")
    p.add_argument("--db", default=None,
                   help="atomic database directory (default: $ALIBZ_DB, "
                        "./db, source checkout db, or bundled installed db)")
    p.add_argument("--workers", type=_positive_int,
                   default=max(1, (os.cpu_count() or 2) - 2),
                   help="parallel worker processes (default: cores - 2)")
    p.add_argument("--n-calls", type=_positive_int, default=DEFAULT_N_CALLS,
                   help="Bayesian-optimisation evaluations per index pass "
                        f"(default {DEFAULT_N_CALLS})")
    p.add_argument("--draws", type=_positive_int, default=DEFAULT_DRAWS,
                   help="amplitude-resampling draws for the uncertainty "
                        f"(default {DEFAULT_DRAWS})")
    p.add_argument("--physical-triage", choices=("off", "report", "prune"),
                   default="off", help="experimental physical priors before "
                   "pass-1 indexing; report records suggestions, prune applies "
                   "them provisionally (default off)")
    p.add_argument("--gas-wavelength-calibration", choices=("off", "report", "apply"),
                   default="apply", help="independent Ar I/O I wavelength references; "
                   "apply only within supported regions (default apply)")
    p.add_argument("--wavelength-registration",
                   choices=("off", "report", "ambient", "apply_element"),
                   default="ambient", help="line-shape-aware registration "
                   "(alibz.wavelength_registration): ambient applies the "
                   "Ar/O/N/H registration to the NIR when >= 3 lines agree and "
                   "records the element registrations; apply_element also "
                   "applies element registrations that pass quality (default ambient)")
    p.add_argument("--subpixel", choices=("gaussian", "parabolic"), default="gaussian",
                   help="sub-pixel line-centre method for the registration and the "
                   "per-peak refinement diagnostics (default gaussian)")
    p.add_argument("--timeout", type=_positive_int, default=DEFAULT_TIMEOUT_S,
                   help="per-spectrum timeout in seconds "
                        f"(default {DEFAULT_TIMEOUT_S})")
    p.add_argument("--limit", type=_positive_int, default=None,
                   help="analyze only the first N files (for quick checks)")
    p.add_argument("--out", default="summary.csv",
                   help="summary CSV filename (default summary.csv)")
    p.add_argument("--notebook", default="fit_inspection.ipynb",
                   help="inspection notebook filename "
                        "(default fit_inspection.ipynb)")
    p.add_argument("--no-notebook", action="store_true",
                   help="skip notebook generation")
    p.add_argument("--no-execute", action="store_true",
                   help="write the notebook without executing it")
    p.add_argument("--force", action="store_true",
                   help="regenerate the notebook even if it already exists "
                        "(overwrites any edits); summary.csv and "
                        "detections.csv are always rewritten")
    p.add_argument("--stimulated-emission",
                   action=argparse.BooleanOptionalAction,
                   default=DEFAULT_STIMULATED_EMISSION,
                   help="apply the induced-emission factor "
                        "(1 - exp(-hc/lambda kT)) to self-absorption "
                        "optical depths")
    p.add_argument("--search", default=DEFAULT_SEARCH,
                   help="outer plasma-state search mode for the indexer "
                        "passes: 'gp' (historical cold-start Bayesian "
                        "optimisation) or 'grid' (profile-likelihood scan "
                        "over T, ne seeding the GP; basin-stable). "
                        f"Default {DEFAULT_SEARCH!r}.")
    p.add_argument("--gp-seed", type=int, default=DEFAULT_GP_SEED,
                   help=argparse.SUPPRESS)  # sensitivity-harness knob
    p.add_argument("--weighted-solve",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="EXPERIMENTAL: whiten the concentration solve by "
                        "the per-peak area uncertainties (chi-squared "
                        "objective). Accurate at the true plasma state but "
                        "inverts the Ca/Mg ratio on the synthetic round "
                        "trip (2026-09-21); off by default.")
    p.add_argument("--stage-consistency-weight", type=float,
                   default=STAGE_CONSISTENCY_WEIGHT,
                   help="weight of the stage-consistency thermometer in "
                        "the plasma-state search: the extra misfit of "
                        "tying each element's ion stages (Saha ratio), in "
                        "data-misfit units; 0 disables it "
                        f"(default {STAGE_CONSISTENCY_WEIGHT})")
    p.add_argument("--no-provenance", action="store_true",
                   help="skip writing run_manifest.json (git state, config "
                        "snapshot, input hashes)")
    p.add_argument("--strict-provenance", action="store_true",
                   help="refuse to run when the worktree has uncommitted "
                        "or untracked changes (default: capture them under "
                        "<data_dir>/provenance/ and warn)")
    args = p.parse_args(argv)

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        p.error(f"not a directory: {data_dir}")
    try:
        dbpath = resolve_dbpath(args.db)
    except FileNotFoundError as exc:
        p.error(str(exc))

    print(f"alibz-analyze: {data_dir}")
    print(f"  db={dbpath}  workers={args.workers}  n_calls={args.n_calls}"
          f"  draws={args.draws}")

    rows = analyze_directory(
        data_dir, pattern=args.pattern, dbpath=dbpath,
        workers=args.workers, n_calls=args.n_calls, draws=args.draws,
        timeout_s=args.timeout, limit=args.limit,
        stimulated_emission=args.stimulated_emission,
        search=args.search, gp_seed=args.gp_seed,
        weighted_solve=args.weighted_solve,
        physical_triage=args.physical_triage,
        gas_wavelength_calibration=args.gas_wavelength_calibration,
        wavelength_registration=args.wavelength_registration,
        subpixel=args.subpixel,
        stage_consistency_weight=args.stage_consistency_weight,
        provenance=not args.no_provenance,
        strict_provenance=args.strict_provenance,
        exclude=(args.out, DETECTIONS_NAME),
    )
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    gas_path = os.path.join(data_dir, "background_gases.json")
    with open(gas_path, "w") as fh:
        json.dump({r["file"]: r.get("background_gases", {}) for r in rows},
                  fh, indent=2, allow_nan=False)
        fh.write("\n")
    calibration_path = os.path.join(data_dir, "wavelength_calibration.json")
    with open(calibration_path, "w") as fh:
        json.dump({r["file"]: r.get("wavelength_calibration", {}) for r in rows},
                  fh, indent=2, allow_nan=False)
        fh.write("\n")
    if args.physical_triage != "off":
        triage_path = os.path.join(data_dir, "physical_triage.json")
        with open(triage_path, "w") as fh:
            json.dump({r["file"]: r.get("physical_triage") for r in rows},
                      fh, indent=2, allow_nan=False)
            fh.write("\n")

    csv_path = os.path.join(data_dir, args.out)
    elements = write_summary_csv(rows, csv_path)
    print(f"\nwrote {csv_path}  ({n_ok}/{len(rows)} spectra ok, "
          f"{len(elements)} elements)")

    det_path = os.path.join(data_dir, DETECTIONS_NAME)
    n_det = write_detections_csv(rows, det_path)
    print(f"wrote {det_path}  ({n_det} per-element detection records "
          f"incl. borderline evidence and upper limits)")

    if not args.no_notebook:
        nb_path = os.path.join(data_dir, args.notebook)
        if os.path.exists(nb_path) and not args.force:
            print(f"notebook exists, keeping it (use --force to regenerate "
                  f"and overwrite your edits): {nb_path}")
        else:
            nb = build_inspection_notebook(
                data_dir, dbpath, pattern=args.pattern, summary_name=args.out,
                n_calls=args.n_calls,
                stimulated_emission=args.stimulated_emission,
                gas_wavelength_calibration=args.gas_wavelength_calibration,
            )
            write_inspection_notebook(nb, nb_path)
            print(f"wrote {nb_path}")
            if not args.no_execute:
                print("executing notebook (one live spectrum, ~1-3 min)...")
                ok, msg = execute_notebook(nb_path)
                print(f"notebook {'executed' if ok else 'not executed'}: {msg}")

    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
