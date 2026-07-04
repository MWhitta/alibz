"""Command-line entry point: ``alibz-analyze <data_dir>``.

Analyzes every spectrum in a directory and writes ``summary.csv`` (element
abundances + uncertainties) and ``fit_inspection.ipynb`` into it.  See
:mod:`alibz.pipeline` for the analysis chain and uncertainty semantics.
"""

import argparse
import os
import sys

from alibz.pipeline import (
    DEFAULT_DRAWS,
    DEFAULT_N_CALLS,
    DEFAULT_PATTERN,
    DEFAULT_TIMEOUT_S,
    analyze_directory,
    build_inspection_notebook,
    execute_notebook,
    resolve_dbpath,
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
                    "1-sigma statistical uncertainty) and "
                    "fit_inspection.ipynb into the directory.",
    )
    p.add_argument("data_dir", help="directory of wavelength,intensity CSVs")
    p.add_argument("--pattern", default=DEFAULT_PATTERN,
                   help=f"spectrum filename glob (default {DEFAULT_PATTERN!r})")
    p.add_argument("--db", default=None,
                   help="atomic database directory (default: $ALIBZ_DB, "
                        "./db, or the repo db)")
    p.add_argument("--workers", type=_positive_int,
                   default=max(1, (os.cpu_count() or 2) - 2),
                   help="parallel worker processes (default: cores - 2)")
    p.add_argument("--n-calls", type=_positive_int, default=DEFAULT_N_CALLS,
                   help="Bayesian-optimisation evaluations per index pass "
                        f"(default {DEFAULT_N_CALLS})")
    p.add_argument("--draws", type=_positive_int, default=DEFAULT_DRAWS,
                   help="amplitude-resampling draws for the uncertainty "
                        f"(default {DEFAULT_DRAWS})")
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
                        "(overwrites any edits); summary.csv is always "
                        "rewritten")
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
        exclude=(args.out,),
    )
    n_ok = sum(1 for r in rows if r["status"] == "ok")

    csv_path = os.path.join(data_dir, args.out)
    elements = write_summary_csv(rows, csv_path)
    print(f"\nwrote {csv_path}  ({n_ok}/{len(rows)} spectra ok, "
          f"{len(elements)} elements)")

    if not args.no_notebook:
        nb_path = os.path.join(data_dir, args.notebook)
        if os.path.exists(nb_path) and not args.force:
            print(f"notebook exists, keeping it (use --force to regenerate "
                  f"and overwrite your edits): {nb_path}")
        else:
            nb = build_inspection_notebook(
                data_dir, dbpath, pattern=args.pattern, summary_name=args.out,
                n_calls=args.n_calls,
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
