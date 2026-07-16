"""Single command surface for MW2-112 analysis and provenance tasks."""

from __future__ import annotations

import argparse
import importlib
import sys


COMMANDS = {
    "relative": (
        "measure the primary within-element relative profiles",
        "scripts.build_mw2_112_relative_profiles"),
    "reassemble": (
        "rebuild element/contrast tables from line data or checkpoints",
        "scripts.reassemble_relative_profiles"),
    "report": (
        "build validation, segmentation, figures, and report",
        "scripts.report_mw2_112_relative_profiles"),
    "regenerate": (
        "reproduce a bundle from raw spectra and frozen priors",
        "scripts.regenerate_mw2_112_relative_profiles"),
    "provenance": (
        "freeze or verify a provenance bundle",
        "scripts.mw2_112_provenance"),
    "peak-pca": (
        "quantify the profile with corpus-prior peak-window PCA",
        "scripts.run_mw2_112_peak_window_pca"),
    "quantitative": (
        "run the experimental CF-LIBS/pilot workflow",
        "scripts.run_mw2_112"),
}


def _parser(commands) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mw2-112",
        description="MW2-112 LIBS profile analysis and reproducibility tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", choices=tuple(commands))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parser.epilog = "commands:\n" + "\n".join(
        f"  {name:<12} {description}"
        for name, (description, _handler) in commands.items())
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser(COMMANDS)
    if not argv:
        parser.print_help()
        return 0
    if argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    namespace = parser.parse_args(argv[:1])
    _description, module_name = COMMANDS[namespace.command]
    handler = importlib.import_module(module_name).main
    return handler(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
