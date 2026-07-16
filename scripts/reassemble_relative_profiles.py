"""Rebuild coherent-stage element/contrast tables from line checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alibz.mw2_112 import (
    DETECTION_SNR,
    MINIMUM_COHERENT_LINES_PER_STAGE,
    MINIMUM_LINE_COHERENCE_SPEARMAN,
    atomic_json,
    load_line_records,
    software_state,
    utc_now,
    write_relative_tables,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    try:
        lines = load_line_records(run_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    write_relative_tables(run_dir, lines)

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["postprocessed_utc"] = utc_now()
    manifest["postprocessing"] = {
        "minimum_line_coherence_spearman": MINIMUM_LINE_COHERENCE_SPEARMAN,
        "minimum_coherent_lines_per_stage": MINIMUM_COHERENT_LINES_PER_STAGE,
        "detection_snr": DETECTION_SNR,
        "source_tree_sha256": software_state(
            Path(__file__).resolve().parents[1])["source_tree_sha256"],
    }
    atomic_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
