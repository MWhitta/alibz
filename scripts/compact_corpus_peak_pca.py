"""Extract a compact, provenance-rich peak-shape PCA basis.

Legacy corpus results contain millions of per-peak dictionaries after the
small PCA arrays.  ``--legacy-pickle`` truncates that top-level value without
loading it, then writes only the reusable basis and scalar provenance.
"""

from __future__ import annotations

import argparse
import json
import pickle
import pickletools
import struct
import tempfile
from pathlib import Path

import numpy as np

from alibz.mw2_112 import hash_file, utc_now


def _compact_legacy_pickle(source: Path, stop_key: str) -> dict:
    """Load top-level keys preceding a huge list value in a protocol-4 pickle."""
    last_frame = None
    value_start = None
    with source.open("rb") as fh:
        iterator = iter(pickletools.genops(fh))
        for opcode, argument, position in iterator:
            if opcode.name == "FRAME":
                last_frame = (position, int(argument))
            if argument == stop_key:
                # Include an optional MEMOIZE for the key, stop at the value.
                next_opcode, _next_argument, next_position = next(iterator)
                if next_opcode.name == "MEMOIZE":
                    _value_opcode, _value_argument, value_start = next(iterator)
                else:
                    value_start = next_position
                break
    if value_start is None or last_frame is None:
        raise ValueError(f"could not locate framed top-level key {stop_key!r}")
    frame_position, _frame_size = last_frame
    frame_payload_start = frame_position + 9
    if value_start <= frame_payload_start:
        raise ValueError("unsupported pickle frame boundary")
    with tempfile.NamedTemporaryFile(suffix=".pkl") as temporary:
        with source.open("rb") as source_fh:
            remaining = value_start
            while remaining:
                chunk = source_fh.read(min(2 ** 20, remaining))
                if not chunk:
                    raise EOFError(source)
                temporary.write(chunk)
                remaining -= len(chunk)
        # Empty value, memoize, SETITEMS for the top-level dict, STOP.
        temporary.write(bytes((0x5D, 0x94, 0x75, 0x2E)))
        temporary.flush()
        with open(temporary.name, "r+b") as output:
            output.seek(frame_position + 1)
            output.write(struct.pack(
                "<Q", value_start - frame_payload_start))
        with open(temporary.name, "rb") as compact:
            return pickle.load(compact)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--legacy-pickle", action="store_true")
    parser.add_argument("--window-multiplier", type=float, default=0.5)
    parser.add_argument("--source-notebook", default="notebooks/peaky_data.ipynb")
    args = parser.parse_args(argv)
    source = args.source.resolve()
    if args.legacy_pickle:
        results = _compact_legacy_pickle(source, "peak_metadata")
    else:
        with source.open("rb") as fh:
            results = pickle.load(fh)

    output = args.output_npz.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        components=np.asarray(results["components"], dtype=float),
        explained_variance_ratio=np.asarray(
            results["explained_variance_ratio"], dtype=float),
        mean_peak=np.asarray(results["mean_peak"], dtype=float),
        mean_peak_zeroed=np.asarray(results["mean_peak_zeroed"], dtype=float),
        score_mean=np.asarray(results["score_stats"]["mean"], dtype=float),
        score_std=np.asarray(results["score_stats"]["std"], dtype=float),
        score_p05=np.asarray(
            results["score_stats"]["percentiles"][5], dtype=float),
        score_p95=np.asarray(
            results["score_stats"]["percentiles"][95], dtype=float),
    )
    width = results["width_stats"]
    half_window = args.window_multiplier * float(width["smallest_mode_mean"])
    decompositions = []
    for index, record in enumerate(results.get("decompositions", [])):
        decompositions.append({
            "pc": index + 1,
            "physical_interpretation": record.get("physical_interpretation"),
            "d_sigma": float(record.get("d_sigma", 0.0)),
            "d_gamma": float(record.get("d_gamma", 0.0)),
            "d_tau": float(record.get("d_tau", 0.0)),
            "gaussian_fraction": float(record.get("gaussian_fraction", 0.0)),
            "lorentzian_fraction": float(record.get("lorentzian_fraction", 0.0)),
            "asymmetry_fraction": float(record.get("asymmetry_fraction", 0.0)),
        })
    manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "source": {
            "path": str(source),
            "sha256": hash_file(source),
            "source_notebook": args.source_notebook,
        },
        "training": {
            "spectra": len(results["csv_files"]),
            "peak_windows": int(results["score_stats"]["n_samples"]),
            "window_multiplier": args.window_multiplier,
            "smallest_mode_fwhm_nm": float(width["smallest_mode_mean"]),
            "half_window_nm": half_window,
            "n_window_points": int(np.asarray(results["components"]).shape[1]),
            "normalization": (
                "linear endpoint baseline; subtract minimum; divide by range; "
                "linear interpolation to fixed grid"
            ),
        },
        "basis": {
            "npz": output.name,
            "sha256": hash_file(output),
            "components": int(np.asarray(results["components"]).shape[0]),
            "cumulative_explained_variance": float(np.sum(
                results["explained_variance_ratio"])),
        },
        "legacy_physical_decomposition": decompositions,
        "notes": [
            "The basis is global across UV/VIS/NIR peak windows.",
            "The 0.1557 nm training half-window, not the later notebook example's "
            "0.0841 nm segment-derived half-window, is required for projection.",
            "Legacy component labels are diagnostics, not unique causal assignments; "
            "the analysis also projects onto explicit shift/broadening/flattening/"
            "splitting templates.",
        ],
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output} and {output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
