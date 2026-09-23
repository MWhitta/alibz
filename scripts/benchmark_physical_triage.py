#!/usr/bin/env python3
"""Benchmark physical candidate triage on pinned real LIBS spectra.

The benchmark scores only independently supported positive elements. Unknown
elements are never converted into false positives. Candidate counts compare the
existing candidate builder with its opt-in physical-triage prune mode using the
same blindly extracted peak table and wavelength correction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, peak_widths

os.environ.setdefault("MPLCONFIGDIR", "/tmp/alibz-mpl")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from alibz.peaky_finder import PeakyFinder
from alibz.peaky_indexer_v3 import PeakyIndexerV3
from alibz.triage import TriageConfig, triage_candidates
from alibz.utils.database import Database
from alibz.utils.sahaboltzmann import SahaBoltzmann
from alibz.utils.wavelength import estimate_wavelength_shift_segments, shift_at


DEFAULT_MANIFEST = "provenance/physical-triage-real-data-20260922.json"
DEFAULT_ARCHIVE = "provenance/physical-triage-real-data-20260922.npz"
SEGMENT_EDGES = (365.0, 620.0)
NULL_TRANSLATIONS_NM = (1.37, -1.91, 2.73)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_manifest(path: Path, verify_hashes: bool = True) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != "physical-triage-real-data-v1":
        raise ValueError("unsupported or missing real-data manifest schema")
    if not manifest.get("datasets"):
        raise ValueError("manifest has no datasets")
    for dataset in manifest["datasets"]:
        expected = dataset.get("truth", {}).get("expected_positive_elements", [])
        if not expected:
            raise ValueError(f"{dataset.get('id')}: no supported positive label")
        for sample in dataset.get("samples", []):
            p = Path(sample["path"])
            if verify_hashes:
                if not p.is_file():
                    raise FileNotFoundError(p)
                if sha256(p) != sample["sha256"]:
                    raise ValueError(f"hash mismatch: {p}")
    return manifest


def _ledger_path(dataset: dict) -> Path | None:
    for evidence in dataset.get("evidence", []):
        if "Acquisition ledger" in evidence.get("role", ""):
            return Path(evidence["path"])
    return None


def expand_samples(dataset: dict) -> list[dict]:
    """Expand the pinned split to all ledger means as technical replicates."""
    selected = {s["run_id"]: dict(s) for s in dataset.get("samples", [])}
    ledger_path = _ledger_path(dataset)
    if ledger_path is None or not ledger_path.is_file():
        return list(selected.values())
    ledger = sorted(json.loads(ledger_path.read_text()), key=lambda x: x["created_at"])
    acq = ledger_path.parent / "acq"
    expanded = []
    for i, row in enumerate(ledger):
        run_id = row["run"]
        sample = selected.get(run_id, {})
        p = acq / run_id / "average.csv"
        item = {
            "id": sample.get("id", f"fe-ledger-{i:02d}"),
            "split": sample.get("split", "nuisance"),
            "path": str(p),
            "run_id": run_id,
            "ledger_index": i,
            "shots_averaged": row["shots"],
            "location_xyz": row.get("location"),
            "delay": row.get("delay"),
            "period": row.get("period"),
            "pulse_period": row.get("pulsePeriod"),
        }
        item.update({k: v for k, v in sample.items() if k not in item})
        if not p.is_file():
            raise FileNotFoundError(p)
        actual = sha256(p)
        if sample.get("sha256") and sample["sha256"] != actual:
            raise ValueError(f"hash mismatch: {p}")
        item["sha256"] = actual
        expanded.append(item)
    return expanded


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    if a.ndim != 2 or a.shape[1] < 2 or len(a) < 3:
        raise ValueError(f"not a two-column spectrum: {path}")
    x, y = np.asarray(a[:, 0], float), np.asarray(a[:, 1], float)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(np.diff(x) <= 0):
        raise ValueError(f"invalid spectrum values or grid: {path}")
    return x, y


def load_replay_archive(path: Path, expected_sha256: str | None = None):
    actual = sha256(path)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(f"archive hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        wavelengths = np.asarray(z["wavelength_nm"], dtype=float)
        intensities = np.asarray(z["intensity_counts"], dtype=float)
        run_ids = np.asarray(z["run_ids"]).astype(str)
        metadata = json.loads(str(np.asarray(z["metadata_json"]).item()))
    if wavelengths.shape != intensities.shape or wavelengths.ndim != 2:
        raise ValueError("archive wavelength/intensity arrays must be matching 2-D arrays")
    if len(run_ids) != len(wavelengths) or len(metadata) != len(wavelengths):
        raise ValueError("archive metadata length mismatch")
    records = []
    for i, meta in enumerate(metadata):
        if meta.get("run_id") != run_ids[i]:
            raise ValueError("archive run-id order mismatch")
        records.append((dict(meta), wavelengths[i], intensities[i]))
    return records, actual


def _analysis_intervals(dataset: dict) -> list[tuple[float, float]]:
    # The dense 960.18-961 nm native-export tail is explicitly a data-quality
    # exclusion in the manifest, not assumed valid detector coverage.
    intervals = dataset.get("grid", {}).get("coverage_intervals_nm", [])
    return [tuple(map(float, intervals[0]))] if intervals else []


def crop_to_coverage(x: np.ndarray, y: np.ndarray,
                     coverage: list[tuple[float, float]]):
    use = np.zeros(len(x), dtype=bool)
    for lo, hi in coverage:
        use |= (x >= lo) & (x <= hi)
    return x[use], y[use]


def quick_peaks(x: np.ndarray, y: np.ndarray,
                coverage: list[tuple[float, float]], sigma_cut: float = 4.0):
    rows, sigmas = [], []
    noise_x, noise_area = [], []
    boundaries = sorted({v for pair in coverage for v in pair} | set(SEGMENT_EDGES))
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        if not any(lo < b and hi > a for a, b in coverage):
            continue
        use = (x >= lo) & (x <= hi)
        idx = np.flatnonzero(use)
        if idx.size < 15:
            continue
        xx, yy = x[idx], y[idx]
        pitch = float(np.median(np.diff(xx)))
        width = max(11, int(round(2.0 / pitch)) | 1)
        width = min(width, (len(yy) // 2) * 2 - 1)
        baseline = median_filter(yy, size=width, mode="nearest")
        residual = yy - baseline
        noise_width = max(11, int(round(1.0 / pitch)) | 1)
        med = median_filter(residual, size=noise_width, mode="nearest")
        mad = median_filter(np.abs(residual - med), size=noise_width, mode="nearest")
        point_noise = np.maximum(1.4826 * mad, 1.0)
        distance = max(1, int(round(0.10 / pitch)))
        found, props = find_peaks(
            residual, height=sigma_cut * point_noise, prominence=3.0 * point_noise,
            distance=distance)
        if not len(found):
            continue
        widths, _, left, right = peak_widths(residual, found, rel_height=0.5)
        for j, p in enumerate(found):
            half = max(2, int(np.ceil(max(widths[j], 2.0))))
            a, b = max(0, p - half), min(len(xx), p + half + 1)
            positive = np.maximum(residual[a:b], 0.0)
            area = float(np.trapezoid(positive, xx[a:b]))
            if not np.isfinite(area) or area <= 0:
                continue
            fwhm = max(pitch, float((right[j] - left[j]) * pitch))
            area_sigma = float(point_noise[p] * np.sqrt(b - a) * pitch)
            rows.append((area, float(xx[p]), fwhm / 2.35482, pitch / 2.0))
            sigmas.append(max(area_sigma, np.finfo(float).eps))
        noise_x.extend(xx.tolist())
        noise_area.extend((point_noise * np.sqrt(max(3, int(round(0.2 / pitch)))) * pitch).tolist())
    peaks = np.asarray(rows, dtype=float).reshape((-1, 4))
    order = np.argsort(peaks[:, 0])[::-1] if len(peaks) else np.empty(0, int)
    return peaks[order], np.asarray(sigmas)[order], (np.asarray(noise_x), np.asarray(noise_area))


def production_peaks(x: np.ndarray, y: np.ndarray):
    finder = PeakyFinder.__new__(PeakyFinder)
    fit = finder.fit_spectrum(x, y, subtract_background=True, plot=False,
                              n_sigma=0)
    peaks = np.asarray(fit["sorted_parameter_array"], dtype=float).reshape((-1, 4))
    amp_sigma = np.maximum(np.sqrt(np.maximum(peaks[:, 0], 0.0)), 1.0)
    return peaks, amp_sigma, None


def blind_shift(peaks_obs: np.ndarray, db: Database):
    shift, n = estimate_wavelength_shift_segments(peaks_obs, db)
    peaks_db = peaks_obs.copy()
    if len(peaks_db):
        peaks_db[:, 1] -= np.asarray(shift_at(shift, peaks_db[:, 1]), dtype=float)
    return shift, int(n), peaks_db


def shift_coverage(coverage, shift):
    return [(float(lo - shift_at(shift, lo)), float(hi - shift_at(shift, hi)))
            for lo, hi in coverage]


def shift_noise_model(noise, shift):
    if noise is None:
        return None
    x, sigma = noise
    return (np.asarray(x) - np.asarray(shift_at(shift, x)), np.asarray(sigma))


def candidate_build(peaks: np.ndarray, amp_sigma: np.ndarray, db, sb,
                    mode: str, coverage, protected=()):
    start = time.perf_counter()
    idx = PeakyIndexerV3(peaks, db=db, sb=sb, amp_sigma=amp_sigma)
    idx.build_candidate_matrix(physical_triage=mode,
                               triage_coverage=coverage,
                               triage_protected_elements=protected)
    elapsed = time.perf_counter() - start
    elements = sorted({sp.element for sp in idx.line_table.species})
    return {
        "mode": mode,
        "runtime_s": elapsed,
        "species": idx.line_table.n_species,
        "lines": idx.line_table.n_lines,
        "elements": elements,
        "triage_integration": getattr(idx, "_triage_report", None),
    }


def _translated_peaks(peaks: np.ndarray, delta: float,
                      coverage: list[tuple[float, float]]) -> np.ndarray:
    out = peaks.copy()
    if not len(out):
        return out
    edges = [coverage[0][0], *SEGMENT_EDGES, coverage[0][1]]
    for lo, hi in zip(edges[:-1], edges[1:]):
        use = (out[:, 1] >= lo) & (out[:, 1] <= hi)
        width = hi - lo
        out[use, 1] = lo + np.mod(out[use, 1] + delta - lo, width)
    return out


def element_diagnostic(result, element: str) -> dict:
    evidence = result.element_evidence.get(element)
    ranked = sorted(result.element_evidence.values(), key=lambda e: e.score,
                    reverse=True)
    rank = next((i + 1 for i, e in enumerate(ranked) if e.element == element), None)
    if evidence is None:
        return {"present": False, "score_rank": rank}
    return {
        "present": True,
        "decision": evidence.decision,
        "score": evidence.score,
        "score_rank": rank,
        "matched_feature_count": evidence.matched_feature_count,
        "matched_stage_count": evidence.matched_stage_count,
        "weak_wavelength_match_count": evidence.weak_wavelength_match_count,
        "reasons": list(evidence.reasons),
        "deferred": element in result.deferred_elements,
        "proposed_removal": element in result.rejected_elements,
    }


def k_area_counterexample(manifest: dict) -> dict | None:
    target = None
    for item in manifest.get("exploratory_unlabelled", []):
        for p in item.get("paths", []):
            if p.endswith("k_wide_line_profiles.csv"):
                target = Path(p)
    if target is None or not target.is_file():
        return None
    pairs = {}
    with target.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["line_id"] not in {"K_I_766.4899", "K_I_769.8964"}:
                continue
            area = float(row.get("area") or 0.0)
            snr = float(row.get("snr") or 0.0)
            if area > 0 and snr >= 5:
                pairs.setdefault(row["test_id"], {})[row["line_id"]] = area
    ratio = np.asarray([v["K_I_766.4899"] / v["K_I_769.8964"]
                        for v in pairs.values() if len(v) == 2])
    thin = 2.006045
    outside = (ratio < 0.8 * thin) | (ratio > 1.2 * thin)
    return {"path": str(target), "sha256": sha256(target), "n": len(ratio),
            "area_ratio_quantiles_5_50_95": np.quantile(ratio, [.05, .5, .95]),
            "thin_ratio": thin, "outside_thin_plus_minus_20pct": int(outside.sum())}


def archive_means(samples: list[dict], out: Path) -> dict:
    wavelengths, intensities, metadata = [], [], []
    for sample in samples:
        x, y = load_csv(Path(sample["path"]))
        wavelengths.append(x)
        intensities.append(y)
        metadata.append({k: sample.get(k) for k in
                         ("id", "split", "run_id", "ledger_index", "shots_averaged",
                          "location_xyz", "delay", "period", "pulse_period", "sha256")})
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "wavelength_nm": np.asarray(wavelengths),
        "intensity_counts": np.asarray(intensities),
        "run_ids": np.asarray([s["run_id"] for s in samples]),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    # numpy.savez_compressed embeds current ZIP timestamps. Write the same NPY
    # members with fixed metadata so the archive hash is reproducible.
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as zf:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED,
                        compresslevel=9)
    return {"path": str(out), "sha256": sha256(out), "run_means": len(samples),
            "method": "deterministic compressed NPZ of each hash-pinned average.csv native axis and intensity"}


def run_benchmark(manifest: dict, *, limit: int | None, peak_method: str,
                  indexer_cases: int, archive_path: Path | None,
                  replay_archive: Path | None = None) -> dict:
    db, sb = Database("db"), SahaBoltzmann("db")
    results, all_samples, spectra_by_run = [], [], {}
    replay_records, replay_sha = (None, None)
    if replay_archive is not None:
        expected_archive_sha = manifest["datasets"][0].get(
            "repo_local_archive", {}).get("sha256")
        replay_records, replay_sha = load_replay_archive(
            replay_archive, expected_archive_sha)
    gas_fn = None
    try:
        from alibz.gas_detection import detect_background_gases
        gas_fn = detect_background_gases
    except (ImportError, AttributeError):
        pass
    for dataset in manifest["datasets"]:
        if replay_records is None:
            samples = expand_samples(dataset)
            spectra = [(s, *load_csv(Path(s["path"]))) for s in samples]
        else:
            selected = {s["run_id"]: s for s in dataset.get("samples", [])}
            spectra = []
            for archived, x, y in replay_records:
                sample = dict(archived)
                sample.update({k: v for k, v in selected.get(
                    archived["run_id"], {}).items() if k != "path"})
                sample["archive_source"] = str(replay_archive)
                spectra.append((sample, x, y))
            samples = [s for s, _, _ in spectra]
        all_samples.extend(samples)
        coverage = _analysis_intervals(dataset)
        expected = set(dataset["truth"]["expected_positive_elements"])
        for sample, x_raw, y_raw in spectra[:limit]:
            spectra_by_run[sample["run_id"]] = (x_raw, y_raw)
            x, y = crop_to_coverage(x_raw, y_raw, coverage)
            t0 = time.perf_counter()
            if peak_method == "quick":
                peaks_obs, amp_sigma, noise = quick_peaks(x, y, coverage)
            else:
                peaks_obs, amp_sigma, noise = production_peaks(x, y)
            extract_s = time.perf_counter() - t0
            shift, n_anchors, peaks_db = blind_shift(peaks_obs, db)
            coverage_db = shift_coverage(coverage, shift)
            noise_db = shift_noise_model(noise, shift)
            cfg = TriageConfig()
            gas = gas_fn(x, y, db, shift_nm=shift, peak_array=peaks_db) if gas_fn else None
            protected = tuple(sorted(
                el for el, evidence in (gas or {}).items()
                if isinstance(evidence, dict) and evidence.get("status") == "detected"))
            t1 = time.perf_counter()
            triage = triage_candidates(peaks_db, db, config=cfg,
                                       wavelength_range=coverage_db[0],
                                       coverage_intervals=coverage_db,
                                       peak_sigma=amp_sigma,
                                       local_noise=noise_db,
                                       protected_elements=protected)
            triage_s = time.perf_counter() - t1
            off = candidate_build(peaks_db, amp_sigma, db, sb, "off", coverage_db)
            prune = candidate_build(peaks_db, amp_sigma, db, sb, "prune", coverage_db,
                                    protected=protected)
            nulls = []
            for delta in NULL_TRANSLATIONS_NM:
                null = triage_candidates(_translated_peaks(peaks_db, delta, coverage_db),
                                         db, config=cfg,
                                         wavelength_range=coverage_db[0],
                                         coverage_intervals=coverage_db,
                                         peak_sigma=amp_sigma, local_noise=noise_db,
                                         protected_elements=protected)
                nulls.append({"translation_nm": delta,
                              "kept_elements": list(null.keep_elements),
                              "rejected_elements": list(null.rejected_elements),
                              "deferred_elements": list(null.deferred_elements),
                              "expected_element_diagnostics": {
                                  el: element_diagnostic(null, el) for el in expected}})
            item = {
                "dataset_id": dataset["id"], "sample": sample,
                "expected_positive_elements": sorted(expected),
                "peak_method": peak_method, "n_peaks": len(peaks_db),
                "peak_extraction_runtime_s": extract_s,
                "wavelength_shift": {"repr": repr(shift),
                                     "global_nm": float(shift),
                                     "anchor_matches": n_anchors},
                "triage_runtime_s": triage_s, "triage": triage.to_dict(),
                "true_positive_diagnostics": {
                    el: {
                        "triage": element_diagnostic(triage, el),
                        "candidate_off_retained": el in off["elements"],
                        "candidate_prune_retained": el in prune["elements"],
                    } for el in expected},
                "peak_uncertainty": ("local integrated-area noise estimate"
                                     if peak_method == "quick" else
                                     "sqrt(area) proxy; missing-line rejection disabled"),
                "candidate_build": {"off": off, "prune": prune},
                "null_translations": nulls, "background_gases": gas,
                "gas_protected_elements": list(protected),
            }
            results.append(item)
    for item in results[:max(0, indexer_cases)]:
        peaks = np.asarray(item["triage"].get("input_peaks", []), dtype=float)
        # Triage does not promise to echo peaks; reconstruct the same table.
        sample = item["sample"]
        x, y = spectra_by_run[sample["run_id"]]
        dataset = next(d for d in manifest["datasets"] if d["id"] == item["dataset_id"])
        coverage = _analysis_intervals(dataset)
        x, y = crop_to_coverage(x, y, coverage)
        extracted = quick_peaks(x, y, coverage) if peak_method == "quick" else production_peaks(x, y)
        shift, _, peaks = blind_shift(extracted[0], db)
        coverage_db = shift_coverage(coverage, shift)
        amp_sigma = extracted[1]
        gas = gas_fn(x, y, db, shift_nm=shift, peak_array=peaks) if gas_fn else None
        protected = tuple(sorted(
            el for el, evidence in (gas or {}).items()
            if isinstance(evidence, dict) and evidence.get("status") == "detected"))
        item["full_indexer"] = {}
        for mode in ("off", "prune"):
            start = time.perf_counter()
            idx = PeakyIndexerV3(peaks, db=db, sb=sb, amp_sigma=amp_sigma)
            fit = idx.run(n_calls=10, verbose=False, physical_triage=mode,
                          triage_coverage=coverage_db,
                          triage_protected_elements=protected)
            item["full_indexer"][mode] = {
                "runtime_s": time.perf_counter() - start,
                "r_squared": fit.r_squared,
                "element_fractions": fit.element_fractions,
            }
    if replay_archive is not None:
        archive = {"path": str(replay_archive), "sha256": replay_sha,
                   "run_means": len(replay_records), "role": "benchmark input replay archive"}
    else:
        archive = archive_means(all_samples, archive_path) if archive_path else None
    expected_total = sum(len(x["true_positive_diagnostics"]) for x in results)
    retained = sum(sum(v["candidate_prune_retained"]
                       for v in x["true_positive_diagnostics"].values())
                   for x in results)
    material_groups = len({x["dataset_id"] for x in results})
    reduced = any(x["candidate_build"]["prune"]["species"] <
                  x["candidate_build"]["off"]["species"] for x in results)
    speedups = [x["candidate_build"]["off"]["runtime_s"] /
                max(x["candidate_build"]["prune"]["runtime_s"], 1e-12)
                for x in results]
    median_speedup = float(np.median(speedups)) if speedups else None
    no_label_loss = bool(expected_total and retained == expected_total)
    diverse = material_groups >= 3
    speed_benefit = bool(median_speedup is not None and median_speedup > 1.0)
    gate = {
        "expected_positive_retention": retained / expected_total if expected_total else None,
        "no_labelled_positive_loss": no_label_loss,
        "candidate_reduction_observed": reduced,
        "median_candidate_build_speedup": median_speedup,
        "median_speed_benefit_observed": speed_benefit,
        "independently_labelled_material_groups": material_groups,
        "diverse_matrix_requirement_met": diverse,
        "broad_validation_status": ("pass" if
            no_label_loss and reduced and speed_benefit and diverse else "fail"),
        "requirements": ["retain every supported positive",
                         "reduce candidates beyond existing gates",
                         "median candidate-build speedup above 1x",
                         "at least three independently labelled material matrices"]
    }
    return {"schema_version": "physical-triage-benchmark-v1",
            "manifest_schema": manifest["schema_version"],
            "configuration": {"peak_method": peak_method, "limit": limit,
                              "indexer_cases": indexer_cases,
                              "triage": asdict(TriageConfig()),
                              "null_translations_nm": NULL_TRANSLATIONS_NM},
            "archive": archive, "gate": gate,
            "k_area_counterexample": k_area_counterexample(manifest),
            "samples": results}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--out", default="reports/physical-triage-real-benchmark.json")
    p.add_argument("--limit", type=int)
    p.add_argument("--indexer-cases", type=int, default=0)
    p.add_argument("--peak-method", choices=("quick", "production"), default="quick")
    p.add_argument("--archive-inputs", default=DEFAULT_ARCHIVE)
    p.add_argument("--replay-archive",
                   help="Use a hash-verified NPZ archive when raw scratch CSVs are unavailable")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path, verify_hashes=not bool(args.replay_archive))
    if args.replay_archive:
        expected_sha = manifest["datasets"][0].get("repo_local_archive", {}).get("sha256")
        replay, _ = load_replay_archive(Path(args.replay_archive), expected_sha)
        expanded_count = len(replay)
    else:
        expanded_count = len(sum((expand_samples(d) for d in manifest["datasets"]), []))
    if args.dry_run:
        print(json.dumps({"status": "ok", "manifest": str(manifest_path),
                          "datasets": len(manifest["datasets"]),
                          "selected_samples": sum(len(d["samples"]) for d in manifest["datasets"]),
                          "expanded_run_means": expanded_count,
                          "replay_archive": args.replay_archive,
                          "broad_validation_status": manifest["gate"]["broad_validation_status"]},
                         indent=2))
        return 0
    result = run_benchmark(manifest, limit=args.limit,
                           peak_method=args.peak_method,
                           indexer_cases=max(0, args.indexer_cases),
                           archive_path=Path(args.archive_inputs) if args.archive_inputs else None,
                           replay_archive=Path(args.replay_archive) if args.replay_archive else None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "ok", "out": str(out),
                      "samples": len(result["samples"]), "gate": result["gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
