"""Rebuild quantitative Se/Th/U line tables from public atomic databases.

The legacy bundle contains only hydrogenic (very-high-ionization) records for
Se, Th and U.  Those records are irrelevant to ordinary LIBS plasmas and make
the Saha stage chain non-contiguous.  This importer replaces those three
element arrays with neutral/singly-ionized Kurucz records, then cross-matches
NIST ASD for preferred Ritz wavelengths, classified levels and transition
probabilities where ASD provides them.

The source hashes below pin the exact snapshots retrieved on 2026-07-13.
Run from the repository root::

    python scripts/update_heavy_atomic_lines.py --source-dir /path/to/cache

Missing files are downloaded into ``--source-dir``.  The database pickle is
written through a temporary file and atomically replaced only after all
records pass validation.  A JSON manifest records provenance and line counts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import stat
import tempfile
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

from alibz.utils.wavelength import air_to_vacuum


CM1_PER_EV = 8065.544005
GF_TO_GA_DENOM = 1.4991938e-16  # gf = denom * lambda_A^2 * g_u A_ul
TARGETS = {"Se": 34, "Th": 90, "U": 92}
UNSUPPORTED = ["Pm", "Po", "At", "Rn", "Pa"]


def _nist_url(spectrum: str) -> str:
    query = {
        "spectra": spectrum,
        "limits_type": "0",
        "low_w": "180",
        "upp_w": "962",
        "unit": "1",
        "submit": "Retrieve Data",
        "de": "0",
        "format": "3",
        "line_out": "0",
        "remove_js": "on",
        "en_unit": "1",
        "output": "0",
        "bibrefs": "1",
        "page_size": "30000",
        "show_obs_wl": "1",
        "show_calc_wl": "1",
        "order_out": "0",
        "show_av": "2",
        "A_out": "1",
        "intens_out": "1",
        "allowed_out": "1",
        "forbid_out": "1",
        "conf_out": "1",
        "term_out": "1",
        "enrg_out": "1",
        "J_out": "1",
        "g_out": "1",
    }
    return "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?" + \
        urllib.parse.urlencode(query)


SOURCES = {
    "kurucz_gf3400_all": {
        "url": "http://kurucz.harvard.edu/linelists/gfall/gf3400.all",
        "sha256": "b676f01b3d701718ea53d2b90281ece84e8c20275f5424d34c632549bcdcc7d8",
    },
    "kurucz_gf9000_all": {
        "url": "http://kurucz.harvard.edu/linelists/gfall/gf9000.all",
        "sha256": "ee7a52d5ffacfff999fa209e65995b4cb508c70bf03bb80ac6ee71276cf6e045",
    },
    "kurucz_gf9001_all": {
        "url": "http://kurucz.harvard.edu/linelists/gfall/gf9001.all",
        "sha256": "a3a1caa9757ecb317185d9977b919e3ca5225972b4968e13b63922f7232974fa",
    },
    "kurucz_gf9200_all": {
        "url": "http://kurucz.harvard.edu/linelists/gfall/gf9200.all",
        "sha256": "ad0fbe0706bc958f715b17d98bbb43bc2b07a725a92f14ca5e029e2749144a8b",
    },
    "kurucz_gf9201_all": {
        "url": "http://kurucz.harvard.edu/linelists/gfall/gf9201.all",
        "sha256": "fb6c71f56270f4ad60b70dbbe4e1e7496dfc8c322a1d6c410fbfc3bbcfb293be",
    },
    "nist_se_i_all.tsv": {
        "url": _nist_url("Se I"),
        "sha256": "cdc3100a0b35d3cf2cb628220167f172f69dabdcc1565c2de23bcd38bfb20397",
    },
    "nist_se_ii_all.tsv": {
        "url": _nist_url("Se II"),
        "sha256": "946b84d8f64f3189110a34c4821ec842170886da4615b2e24d9991777566fb1b",
    },
    "nist_th_i_all.tsv": {
        "url": _nist_url("Th I"),
        "sha256": "1003d72b8d1e17990c633be59aedff3a48d85ae0f7ae90fcfba407dd1f4d1f40",
    },
    "nist_th_ii_all.tsv": {
        "url": _nist_url("Th II"),
        "sha256": "e9358f76dad5a454747152e04912eb8f1a5bdb46509094b5088198f3dd050eed",
    },
    "nist_u_i_all.tsv": {
        "url": _nist_url("U I"),
        "sha256": "e557c7eddfdcbc1c64c6a8b42056f8392a2e3eb0c8d01b2f9add414e5230d8d0",
    },
    "nist_u_ii_all.tsv": {
        "url": _nist_url("U II"),
        "sha256": "06f309db644c2f7a6d17cbf05249bf0b72847c14d153e905b058ec209be393da",
    },
}

KURUCZ_FILES = {
    "Se": ["kurucz_gf3400_all"],
    "Th": ["kurucz_gf9000_all", "kurucz_gf9001_all"],
    "U": ["kurucz_gf9200_all", "kurucz_gf9201_all"],
}
NIST_FILES = {
    "Se": [(1, "nist_se_i_all.tsv"), (2, "nist_se_ii_all.tsv")],
    "Th": [(1, "nist_th_i_all.tsv"), (2, "nist_th_ii_all.tsv")],
    "U": [(1, "nist_u_i_all.tsv"), (2, "nist_u_ii_all.tsv")],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sources(source_dir: Path) -> dict[str, Path]:
    source_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, spec in SOURCES.items():
        path = source_dir / name
        if not path.exists():
            print(f"downloading {spec['url']}")
            with urllib.request.urlopen(spec["url"], timeout=120) as response:
                path.write_bytes(response.read())
        got = _sha256(path)
        if got != spec["sha256"]:
            raise RuntimeError(
                f"source hash mismatch for {name}: expected {spec['sha256']}, got {got}"
            )
        paths[name] = path
    return paths


def _j_text(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:g}"


def _parse_j(value: str):
    value = value.strip()
    if not value:
        return None
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            return float(num) / float(den)
        return float(value)
    except ValueError:
        return None


def _clean(value: str) -> str:
    value = value.strip().strip('"').strip()
    if value.startswith("="):
        value = value[1:].strip('"')
    return value.strip()


def _number(value: str):
    value = _clean(value).replace("[", "").replace("]", "")
    value = value.replace("(", "").replace(")", "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_kurucz(path: Path, atomic_number: int):
    """Parse Kurucz GFALL fixed-width records into normalized dictionaries."""
    rows = []
    for line_number, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if len(raw) < 80:
            continue
        try:
            reported_nm = float(raw[0:11])
            loggf = float(raw[11:18])
            species = float(raw[18:24])
            lower_cm1 = abs(float(raw[24:36]))
            j_lower = float(raw[36:41])
            lower_label = raw[41:52].strip()
            upper_cm1 = abs(float(raw[52:64]))
            j_upper = float(raw[64:69])
            upper_label = raw[69:80].strip()
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: malformed GFALL row") from exc

        z = int(species)
        stage = int(round(100.0 * (species - z))) + 1
        if z != atomic_number or stage not in (1, 2):
            raise ValueError(
                f"{path}:{line_number}: unexpected species code {species}"
            )
        if reported_nm <= 0 or upper_cm1 <= lower_cm1:
            continue

        wavelength_a = 10.0 * reported_nm
        gA = 10.0 ** loggf / (GF_TO_GA_DENOM * wavelength_a ** 2)
        rows.append({
            "stage": stage,
            "reported_nm": reported_nm,
            "vacuum_nm": float(air_to_vacuum(reported_nm)),
            "gA": gA,
            "Ei": lower_cm1 / CM1_PER_EV,
            "Ek": upper_cm1 / CM1_PER_EV,
            "conf_i": lower_label,
            "term_i": "",
            "J_i": j_lower,
            "conf_k": upper_label,
            "term_k": "",
            "J_k": j_upper,
            "g_i": int(round(2.0 * j_lower + 1.0)),
            "g_k": int(round(2.0 * j_upper + 1.0)),
            "kurucz_ref": raw[98:102].strip() if len(raw) >= 102 else "",
            "nist_match": False,
            "nist_ga": False,
        })
    return rows


def parse_nist(path: Path, stage: int):
    """Parse ASD tab output, including the vacuum/air header transition."""
    rows = []
    header = None
    for raw in path.read_text(errors="replace").splitlines():
        if raw.startswith("obs_wl_"):
            header = next(csv.reader([raw], delimiter="\t"))
            continue
        if header is None or not raw.startswith('"'):
            continue
        values = next(csv.reader([raw], delimiter="\t"))
        record = dict(zip(header, values))

        candidates = []
        for prefix in ("ritz_wl_", "obs_wl_"):
            for key, value in record.items():
                if key.startswith(prefix):
                    number = _number(value)
                    if number is not None:
                        candidates.append((prefix.startswith("ritz"), key, number))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        is_ritz, key, reported_nm = candidates[0]
        is_air = "_air(" in key

        rows.append({
            "stage": stage,
            "reported_nm": reported_nm,
            "vacuum_nm": float(air_to_vacuum(reported_nm)) if is_air else reported_nm,
            "ritz": is_ritz,
            "wavelength_kind": ("ritz" if is_ritz else "observed")
            + ("_air" if is_air else "_vacuum"),
            "intensity": _clean(record.get("intens", "")),
            "gA": _number(record.get("gA(s^-1)", "")),
            "Ei": _number(record.get("Ei(eV)", "")),
            "Ek": _number(record.get("Ek(eV)", "")),
            "conf_i": _clean(record.get("conf_i", "")),
            "term_i": _clean(record.get("term_i", "")),
            "J_i": _parse_j(_clean(record.get("J_i", ""))),
            "conf_k": _clean(record.get("conf_k", "")),
            "term_k": _clean(record.get("term_k", "")),
            "J_k": _parse_j(_clean(record.get("J_k", ""))),
            "g_i": _number(record.get("g_i", "")),
            "g_k": _number(record.get("g_k", "")),
            "tp_ref": _clean(record.get("tp_ref", "")),
            "line_ref": _clean(record.get("line_ref", "")),
        })
    return rows


def _match_nist(kurucz_rows, nist_rows, tolerance_nm=0.006):
    """Cross-match one Kurucz record at a time to its best ASD line."""
    by_stage = {}
    for stage in (1, 2):
        candidates = [row for row in nist_rows if row["stage"] == stage]
        candidates.sort(key=lambda row: row["reported_nm"])
        by_stage[stage] = candidates

    n_matches = 0
    n_ritz = 0
    n_ga = 0
    for row in kurucz_rows:
        candidates = by_stage[row["stage"]]
        if not candidates:
            continue
        wavelengths = np.fromiter((c["reported_nm"] for c in candidates), float)
        lo = int(np.searchsorted(wavelengths, row["reported_nm"] - tolerance_nm))
        hi = int(np.searchsorted(wavelengths, row["reported_nm"] + tolerance_nm, side="right"))
        if lo == hi:
            continue

        best = None
        best_score = math.inf
        for candidate in candidates[lo:hi]:
            dw = abs(candidate["reported_nm"] - row["reported_nm"])
            score = dw / tolerance_nm
            if candidate["Ei"] is not None and candidate["Ek"] is not None:
                de = abs(candidate["Ei"] - row["Ei"]) + \
                    abs(candidate["Ek"] - row["Ek"])
                if de > 0.10:
                    continue
                score += de / 0.02
            if score < best_score:
                best, best_score = candidate, score
        if best is None:
            continue

        row["nist_match"] = True
        n_matches += 1
        if best["ritz"]:
            row["reported_nm"] = best["reported_nm"]
            row["vacuum_nm"] = best["vacuum_nm"]
            n_ritz += 1
        if best["gA"] is not None and best["gA"] > 0:
            row["gA"] = best["gA"]
            row["nist_ga"] = True
            n_ga += 1
        for key in ("Ei", "Ek", "J_i", "J_k"):
            if best[key] is not None:
                row[key] = best[key]
        for key in ("conf_i", "term_i", "conf_k", "term_k"):
            if best[key]:
                row[key] = best[key]
        for key in ("g_i", "g_k"):
            if best[key] is not None and best[key] > 0:
                row[key] = int(round(best[key]))
    return {"matches": n_matches, "ritz_updates": n_ritz, "gA_updates": n_ga}


def _as_database_array(rows):
    output = []
    for row in sorted(rows, key=lambda item: (item["stage"], item["vacuum_nm"])):
        if not (row["gA"] > 0 and row["Ek"] > row["Ei"] >= 0):
            raise ValueError(f"invalid quantitative line: {row}")
        output.append([
            str(float(row["stage"])),
            f"{row['vacuum_nm']:.9f}",
            "0.0",
            f"{row['gA']:.12g}",
            f"{row['Ei']:.10f}",
            f"{row['Ek']:.10f}",
            row["conf_i"],
            row["term_i"],
            _j_text(row["J_i"]),
            row["conf_k"],
            row["term_k"],
            _j_text(row["J_k"]),
            str(int(row["g_i"])),
            str(int(row["g_k"])),
        ])
    return np.asarray(output, dtype="<U96")


def _write_observed_catalog(path: Path, records):
    fields = [
        "element", "ion_stage", "wavelength_vacuum_nm", "wavelength_kind",
        "intensity", "gA_s-1", "Ei_eV", "Ek_eV", "g_i", "g_k",
        "tp_ref", "line_ref", "quantitative_ready",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for record in sorted(
            records,
            key=lambda row: (TARGETS[row["element"]], row["stage"], row["vacuum_nm"]),
        ):
            ready = all(
                record[key] is not None
                for key in ("gA", "Ei", "Ek", "g_i", "g_k")
            )
            writer.writerow({
                "element": record["element"],
                "ion_stage": record["stage"],
                "wavelength_vacuum_nm": f"{record['vacuum_nm']:.9f}",
                "wavelength_kind": record["wavelength_kind"],
                "intensity": record["intensity"],
                "gA_s-1": "" if record["gA"] is None else f"{record['gA']:.12g}",
                "Ei_eV": "" if record["Ei"] is None else f"{record['Ei']:.10f}",
                "Ek_eV": "" if record["Ek"] is None else f"{record['Ek']:.10f}",
                "g_i": "" if record["g_i"] is None else int(round(record["g_i"])),
                "g_k": "" if record["g_k"] is None else int(round(record["g_k"])),
                "tp_ref": record["tp_ref"],
                "line_ref": record["line_ref"],
                "quantitative_ready": "1" if ready else "0",
            })


def _write_quantitative_catalog(path: Path, records):
    fields = [
        "element", "ion_stage", "wavelength_vacuum_nm", "gA_s-1",
        "Ei_eV", "Ek_eV", "conf_i", "term_i", "J_i", "conf_k",
        "term_k", "J_k", "g_i", "g_k", "kurucz_ref", "nist_match",
        "nist_gA",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for record in sorted(
            records,
            key=lambda row: (TARGETS[row["element"]], row["stage"], row["vacuum_nm"]),
        ):
            writer.writerow({
                "element": record["element"],
                "ion_stage": record["stage"],
                "wavelength_vacuum_nm": f"{record['vacuum_nm']:.9f}",
                "gA_s-1": f"{record['gA']:.12g}",
                "Ei_eV": f"{record['Ei']:.10f}",
                "Ek_eV": f"{record['Ek']:.10f}",
                "conf_i": record["conf_i"],
                "term_i": record["term_i"],
                "J_i": _j_text(record["J_i"]),
                "conf_k": record["conf_k"],
                "term_k": record["term_k"],
                "J_k": _j_text(record["J_k"]),
                "g_i": int(record["g_i"]),
                "g_k": int(record["g_k"]),
                "kurucz_ref": record["kurucz_ref"],
                "nist_match": "1" if record["nist_match"] else "0",
                "nist_gA": "1" if record["nist_ga"] else "0",
            })


def rebuild(
    database_path: Path,
    source_dir: Path,
    manifest_path: Path,
    observed_path: Path,
    quantitative_path: Path,
):
    paths = _sources(source_dir)
    target_mode = (stat.S_IMODE(database_path.stat().st_mode)
                   if database_path.exists() else 0o644)
    with database_path.open("rb") as handle:
        atom_dict = pickle.load(handle)

    summary = {}
    observed_records = []
    quantitative_records = []
    for element, atomic_number in TARGETS.items():
        kurucz_rows = []
        for filename in KURUCZ_FILES[element]:
            kurucz_rows.extend(parse_kurucz(paths[filename], atomic_number))
        nist_rows = []
        for stage, filename in NIST_FILES[element]:
            nist_rows.extend(parse_nist(paths[filename], stage))
        for row in nist_rows:
            row["element"] = element
        observed_records.extend(nist_rows)
        match = _match_nist(kurucz_rows, nist_rows)
        for row in kurucz_rows:
            row["element"] = element
        quantitative_records.extend(kurucz_rows)
        array = _as_database_array(kurucz_rows)

        stages = array[:, 0].astype(float).astype(int)
        wavelengths = array[:, 1].astype(float)
        stage_set = [int(value) for value in sorted(set(stages))]
        if stage_set != list(range(1, max(stage_set) + 1)):
            raise ValueError(f"{element}: non-contiguous stages {stage_set}")
        atom_dict[element] = array
        summary[element] = {
            "quantitative_lines": int(len(array)),
            "instrument_lines_180_962_nm": int(np.sum(
                (wavelengths >= 180.0) & (wavelengths <= 962.5)
            )),
            "stages_1_based": stage_set,
            "nist_observed_or_ritz_rows": len(nist_rows),
            "nist_matches": match["matches"],
            "nist_ritz_updates": match["ritz_updates"],
            "nist_gA_updates": match["gA_updates"],
            "kurucz_reference_codes": dict(sorted(Counter(
                row["kurucz_ref"] or "unlabelled" for row in kurucz_rows
            ).items())),
        }

    if len(atom_dict) != 92:
        raise ValueError(f"expected 92 element arrays, found {len(atom_dict)}")

    _write_observed_catalog(observed_path, observed_records)
    _write_quantitative_catalog(quantitative_path, quantitative_records)

    database_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=database_path.name + ".", suffix=".tmp", dir=database_path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(atom_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.chmod(tmp_name, target_mode)
        with open(tmp_name, "rb") as handle:
            check = pickle.load(handle)
        for element in TARGETS:
            if check[element].shape != atom_dict[element].shape:
                raise RuntimeError(f"round-trip validation failed for {element}")
        os.replace(tmp_name, database_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

    manifest = {
        "schema_version": 1,
        "retrieved_utc": "2026-07-13",
        "database": "el_lines92.pickle",
        "observed_catalog": observed_path.name,
        "observed_catalog_rows": len(observed_records),
        "quantitative_catalog": quantitative_path.name,
        "quantitative_catalog_rows": len(quantitative_records),
        "element_order": "H through U (92 positions)",
        "unsupported_elements": UNSUPPORTED,
        "supported_count": 92 - len(UNSUPPORTED),
        "quantitative_policy": (
            "Forward-model rows require wavelength, gA/log(gf), lower/upper "
            "energies and level degeneracies; observed-only NIST rows are "
            "retained in the companion catalog but not assigned fabricated strengths."
        ),
        "wavelength_policy": (
            "Stored wavelengths are vacuum; Database converts >=200 nm to "
            "standard air at load time."
        ),
        "sources": {
            name: {"url": spec["url"], "sha256": spec["sha256"]}
            for name, spec in SOURCES.items()
        },
        "elements": summary,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("db/el_lines92.pickle"))
    parser.add_argument(
        "--source-dir", type=Path,
        default=Path(os.environ.get("ALIBZ_ATOMIC_SOURCE_CACHE", "/private/tmp/alibz_atomic_sources")),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("db/atomic_line_sources.json")
    )
    parser.add_argument(
        "--observed", type=Path, default=Path("db/observed_lines_nist.tsv")
    )
    parser.add_argument(
        "--quantitative", type=Path,
        default=Path("db/quantitative_lines_se_th_u.tsv"),
    )
    args = parser.parse_args()
    summary = rebuild(
        args.db, args.source_dir, args.manifest, args.observed,
        args.quantitative,
    )
    for element, stats in summary.items():
        print(element, json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
