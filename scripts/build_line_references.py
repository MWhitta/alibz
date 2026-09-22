#!/usr/bin/env python3
"""Generate Pantheum-style per-element atomic line reference JSON files.

This reproduces the recipe used to build the bundled
``pantheum/alibz/references/fe-v1.json`` file and applies it to every other
element the alibz database supports, so Pantheum's acquisition optimizer can
score arbitrary compositions instead of only Fe.

Recipe (reverse-engineered from fe-v1.json and verified to reproduce it
exactly -- see ``--check-fe``):

1.  Source data is ``db/el_lines92.pickle`` (loaded directly, *not* through
    ``alibz.utils.database.Database``, because ``Database.__init__``
    rewrites the wavelength column from vacuum to air in place and the
    selection math below needs the original Ritz VACUUM wavelengths). Each
    element's row array has columns::

        0  ion stage (1 = neutral, 2 = singly ionized, ...)
        1  vacuum wavelength [nm]
        2  legacy/unused field (ignored; "0.0" placeholder in newer imports)
        3  gA [s^-1]                       -> ``gA_s_1``
        4  lower level energy Ei [eV]      -> ``lower_energy_eV``
        5  upper level energy Ek [eV]      -> ``upper_energy_eV``
        6  lower configuration
        7  lower term
        8  lower J
        9  upper configuration             -> ``upper_level[0]``
        10 upper term                      -> ``upper_level[1]``
        11 upper J                         -> ``upper_level[2]``
        12 g_lower (statistical weight)
        13 g_upper (statistical weight)

    The row's position in this array is the numeric part of the bundled
    ``id`` (e.g. row 7069 of the Fe array -> ``"Fe-7069"``).

2.  Candidate pool: rows with ion stage 1 or 2 (neutral/singly ionized --
    the stages a LIBS plasma actually populates at useful SNR).

3.  Ranking "strength" at a reference temperature T = 10000 K::

        strength = gA * exp(-Ek / (kB * T)),   kB = 8.617333262e-5 eV/K

    (CODATA Boltzmann constant in eV/K.)  This is a plain Boltzmann-weighted
    emissivity proxy, not a plasma model.

4.  Greedy "separated" selection: walk the candidate pool in descending
    strength order, keep a line if its VACUUM wavelength is at least
    ``MIN_SEPARATION_NM`` (0.34 nm) from every already-selected line's
    vacuum wavelength, stop once ``TARGET_LINE_COUNT`` (96) lines are kept
    or the pool is exhausted. For Fe this reproduces all 96 bundled ids in
    the bundled order exactly.

5.  ``potential_blends``: for each selected line, other lines in the SAME
    ion-stage-1/2 candidate pool (any selected or not) whose vacuum
    wavelength is within ``BLEND_WINDOW_NM`` (0.2 nm) of the selected line
    AND whose strength is at least ``BLEND_RATIO_FLOOR`` (0.05, i.e. 5%) of
    the selected line's own strength. This reproduces every one of the 118
    blend entries (67 non-empty lists) in fe-v1.json exactly.

6.  Derived per-line fields:

        A_s_1               = gA_s_1 / g_upper
        oscillator_strength = gA_s_1 * vacuum_lambda_nm^2 / (6.670e13 * g_lower)
        wavelength_nm       = vacuum_to_air(vacuum_lambda_nm), rounded to 6
                               decimals the same way Database.__init__ does
        upper_key           = f"{ion_stage}|{upper_energy_eV:.8f}|{cfg}|{term}|{J}"

See ``reports/2026-09-22-line-references.md`` for the derivation, residuals
(none found for Fe) and the run that produced
``pantheum/alibz/references/*.json``.

Usage
-----

Regenerate Fe and diff it against the bundled file (exit non-zero on any
mismatch)::

    python scripts/build_line_references.py --check-fe \\
        /path/to/pantheum-I/pantheum/alibz/references/fe-v1.json

Write reference files for every supported element (skips Fe unless
``--include-fe`` is passed) with real files written::

    python scripts/build_line_references.py --out DIR --elements all

Write only a couple of elements::

    python scripts/build_line_references.py --out DIR --elements Cr,Ca,Ni

Preview counts without writing anything::

    python scripts/build_line_references.py --out DIR --elements all --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from alibz.utils.database import Database  # noqa: E402
from alibz.utils.wavelength import vacuum_to_air  # noqa: E402

# --- recipe constants (see module docstring) --------------------------------
KB_EV_PER_K = 8.617333262e-5  # CODATA Boltzmann constant, eV/K
REFERENCE_TEMPERATURE_K = 10000.0
TARGET_LINE_COUNT = 96
MIN_SEPARATION_NM = 0.34          # vacuum nm, greedy selection
BLEND_WINDOW_NM = 0.2             # vacuum nm, potential_blends search radius
BLEND_RATIO_FLOOR = 0.05          # candidate/parent strength ratio floor
OSCILLATOR_STRENGTH_DENOM = 6.670e13
MIN_CANDIDATES_TO_KEEP_ELEMENT = 3
DEFAULT_OBSERVABLE_RANGE_NM = (180.0, 961.0)  # SciAps Z300 export grid; AIR nm

ATOMIC_UNCERTAINTY_NOTE = (
    "not retained in bundled source; not calibration-grade validation"
)
WAVELENGTH_MEDIUM = "air above 200 nm (alibz Database conversion)"
OSCILLATOR_STRENGTH_CONVERSION = (
    "f_ik = gA * vacuum_lambda_nm^2 / (6.670e13 * g_i); "
    "vacuum wavelength from upper-minus-lower energy"
)
PHYSICS_SOURCE = "https://physics.nist.gov/Pubs/AtSpec/node17.html"


def _strip_quotes(value: str) -> str:
    return value.strip().strip('"')


def _load_raw_atom_dict(dbpath: Path) -> dict:
    """Load el_lines92.pickle directly (vacuum wavelengths, untouched)."""
    with open(dbpath / "el_lines92.pickle", "rb") as fh:
        return pickle.load(fh)


def _pickle_sha256(dbpath: Path) -> str:
    h = hashlib.sha256()
    with open(dbpath / "el_lines92.pickle", "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo_root: Path) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _build_line_record(element: str, idx: int, row: np.ndarray, strength: float) -> dict:
    stage = int(float(row[0]))
    vacuum_nm = float(row[1])
    gA = float(row[3])
    Ei = float(row[4])
    Ek = float(row[5])
    lower_cfg = _strip_quotes(row[6])
    lower_term = _strip_quotes(row[7])
    lower_J = _strip_quotes(row[8])
    upper_cfg = _strip_quotes(row[9])
    upper_term = _strip_quotes(row[10])
    upper_J = _strip_quotes(row[11])
    g_lower = float(row[12])
    g_upper = float(row[13])

    A_s_1 = gA / g_upper
    oscillator_strength = gA * vacuum_nm ** 2 / (OSCILLATOR_STRENGTH_DENOM * g_lower)
    air_nm = float(f"{float(vacuum_to_air(np.array([vacuum_nm]))[0]):.6f}")
    upper_key = f"{stage}|{Ek:.8f}|{upper_cfg}|{upper_term}|{upper_J}"

    return {
        "id": f"{element}-{idx}",
        "element": element,
        "ion_stage": stage,
        "wavelength_nm": air_nm,
        "gA_s_1": gA,
        "A_s_1": A_s_1,
        "oscillator_strength": oscillator_strength,
        "lower_energy_eV": Ei,
        "upper_energy_eV": Ek,
        "g_lower": g_lower,
        "g_upper": g_upper,
        "upper_level": [upper_cfg, upper_term, upper_J],
        "upper_key": upper_key,
        "strength_at_10000K": strength,
        "potential_blends": [],  # filled in by caller
        "atomic_uncertainty": ATOMIC_UNCERTAINTY_NOTE,
        "_lower_cfg": lower_cfg,
        "_lower_term": lower_term,
        "_lower_J": lower_J,
        "_vacuum_nm": vacuum_nm,
    }


def select_element_lines(element: str, raw_arr: np.ndarray,
                          range_nm=DEFAULT_OBSERVABLE_RANGE_NM):
    """Apply the selection + blend recipe to one element's raw line array.

    ``range_nm`` is an observable-wavelength gate ``(lo, hi)`` applied to
    the AIR wavelength (the same convention as the ``wavelength_nm`` field)
    BEFORE the greedy selection; it restricts both the line-selection pool
    and the potential-blend pool, so lines outside the instrument's export
    grid (e.g. vacuum-UV lines a SciAps Z300 180-961 nm grid can never see)
    cannot consume a candidate window or a blend slot.

    Returns ``(records, n_candidates)`` where ``records`` is the ordered
    list of selected line dicts (ascending row-index / wavelength order,
    matching fe-v1.json's order) with ``potential_blends`` populated, and
    ``n_candidates`` is the size of the ion-stage-1/2, in-range pool
    considered.
    """
    if raw_arr.size == 0:
        return [], 0

    stage = raw_arr[:, 0].astype(float)
    vac = raw_arr[:, 1].astype(float)
    gA = raw_arr[:, 3].astype(float)
    Ek = raw_arr[:, 5].astype(float)
    air_all = np.asarray(vacuum_to_air(vac), dtype=float)

    range_lo, range_hi = range_nm
    mask = (
        ((stage == 1.0) | (stage == 2.0))
        & (air_all >= range_lo) & (air_all <= range_hi)
    )
    pool_idx = np.where(mask)[0]
    n_candidates = int(pool_idx.size)
    if n_candidates < MIN_CANDIDATES_TO_KEEP_ELEMENT:
        return [], n_candidates

    kT = KB_EV_PER_K * REFERENCE_TEMPERATURE_K
    strength = gA * np.exp(-Ek / kT)

    pool_vac = vac[pool_idx]
    pool_strength = strength[pool_idx]
    pool_order = np.argsort(pool_vac, kind="stable")
    pool_vac_sorted = pool_vac[pool_order]
    pool_idx_sorted = pool_idx[pool_order]
    pool_strength_sorted = pool_strength[pool_order]

    order = pool_idx[np.argsort(-strength[pool_idx], kind="stable")]

    selected = []
    selected_vac = []
    for i in order:
        w = vac[i]
        if any(abs(w - sw) < MIN_SEPARATION_NM for sw in selected_vac):
            continue
        selected.append(int(i))
        selected_vac.append(w)
        if len(selected) >= TARGET_LINE_COUNT:
            break
    selected.sort()

    records = []
    for idx in selected:
        row = raw_arr[idx]
        rec = _build_line_record(element, idx, row, float(strength[idx]))
        w = vac[idx]
        lo = np.searchsorted(pool_vac_sorted, w - BLEND_WINDOW_NM, side="left")
        hi = np.searchsorted(pool_vac_sorted, w + BLEND_WINDOW_NM, side="right")
        blend_floor = BLEND_RATIO_FLOOR * strength[idx]
        blends = []
        for j in range(lo, hi):
            cand_idx = int(pool_idx_sorted[j])
            if cand_idx == idx:
                continue
            if pool_strength_sorted[j] < blend_floor:
                continue
            blends.append(cand_idx)
        blends.sort()
        for b in blends:
            row_b = raw_arr[b]
            air_b = float(f"{float(vacuum_to_air(np.array([float(row_b[1])]))[0]):.6f}")
            rec["potential_blends"].append({"id": f"{element}-{b}", "wavelength_nm": air_b})
        for junk_key in ("_lower_cfg", "_lower_term", "_lower_J", "_vacuum_nm"):
            rec.pop(junk_key, None)
        records.append(rec)

    return records, n_candidates


def _range_json(range_nm):
    lo, hi = range_nm
    return [int(lo) if float(lo).is_integer() else lo,
            int(hi) if float(hi).is_integer() else hi]


def build_reference_document(element: str, records: list, commit: str, sha256: str,
                              range_nm=DEFAULT_OBSERVABLE_RANGE_NM) -> dict:
    lo, hi = range_nm
    return {
        "schema_version": 1,
        "element": element,
        "label": f"Bundled {element} I/II atomic candidates",
        "wavelength_medium": WAVELENGTH_MEDIUM,
        "observable_range_nm": _range_json(range_nm),
        "source": {
            "repository": "https://github.com/MWhitta/alibz",
            "commit": commit,
            "path": "db/el_lines92.pickle",
            "sha256": sha256,
            "provenance_status": (
                "legacy bundled line table; generated by "
                "scripts/build_line_references.py from the same source as "
                "fe-v1.json; original per-line citations and uncertainties "
                "unavailable"
            ),
        },
        "selection": (
            f"{len(records)} strongest separated {element} I/II candidate "
            f"windows at 10000 K, restricted to the {lo:g}-{hi:g} nm "
            "observable (air) range; ranking only, not a plasma model or "
            "intensity truth"
        ),
        "oscillator_strength_conversion": OSCILLATOR_STRENGTH_CONVERSION,
        "physics_source": PHYSICS_SOURCE,
        "lines": records,
    }


def _numbers_close(a, b, rel_tol=1e-6):
    if a == b:
        return True
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom <= rel_tol


NUMERIC_FIELDS = [
    "wavelength_nm", "gA_s_1", "A_s_1", "oscillator_strength",
    "lower_energy_eV", "upper_energy_eV", "g_lower", "g_upper",
    "strength_at_10000K",
]


def check_fe(db: Database, dbpath: Path, fe_path: Path,
             range_nm=DEFAULT_OBSERVABLE_RANGE_NM) -> bool:
    raw = _load_raw_atom_dict(dbpath)
    records, _ = select_element_lines("Fe", raw["Fe"], range_nm=range_nm)

    with open(fe_path) as fh:
        bundled = json.load(fh)
    bundled_lines = bundled["lines"]

    ok = True
    gen_ids = [r["id"] for r in records]
    bun_ids = [l["id"] for l in bundled_lines]
    if gen_ids != bun_ids:
        ok = False
        gen_set, bun_set = set(gen_ids), set(bun_ids)
        print(f"ID SET/ORDER MISMATCH: generated {len(gen_ids)} vs bundled {len(bun_ids)}")
        print(f"  missing from generated: {sorted(bun_set - gen_set)}")
        print(f"  extra in generated:     {sorted(gen_set - bun_set)}")
        if gen_set == bun_set:
            print("  (same set, different order)")
    else:
        print(f"IDs: {len(gen_ids)}/{len(gen_ids)} match, same order -- OK")

    by_id_gen = {r["id"]: r for r in records}
    by_id_bun = {l["id"]: l for l in bundled_lines}
    n_field_mismatch = 0
    n_blend_mismatch = 0
    for lid in bun_ids:
        if lid not in by_id_gen:
            continue
        g = by_id_gen[lid]
        b = by_id_bun[lid]
        for field in NUMERIC_FIELDS:
            if not _numbers_close(g[field], b[field]):
                n_field_mismatch += 1
                ok = False
                print(f"  FIELD MISMATCH {lid}.{field}: generated={g[field]!r} bundled={b[field]!r}")
        g_blend_ids = sorted(x["id"] for x in g["potential_blends"])
        b_blend_ids = sorted(x["id"] for x in b["potential_blends"])
        if g_blend_ids != b_blend_ids:
            n_blend_mismatch += 1
            ok = False
            print(f"  BLEND MISMATCH {lid}: generated={g_blend_ids} bundled={b_blend_ids}")

    print(f"Numeric field mismatches: {n_field_mismatch}")
    print(f"Blend-list mismatches:    {n_blend_mismatch}")
    print("RESULT:", "MATCH" if ok else "MISMATCH")
    return ok


def _resolve_elements(db: Database, spec: str) -> list:
    if spec.strip().lower() == "all":
        return list(db.elements)
    return [s.strip() for s in spec.split(",") if s.strip()]


def generate_all(db: Database, dbpath: Path, elements: list, out_dir: Path,
                  dry_run: bool, include_fe: bool, commit: str, sha256: str,
                  range_nm=DEFAULT_OBSERVABLE_RANGE_NM):
    raw = _load_raw_atom_dict(dbpath)
    written = []
    skipped = []
    counts = []
    deleted = []

    for el in elements:
        stale_path = out_dir / f"{el.lower()}-v1.json"

        if el not in db.elements:
            skipped.append((el, "unknown element symbol"))
            continue
        if el in db.no_lines:
            skipped.append((el, "no_lines"))
            continue
        if not db.is_supported(el):
            skipped.append((el, "unsupported (Database.is_supported False)"))
            continue
        arr = raw.get(el)
        if arr is None or arr.size == 0:
            skipped.append((el, "empty line array"))
            continue

        records, n_candidates = select_element_lines(el, arr, range_nm=range_nm)
        if n_candidates < MIN_CANDIDATES_TO_KEEP_ELEMENT:
            skipped.append((el, f"only {n_candidates} ion-stage-1/2 candidates in "
                                 f"{range_nm[0]:g}-{range_nm[1]:g} nm (<{MIN_CANDIDATES_TO_KEEP_ELEMENT})"))
            if el != "Fe" and stale_path.exists() and not dry_run:
                stale_path.unlink()
                deleted.append((el, str(stale_path)))
            continue

        counts.append((el, len(records), n_candidates))

        if el == "Fe" and not include_fe:
            skipped.append((el, "Fe excluded by default; pass --include-fe"))
            continue

        if dry_run:
            continue

        doc = build_reference_document(el, records, commit, sha256, range_nm=range_nm)
        out_path = out_dir / f"{el.lower()}-v1.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
        written.append((el, out_path, len(records), out_path.stat().st_size))

    return written, skipped, counts, deleted


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check-fe", metavar="PATH",
                         help="Regenerate Fe and diff against the bundled fe-v1.json at PATH.")
    parser.add_argument("--out", metavar="DIR",
                         help="Directory to write <symbol>-v1.json reference files into.")
    parser.add_argument("--elements", default="all",
                         help="'all' or a comma-separated list of element symbols (default: all).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print per-element counts, write nothing.")
    parser.add_argument("--include-fe", action="store_true",
                         help="Allow overwriting fe-v1.json-equivalent output; default excludes Fe.")
    parser.add_argument("--dbpath", default="db",
                         help="alibz database directory (default: db).")
    parser.add_argument("--range", default=None, metavar="LO,HI",
                         help=(
                             "Observable AIR-wavelength gate 'LO,HI' in nm, applied to "
                             "the candidate pool before selection and to the blend "
                             "pool (default: %g,%g -- the SciAps Z300 export grid)."
                             % DEFAULT_OBSERVABLE_RANGE_NM
                         ))
    args = parser.parse_args()

    if args.range is not None:
        lo_s, hi_s = args.range.split(",")
        range_nm = (float(lo_s), float(hi_s))
    else:
        range_nm = DEFAULT_OBSERVABLE_RANGE_NM

    db = Database(args.dbpath)
    dbpath = db.dbpath
    commit = _git_commit(REPO_ROOT)
    sha256 = _pickle_sha256(dbpath)

    if args.check_fe:
        ok = check_fe(db, dbpath, Path(args.check_fe), range_nm=range_nm)
        sys.exit(0 if ok else 1)

    if args.out is None:
        parser.error("either --check-fe or --out is required")

    elements = _resolve_elements(db, args.elements)
    out_dir = Path(args.out)
    written, skipped, counts, deleted = generate_all(
        db, dbpath, elements, out_dir, args.dry_run, args.include_fe, commit, sha256,
        range_nm=range_nm,
    )

    print(f"Observable range: {range_nm[0]:g}-{range_nm[1]:g} nm (air)")
    print(f"{'DRY RUN: ' if args.dry_run else ''}{len(counts)} elements produced candidate line sets")
    for el, n, n_cand in counts:
        print(f"  {el:>3}: {n:3d} lines (from {n_cand} ion-stage-1/2 candidates)")
    if not args.dry_run:
        total_bytes = sum(sz for _, _, _, sz in written)
        print(f"Wrote {len(written)} files, {total_bytes} bytes total, to {out_dir}")
    if deleted:
        print(f"Deleted {len(deleted)} stale files (dropped below {MIN_CANDIDATES_TO_KEEP_ELEMENT} candidates):")
        for el, path in deleted:
            print(f"  {el}: {path}")
    if skipped:
        print(f"Skipped {len(skipped)} elements:")
        for el, reason in skipped:
            print(f"  {el}: {reason}")


if __name__ == "__main__":
    main()
