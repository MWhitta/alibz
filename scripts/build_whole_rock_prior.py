"""Build the versioned Gard et al. (2019) whole-rock composition prior.

The raw Zenodo CSV files are intentionally not committed.  This command
converts major oxides to elemental mass ppm, preserves positive trace-element
reports, treats negative values as left-censor limits, fits empirical
marginals and rock-stratified Gaussian copulas, and writes a compact replayable
artifact plus a human-readable element-range table.
"""

from __future__ import annotations

import argparse
from array import array
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtri

from alibz.elements import ATOMIC_NUMBER, ELEMENTS_BY_ATOMIC_NUMBER
from alibz.synthetic import ATOMIC_MASS_U, N_ELEMENTS
from alibz.whole_rock import (
    DEFAULT_TRAINING_EXCLUSIONS,
    WHOLE_ROCK_MODEL_SCHEMA,
    WholeRockCompositionModel,
)


SOURCE_DOI = "10.5281/zenodo.2592823"
PAPER_DOI = "10.5194/essd-11-1553-2019"
SOURCE_URLS = {
    "major": "https://zenodo.org/api/records/2592823/files/major.csv/content",
    "trace": "https://zenodo.org/api/records/2592823/files/trace.csv/content",
    "sample": "https://zenodo.org/api/records/2592823/files/sample.csv/content",
    "rockgroup": "https://zenodo.org/api/records/2592823/files/rockgroup.csv/content",
}
EXPECTED_MD5 = {
    "major": "c7406fd61feaf10d984a637346977fc3",
    "trace": "cf5aba704009051484c76e0dd171145b",
    "sample": "479f6dd6f5a1d6e3075b4ad3e6389cac",
    "rockgroup": "827e9ff1af4564c129e801f72bea5cc1",
}

STRATA = (
    "igneous_volcanic",
    "igneous_plutonic",
    "igneous_other",
    "metamorphic_igneous",
    "metamorphic_sedimentary",
    "metamorphic_unspecified",
    "sedimentary_clastic",
    "sedimentary_chemical_biogenic",
    "sedimentary_unspecified",
    "unclassified",
)
VOLATILE_STRATA = ("carbonate_rich", "volatile_rich")

# Raw column, element, number of element atoms, number of oxygen atoms.
OXIDES = (
    ("sio2", "Si", 1, 2),
    ("tio2", "Ti", 1, 2),
    ("al2o3", "Al", 2, 3),
    ("cr2o3", "Cr", 2, 3),
    ("mgo", "Mg", 1, 1),
    ("cao", "Ca", 1, 1),
    ("mno", "Mn", 1, 1),
    ("nio", "Ni", 1, 1),
    ("k2o", "K", 2, 1),
    ("na2o", "Na", 2, 1),
    ("sro", "Sr", 1, 1),
    ("p2o5", "P", 2, 5),
    ("so3", "S", 1, 3),
    ("bao", "Ba", 1, 1),
)


def _hash_file(path, algorithm):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sources(paths):
    hashes = {}
    for name, path in paths.items():
        md5 = _hash_file(path, "md5")
        if md5 != EXPECTED_MD5[name]:
            raise ValueError(
                f"{name} source checksum changed: expected {EXPECTED_MD5[name]}, got {md5}"
            )
        hashes[name] = {
            "input_filename": Path(path).name,
            "url": SOURCE_URLS[name],
            "md5": md5,
            "sha256": _hash_file(path, "sha256"),
        }
    return hashes


def _load_numeric_csv(path):
    with Path(path).open("r", newline="", encoding="latin-1") as handle:
        header = next(csv.reader(handle))
    values = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        dtype=np.float32,
        filling_values=np.nan,
        invalid_raise=True,
    )
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != len(header):
        raise ValueError(f"column mismatch while reading {path}")
    return tuple(header), values


def _rock_stratum(group, origin):
    group = group.strip().lower()
    origin = origin.strip().lower()
    if group == "igneous":
        if origin in ("volcanic", "metavolcanic"):
            return "igneous_volcanic"
        if origin in ("plutonic", "metaplutonic"):
            return "igneous_plutonic"
        return "igneous_other"
    if group == "metamorphic":
        if origin in ("metaigneous", "metaplutonic", "metavolcanic", "plutonic"):
            return "metamorphic_igneous"
        if origin == "metasedimentary":
            return "metamorphic_sedimentary"
        return "metamorphic_unspecified"
    if group == "sedimentary":
        if origin == "clastic":
            return "sedimentary_clastic"
        if origin in ("chemical", "biogenic"):
            return "sedimentary_chemical_biogenic"
        return "sedimentary_unspecified"
    return "unclassified"


def _load_rockgroup_map(path):
    out = {}
    with Path(path).open("r", newline="", encoding="latin-1") as handle:
        for row in csv.DictReader(handle):
            out[int(row["rgroup_id"])] = _rock_stratum(
                row["rock_group"], row["rock_origin"]
            )
    return out


def _integer_or_zero(value):
    return int(value) if value else 0


def _load_sample_links(path, rockgroup_map):
    major = array("i")
    trace = array("i")
    group = array("h")
    stratum_index = {name: i for i, name in enumerate(STRATA)}
    with Path(path).open("r", newline="", encoding="latin-1") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        columns = {name: i for i, name in enumerate(header)}
        for row in reader:
            major.append(_integer_or_zero(row[columns["major_id"]]))
            trace.append(_integer_or_zero(row[columns["trace_id"]]))
            rgroup = _integer_or_zero(row[columns["rgroup_id"]])
            label = rockgroup_map.get(rgroup, "unclassified")
            group.append(stratum_index[label])
    return (
        np.frombuffer(major, dtype=np.int32).copy(),
        np.frombuffer(trace, dtype=np.int32).copy(),
        np.frombuffer(group, dtype=np.int16).copy(),
    )


def _id_lookup(ids):
    integer = np.asarray(ids, dtype=np.int64)
    if np.any(integer <= 0):
        raise ValueError("chemistry table ids must be positive")
    lookup = np.full(int(integer.max()) + 1, -1, dtype=np.int32)
    if np.unique(integer).size != integer.size:
        raise ValueError("chemistry table ids must be unique")
    lookup[integer] = np.arange(integer.size, dtype=np.int32)
    return lookup


def _positive(values):
    return np.where(np.isfinite(values) & (values > 0), values, 0.0)


def _major_element_ppm(header, raw):
    columns = {name: i for i, name in enumerate(header)}
    n = raw.shape[0]
    oxide_values = {}
    for name, _element, _n_el, _n_o in OXIDES:
        oxide_values[name] = _positive(raw[:, columns[name]]).astype(np.float64)

    feo = _positive(raw[:, columns["feo"]]).astype(np.float64)
    fe2o3 = _positive(raw[:, columns["fe2o3"]]).astype(np.float64)
    feo_total = _positive(raw[:, columns["feo_tot"]]).astype(np.float64)
    fe2o3_total = _positive(raw[:, columns["fe2o3_tot"]]).astype(np.float64)
    fe = ATOMIC_MASS_U[ATOMIC_NUMBER["Fe"] - 1]
    oxygen = ATOMIC_MASS_U[ATOMIC_NUMBER["O"] - 1]
    feo_mass = fe + oxygen
    fe2o3_mass = 2.0 * fe + 3.0 * oxygen
    fe2o3_to_feo = 2.0 * feo_mass / fe2o3_mass
    feo_equivalent = np.where(
        feo_total > 0,
        feo_total,
        np.where(
            fe2o3_total > 0,
            fe2o3_total * fe2o3_to_feo,
            feo + fe2o3 * fe2o3_to_feo,
        ),
    )

    total = feo_equivalent.copy()
    for values in oxide_values.values():
        total += values
    valid = np.isfinite(total) & (total >= 85.0) & (total <= 120.0)
    scale = np.zeros(n, dtype=np.float64)
    scale[valid] = 100.0 / total[valid]
    concentration = np.full((n, N_ELEMENTS), np.nan, dtype=np.float32)
    oxygen_ppm = np.zeros(n, dtype=np.float64)

    def add_oxide(values, element, n_element, n_oxygen):
        ei = ATOMIC_NUMBER[element] - 1
        element_mass = ATOMIC_MASS_U[ei]
        molecular_mass = n_element * element_mass + n_oxygen * oxygen
        normalized = values * scale
        present = valid & (values > 0)
        concentration[present, ei] = (
            normalized[present] * 10_000.0
            * n_element * element_mass / molecular_mass
        ).astype(np.float32)
        oxygen_ppm[present] += (
            normalized[present] * 10_000.0
            * n_oxygen * oxygen / molecular_mass
        )

    for name, element, n_element, n_oxygen in OXIDES:
        add_oxide(oxide_values[name], element, n_element, n_oxygen)
    add_oxide(feo_equivalent, "Fe", 1, 1)
    concentration[valid, ATOMIC_NUMBER["O"] - 1] = oxygen_ppm[valid]
    return concentration, valid


def _volatile_major_element_ppm(
    header,
    raw,
    volatile_threshold_wt=5.0,
    carbonate_co2_threshold_wt=20.0,
):
    """Convert directly speciated carbonate/volatile analyses to elements.

    Carbonate columns replace their corresponding simple oxide so Ca or Mg is
    not counted twice.  H2O-total is preferred over H2O+ plus H2O-.  LOI is
    deliberately not assigned to an element because it is not speciated.
    """
    columns = {name: i for i, name in enumerate(header)}
    n = raw.shape[0]
    oxide_values = {
        name: _positive(raw[:, columns[name]]).astype(np.float64)
        for name, _element, _n_el, _n_o in OXIDES
    }
    caco3 = _positive(raw[:, columns["caco3"]]).astype(np.float64)
    mgco3 = _positive(raw[:, columns["mgco3"]]).astype(np.float64)
    co2 = _positive(raw[:, columns["co2"]]).astype(np.float64)
    h2o_total = _positive(raw[:, columns["h2o_tot"]]).astype(np.float64)
    h2o_plus = _positive(raw[:, columns["h2o_plus"]]).astype(np.float64)
    h2o_minus = _positive(raw[:, columns["h2o_minus"]]).astype(np.float64)
    h2o = np.where(h2o_total > 0, h2o_total, h2o_plus + h2o_minus)
    loi = _positive(raw[:, columns["loi"]]).astype(np.float64)

    # Explicit carbonate analytes are full compounds, not additional CaO or
    # MgO columns.  When present they also supersede a separately exported CO2
    # field to avoid duplicate carbonate mass.
    oxide_values["cao"][caco3 > 0] = 0.0
    oxide_values["mgo"][mgco3 > 0] = 0.0
    has_carbonate_analyte = (caco3 > 0) | (mgco3 > 0)
    co2_component = np.where(has_carbonate_analyte, 0.0, co2)

    feo = _positive(raw[:, columns["feo"]]).astype(np.float64)
    fe2o3 = _positive(raw[:, columns["fe2o3"]]).astype(np.float64)
    feo_total = _positive(raw[:, columns["feo_tot"]]).astype(np.float64)
    fe2o3_total = _positive(raw[:, columns["fe2o3_tot"]]).astype(np.float64)
    fe = ATOMIC_MASS_U[ATOMIC_NUMBER["Fe"] - 1]
    oxygen = ATOMIC_MASS_U[ATOMIC_NUMBER["O"] - 1]
    carbon = ATOMIC_MASS_U[ATOMIC_NUMBER["C"] - 1]
    hydrogen = ATOMIC_MASS_U[ATOMIC_NUMBER["H"] - 1]
    calcium = ATOMIC_MASS_U[ATOMIC_NUMBER["Ca"] - 1]
    magnesium = ATOMIC_MASS_U[ATOMIC_NUMBER["Mg"] - 1]
    feo_mass = fe + oxygen
    fe2o3_mass = 2.0 * fe + 3.0 * oxygen
    fe2o3_to_feo = 2.0 * feo_mass / fe2o3_mass
    feo_equivalent = np.where(
        feo_total > 0,
        feo_total,
        np.where(
            fe2o3_total > 0,
            fe2o3_total * fe2o3_to_feo,
            feo + fe2o3 * fe2o3_to_feo,
        ),
    )

    co2_mass = carbon + 2.0 * oxygen
    caco3_mass = calcium + carbon + 3.0 * oxygen
    mgco3_mass = magnesium + carbon + 3.0 * oxygen
    carbonate_co2_equivalent = (
        co2_component
        + caco3 * co2_mass / caco3_mass
        + mgco3 * co2_mass / mgco3_mass
    )
    specified_volatile = (
        h2o + carbonate_co2_equivalent + oxide_values["so3"]
    )
    carbonate = carbonate_co2_equivalent >= float(carbonate_co2_threshold_wt)
    volatile = (~carbonate) & (
        specified_volatile >= float(volatile_threshold_wt)
    )
    group = np.full(n, -1, dtype=np.int16)
    group[carbonate] = 0
    group[volatile] = 1

    total = feo_equivalent + h2o + co2_component + caco3 + mgco3
    for values in oxide_values.values():
        total += values
    valid = (
        (group >= 0) & np.isfinite(total) & (total >= 85.0) & (total <= 120.0)
    )
    group[~valid] = -1
    scale = np.zeros(n, dtype=np.float64)
    scale[valid] = 100.0 / total[valid]
    concentration = np.full((n, N_ELEMENTS), np.nan, dtype=np.float32)

    def add_component(values, atoms):
        molecular_mass = sum(
            ATOMIC_MASS_U[ATOMIC_NUMBER[element] - 1] * count
            for element, count in atoms.items()
        )
        present = valid & (values > 0)
        normalized_ppm = values[present] * scale[present] * 10_000.0
        for element, count in atoms.items():
            ei = ATOMIC_NUMBER[element] - 1
            contribution = (
                normalized_ppm * count * ATOMIC_MASS_U[ei] / molecular_mass
            )
            current = concentration[present, ei]
            current = np.where(np.isfinite(current), current, 0.0)
            concentration[present, ei] = (current + contribution).astype(np.float32)

    for name, element, n_element, n_oxygen in OXIDES:
        add_component(oxide_values[name], {element: n_element, "O": n_oxygen})
    add_component(feo_equivalent, {"Fe": 1, "O": 1})
    add_component(h2o, {"H": 2, "O": 1})
    add_component(co2_component, {"C": 1, "O": 2})
    add_component(caco3, {"Ca": 1, "C": 1, "O": 3})
    add_component(mgco3, {"Mg": 1, "C": 1, "O": 3})

    stats = {
        "carbonate_candidates": int(carbonate.sum()),
        "volatile_candidates": int(volatile.sum()),
        "carbonate_valid": int(np.sum(group == 0)),
        "volatile_valid": int(np.sum(group == 1)),
        "loi_only_at_or_above_threshold": int(np.sum(
            (loi >= float(volatile_threshold_wt))
            & (specified_volatile < float(volatile_threshold_wt))
        )),
    }
    return concentration, valid, group, stats


def _join_precomputed_chemistry(
    major_header,
    major_raw,
    major_concentration,
    valid_major_rows,
    trace_header,
    trace_raw,
    sample_major_id,
    sample_trace_id,
):
    major_lookup = _id_lookup(major_raw[:, 0])
    trace_lookup = _id_lookup(trace_raw[:, 0])
    n_samples = sample_major_id.size
    values = np.full((n_samples, N_ELEMENTS), np.nan, dtype=np.float32)
    limits = np.full((n_samples, N_ELEMENTS), np.nan, dtype=np.float32)
    valid_major_sample = np.zeros(n_samples, dtype=bool)

    has_major = (sample_major_id > 0) & (sample_major_id < major_lookup.size)
    major_rows = np.full(n_samples, -1, dtype=np.int32)
    major_rows[has_major] = major_lookup[sample_major_id[has_major]]
    has_major &= major_rows >= 0
    valid_major_sample[has_major] = valid_major_rows[major_rows[has_major]]
    values[valid_major_sample] = major_concentration[
        major_rows[valid_major_sample]
    ]

    has_trace = (sample_trace_id > 0) & (sample_trace_id < trace_lookup.size)
    trace_rows = np.full(n_samples, -1, dtype=np.int32)
    trace_rows[has_trace] = trace_lookup[sample_trace_id[has_trace]]
    has_trace &= trace_rows >= 0
    sample_index = np.flatnonzero(has_trace)
    row_index = trace_rows[has_trace]
    element_lookup = {el.lower(): i for i, el in enumerate(ELEMENTS_BY_ATOMIC_NUMBER)}

    for ci, name in enumerate(trace_header):
        if not name.endswith("_ppm"):
            continue
        symbol = name[:-4].lower()
        if symbol not in element_lookup:
            continue
        ei = element_lookup[symbol]
        observed = trace_raw[row_index, ci]
        positive = np.isfinite(observed) & (observed > 0)
        destination = sample_index[positive]
        fill = ~np.isfinite(values[destination, ei])
        values[destination[fill], ei] = observed[positive][fill]
        censored = np.isfinite(observed) & (observed < 0)
        destination = sample_index[censored]
        fill = ~np.isfinite(values[destination, ei])
        limits[destination[fill], ei] = -observed[censored][fill]

    return values, limits, valid_major_sample


def _join_chemistry(
    major_header,
    major_raw,
    trace_header,
    trace_raw,
    sample_major_id,
    sample_trace_id,
):
    major_concentration, valid_major_rows = _major_element_ppm(
        major_header, major_raw
    )
    return _join_precomputed_chemistry(
        major_header,
        major_raw,
        major_concentration,
        valid_major_rows,
        trace_header,
        trace_raw,
        sample_major_id,
        sample_trace_id,
    )


def _quantiles(values, probability):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return np.full(probability.size, np.nan)
    return np.quantile(np.log10(values), probability)


def _nearest_correlation(matrix, floor=1e-4):
    matrix = np.asarray(matrix, dtype=float)
    matrix = np.nan_to_num(0.5 * (matrix + matrix.T), nan=0.0)
    np.fill_diagonal(matrix, 1.0)
    values, vectors = np.linalg.eigh(matrix)
    matrix = (vectors * np.clip(values, floor, None)) @ vectors.T
    scale = np.sqrt(np.clip(np.diag(matrix), floor, None))
    matrix /= scale[:, None] * scale[None, :]
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _fit_correlation(values, row_index, quantiles, probability, modeled, rng,
                     max_rows, min_overlap, shrinkage):
    if row_index.size > max_rows:
        row_index = np.sort(rng.choice(row_index, max_rows, replace=False))
    subset = values[row_index]
    z = np.zeros(subset.shape, dtype=np.float32)
    mask = np.zeros(subset.shape, dtype=np.float32)
    for ei in np.flatnonzero(modeled):
        observed = subset[:, ei]
        present = np.isfinite(observed) & (observed > 0)
        if not present.any():
            continue
        q = quantiles[ei]
        finite = np.isfinite(q)
        if finite.sum() < 2:
            continue
        u = np.interp(
            np.log10(observed[present]),
            q[finite],
            probability[finite],
        )
        z[present, ei] = ndtri(np.clip(u, 1e-5, 1.0 - 1e-5)).astype(np.float32)
        mask[present, ei] = 1.0

    count = mask.T @ mask
    cross = z.T @ z
    sum_one = z.T @ mask
    sum_square = (z * z).T @ mask
    safe_n = np.maximum(count, 1.0)
    covariance = cross - (sum_one * sum_one.T) / safe_n
    variance_i = sum_square - (sum_one * sum_one) / safe_n
    denominator = np.sqrt(np.maximum(variance_i * variance_i.T, 1e-12))
    correlation = covariance / denominator
    enough = count >= float(min_overlap)
    weight = np.maximum(count - 3.0, 0.0)
    weight /= weight + float(shrinkage)
    correlation = np.where(enough, correlation * weight, 0.0)
    correlation[~modeled, :] = 0.0
    correlation[:, ~modeled] = 0.0
    np.fill_diagonal(correlation, 1.0)
    return _nearest_correlation(correlation).astype(np.float32), count.astype(np.int32)


def _fit_model(values, limits, group_index, probability, min_reports,
               max_correlation_rows, min_overlap, shrinkage, seed,
               stratum_names=STRATA):
    stratum_names = tuple(stratum_names)
    strata = ("global",) + stratum_names
    n_strata = len(strata)
    shape = (n_strata, N_ELEMENTS)
    detected_count = np.zeros(shape, dtype=np.int64)
    censored_count = np.zeros(shape, dtype=np.int64)
    reporting_rate = np.zeros(shape, dtype=np.float32)
    censored_fraction = np.zeros(shape, dtype=np.float32)
    detected_q = np.full(shape + (probability.size,), np.nan, dtype=np.float32)
    limit_q = np.full_like(detected_q, np.nan)
    fallback = np.zeros(shape, dtype=bool)
    sample_count = np.zeros(n_strata, dtype=np.int64)
    row_sets = [np.arange(values.shape[0], dtype=np.int64)]
    row_sets.extend(
        np.flatnonzero(group_index == i) for i in range(len(stratum_names))
    )

    for si, rows in enumerate(row_sets):
        sample_count[si] = rows.size
        for ei in range(N_ELEMENTS):
            positive = values[rows, ei]
            positive = positive[np.isfinite(positive) & (positive > 0)]
            censored = limits[rows, ei]
            censored = censored[np.isfinite(censored) & (censored > 0)]
            detected_count[si, ei] = positive.size
            censored_count[si, ei] = censored.size
            reporting_rate[si, ei] = (
                (positive.size + censored.size) / rows.size if rows.size else 0.0
            )
            total_reported = positive.size + censored.size
            censored_fraction[si, ei] = (
                censored.size / total_reported if total_reported else 0.0
            )
            detected_q[si, ei] = _quantiles(positive, probability)
            limit_q[si, ei] = _quantiles(censored, probability)

    global_modeled = (
        (detected_count[0] >= min_reports)
        | ((detected_count[0] > 0) & (censored_count[0] >= min_reports))
    )
    modeled = np.zeros(shape, dtype=bool)
    modeled[0] = global_modeled
    for si in range(1, n_strata):
        adequate = (
            (detected_count[si] >= min_reports)
            | ((detected_count[si] > 0) & (censored_count[si] >= min_reports))
        )
        modeled[si] = global_modeled
        use_global = global_modeled & ~adequate
        fallback[si, use_global] = True
        detected_q[si, use_global] = detected_q[0, use_global]
        limit_q[si, use_global] = limit_q[0, use_global]
        censored_fraction[si, use_global] = censored_fraction[0, use_global]

    training_mask = np.ones(N_ELEMENTS, dtype=bool)
    for element in DEFAULT_TRAINING_EXCLUSIONS:
        training_mask[ATOMIC_NUMBER[element] - 1] = False
    modeled &= training_mask[None, :]

    correlation = np.zeros((n_strata, N_ELEMENTS, N_ELEMENTS), dtype=np.float32)
    overlap_minimum = np.zeros_like(correlation, dtype=np.int32)
    rng = np.random.default_rng(seed)
    for si, rows in enumerate(row_sets):
        correlation[si], overlap_minimum[si] = _fit_correlation(
            values,
            rows,
            detected_q[si],
            probability,
            modeled[si],
            rng,
            max_correlation_rows,
            min_overlap,
            shrinkage,
        )
    return {
        "schema": np.asarray(WHOLE_ROCK_MODEL_SCHEMA),
        "elements": np.asarray(ELEMENTS_BY_ATOMIC_NUMBER),
        "strata": np.asarray(strata),
        "stratum_sample_count": sample_count,
        "quantile_probability": probability,
        "detected_log10_ppm_quantile": detected_q,
        "limit_log10_ppm_quantile": limit_q,
        "detected_count": detected_count,
        "censored_count": censored_count,
        "reporting_rate": reporting_rate,
        "censored_fraction": censored_fraction,
        "modeled_mask": modeled,
        "marginal_fallback_to_global": fallback,
        "correlation": correlation,
        "pairwise_overlap": overlap_minimum,
        "training_mask": training_mask,
        "source_doi": np.asarray(SOURCE_DOI),
    }


def _summary_nuclei_quantiles(model, draws, seed):
    probability = np.asarray([0.001, 0.025, 0.5, 0.975, 0.999])
    out = {}
    for si, stratum in enumerate(model.strata):
        samples = np.zeros((draws, N_ELEMENTS), dtype=np.float32)
        for draw in range(draws):
            sample = model.sample(
                seed + si * draws + draw,
                stratum=stratum,
                min_nuclei_fraction=0.0,
            )
            samples[draw] = sample.nuclei_fraction
        out[stratum] = np.quantile(samples, probability, axis=0)
    return probability, out


def _write_ranges(path, arrays, nuclei_probability, nuclei_quantiles):
    probability = arrays["quantile_probability"]
    selected = np.asarray([0.001, 0.005, 0.025, 0.5, 0.975, 0.995, 0.999])
    fields = [
        "stratum", "element", "atomic_number", "n_samples", "n_detected",
        "n_censored", "reporting_rate", "censored_fraction", "modeled",
        "fallback_to_global", "detected_p001_ppm", "detected_p005_ppm",
        "detected_p025_ppm", "detected_p500_ppm", "detected_p975_ppm",
        "detected_p995_ppm", "detected_p999_ppm", "nuclei_p001",
        "nuclei_p025", "nuclei_p500", "nuclei_p975", "nuclei_p999",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for si, stratum in enumerate(arrays["strata"].tolist()):
            for ei, element in enumerate(ELEMENTS_BY_ATOMIC_NUMBER):
                detected = arrays["detected_log10_ppm_quantile"][si, ei]
                finite = np.isfinite(detected)
                if finite.any():
                    log_values = np.interp(
                        selected, probability[finite], detected[finite]
                    )
                    ppm = 10.0 ** log_values
                else:
                    ppm = np.full(selected.size, np.nan)
                nq = nuclei_quantiles[stratum][:, ei]
                row = {
                    "stratum": stratum,
                    "element": element,
                    "atomic_number": ei + 1,
                    "n_samples": int(arrays["stratum_sample_count"][si]),
                    "n_detected": int(arrays["detected_count"][si, ei]),
                    "n_censored": int(arrays["censored_count"][si, ei]),
                    "reporting_rate": f"{arrays['reporting_rate'][si, ei]:.8g}",
                    "censored_fraction": f"{arrays['censored_fraction'][si, ei]:.8g}",
                    "modeled": int(arrays["modeled_mask"][si, ei]),
                    "fallback_to_global": int(
                        arrays["marginal_fallback_to_global"][si, ei]
                    ),
                }
                for name, value in zip(fields[10:17], ppm):
                    row[name] = "" if not np.isfinite(value) else f"{value:.9g}"
                for name, value in zip(fields[17:], nq):
                    row[name] = f"{value:.9g}"
                writer.writerow(row)


def build(args):
    paths = {
        "major": args.major,
        "trace": args.trace,
        "sample": args.sample,
        "rockgroup": args.rockgroup,
    }
    source_hashes = _verify_sources(paths)
    rockgroups = _load_rockgroup_map(args.rockgroup)
    sample_major, sample_trace, sample_group = _load_sample_links(
        args.sample, rockgroups
    )
    major_header, major_raw = _load_numeric_csv(args.major)
    trace_header, trace_raw = _load_numeric_csv(args.trace)
    volatile_stats = None
    if args.composition_mode == "anhydrous":
        values, limits, valid_major = _join_chemistry(
            major_header,
            major_raw,
            trace_header,
            trace_raw,
            sample_major,
            sample_trace,
        )
        stratum_names = STRATA
        major_analysis_count = int(valid_major.sum())
    else:
        concentration, valid_rows, row_group, volatile_stats = (
            _volatile_major_element_ppm(
                major_header,
                major_raw,
                volatile_threshold_wt=args.volatile_threshold_wt,
                carbonate_co2_threshold_wt=args.carbonate_co2_threshold_wt,
            )
        )
        major_lookup = _id_lookup(major_raw[:, 0])
        has_major = (sample_major > 0) & (sample_major < major_lookup.size)
        major_rows = np.full(sample_major.size, -1, dtype=np.int32)
        major_rows[has_major] = major_lookup[sample_major[has_major]]
        has_major &= major_rows >= 0
        selected = np.zeros(sample_major.size, dtype=bool)
        selected[has_major] = valid_rows[major_rows[has_major]]
        sample_group = row_group[major_rows[selected]]
        sample_major = sample_major[selected]
        sample_trace = sample_trace[selected]
        values, limits, valid_major = _join_precomputed_chemistry(
            major_header,
            major_raw,
            concentration,
            valid_rows,
            trace_header,
            trace_raw,
            sample_major,
            sample_trace,
        )
        stratum_names = VOLATILE_STRATA
        major_analysis_count = int(valid_rows.sum())
    probability = np.linspace(0.001, 0.999, args.quantile_knots)
    arrays = _fit_model(
        values,
        limits,
        sample_group,
        probability,
        args.min_reports,
        args.max_correlation_rows,
        args.min_pairwise_overlap,
        args.correlation_shrinkage,
        args.seed,
        stratum_names=stratum_names,
    )
    arrays["composition_mode"] = np.asarray(args.composition_mode)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_model, **arrays)

    group_counts = {
        name: int(np.sum(sample_group == i)) for i, name in enumerate(stratum_names)
    }
    if "unclassified" in stratum_names:
        known = sample_group != stratum_names.index("unclassified")
    else:
        known = np.ones(sample_group.size, dtype=bool)
    known_group_count = int(known.sum())
    igneous_count = sum(
        value for name, value in group_counts.items() if name.startswith("igneous_")
    )
    paper_source = None
    if args.paper and args.paper.exists():
        paper_source = {
            "input_filename": args.paper.name,
            "sha256": _hash_file(args.paper, "sha256"),
        }
    manifest = {
        "schema": WHOLE_ROCK_MODEL_SCHEMA,
        "paper": {
            "citation": "Gard, Hasterok, and Halpin (2019), Earth Syst. Sci. Data 11, 1553-1566",
            "doi": PAPER_DOI,
            "local_source": paper_source,
            "database_sample_count_reported": 1_022_092,
        },
        "data_release": {"doi": SOURCE_DOI, "files": source_hashes},
        "fit": {
            "composition_mode": args.composition_mode,
            "sample_rows": int(values.shape[0]),
            "valid_major_rows_85_to_120_wt_percent": major_analysis_count,
            "stratum_counts": group_counts,
            "known_group_count": known_group_count,
            "igneous_fraction_among_known": (
                igneous_count / known_group_count
                if args.composition_mode == "anhydrous" and known_group_count
                else None
            ),
            "quantile_knots": int(args.quantile_knots),
            "minimum_reports_per_marginal": int(args.min_reports),
            "maximum_rows_per_correlation_fit": int(args.max_correlation_rows),
            "minimum_pairwise_overlap": int(args.min_pairwise_overlap),
            "correlation_shrinkage_pseudocount": float(args.correlation_shrinkage),
            "seed": int(args.seed),
            "major_qc": (
                "positive LOI-free oxide sum in [85, 120] wt%, normalized to 100"
                if args.composition_mode == "anhydrous" else
                "directly speciated major+volatile total in [85, 120] wt%, normalized to 100"
            ),
            "negative_values": "left-censored; absolute value is the reported detection limit",
            "missing_values": "unknown/unreported, never converted to chemical zero",
            "iron": "FeO total preferred, then Fe2O3 total converted to FeO equivalent, then FeO+Fe2O3",
            "joint_model": "empirical log10(ppm) quantiles plus pairwise-complete shrinkage Gaussian copula",
        },
        "training": {
            "fixed_output_positions": 92,
            "excluded_elements": list(DEFAULT_TRAINING_EXCLUSIONS),
            "stratum_default": "balanced (corpus-weighted is available explicitly)",
            "bulk_to_plasma_transfer": "not calibrated; composition supplies a nuclei prior only",
            "ion_stage_model": "independent I-III fractions; no Saha coupling",
        },
        "biases": [
            "The compilation is a literature/database aggregate, not a probability sample of Earth's rocks.",
            "North America, Canada, Australia, and New Zealand are overrepresented; Africa is underrepresented.",
            "Igneous and especially volcanic samples are overrepresented.",
            "Analytical methods and detection limits are heterogeneous, and some duplicate analyses were merged upstream.",
            "Reporting rate is not elemental absence probability.",
            "The LOI-free major QC excludes many volatile-rich/carbonate compositions from complete-composition fitting.",
        ],
    }
    if args.composition_mode == "carbonate_volatile":
        manifest["fit"].update({
            "carbonate_co2_equivalent_threshold_wt_percent": float(
                args.carbonate_co2_threshold_wt
            ),
            "noncarbonate_speciated_volatile_threshold_wt_percent": float(
                args.volatile_threshold_wt
            ),
            "volatile_selection_counts": volatile_stats,
            "speciation_rule": (
                "H2O-total preferred over H2O+ plus H2O-; carbonate analytes "
                "replace CaO/MgO and separately exported CO2; LOI never assigned"
            ),
        })
        manifest["biases"][-1] = (
            "LOI-only rows are excluded because carbon, hydrogen, sulfur, and oxygen cannot be recovered from unspeciated mass loss."
        )
    model = WholeRockCompositionModel(args.out_model)
    nq_probability, nq = _summary_nuclei_quantiles(
        model, args.summary_draws, args.seed ^ 0x5A17
    )
    _write_ranges(args.out_ranges, arrays, nq_probability, nq)
    manifest["artifacts"] = {
        "model": {
            "filename": args.out_model.name,
            "sha256": _hash_file(args.out_model, "sha256"),
        },
        "element_ranges": {
            "filename": args.out_ranges.name,
            "sha256": _hash_file(args.out_ranges, "sha256"),
        },
    }
    args.out_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--major", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--rockgroup", type=Path, required=True)
    parser.add_argument("--paper", type=Path, default=Path("essd-11-1553-2019.pdf"))
    parser.add_argument(
        "--composition-mode",
        choices=("anhydrous", "carbonate_volatile"),
        default="anhydrous",
    )
    parser.add_argument("--carbonate-co2-threshold-wt", type=float, default=20.0)
    parser.add_argument("--volatile-threshold-wt", type=float, default=5.0)
    parser.add_argument(
        "--out-model", type=Path
    )
    parser.add_argument(
        "--out-ranges", type=Path
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
    )
    parser.add_argument("--quantile-knots", type=int, default=101)
    parser.add_argument("--min-reports", type=int, default=100)
    parser.add_argument("--max-correlation-rows", type=int, default=100_000)
    parser.add_argument("--min-pairwise-overlap", type=int, default=200)
    parser.add_argument("--correlation-shrinkage", type=float, default=500.0)
    parser.add_argument("--summary-draws", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20191553)
    args = parser.parse_args(argv)
    if args.composition_mode == "anhydrous":
        args.out_model = args.out_model or Path("db/whole_rock_prior_v1.npz")
        args.out_ranges = args.out_ranges or Path("db/whole_rock_element_ranges.csv")
        args.out_manifest = args.out_manifest or Path(
            "db/whole_rock_prior_v1_manifest.json"
        )
    else:
        args.out_model = args.out_model or Path(
            "db/whole_rock_carbonate_volatile_prior_v1.npz"
        )
        args.out_ranges = args.out_ranges or Path(
            "db/whole_rock_carbonate_volatile_ranges.csv"
        )
        args.out_manifest = args.out_manifest or Path(
            "db/whole_rock_carbonate_volatile_prior_v1_manifest.json"
        )
    if args.quantile_knots < 11:
        parser.error("--quantile-knots must be at least 11")
    if args.min_reports < 2 or args.min_pairwise_overlap < 2:
        parser.error("report/overlap thresholds must be at least two")
    if args.summary_draws < 100:
        parser.error("--summary-draws must be at least 100")
    if args.volatile_threshold_wt <= 0 or args.carbonate_co2_threshold_wt <= 0:
        parser.error("volatile thresholds must be positive")
    return args


def main(argv=None):
    manifest = build(parse_args(argv))
    print(json.dumps({
        "schema": manifest["schema"],
        "sample_rows": manifest["fit"]["sample_rows"],
        "valid_major_rows": manifest["fit"]["valid_major_rows_85_to_120_wt_percent"],
    }, indent=2))


if __name__ == "__main__":
    main()
