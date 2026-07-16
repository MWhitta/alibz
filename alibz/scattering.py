"""Natural-isotope X-ray and neutron contrast screens for LIBS results.

These are deliberately element-level *contrast proxies*, not phase structure
factors or scattering-length densities.  LIBS emitter fractions lack density,
crystal structure, and an absolute bulk calibration.  The functions here keep
the physical mechanisms separate so a convenient scalar cannot be mistaken for
a diffraction refinement.
"""

from __future__ import annotations

import csv
from typing import Iterable, Sequence

import numpy as np
import periodictable as pt


REFERENCE_NEUTRON_WAVELENGTH_A = 1.798
DEFAULT_XRAY_Q_INV_A = (0.0, 2.0, 4.0, 6.0)


def natural_element_properties(element: str,
                               q_values: Sequence[float] =
                               DEFAULT_XRAY_Q_INV_A,
                               neutron_wavelength_a: float =
                               REFERENCE_NEUTRON_WAVELENGTH_A) -> dict:
    """Return natural-isotope atomic and scattering properties.

    Neutron absorption tabulations use the 2200 m/s reference wavelength
    (1.798 Angstrom).  The returned absorption is scaled by wavelength using
    the usual 1/v approximation and is explicitly labeled as such.
    """
    atom = getattr(pt, element, None)
    if atom is None or not getattr(atom, "number", None):
        raise KeyError(f"unknown element: {element}")
    q_values = tuple(float(q) for q in q_values)
    f0 = {}
    for q in q_values:
        value = float(atom.xray.f0(q))
        f0[q] = value if np.isfinite(value) else None

    neutron = atom.neutron
    b_c = getattr(neutron, "b_c", None)
    coherent = getattr(neutron, "coherent", None)
    incoherent = getattr(neutron, "incoherent", None)
    absorption_ref = getattr(neutron, "absorption", None)
    scale = float(neutron_wavelength_a) / REFERENCE_NEUTRON_WAVELENGTH_A
    absorption = (None if absorption_ref is None
                  else float(absorption_ref) * scale)
    return {
        "element": element,
        "Z": int(atom.number),
        "xray_f0": f0,
        "neutron_b_c_fm": None if b_c is None else float(b_c),
        "neutron_coherent_b": (None if coherent is None
                               else float(coherent)),
        "neutron_incoherent_b": (None if incoherent is None
                                 else float(incoherent)),
        "neutron_absorption_b": absorption,
        "neutron_absorption_reference_b": (None if absorption_ref is None
                                            else float(absorption_ref)),
        "neutron_wavelength_a": float(neutron_wavelength_a),
    }


def _resolved_fractions(row: dict) -> tuple[dict, dict]:
    fractions = dict(row.get("fractions") or {})
    statuses = {}
    for det in row.get("detections") or []:
        element = det.get("element")
        if not element:
            continue
        statuses[element] = det.get("status", "")
        value = det.get("fraction_resolved")
        if value is not None:
            fractions[element] = float(value)
    return fractions, statuses


def scattering_contributions(rows: Iterable[dict],
                             q_values: Sequence[float] =
                             DEFAULT_XRAY_Q_INV_A,
                             neutron_wavelength_a: float =
                             REFERENCE_NEUTRON_WAVELENGTH_A) -> list[dict]:
    """Build long-format per-sample, per-element contrast contributions."""
    q_values = tuple(float(q) for q in q_values)
    out = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        fractions, statuses = _resolved_fractions(row)
        for element, fraction in fractions.items():
            fraction = float(fraction)
            if not np.isfinite(fraction) or fraction <= 0:
                continue
            try:
                prop = natural_element_properties(
                    element, q_values=q_values,
                    neutron_wavelength_a=neutron_wavelength_a)
            except KeyError:
                continue
            rec = {
                "test_id": row.get("test_id", ""),
                "height_um": row.get("height_um", ""),
                "file": row.get("file", ""),
                "sample": row.get("sample", ""),
                "element": element,
                "detection_status": statuses.get(element, ""),
                "point_qc_status": row.get("qc_status", ""),
                "fraction_resolved": fraction,
                "Z": prop["Z"],
                "transition_metal": int(
                    (21 <= prop["Z"] <= 30)
                    or (39 <= prop["Z"] <= 48)
                    or (72 <= prop["Z"] <= 80)),
                "neutron_b_c_fm": prop["neutron_b_c_fm"],
                "neutron_coherent_b": prop["neutron_coherent_b"],
                "neutron_incoherent_b": prop["neutron_incoherent_b"],
                "neutron_absorption_b": prop["neutron_absorption_b"],
                "neutron_wavelength_a": prop["neutron_wavelength_a"],
            }
            rec["primary_eligible"] = int(
                rec["detection_status"] == "detected"
                and rec["point_qc_status"] != "fail")
            for q, f0 in prop["xray_f0"].items():
                label = f"{q:g}".replace(".", "p")
                rec[f"xray_f0_q{label}"] = f0
                rec[f"xray_power_q{label}"] = (
                    None if f0 is None else fraction * f0 * f0)
            b_c = prop["neutron_b_c_fm"]
            rec["neutron_signed_amplitude"] = (
                None if b_c is None else fraction * b_c)
            for source, target in (
                ("neutron_coherent_b", "neutron_coherent_contribution"),
                ("neutron_incoherent_b", "neutron_incoherent_contribution"),
                ("neutron_absorption_b", "neutron_absorption_contribution"),
            ):
                value = prop[source]
                rec[target] = None if value is None else fraction * value
            out.append(rec)
    return out


def write_scattering_csv(records: Sequence[dict], path: str) -> None:
    """Write contrast records without adding a pandas/Parquet dependency."""
    base = [
        "test_id", "height_um", "file", "sample", "element",
        "detection_status", "point_qc_status", "primary_eligible",
        "fraction_resolved", "Z", "transition_metal",
    ]
    extra = sorted({key for row in records for key in row} - set(base))
    header = base + extra
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
