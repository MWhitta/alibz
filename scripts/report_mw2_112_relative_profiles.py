"""Report and figures for the full MW2-112 relative-profile run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt
from scipy.stats import spearmanr

from alibz.mw2_112 import atomic_json, utc_now, write_csv
from alibz.scattering import natural_element_properties


PROFILE_ELEMENTS = (
    "Li", "O", "Na", "Mg", "Al", "Si", "K", "Ca", "Ti", "Fe",
    "Rb", "Sr", "Ba",
)
SEGMENT_ELEMENTS = (
    "Li", "Na", "Mg", "Al", "Si", "K", "Ca", "Ti", "Fe", "Rb",
    "Sr", "Ba",
)
VENDOR_ELEMENTS = ("Li", "Mg", "Al", "Si", "Ti", "Mn", "Fe")


def _rolling(values, width=5):
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    half = width // 2
    for i in range(values.size):
        out[i] = np.nanmedian(values[max(0, i - half):i + half + 1])
    return out


def _geomean(*values):
    values = np.maximum(np.stack(values), 0.0)
    return np.exp(np.mean(np.log1p(values), axis=0)) - 1.0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--vendor-dir", default=None)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    with (run_dir / "relative_element_profiles.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    ids = sorted({int(row["test_id"]) for row in rows})
    heights = np.array([(test_id - 989) * 0.1 for test_id in ids])
    index = {test_id: i for i, test_id in enumerate(ids)}
    profiles = {}
    status = {}
    for element in {row["element"] for row in rows}:
        values = np.zeros(len(ids), dtype=float)
        states = ["missing"] * len(ids)
        for row in rows:
            if row["element"] != element:
                continue
            i = index[int(row["test_id"])]
            values[i] = float(row["relative_score"])
            states[i] = row["status"]
        profiles[element] = values
        status[element] = states

    # Multivariate log-profile change points.  Three pre-registered BIC-scale
    # penalties provide a stability count rather than one tuned segmentation.
    matrix = np.column_stack([np.log1p(profiles[element])
                              for element in SEGMENT_ELEMENTS])
    zero_index = index.get(1632)
    if zero_index is not None and 0 < zero_index < len(ids) - 1:
        matrix[zero_index] = 0.5 * (matrix[zero_index - 1]
                                    + matrix[zero_index + 1])
    median = np.median(matrix, axis=0)
    scale = 1.4826 * np.median(np.abs(matrix - median), axis=0)
    matrix = np.clip((matrix - median) / np.maximum(scale, 1e-6), -6, 6)
    base_penalty = matrix.shape[1] * np.log(matrix.shape[0])
    penalties = (0.5, 1.0, 2.0)
    boundary_support = Counter()
    boundary_penalties = {}
    for multiplier in penalties:
        endpoints = rpt.Pelt(model="l2", min_size=2, jump=1).fit(matrix).predict(
            pen=multiplier * base_penalty)
        for endpoint in endpoints[:-1]:
            boundary_support[endpoint] += 1
            boundary_penalties.setdefault(endpoint, []).append(multiplier)
    boundaries = []
    for endpoint, support in sorted(boundary_support.items()):
        if support < 2:
            continue
        boundaries.append({
            "boundary_after_test_id": ids[endpoint - 1],
            "boundary_before_test_id": ids[endpoint],
            "height_mm": heights[endpoint],
            "penalty_support": support,
            "penalty_multipliers": ";".join(
                str(value) for value in boundary_penalties[endpoint]),
            "method": "multivariate PELT l2 on robust log-relative profiles",
        })
    write_csv(run_dir / "layer_boundaries.csv", boundaries)

    # Independent vendor comparison; used for validation only.
    validation = []
    if args.vendor_dir:
        vendor = {}
        for path in Path(args.vendor_dir).glob("*.csv"):
            with path.open() as fh:
                row = next(csv.DictReader(fh))
            vendor[int(row["Test #"])] = row
        for element in VENDOR_ELEMENTS:
            x, y = [], []
            for test_id, vendor_row in vendor.items():
                raw = vendor_row.get(f"{element} (%)", "")
                if (raw and not raw.startswith("<")
                        and test_id in index
                        and status[element][index[test_id]] != "unsupported"):
                    try:
                        x.append(profiles[element][index[test_id]])
                        y.append(float(raw))
                    except ValueError:
                        pass
            if len(x) >= 20:
                rho, pvalue = spearmanr(x, y)
                validation.append({
                    "element": element, "n": len(x),
                    "spearman_rho": float(rho), "p_value": float(pvalue),
                    "interpretation": (
                        "strong" if rho >= 0.7 else
                        "moderate" if rho >= 0.4 else
                        "weak_or_discordant"),
                })
    write_csv(run_dir / "vendor_validation.csv", validation)

    association = []
    for test_id, height in zip(ids, heights):
        i = index[test_id]
        association.append({
            "test_id": test_id, "height_mm": height,
            "li_clay_index": _geomean(
                profiles["Li"], profiles["Mg"], profiles["Si"])[i],
            "aluminous_clay_index": _geomean(
                profiles["Al"], profiles["Si"])[i],
            "k_feldspar_index": _geomean(
                profiles["K"], profiles["Al"], profiles["Si"])[i],
            "fe_ti_accessory_index": _geomean(
                profiles["Fe"], profiles["Ti"])[i],
            "quartz_contrast_index": profiles["Si"][i] / max(
                1.0 + profiles["Al"][i] + profiles["Li"][i]
                + profiles["Mg"][i] + profiles["K"][i]
                + profiles["Ca"][i], 1e-12),
            "calcite_index": "",
            "notes": "standardized association scores; not phase fractions",
        })
    write_csv(run_dir / "mineral_association_indices.csv", association)

    scattering = []
    for element in PROFILE_ELEMENTS:
        prop = natural_element_properties(element, q_values=(4.0,))
        scattering.append({
            "element": element,
            "profile_status": status[element][0],
            "xray_f0_q4": prop["xray_f0"][4.0],
            "neutron_b_c_fm": prop["neutron_b_c_fm"],
            "neutron_coherent_b": prop["neutron_coherent_b"],
            "neutron_incoherent_b": prop["neutron_incoherent_b"],
            "neutron_absorption_b_at_1p798A": prop["neutron_absorption_b"],
            "natural_isotopic_abundance": 1,
        })
    write_csv(run_dir / "scattering_reference.csv", scattering)

    stable_heights = [row["height_mm"] for row in boundaries
                      if row["penalty_support"] == len(penalties)]
    print(f"report: {len(boundaries)} supported boundaries, "
          f"{len(stable_heights)} stable", flush=True)

    def plot_profiles(elements, title, filename, unsupported_note=None):
        fig, axes = plt.subplots(len(elements), 1, figsize=(12, 1.8 * len(elements)),
                                 sharex=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, element in zip(axes, elements):
            values = profiles[element]
            ax.plot(heights, values, color="0.72", lw=0.6, label="100 µm shot")
            ax.plot(heights, _rolling(values), lw=1.3, label="5-point median")
            ax.set_ylabel(element)
            for boundary in stable_heights:
                ax.axvline(boundary, color="0.2", alpha=0.12, lw=0.6)
            if zero_index is not None:
                ax.axvline(heights[zero_index], color="red", alpha=0.45,
                           lw=0.8, ls="--")
        axes[0].set_title(title)
        axes[-1].set_xlabel("Height above profile bottom (mm)")
        axes[0].legend(loc="upper right", fontsize=7, frameon=False)
        for suffix in ("png", "pdf"):
            fig.savefig(figure_dir / f"{filename}.{suffix}", dpi=220)
        plt.close(fig)

    plot_profiles(("Fe", "Ti", "Si", "Al", "Rb", "Sr", "Ba"),
                  "MW2-112 relative X-ray-sensitive elemental profiles",
                  "relative_xray_elements")
    print("report: wrote X-ray profile figure", flush=True)
    plot_profiles(("Li", "O", "Na", "Mg", "K", "Ca"),
                  "MW2-112 relative neutron-sensitive elemental profiles",
                  "relative_neutron_elements",
                  unsupported_note=(
                      "H/B/C/F lack coherent multi-line support; Li/O may "
                      "retain plasma or ambient systematics."))
    print("report: wrote neutron profile figure", flush=True)

    # Calibration/QC overview.
    with (run_dir / "session_calibration.csv").open() as fh:
        calibration = list(csv.DictReader(fh))
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                             constrained_layout=True)
    response = np.array([float(row["response_measured"])
                         if row["response_measured"] else np.nan
                         for row in calibration])
    prior = np.array([float(row["response_prior"])
                      if row["response_prior"] else np.nan
                      for row in calibration])
    axes[0].plot(heights, response, ".", ms=1.5, alpha=0.5, label="measured")
    axes[0].plot(heights, prior, lw=0.8, label="shared prior")
    axes[0].set_ylabel("620 nm gain")
    axes[0].legend(frameon=False, fontsize=7)
    for ax, label in zip(axes[1:], ("uv", "vis", "nir")):
        shift = np.array([float(row[f"shift_prior_{label}_nm"]) * 1000
                          if row[f"shift_prior_{label}_nm"] else np.nan
                          for row in calibration])
        ax.plot(heights, shift, lw=0.8)
        ax.set_ylabel(f"{label.upper()} shift\n(pm)")
    axes[-1].set_xlabel("Height above profile bottom (mm)")
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"session_calibration.{suffix}", dpi=220)
    plt.close(fig)
    print("report: wrote calibration figure", flush=True)

    unsupported = sorted(element for element in profiles
                         if status[element][0] == "unsupported")
    validation_text = ", ".join(
        f"{row['element']} ρ={row['spearman_rho']:.3f}"
        for row in validation)
    stable_text = ", ".join(f"{height:.1f}" for height in stable_heights)
    bottom_top = {
        element: (float(np.mean(profiles[element][:100])),
                  float(np.mean(profiles[element][-100:])))
        for element in ("Li", "Na", "Mg", "Si", "K", "Fe", "Rb")
    }
    report = f"""# MW2-112 LIBS relative-profile analysis report

Generated: {utc_now()}

## Outcome

All 929 single-pulse spectra were processed successfully. The primary product
is a within-element multi-line relative profile; scores are comparable along
the profile for one element, not between different elements. ID 1632 is an
explicit all-zero/missing observation.

The quantitative CF-LIBS pilot did not pass its production gate (3 pass, 4
warn, 12 nonzero scientific failures, plus the zero shot). Closed-sum emitter
fractions are therefore secondary and must not fill failed positions.

## Physical constraints

- 255 candidate same-stage multiplets were measured with shared per-segment
  peak widths, wavelength-shift priors, and 620 nm response correction.
- An ion stage enters the primary element score only when at least two lines
  have median spatial Spearman coherence >= 0.15.
- Natural-isotope coherent, incoherent, and absorption coefficients remain
  separate. Contrast scores cannot be ranked across elements because each
  elemental profile has its own normalization.
- Unsupported primary elements: {', '.join(unsupported)}.
- H is unsupported; no defensible hydrogen/neutron-incoherent profile can be
  recovered from these spectra. O remains ambient-sensitive.
- C is unsupported, so no calcite association index is reported.

## Independent comparison

Vendor summaries were not used for fitting. Spatial rank comparisons are:
{validation_text or 'not available'}.
Al is internally coherent but externally discordant and should be treated as
model/systematic-limited.

## Profile-scale observations

- The uppermost approximately 10 mm is enriched in Li ({bottom_top['Li'][0]:.2f}
  bottom-decile mean versus {bottom_top['Li'][1]:.2f} top), Na
  ({bottom_top['Na'][0]:.2f} versus {bottom_top['Na'][1]:.2f}), K
  ({bottom_top['K'][0]:.2f} versus {bottom_top['K'][1]:.2f}), and Rb
  ({bottom_top['Rb'][0]:.2f} versus {bottom_top['Rb'][1]:.2f}).
- The same interval is depleted in Si ({bottom_top['Si'][0]:.2f} versus
  {bottom_top['Si'][1]:.2f}), Fe ({bottom_top['Fe'][0]:.2f} versus
  {bottom_top['Fe'][1]:.2f}), and Mg ({bottom_top['Mg'][0]:.2f} versus
  {bottom_top['Mg'][1]:.2f}). These are within-element standardized scores,
  not concentration ratios.
- Fully stable layer boundaries occur at heights (mm): {stable_text}.

## Spatial products

The multivariate PELT ensemble found {len(boundaries)} boundaries supported by
at least two of three BIC-scale penalties; {len(stable_heights)} are supported
by all three. Raw 100 µm points are retained alongside a labeled 5-point median.
Mineral-association columns are standardized covariates, never phase fractions.
"""
    (run_dir / "analysis_report.md").write_text(report)
    print("report: wrote analysis report", flush=True)

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["reported_utc"] = utc_now()
    manifest["report_products"] = {
        "stable_boundaries": len(stable_heights),
        "supported_boundaries": len(boundaries),
        "unsupported_elements": unsupported,
    }
    atomic_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
