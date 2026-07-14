#!/usr/bin/env python
"""Render a reproducible full-band individual-shot spectrum demonstration."""

import argparse
from pathlib import Path

import numpy as np

from alibz.elements import ATOMIC_NUMBER
from alibz.synthetic import (
    AtomicStrengthUncertainty,
    HierarchicalPlasmaSampler,
    InstrumentResponse,
    SyntheticScene,
    SyntheticSpectrumGenerator,
    dry_air_component,
)
from alibz.synthetic_calibration import IndividualShotCalibration
from alibz.utils.colors import spectral_background, wavelength_to_rgb


DEMO_COMPOSITION = (
    ("Ca", 2, 0.20), ("Al", 1, 0.15), ("Fe", 1, 0.20),
    ("Mg", 1, 0.10), ("Si", 1, 0.10), ("K", 1, 0.05),
    ("Na", 1, 0.05), ("Ti", 1, 0.05), ("Sr", 2, 0.04),
    ("Li", 1, 0.01), ("C", 1, 0.05),
)


def build_scene():
    abundance = np.zeros((92, 3), dtype=float)
    for element, stage, value in DEMO_COMPOSITION:
        abundance[ATOMIC_NUMBER[element] - 1, stage - 1] = value
    sampler = HierarchicalPlasmaSampler()
    target = sampler.component(
        abundance,
        seed=812,
        emission_scale=0.2,
        continuum_scale=300.0,
    )
    gas_base = dry_air_component(
        0.9,
        {
            "N": (0.55, 0.40, 0.05),
            "O": (0.55, 0.40, 0.05),
            "Ar": (0.45, 0.50, 0.05),
        },
        emission_scale=0.03,
        continuum_scale=30.0,
    )
    gas = sampler.component(
        gas_base.stage_abundance,
        seed=813,
        emission_scale=gas_base.emission_scale,
        continuum_scale=gas_base.continuum_scale,
        label=gas_base.label,
    )
    return SyntheticScene(target, gas, seed=814, metadata={"demo": True})


def _style_spectrum_axis(ax, lo, hi, label=None, background_alpha=0.13):
    """Apply the repository-wide wavelength-aware spectral colour scheme."""
    ax.set_xlim(lo, hi)
    ax.set_facecolor("#f8fafc")
    spectral_background(ax, lo=lo, hi=hi, alpha=background_alpha)
    ax.grid(axis="y", color="white", lw=0.65, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#64748b")
    ax.tick_params(axis="x", colors="#334155")
    if label is None:
        ax.spines["left"].set_color("#64748b")
        ax.tick_params(axis="y", colors="#334155")
    else:
        panel_color = wavelength_to_rgb(0.5 * (lo + hi))
        ax.spines["left"].set_color(panel_color)
        ax.spines["left"].set_linewidth(2.0)
        ax.tick_params(axis="y", colors=panel_color)
        ax.yaxis.label.set_color(panel_color)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        default="jobs/synthetic_instrument_mw2_individual.json",
    )
    parser.add_argument("--out-prefix", default="jobs/explicit_synthetic_demo")
    parser.add_argument("--no-noise", action="store_true")
    args = parser.parse_args(argv)

    calibration_path = Path(args.calibration)
    if calibration_path.exists():
        response = IndividualShotCalibration.load(calibration_path).response
    else:
        response = InstrumentResponse.provisional_current_instrument()
    generator = SyntheticSpectrumGenerator(
        "db",
        response,
        atomic_uncertainty=AtomicStrengthUncertainty(enabled=True),
    )
    result = generator.render(build_scene(), add_noise=not args.no_noise)

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    np.savetxt(
        csv_path,
        np.column_stack((result.wavelength_nm, result.intensity_counts)),
        delimiter=",",
        header="wavelength,intensity",
        comments="",
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        4, 1, figsize=(12, 9), constrained_layout=True,
        facecolor="#f8fafc",
    )
    trace_color = "#111827"
    axes[0].plot(
        result.wavelength_nm, result.intensity_counts,
        color=trace_color, lw=0.45,
    )
    axes[0].set_ylabel("counts")
    axes[0].set_title(
        "Explicit-stage synthetic individual shot — full band",
        color="#0f172a", fontweight="semibold",
    )
    _style_spectrum_axis(axes[0], 190, 910, background_alpha=0.15)
    for ax, (lo, hi, label) in zip(
        axes[1:],
        ((190, 365, "UV"), (365, 620, "VIS"), (620, 910, "NIR")),
    ):
        mask = (result.wavelength_nm >= lo) & (result.wavelength_nm < hi)
        ax.plot(
            result.wavelength_nm[mask], result.intensity_counts[mask],
            color=trace_color, lw=0.55,
        )
        ax.set_ylabel(f"{label} counts")
        _style_spectrum_axis(ax, lo, hi, label=label)
    axes[-1].set_xlabel("wavelength (nm)")
    png_path = prefix.with_suffix(".png")
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    values = result.intensity_counts
    print(f"CSV: {csv_path.resolve()}")
    print(f"plot: {png_path.resolve()}")
    print(
        f"channels={values.size}, min={values.min():.3g}, "
        f"median={np.median(values):.3g}, p99={np.quantile(values, 0.99):.3g}, "
        f"max={values.max():.3g}, negative={100 * np.mean(values < 0):.3g}%"
    )
    for warning in result.manifest.get("warnings", []):
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
