"""Render the measured benefit/cost and potassium-ratio counterexample."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path(
        "reports/physical-triage-real-benchmark-20260922.json"))
    parser.add_argument("--out", type=Path, default=Path(
        "reports/figures/physical-triage-20260922.png"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = json.loads(args.benchmark.read_text())
    samples = report["samples"]
    k_source = Path(report["k_area_counterexample"]["path"])
    if not samples or not k_source.is_file():
        raise ValueError("benchmark samples and measured potassium table are required")
    if args.dry_run:
        print(f"{len(samples)} real run means; output {args.out}")
        return 0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    colors = ("#457b9d", "#e09f3e")
    for ax, key, title, unit in (
        (axes[0], "species", "Final candidate pool is unchanged", "Species after existing filters"),
        (axes[1], "runtime_s", "Extra triage adds computation", "Candidate construction (seconds)"),
    ):
        groups = [np.array([s["candidate_build"][mode][key] for s in samples])
                  for mode in ("off", "prune")]
        medians = [np.median(v) for v in groups]
        ax.bar([0, 1], medians, color=colors, width=.55)
        for i, values in enumerate(groups):
            ax.scatter(i + np.linspace(-.12, .12, len(values)), values,
                       color="#293241", s=10, alpha=.55)
        ax.set_xticks([0, 1], ["Existing", "+ triage"])
        ax.set_ylabel(unit)
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
    measurements = {}
    with k_source.open() as handle:
        for row in csv.DictReader(handle):
            measurements.setdefault(row["test_id"], {})[row["line_id"]] = row
    ratios = []
    for rows in measurements.values():
        pair = [rows["K_I_766.4899"], rows["K_I_769.8964"]]
        if all(float(r["snr"] or 0) >= 5 and float(r["area"] or 0) > 0 for r in pair):
            ratios.append(float(pair[0]["area"]) / float(pair[1]["area"]))
    ax = axes[2]
    ax.hist(ratios, bins=28, color=colors[0])
    thin = report["k_area_counterexample"]["thin_ratio"]
    ax.axvspan(thin * .8, thin * 1.2, color=colors[1], alpha=.2,
               label="Thin ratio ±20%")
    ax.axvline(thin, color=colors[1], linestyle="--")
    ax.set_title("A strict thin-ratio veto loses K evidence", fontsize=11)
    ax.set_xlabel("K I integrated-area ratio, 766 / 769 nm")
    ax.set_ylabel("Measured spectra")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Physical triage: {len(samples)} Fe run means and {len(ratios)} K doublets",
                 fontsize=13)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
