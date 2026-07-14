"""Visual adjudication of suspect LIBS composition results.

For each flagged spectrum (bad T, poor r2, or a geochemically implausible
dominant element) re-run the pipeline and show WHERE the suspect element's
signal comes from: its supporting peaks on the full spectrum, then zooms on
the strongest drivers with the database lines of the suspect element AND its
rivals overlaid — so a human can judge whether the dominance is real or a
misassignment.  Writes one PNG per spectrum.

Usage: point FLAGGED_JSON at a list of
    {"path", "suspect_element", "fraction", "T", "r2", "reasons"}
(see the companion flagging step) and run with the repo venv from the repo
root.  Output PNGs go to OUT_DIR.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = "/Users/mwhittaker/Projects/github/alibz"
os.chdir(REPO)
sys.path.insert(0, REPO)

from alibz.pipeline import analyze_spectrum, load_spectrum_csv       # noqa: E402
from alibz.utils.database import Database                            # noqa: E402
from alibz.utils.voigt import multi_voigt                           # noqa: E402
from alibz.utils.wavelength import shift_at                         # noqa: E402
from alibz.elements import element_color                            # noqa: E402
from alibz.utils.colors import spectral_line                       # noqa: E402

FLAGGED_JSON = os.environ.get(
    "FLAGGED_JSON",
    "/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/"
    "46877cda-9c25-4a24-9d3d-3c6563cfbe02/scratchpad/flagged.json")
OUT_DIR = os.environ.get(
    "OUT_DIR",
    "/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/"
    "46877cda-9c25-4a24-9d3d-3c6563cfbe02/scratchpad/adjudication")
os.makedirs(OUT_DIR, exist_ok=True)

_db = Database("db")


def _db_lines(el, lo, hi, shift, min_gA=3e5):
    arr = _db.lines(el)
    if arr.size == 0:
        return []
    ion = arr[:, 0].astype(float)
    wl = arr[:, 1].astype(float)
    gA = arr[:, 3].astype(float)
    wl_i = wl + np.array([float(shift_at(shift, w)) for w in wl])
    m = (wl_i >= lo) & (wl_i <= hi) & (gA >= min_gA) & (ion <= 2)
    return sorted(zip(wl_i[m], ion[m].astype(int), gA[m]),
                  key=lambda t: -t[2])


def adjudicate(entry, idx):
    x, y = load_spectrum_csv(entry["path"])
    a = analyze_spectrum(x, y, "db")
    res = a["result"]
    final = a["final"]
    shift = a["shift"]
    sus = entry["suspect_element"]
    bg = np.asarray(final.get("background", np.zeros_like(y)), dtype=float)
    ybg = y - bg
    pk = np.atleast_2d(final["sorted_parameter_array"])
    model = multi_voigt(x, np.ravel(pk[:, :4]))

    support = sorted(a.get("support", {}).get(sus, []), reverse=True)
    det = next((d for d in a["detections"] if d["element"] == sus), {})
    conf = det.get("confounder")
    fr = {e: f for e, f in res.element_fractions.items() if f > 0}
    rivals = [e for e in sorted(fr, key=fr.get, reverse=True)
              if e != sus][:4]

    drivers = support[:3]                      # strongest suspect peaks
    n_zoom = max(len(drivers), 1)
    fig = plt.figure(figsize=(15, 8.5))
    gs = GridSpec(2, max(n_zoom, 3), height_ratios=[2.0, 1.7], hspace=0.32,
                  wspace=0.22)

    # --- full spectrum -----------------------------------------------------
    ax0 = fig.add_subplot(gs[0, :])
    spectral_line(ax0, x, np.maximum(ybg, 1e-1), lw=0.5, label="data - bg")
    ax0.plot(x, np.maximum(model, 1e-1), color="0.35", lw=0.5, label="model")
    for c, wl, amp in support:
        ax0.axvline(wl, color=element_color(sus), lw=0.8, alpha=0.55)
    for c, wl, amp in drivers:
        ax0.annotate(f"{wl:.1f}", (wl, max(amp, 1.0)), fontsize=8,
                     rotation=90, ha="right", va="bottom",
                     color=element_color(sus))
    ax0.set_yscale("log")
    ax0.set_ylim(bottom=1.0)
    ax0.set_xlabel("wavelength [nm]")
    ax0.set_ylabel("counts")
    stat = det.get("status", "?")
    sd = res.stage_disagreement.get(sus, float("nan"))
    ax0.set_title(
        f"{entry['sample']}\n"
        f"SUSPECT {sus} = {entry['fraction']:.2f}  "
        f"(z={det.get('z','?')}, {stat}, {det.get('n_lines','?')} lines"
        + (f", confounder {conf}" if conf else "")
        + f")   |   T={res.temperature:.0f} K  r2={res.r_squared:.3f}"
        f"  stage_disagreement={sd:.2f}\n"
        f"flags: {', '.join(entry['reasons'])}",
        fontsize=10, loc="left")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.text(0.005, 0.97,
             f"{sus} supporting peaks (contribution): "
             + ", ".join(f"{wl:.1f}({c:.0f})" for c, wl, amp in support[:8]),
             transform=ax0.transAxes, fontsize=7.5, va="top",
             color=element_color(sus))

    # --- zoom on each strongest driver ------------------------------------
    for j, (c, wl0, amp) in enumerate(drivers):
        ax = fig.add_subplot(gs[1, j])
        lo, hi = wl0 - 0.9, wl0 + 0.9
        m = (x >= lo) & (x <= hi)
        ax.plot(x[m], ybg[m], "k-", lw=1.0, label="data - bg")
        ax.plot(x[m], model[m], color="0.4", lw=1.0, label="model")
        # suspect element db lines (the claimed identity)
        for w, ion, gA in _db_lines(sus, lo, hi, shift)[:6]:
            ax.axvline(w, color=element_color(sus), ls="--", lw=1.3)
            ax.annotate(f"{sus}{'I'*ion}", (w, ax.get_ylim()[1]), fontsize=7,
                        rotation=90, ha="center", va="top",
                        color=element_color(sus))
        # confounder + rival db lines (could a rival explain this peak?)
        for el, style in ([(conf, ":")] if conf else []) + \
                [(r, ":") for r in rivals if r != conf]:
            for w, ion, gA in _db_lines(el, lo, hi, shift)[:4]:
                ax.axvline(w, color=element_color(el), ls=style, lw=1.0,
                           alpha=0.7)
                ax.annotate(f"{el}{'I'*ion}", (w, ax.get_ylim()[1] * 0.55),
                            fontsize=6, rotation=90, ha="center", va="top",
                            color=element_color(el), alpha=0.8)
        ax.axvline(wl0, color=element_color(sus), lw=0.6, alpha=0.3)
        ax.set_title(f"driver {wl0:.2f} nm  (contrib {c:.0f})", fontsize=9)
        ax.set_xlabel("nm")
        if j == 0:
            ax.legend(fontsize=7, loc="upper right")
    if not drivers:
        ax = fig.add_subplot(gs[1, :])
        ax.text(0.5, 0.5, f"no fitted peaks dominated by {sus}",
                ha="center", va="center")
        ax.axis("off")

    out = os.path.join(OUT_DIR, f"{idx:02d}_{sus}_"
                       + entry["sample"][:26].replace(" ", "_").replace("/", "-")
                       + ".png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out, dict(sus=sus, frac=entry["fraction"], z=det.get("z"),
                     status=det.get("status"), conf=conf,
                     n_lines=det.get("n_lines"),
                     T=round(float(res.temperature)),
                     r2=round(float(res.r_squared), 3),
                     drivers=[(round(wl, 2), round(c)) for c, wl, _ in drivers])


def main():
    flagged = [e for e in json.load(open(FLAGGED_JSON)) if e.get("path")]
    print(f"{len(flagged)} flagged spectra -> {OUT_DIR}", flush=True)
    manifest = []
    for i, e in enumerate(flagged, 1):
        try:
            out, meta = adjudicate(e, i)
            manifest.append(dict(png=out, **meta, sample=e["sample"]))
            print(f"[{i}/{len(flagged)}] {e['sample'][:34]}: "
                  f"{meta['sus']} {meta['frac']} z={meta['z']} "
                  f"conf={meta['conf']} -> {os.path.basename(out)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[{i}] {e['sample'][:34]}: FAILED {exc}", flush=True)
    json.dump(manifest, open(os.path.join(OUT_DIR, "manifest.json"), "w"),
              indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
