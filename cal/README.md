# Calibration campaign: Z300 + alibz on high-purity elemental standards

Goal: calibrate the SciAps Z300 acquisition (delay, period, raster, shot count)
and the alibz analysis pipeline (wavelength registration, line matching,
self-absorption, broadening, plasma T/nₑ) against targets whose composition is
known: pure-metal standards, one element at a time. Each study gets a dated
report in `reports/`, a reproducible script in `scripts/`, and figures in
`figures/`. Instrument facts learned along the way go in
`docs/z300_device_notes.md`, not here.

Scripts compute the repo root as three levels up and default their outputs into
`cal/`. Run them from the repo root with `.venv/bin/python cal/scripts/<name>.py`.

## Studies

| date | element | study | report | script | figures |
|---|---|---|---|---|---|
| 2026-09-21 | Fe (Aesar 99.98%) | Purity checkout, 4x4 grid x 3 reps | `reports/2026-09-21-fe-purity-checkout.md` | `scripts/fe_line_qc.py`, `scripts/fe_grid_qc.py` | — |
| 2026-09-22 | Fe | First raster delay/period study (`opt-b61065d7`) | `reports/2026-09-22-first-raster-study-analysis.md` | — | — |
| 2026-09-22 | Fe | Plasma and surface analysis (T, depth, wavelength shift) | `reports/2026-09-22-fe-calibration-plasma-analysis.md` | `scripts/fe_plasma_analysis.py` | `figures/fe-plasma-20260922/` |
| 2026-09-22 | V | First replicated delay/period study (`opt-9d384296`, V_pure_run1, 5/10/20 x 10/25/50) | `reports/2026-09-22-vanadium-sequence-results.md` | — | — |
| 2026-09-22 | Fe + V | Line matching, self-absorption, broadening | `reports/2026-09-22-line-matching-sa-broadening.md` | `scripts/line_physics_analysis.py` | `figures/line-physics-20260922/` |
| 2026-09-23 | V | Short delay/period timing study (`opt-c86274e9`, V_pure_run2, 1/5/10/20/50 x 1/10/25/50) | `reports/2026-09-23-vanadium-timing-study.md` | `scripts/v_timing_study.py` | `figures/v-timing-20260923/` |

Related work outside `cal/`: wavelength registration
(`reports/2026-09-23-wavelength-registration.md`, `alibz/wavelength_registration.py`)
and the Fe calibration boundary note (`reports/2026-09-23-fe-calibration-boundary.md`).

## Data

Pantheum on `moissanite` holds the per-shot native spectra:
`~/.local/state/pantheum/alibz/acquisitions/<run_id>/shots/shot-N.csv`, with
the session/batch ledger in `~/.local/state/pantheum/alibz/alibz.sqlite`
(`optimization_sessions`, `optimization_batches`, `acquisitions`). Ledgers
extracted for a study go in `cal/data/`.

## Lessons carried between studies

- Use the native grid only (7,914 px, three segments). No resampling.
- Fresh sites and replicates are what make a delay/period grid readable.
  A fixed site order confounds delay with crater depth (one delay row = one
  raster lap). Randomise or interleave condition order across laps.
- Timing is in vendor units whose physical scale has not been verified.
