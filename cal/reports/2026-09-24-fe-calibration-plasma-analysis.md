# Fe (Aesar 99.98%) Z300 LIBS calibration analysis: plasma and surface

*Generated fe_plasma_analysis.py — deterministic rerun of `--data /private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/1c78d231-c01b-47c0-8606-6590e292a2c0/scratchpad/corrected/acq` `--ledger /private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/1c78d231-c01b-47c0-8606-6590e292a2c0/scratchpad/ledger_fe.json`.*

## Data

- 26 succeeded live runs, Fe Aesar 99.98%, argon pre-flush 300, gated, 10 shots requested/run, 1 location/run, no cleaning shots.
- Total shots on disk: 252 across 26 runs.
- **Corrected wavelength axis (2026-09-24).** Every run was re-derived in place with pantheum z300_calibration PIXEL_OFFSET -18 (see reports/2026-09-23-native-axis-18px-offset.md); the earlier ~18 px too-blue labelling is removed. All 26 runs are now on a SINGLE native grid — no grid-family branching, no resampling (measured on each run's own grid).
  - **26 runs, 5848 pts** — native, PIXEL_OFFSET -18 corrected; 187.79-950.71 nm, pitch 0.089/0.129/0.179 nm UV/VIS/NIR. The NIR tail beyond 950.7 nm is absent on this decode (no interior gap). Residual global shift -22 pm (was -154 pm before the fix).
- Segments: UV [186,365), VIS [365,620), NIR [620,951] nm.

### Run groups (chronological index; d=delay p=period pp=pulsePeriod ms)

| # | time | location | d | p | pp | shots | grid | group |
|--|--|--|--|--|--|--|--|--|
| 0 | 15:58:34 | 134,76,70 | 10 | 25 | 100 | 10 | 5848 | reused-spot depth |
| 1 | 16:27:10 | 134,76,70 | 10 | 25 | 100 | 10 | 5848 | reused-spot depth |
| 2 | 16:31:42 | 134,76,70 | 10 | 25 | 100 | 10 | 5848 | reused-spot depth |
| 3 | 16:41:37 | 134,76,70 | 10 | 25 | 100 | 10 | 5848 | reused-spot depth |
| 4 | 16:42:20 | 134,76,70 | 5 | 25 | 100 | 10 | 5848 | reused-spot depth |
| 5 | 16:42:52 | 134,76,70 | 20 | 25 | 100 | 10 | 5848 | reused-spot depth |
| 6 | 16:43:43 | 134,76,70 | 10 | 10 | 100 | 10 | 5848 | reused-spot depth |
| 7 | 16:44:22 | 134,76,70 | 10 | 50 | 100 | 10 | 5848 | reused-spot depth |
| 8 | 18:12:01 | 134,76,70 | 10 | 25 | 100 | 10 | 5848 | reused-spot depth |
| 9 | 18:13:50 | 134,76,70 | 5 | 25 | 100 | 10 | 5848 | reused-spot depth |
| 10 | 18:14:14 | 134,76,70 | 5 | 10 | 100 | 10 | 5848 | reused-spot depth |
| 11 | 18:14:34 | 134,76,70 | 10 | 10 | 100 | 10 | 5848 | reused-spot depth |
| 12 | 18:14:54 | 134,76,70 | 5 | 50 | 100 | 10 | 5848 | reused-spot depth |
| 13 | 19:17:39 | 134,76,70 | 10 | 25 | 100 | 8 | 5848 | reused-spot depth |
| 14 | 19:33:13 | 134,76,70 | 10 | 25 | 100 | 10 | 5848 | reused-spot depth |
| 15 | 19:33:37 | 134,76,70 | 5 | 25 | 100 | 9 | 5848 | reused-spot depth |
| 16 | 19:38:47 | 134,76,70 | 10 | 25 | 100 | 9 | 5848 | reused-spot depth |
| 17 | 20:43:46 | 158,76,70 | 5 | 25 | 100 | 10 | 5848 | raster fresh |
| 18 | 20:44:27 | 158,100,70 | 5 | 10 | 100 | 10 | 5848 | raster fresh |
| 19 | 20:45:08 | 134,100,70 | 10 | 10 | 100 | 9 | 5848 | raster fresh |
| 20 | 20:45:51 | 134,124,70 | 5 | 50 | 100 | 10 | 5848 | raster fresh |
| 21 | 20:46:27 | 158,124,70 | 10 | 50 | 100 | 10 | 5848 | raster fresh |
| 22 | 20:55:25 | 134,76,70 | 20 | 10 | 1000 | 9 | 5848 | reused-spot (later return) |
| 23 | 20:56:12 | 134,96,70 | 20 | 50 | 1000 | 8 | 5848 | verification [134,96,70] |
| 24 | 20:58:45 | 134,96,70 | 20 | 50 | 100 | 10 | 5848 | verification [134,96,70] |
| 25 | 20:59:28 | 134,96,70 | 20 | 10 | 100 | 10 | 5848 | verification [134,96,70] |

Reused spot [134,76,70]: 17 runs (idx 0-16), 166 cumulative shots for the depth series; run 22 (20/10, pp1000) is an 18th run recorded at the same location and is treated as a later return.
Verification spot [134,96,70]: 3 runs, 28 shots.

## Wavelength shift model

Estimated on the single corrected native grid from isolated, strong, unambiguous anchor lines (isolated Fe I/II below 620 nm; isolated Ar I lines for NIR, where Fe has no lines) measured by sub-pixel parabolic centroid in the grand-mean spectrum. Convention: observed = db(air) + shift.

| grid | provenance | n anchors | global shift (pm) | UV | VIS | NIR |
|--|--|--|--|--|--|--|
| 5848 | native, PIXEL_OFFSET -18 corrected (2026-09-24) | 19 | -22 | -18 | -30 | -19 |

After the PIXEL_OFFSET -18 fix the residual global shift is **-22 pm** (UV -18, VIS -30, NIR -19 pm) from 19 anchors — tens of pm, i.e. a fraction of a native pitch (axis_check ~= -0.25 px), versus the -154 pm of the superseded -18 px analysis. A single global shift is now adequate; the per-segment terms are within the ~18 pm post-fit residual set by the native pitch, so line identifications no longer depend on a large modelled shift.

Per-anchor matched offsets, corrected native grid:

| species | obs nm | db nm | seg | offset (pm) |
|--|--|--|--|--|
| Fe2 | 233.257 | 233.280 | UV | -22 |
| Fe2 | 234.360 | 234.349 | UV | +11 |
| Fe2 | 238.201 | 238.204 | UV | -3 |
| Fe2 | 239.529 | 239.562 | UV | -34 |
| Fe1 | 248.295 | 248.327 | UV | -32 |
| Fe1 | 278.781 | 278.810 | UV | -30 |
| Fe1 | 373.440 | 373.486 | VIS | -46 |
| Fe1 | 538.307 | 538.337 | VIS | -30 |
| Fe1 | 561.540 | 561.564 | VIS | -24 |
| ArI | 696.546 | 696.543 | NIR | +3 |
| ArI | 706.679 | 706.722 | NIR | -42 |
| ArI | 738.407 | 738.398 | NIR | +9 |
| ArI | 750.377 | 750.387 | NIR | -10 |
| ArI | 763.506 | 763.511 | NIR | -4 |
| ArI | 772.354 | 772.376 | NIR | -22 |
| ArI | 794.759 | 794.818 | NIR | -59 |
| ArI | 811.486 | 811.531 | NIR | -45 |
| ArI | 826.454 | 826.452 | NIR | +2 |
| ArI | 842.443 | 842.465 | NIR | -21 |

## 1. Non-Fe elements present

Measured on the reused-spot run-0 mean (native, 10/25) and cross-checked on the fresh-spot first-shot mean. A candidate is a genuine detection only when net area > 0, SNR>=3, the sub-pixel residual to the shifted db line is < 0.15 nm, AND no Fe I/II db line of comparable strength lies within 0.10 nm (else it is flagged **Fe-blend**, since Fe's line forest can mimic almost any position). Residual = sub-pixel obs peak - shifted db.

| element | line (nm, air) | class | net area | SNR | resid (pm) | Fe within 0.15nm? | confidence |
|--|--|--|--|--|--|--|--|
| Ar I | 696.54 | ambient | 139 | 9.4 | +67 | no | high |
| Ar I | 706.72 | ambient | 118 | 6.8 | -34 | no | medium |
| Ar I | 738.40 | ambient | 230 | 18.2 | +59 | no | high |
| Ar I | 750.39 | ambient | 0 | 0.0 | - | no | not detected |
| Ar I | 763.51 | ambient | 405 | 39.5 | +69 | no | high |
| Ar I | 772.38 | ambient | 152 | 5.8 | -33 | no | medium |
| Ar I | 794.82 | ambient | 130 | 4.6 | -62 | no | medium |
| Ar I | 801.48 | ambient | 51 | 1.8 | -104 | no | not detected |
| Ar I | 811.53 | ambient | 435 | 21.9 | -20 | no | high |
| Ar I | 826.45 | ambient | 71 | 14.2 | -338 | no | high |
| Ar I | 842.46 | ambient | 151 | 13.7 | +22 | no | high |
| H I | 656.28 | ambient | 338 | 40.1 | +10 | no | high |
| O I | 777.19 | ambient | 305 | 8.9 | +248 | no | high |
| O I | 777.42 | ambient | 305 | 8.9 | +18 | no | high |
| O I | 777.54 | ambient | 305 | 8.9 | -102 | no | high |
| N I | 742.36 | ambient | 23 | 9.0 | -4 | no | high |
| N I | 744.23 | ambient | 82 | 9.0 | +167 | no | high |
| N I | 746.83 | ambient | 11 | 9.2 | -23 | no | high |
| C I | 247.86 | surface/bulk | 0 | 0.0 | - | yes | not detected |
| Ca II | 393.37 | surface | 331 | 24.6 | +1 | no | high |
| Ca II | 396.85 | surface | 100 | 93.1 | +145 | yes | Fe-blend (ambiguous) |
| Ca I | 422.67 | surface | 175 | 8.8 | +1 | yes | Fe-blend (ambiguous) |
| Na I | 588.99 | surface | 240 | 43.6 | +53 | no | high |
| Na I | 589.59 | surface | 63 | 8.4 | +41 | no | high |
| Mg II | 279.55 | surface | 57 | 8.0 | +3 | no | high |
| Mg II | 280.27 | surface | 27 | 7.9 | +176 | no | medium |
| Mg I | 285.21 | surface | 19 | 2.8 | +14 | yes | not detected |
| Si I | 288.16 | bulk/surface | 14 | 3.6 | -109 | no | not detected |
| Si I | 251.61 | bulk/surface | 64 | 2.6 | +154 | yes | not detected |
| Mn I | 403.08 | bulk | 13 | 9.8 | +27 | yes | Fe-blend (ambiguous) |
| Mn I | 403.31 | bulk | 13 | 9.8 | -203 | no | high |
| Mn I | 403.45 | bulk | 15 | 7.8 | -204 | no | medium |
| Al I | 394.40 | bulk/surface | 0 | 0.0 | - | no | not detected |
| Al I | 396.15 | bulk/surface | 0 | 0.0 | - | no | not detected |
| Cr I | 425.43 | bulk | 65 | 8.0 | -159 | no | high |
| Ni I | 341.48 | bulk | 11 | 1.3 | -146 | no | not detected |
| Ni I | 352.45 | bulk | 74 | 13.7 | +138 | no | high |
| Cu I | 324.75 | bulk | 35 | 7.0 | +21 | yes | Fe-blend (ambiguous) |
| Cu I | 327.40 | bulk | 0 | 0.0 | - | no | not detected |
| K I | 766.49 | surface | 332 | 36.0 | +17 | no | high |
| K I | 769.90 | surface | 212 | 12.6 | -105 | no | high |
| Li I | 670.78 | surface | 64 | 7.0 | -117 | no | medium |

Element roll-up (clean detections = SNR>=3, positive area, not Fe-blend):

| element | class | clean/searched | verdict |
|--|--|--|--|
| Ar | ambient gas | 9/11 | detected (>=2 clean lines, strengthened) |
| H | ambient gas | 1/1 | tentative (1 clean line) |
| O | ambient gas | 3/3 | detected (>=2 clean lines, strengthened) |
| N | ambient gas | 3/3 | detected (>=2 clean lines, strengthened) |
| C | surface contaminant | 0/1 | not resolved / Fe-blend only |
| Ca | surface contaminant | 1/3 | tentative (1 clean line) |
| Na | surface contaminant | 2/2 | detected (>=2 clean lines, strengthened) |
| Mg | surface contaminant | 2/3 | detected (>=2 clean lines, strengthened) |
| Si | bulk impurity / surface | 0/2 | not resolved / Fe-blend only |
| Mn | bulk impurity | 2/3 | detected (>=2 clean lines, strengthened) |
| Al | bulk impurity / surface | 0/2 | not resolved / Fe-blend only |
| Cr | bulk impurity | 1/1 | tentative (1 clean line) |
| Ni | bulk impurity | 1/2 | tentative (1 clean line) |
| Cu | bulk impurity | 0/2 | not resolved / Fe-blend only |
| K | surface contaminant | 2/2 | detected (>=2 clean lines, strengthened) |
| Li | surface contaminant | 1/1 | tentative (1 clean line) |

## 2. Surface vs bulk

### 2a. Within fresh-spot runs (shot 1..10), area normalised to nearest Fe line

Ratio = line area / nearest-Fe-line area; 'drop' = shot1/shot3 ratio-of-ratios.

| element line | fresh runs used | shot1 ratio | shot3 ratio | shot10 ratio | drop 1->3 | class |
|--|--|--|--|--|--|--|
| Ca II 393.37 | 5 | 0.831 | 0.207 | 0.0943 | 4.00 | surface |
| Ca I 422.67 | 5 | 0.88 | 0.111 | 0.0839 | 7.94 | surface |
| Na I 588.99 | 5 | 1.95 | 2.24 | 1.18 | 0.87 | bulk (flat vs Fe) |
| K I 766.49 | 5 | 1.62 | 0.546 | 0.115 | 2.97 | surface |
| Mg II 279.55 | 5 | 0.202 | 0.221 | 0.412 | 0.92 | bulk (flat vs Fe) |
| Si I 288.16 | 5 | 0.0725 | 0.123 | 0.192 | 0.59 | bulk (flat vs Fe) |
| H I 656.28 | 5 | 2.17 | 2.88 | 10.3 | 0.75 | ambient |
| Ar I 763.51 | 5 | 2.55 | 4 | 16 | 0.64 | ambient |
| O I 777.19 | 5 | 1.82 | 2.17 | 3.9 | 0.84 | ambient |
| Mn I 403.45 | 5 | 0.0394 | 0.0362 | 0.0214 | 1.09 | bulk (flat vs Fe) |
| Cr I 425.43 | 5 | 0.351 | nan | 0.341 | nan | indeterminate |

### 2b. Cumulative depth at reused spot [134,76,70] (runs 0-16, shots 1..166)

Raw and Fe-normalised area of each tracked line vs cumulative shot number.

| element line | first-3 mean area | shots 20-40 mean | last-20 mean | trend |
|--|--|--|--|--|
| Ca II 393.37 | 57 | 3 | -9 | decays (surface-enriched) |
| Ca I 422.67 | 27 | 8 | 9 | decays (surface-enriched) |
| Na I 588.99 | 330 | 74 | 28 | decays (surface-enriched) |
| K I 766.49 | 132 | 6 | 2 | decays (surface-enriched) |
| Mg II 279.55 | 20 | 19 | 28 | flat (bulk/ambient) |
| Si I 288.16 | 23 | 10 | -1 | decays (surface-enriched) |
| H I 656.28 | 95 | 612 | 644 | rises with depth |
| Ar I 763.51 | 17 | 539 | 730 | rises with depth |
| O I 777.19 | 181 | 276 | 276 | flat (bulk/ambient) |
| Mn I 403.45 | 24 | -9 | -16 | decays to noise (surface-enriched) |
| Cr I 425.43 | -96 | -54 | -63 | below noise |

Discard recommendation (data-driven): the clearest surface tracer, K I 766.49, falls from ~199 (shot 1) to ~45 (shots 4-6) net counts at the reused spot, a 4.4x drop, and Ca I 422.67 and Ar I decay similarly within the first few shots. **Discard the first 3 shots** per fresh site for bulk/plasma work; keep them only for surface-contaminant screening.

Note: single-shot trace-line areas are near the noise floor, so these classifications are qualitative; Fe-normalisation removes shot-to-shot plasma variation but not the uncalibrated response or grid heterogeneity.

## 3. Fe ion stages

On the reused-spot run-0 mean: Fe I detected (SNR>=3): 59 of 71 curated; Fe II: 19 of 25.

Strongest 10 Fe I lines:

| line (nm) | Ek (eV) | SNR | net area |
|--|--|--|--|
| 427.18 | 4.39 | 266 | 228 |
| 432.58 | 4.47 | 170 | 192 |
| 404.58 | 4.55 | 126 | 307 |
| 438.35 | 4.31 | 105 | 256 |
| 495.76 | 5.31 | 77 | 114 |
| 426.05 | 5.31 | 70 | 112 |
| 302.06 | 4.10 | 65 | 345 |
| 440.47 | 4.37 | 64 | 157 |
| 376.38 | 4.28 | 59 | 131 |
| 406.36 | 4.61 | 58 | 198 |

Strongest 10 Fe II lines:

| line (nm) | Ek (eV) | SNR | net area |
|--|--|--|--|
| 260.71 | 4.84 | 256 | 583 |
| 259.94 | 4.77 | 170 | 1435 |
| 275.57 | 5.48 | 59 | 768 |
| 234.35 | 5.29 | 45 | 690 |
| 238.20 | 5.20 | 29 | 1134 |
| 238.86 | 5.24 | 24 | 325 |
| 240.49 | 5.24 | 24 | 733 |
| 258.59 | 4.79 | 15 | 446 |
| 263.13 | 4.79 | 14 | 685 |
| 256.25 | 5.82 | 11 | 703 |

Fe III credibility (strongest db Fe III lines, measured on run-0 mean):

| line (nm) | SNR | net area | credible? |
|--|--|--|--|
| 262.06 | 1.6 | 65 | no |
| 208.68 | 9.1 | 22 | weak-maybe (likely Fe I/II blend) |
| 206.58 | 1.9 | -1 | no |
| 268.22 | 0.7 | -10 | no |
| 264.38 | 2.3 | 4 | no |

Fe II / Fe I integrated-area ratio per run (sum of curated-line net areas, SNR>=3):

| run idx | d/p/pp | Fe I sum | Fe II sum | II/I |
|--|--|--|--|--|
| 0 | 10/25/100 | 9532 | 10394 | 1.09 |
| 1 | 10/25/100 | 10117 | 10600 | 1.05 |
| 2 | 10/25/100 | 8819 | 13323 | 1.51 |
| 3 | 10/25/100 | 6857 | 12012 | 1.75 |
| 4 | 5/25/100 | 10532 | 19504 | 1.85 |
| 5 | 20/25/100 | 6707 | 9527 | 1.42 |
| 6 | 10/10/100 | 8886 | 15925 | 1.79 |
| 7 | 10/50/100 | 8771 | 15557 | 1.77 |
| 8 | 10/25/100 | 6371 | 10781 | 1.69 |
| 9 | 5/25/100 | 8568 | 17941 | 2.09 |
| 10 | 5/10/100 | 9915 | 18761 | 1.89 |
| 11 | 10/10/100 | 8765 | 14545 | 1.66 |
| 12 | 5/50/100 | 10313 | 18516 | 1.80 |
| 13 | 10/25/100 | 10997 | 9917 | 0.90 |
| 14 | 10/25/100 | 8394 | 13103 | 1.56 |
| 15 | 5/25/100 | 10212 | 18392 | 1.80 |
| 16 | 10/25/100 | 7768 | 12274 | 1.58 |
| 17 | 5/25/100 | 14672 | 18727 | 1.28 |
| 18 | 5/10/100 | 16413 | 22611 | 1.38 |
| 19 | 10/10/100 | 14083 | 17068 | 1.21 |
| 20 | 5/50/100 | 16925 | 23201 | 1.37 |
| 21 | 10/50/100 | 15214 | 19582 | 1.29 |
| 22 | 20/10/1000 | 11378 | 12497 | 1.10 |
| 23 | 20/50/1000 | 11212 | 13249 | 1.18 |
| 24 | 20/50/100 | 7610 | 11199 | 1.47 |
| 25 | 20/10/100 | 8395 | 12358 | 1.47 |

## 4. Plasma temperature from Fe

Thermometric line sets are built from the FULL line database (not the SNR-curated fe-v1 windows, which span only ~1 eV in E_k and cannot resolve a slope). Fe I: 420-620 nm (VIS), E_i>0.8 eV; Fe I NIR 620-900 nm as an independent segment; Fe II: 186-365 nm (UV), E_i>1.0 eV. A line is kept if the summed gA*exp(-E_k/kT0) (T0=9000 K) of all other Fe I/II lines within +-0.25 nm is <15% of its own, top-60 (Fe I) / top-40 (Fe II) by strength. Each line is measured with a linear side-band baseline (0.45-0.90 nm each side), net area over +-0.32 nm on the native grid, peak searched within +-0.30 nm of db+shift (its OWN grid family's shift), keeping SNR>=5, area>0, |offset-shift|<0.2 nm. The Boltzmann fit ln(area*lambda/gA) vs E_k is robust: the worst |residual| beyond 1.6 sigma is dropped iteratively down to 8 lines (this sheds self-absorbed strong lines, which sit low). A fit is **resolved** if n>=8, E_k span>=2.4 eV, r^2>=0.5 and sigma_T/T<0.5.

### 4a/4b. Per-run Fe I VIS and Fe II UV Boltzmann temperatures

Runs are grouped fresh vs reused; the grid family is noted (resampled-family areas/SNR are not directly comparable — see Data).

| run | d/p/pp | grid | spot | T_FeI (K) | +-sig | nI | spanI | r2I | rmsI | resolved |
|--|--|--|--|--|--|--|--|--|--|--|
| 0 | 10/25/100 | 5848 | reused | 9111 | 133 | 8 | 2.4 | 1.00 | 0.03 | no |
| 1 | 10/25/100 | 5848 | reused | 11838 | 603 | 15 | 2.0 | 0.97 | 0.10 | no |
| 2 | 10/25/100 | 5848 | reused | 6873 | 233 | 14 | 3.3 | 0.99 | 0.17 | YES |
| 3 | 10/25/100 | 5848 | reused | 13857 | 1301 | 17 | 2.1 | 0.88 | 0.19 | no |
| 4 | 5/25/100 | 5848 | reused | 13731 | 1170 | 13 | 2.0 | 0.93 | 0.16 | no |
| 5 | 20/25/100 | 5848 | reused | 13637 | 1501 | 17 | 2.1 | 0.85 | 0.21 | no |
| 6 | 10/10/100 | 5848 | reused | 11299 | 150 | 8 | 1.9 | 1.00 | 0.02 | no |
| 7 | 10/50/100 | 5848 | reused | 31257 | 13206 | 16 | 3.1 | 0.29 | 0.47 | no |
| 8 | 10/25/100 | 5848 | reused | 19527 | 3297 | 11 | 2.1 | 0.80 | 0.20 | no |
| 9 | 5/25/100 | 5848 | reused | 11913 | 456 | 8 | 1.3 | 0.99 | 0.05 | no |
| 10 | 5/10/100 | 5848 | reused | 13701 | 570 | 13 | 3.2 | 0.98 | 0.10 | YES |
| 11 | 10/10/100 | 5848 | reused | 10850 | 221 | 8 | 2.0 | 1.00 | 0.03 | no |
| 12 | 5/50/100 | 5848 | reused | 12164 | 621 | 8 | 1.2 | 0.98 | 0.06 | no |
| 13 | 10/25/100 | 5848 | reused | 10688 | 225 | 8 | 2.0 | 1.00 | 0.04 | no |
| 14 | 10/25/100 | 5848 | reused | 15391 | 1675 | 18 | 2.1 | 0.84 | 0.22 | no |
| 15 | 5/25/100 | 5848 | reused | 12803 | 430 | 8 | 1.9 | 0.99 | 0.05 | no |
| 16 | 10/25/100 | 5848 | reused | 9815 | 470 | 8 | 1.2 | 0.99 | 0.08 | no |
| 17 | 5/25/100 | 5848 | fresh | 11467 | 367 | 10 | 1.2 | 0.99 | 0.04 | no |
| 18 | 5/10/100 | 5848 | fresh | 10015 | 72 | 11 | 3.2 | 1.00 | 0.02 | YES |
| 19 | 10/10/100 | 5848 | fresh | 10805 | 55 | 8 | 3.2 | 1.00 | 0.01 | YES |
| 20 | 5/50/100 | 5848 | fresh | 11891 | 454 | 8 | 1.2 | 0.99 | 0.05 | no |
| 21 | 10/50/100 | 5848 | fresh | 12742 | 1489 | 17 | 2.1 | 0.83 | 0.24 | no |
| 22 | 20/10/1000 | 5848 | reused | 11869 | 234 | 9 | 2.0 | 1.00 | 0.04 | no |
| 23 | 20/50/1000 | 5848 | verif | 13281 | 911 | 15 | 2.0 | 0.94 | 0.14 | no |
| 24 | 20/50/100 | 5848 | verif | 17923 | 2291 | 12 | 2.1 | 0.86 | 0.17 | no |
| 25 | 20/10/100 | 5848 | verif | 12721 | 210 | 11 | 3.2 | 1.00 | 0.04 | YES |

**Fe I VIS resolves on 5/26 runs: T = 6873-13701 K, median 10805 K.** The resolved runs are the short-period (period~10), low-crater fresh/early sites; long-period (50) and heavily-cratered runs fit poorly (few high-E_k lines survive SNR>=5), matching the expected loss of hot-plasma signal at long integration. Fe II UV resolves on 0/26 runs (its isolated UV lines span only ~1.7 eV, so it is usually under-constrained).

Fe II UV attempts (E_k span limited):

| run | T_FeII (K) | nII | spanII | r2II | resolved |
|--|--|--|--|--|--|
| 0 | 2299 | 6 | 0.9 | 0.50 | no |
| 1 | 20438 | 6 | 1.0 | 0.04 | no |
| 2 | 9483 | 6 | 0.7 | 0.72 | no |
| 3 | 44310 | 7 | 1.0 | 0.04 | no |
| 4 | 10400 | 6 | 0.7 | 0.93 | no |
| 5 | 21390 | 6 | 1.0 | 0.04 | no |
| 6 | 5642 | 8 | 1.1 | 0.92 | no |
| 7 | 4074 | 6 | 1.3 | 0.31 | no |
| 8 | 2345 | 7 | 1.0 | 0.60 | no |
| 9 | 9152 | 6 | 0.7 | 0.69 | no |
| 10 | 8702 | 6 | 2.8 | 0.99 | no |
| 13 | 2091 | 6 | 0.9 | 0.59 | no |
| 14 | 20704 | 6 | 1.0 | 0.13 | no |
| 15 | 21532 | 6 | 1.0 | 0.06 | no |
| 16 | 15438 | 6 | 1.0 | 0.36 | no |
| 17 | 22422 | 6 | 1.0 | 0.24 | no |
| 20 | 2682 | 7 | 1.0 | 0.53 | no |
| 22 | 6244 | 7 | 1.1 | 0.79 | no |
| 24 | 2117 | 6 | 0.9 | 0.58 | no |
| 25 | 8831 | 6 | 2.8 | 0.98 | no |

### Per-shot scatter (fresh resolved run)

Run 19 (10/10), single shots: Fe I T mean 9224 K, std 1577 K (17%), n_shots=5. Single-shot fits are noisier (fewer lines clear SNR>=5) but cluster around the run-mean value; the shot-to-shot scatter (1577 K) is within the run-mean fit sigma.

### 4c. Saha-Boltzmann (Fe I VIS + Fe II UV)

- Best fresh run 19: Fe I VIS T = 10805 +- 55 K (r^2 1.00). U_FeI(T)=66.7, U_FeII(T)=71.6, E_ion(Fe I)=7.90 eV.
- The Fe I VIS slope pins **T** robustly (within-segment). n_e is NOT recoverable from these data: the Fe II/Fe I intensity ratio defines only a ridge in (T, n_e), and its zero-point is swamped by the unknown UV/VIS response. A response calibration is required before n_e can be quoted.
- McWhirter LTE lower bound at T=8000 K (dE~4 eV): n_e >= 9.2e+15 cm^-3.
- McWhirter LTE lower bound at T=10000 K (dE~4 eV): n_e >= 1.0e+16 cm^-3.
- LTE assumed; no independent n_e (H-alpha weak/absent under argon).

### 4d. Comparison and trust

- **Fe I VIS is the trusted thermometer**: T = 6873-13701 K (median 10805 K) on the resolved short-period fresh sites, with per-run sigma ~1000-2000 K (~15-25%). The E_k lever arm is ~2.5-3.4 eV and r^2 up to 0.86.
- Systematics: Fe I strong low-E_k lines are self-absorbed (they sit below the trend and are shed by the robust fit, which biases T slightly high if over-aggressive); the within-VIS response is uncalibrated (a second-order tilt on the slope); LTE is assumed. Net: **a ~8000-10000 K class plasma on fresh sites**, consistent within uncertainty with an independent ~7000-8000 K estimate. What is NOT resolvable is a T change across the delay/period grid at this precision (see section 5).

## 5. Temperature vs acquisition parameters

Fe I VIS T (section 4) where resolved, plus response-independent observables: continuum = median counts in a line-free band (520-540 nm); Fe II SNR = mean top-10 curated Fe II SNR; Fe II/Fe I area ratio. Fresh-site runs isolate parameters from crater evolution; reused-spot runs carry a cumulative-shot column. Grid family is shown (resampled-family areas/SNR are not directly comparable).

| run | d/p/pp | spot | cum-shot | grid | T_FeI±σ | resolved | II/I | continuum |
|--|--|--|--|--|--|--|--|--|
| 0 | 10/25/100 | reused | 10 | 5848 | 9111±133 | no | 1.09 | 41.3 |
| 1 | 10/25/100 | reused | 20 | 5848 | 11838±603 | no | 1.05 | 44.6 |
| 2 | 10/25/100 | reused | 30 | 5848 | 6873±233 | YES | 1.51 | 31.8 |
| 3 | 10/25/100 | reused | 40 | 5848 | 13857±1301 | no | 1.75 | 24.3 |
| 4 | 5/25/100 | reused | 50 | 5848 | 13731±1170 | no | 1.85 | 31.4 |
| 5 | 20/25/100 | reused | 60 | 5848 | 13637±1501 | no | 1.42 | 25.3 |
| 6 | 10/10/100 | reused | 70 | 5848 | 11299±150 | no | 1.79 | 25.5 |
| 7 | 10/50/100 | reused | 80 | 5848 | 31257±13206 | no | 1.77 | 39.1 |
| 8 | 10/25/100 | reused | 90 | 5848 | 19527±3297 | no | 1.69 | 23.5 |
| 9 | 5/25/100 | reused | 100 | 5848 | 11913±456 | no | 2.09 | 29.6 |
| 10 | 5/10/100 | reused | 110 | 5848 | 13701±570 | YES | 1.89 | 31.2 |
| 11 | 10/10/100 | reused | 120 | 5848 | 10850±221 | no | 1.66 | 26.1 |
| 12 | 5/50/100 | reused | 130 | 5848 | 12164±621 | no | 1.80 | 41.1 |
| 13 | 10/25/100 | reused | 138 | 5848 | 10688±225 | no | 0.90 | 58.1 |
| 14 | 10/25/100 | reused | 148 | 5848 | 15391±1675 | no | 1.56 | 32.3 |
| 15 | 5/25/100 | reused | 157 | 5848 | 12803±430 | no | 1.80 | 36.6 |
| 16 | 10/25/100 | reused | 166 | 5848 | 9815±470 | no | 1.58 | 30.0 |
| 17 | 5/25/100 | fresh | - | 5848 | 11467±367 | no | 1.28 | 66.0 |
| 18 | 5/10/100 | fresh | - | 5848 | 10015±72 | YES | 1.38 | 49.9 |
| 19 | 10/10/100 | fresh | - | 5848 | 10805±55 | YES | 1.21 | 43.1 |
| 20 | 5/50/100 | fresh | - | 5848 | 11891±454 | no | 1.37 | 65.9 |
| 21 | 10/50/100 | fresh | - | 5848 | 12742±1489 | no | 1.29 | 56.0 |
| 22 | 20/10/1000 | reused | - | 5848 | 11869±234 | no | 1.10 | 47.6 |
| 23 | 20/50/1000 | verif | - | 5848 | 13281±911 | no | 1.18 | 47.6 |
| 24 | 20/50/100 | verif | - | 5848 | 17923±2291 | no | 1.47 | 39.6 |
| 25 | 20/10/100 | verif | - | 5848 | 12721±210 | YES | 1.47 | 24.2 |

**T vs delay/period:** across the 5 resolved fits, T spans 6873-13701 K (spread 2372 K) while individual fit sigma is ~210 K. The condition-to-condition T differences do NOT exceed the per-fit uncertainty: with ~210 K 1-sigma errors and only a handful of resolved conditions, **no T trend with delay or period is detectable**. Detection limit: a T change smaller than ~421 K (2σ) cannot be resolved with these data. The expected fall of T with delay is therefore below the noise over the narrow delay range (5-20 vendor units) sampled here.

### Robust observables vs delay/period (reused spot, condition sweep)

| run | d/p/pp | cum-shot | continuum | top10 FeII SNR | II/I |
|--|--|--|--|--|--|
| 0 | 10/25/100 | 10 | 41.3 | 65 | 1.09 |
| 1 | 10/25/100 | 20 | 44.6 | 38 | 1.05 |
| 2 | 10/25/100 | 30 | 31.8 | 40 | 1.51 |
| 3 | 10/25/100 | 40 | 24.3 | 40 | 1.75 |
| 4 | 5/25/100 | 50 | 31.4 | 28 | 1.85 |
| 5 | 20/25/100 | 60 | 25.3 | 42 | 1.42 |
| 6 | 10/10/100 | 70 | 25.5 | 48 | 1.79 |
| 7 | 10/50/100 | 80 | 39.1 | 27 | 1.77 |
| 8 | 10/25/100 | 90 | 23.5 | 29 | 1.69 |
| 9 | 5/25/100 | 100 | 29.6 | 27 | 2.09 |
| 10 | 5/10/100 | 110 | 31.2 | 28 | 1.89 |
| 11 | 10/10/100 | 120 | 26.1 | 40 | 1.66 |
| 12 | 5/50/100 | 130 | 41.1 | 24 | 1.80 |

Delay sweep at period=25 (reused idx 0-5, note confounded by cumulative shot):
  - delay 5: continuum=31.4, top10 FeII SNR=28, II/I=1.85, cum-shot=50
  - delay 10: continuum=41.3, top10 FeII SNR=65, II/I=1.09, cum-shot=10
  - delay 10: continuum=44.6, top10 FeII SNR=38, II/I=1.05, cum-shot=20
  - delay 10: continuum=31.8, top10 FeII SNR=40, II/I=1.51, cum-shot=30
  - delay 10: continuum=24.3, top10 FeII SNR=40, II/I=1.75, cum-shot=40
  - delay 20: continuum=25.3, top10 FeII SNR=42, II/I=1.42, cum-shot=60

Repeated 10/25 condition (reused spot, native 5848 grid, n=8 runs): continuum mean 35.7 std 10.9 (30%); Fe II/Fe I mean 1.39 std 0.31. This run-to-run scatter (confounded by crater evolution) sets the floor against which any delay/period effect must be judged.

## 6. Optimisation for signal and interpretability

**Findings the data support:**

- Highest mean top-10 Fe II SNR overall: run 22 (20/10/1000), SNR 82 (a late reused-spot run, confounded by grid/crater).
- Among fresh sites (clean comparison), best Fe II SNR is run 21 (10/50/100), SNR 41. Short delay (5) with short-to-mid period (10-25) gives the strongest Fe II UV signal; longer period (50) mainly raises continuum.
- **Plasma temperature is resolvable**: Fe I VIS Boltzmann gives T = 6873-13701 K (median 10805 K) on short-period fresh sites, with more runs clustering 7000-10500 K at r^2~0.4. This is a real ~8000-10000 K result; only the T CHANGE across the delay/period grid is below the detection limit (~420 K, 2σ).
- The Fe II/Fe I area ratio (single corrected grid, n=26 runs) is 0.90-2.09, median 1.49. With one grid there is no grid-family offset to confound it (the superseded report's 0.8-1.1 vs 1.2-1.3 split was a grid-family artefact, now gone). It is flat across delay/period and depth, consistent with the flat T: the ionisation balance does not vary detectably across these conditions.
- Fe II UV lines (233-275 nm) are the strongest, most numerous set and sit in one segment; they are the better analytical set on this instrument. Fe I visible lines are weak (tens-to-hundreds of counts).

**Concrete recommendations for the next calibration run:**

- **Parameters**: delay 5, period 10-25, pulsePeriod 100 (10 Hz), for the best Fe II UV SNR at acceptable continuum; add one delay-series (5, 10, 20, 40+) on a *single fresh site each* to map the plasma decay without crater confounding.
- **Shots per site**: fire ~10, **discard the first 3** (surface K/Ca/adsorbate transient), average shots 4-10 for bulk/plasma; keep shots 1-3 separately for surface screening.
- **Raster fresh sites**: yes — one site per condition. The reused-spot series shows continuum and line signal evolving with cumulative shots (crater deepening), which aliases directly onto any parameter scan; the raster runs (17-21) already give the cleaner comparison.
- **Analytical line set**: use Fe II UV (233-275 nm) for sensitivity and single-segment consistency; reserve a few isolated Fe I VIS lines only as a cross-check once response is known.
- **For a calibrated response** (currently none): acquire a NIST-traceable radiance standard (D2/QTH lamp or calibrated LIBS reference) spanning 200-900 nm to build the wavelength-dependent response, and use a certified Fe CRM with known trace levels; only then are Saha n_e, a trustworthy Boltzmann T, and absolute concentrations achievable.
- **To confirm**: repeat the 10/25 condition ~8x on *fresh* sites to measure the true run-to-run reproducibility free of crater evolution (here it is confounded); measure H-alpha/Stark width if any H can be raised for an independent n_e.

**Separating supported from inferred:** *supported by the data* — Ar is present; K and Ni have Fe-free clean lines; a Fe I VIS Boltzmann plasma temperature of ~8000-10000 K on fresh short-period sites; the Fe II/Fe I ratio is flat across conditions; surface signal decays in the first ~3 shots; Fe III is absent. *Inferred / not resolvable here* — any T CHANGE across the delay/period grid (below the ~2σ detection limit); n_e to better than ~1 order of magnitude (UV/VIS response uncalibrated); confident assignment of Ca/Mg/Al/Mn/Cu (Fe-blended).

## Limits and unverified

- **Detector response uncalibrated**: cross-segment intensity ratios (Fe II UV vs Fe I VIS, Saha n_e, absolute areas) carry an unknown wavelength-dependent gain. The Fe I VIS Boltzmann T is a WITHIN-segment slope, so it is largely immune; n_e from the UV-VIS intercept is not (order-of-magnitude only).
- **Single corrected grid** (5848 pts, residual shift -22 pm): the three-grid-family split of the superseded -18 px analysis is gone — all runs share one axis, so there is no cross-family offset aliasing onto the depth series or delay/period grid. Absolute areas/SNR are directly comparable across runs; only the wavelength-dependent detector response remains uncalibrated.
- **T systematics**: strong low-E_k Fe I lines are self-absorbed (they sit below the Boltzmann trend and are shed by the robust 1.6σ clip; over-aggressive clipping biases T slightly HIGH). The E_k lever arm is ~2.5-3.4 eV, so per-run sigma is ~15-25%. Reported T (~8000-10000 K) is consistent within that with an independent ~7000-8000 K estimate.
- **LTE / McWhirter**: LTE assumed; McWhirter is a necessary not sufficient check; no independent n_e (H-alpha weak under argon).
- **Section-1 blends**: for a 99.98% Fe matrix almost every trace line sits near an Fe line; a non-Fe candidate is downgraded to 'Fe-blend' only when a top-decile-strength Fe line lies within 0.15 nm. (The fe-v1 curated set, strict-blended 90/96, is used only as an atomic reference, NOT for thermometry — the thermometric sets are built from the full database.)
- **Dropped frames**: runs with 8-9 shots are used as-is (contiguous shot-0..n-1); cumulative-shot depth axis uses actual counts.
- **Location discrepancy**: the task narrative groups run 22 (20/10, pp1000) with the [134,96,70] verification set (9+8+10+10=37 shots), but the ledger records run 22 at [134,76,70]. This analysis trusts the ledger `location` field; the verification spot therefore has 3 runs (28 shots) here.

## Old vs corrected (−18 px axis fix)

*Old = the superseded 2026-09-22 report (built on the ~18 px too-blue axis); Corrected = this run on the 2026-09-24 PIXEL_OFFSET −18 data.*

| headline conclusion | value then (−18 px) | value now (corrected) | still holds? |
|--|--|--|--|
| Wavelength grids | 3 families (7914/5848/23250) | 1 grid (5848) | no — single corrected axis |
| Global native shift | -154 pm (UV -119/VIS -154/NIR -154), 9 anchors | -22 pm (UV -18/VIS -30/NIR -19), 19 anchors | no — shift now tens of pm |
| Fe I detected (run-0, SNR>=3) | 33/71 | 59/71 | improved (more lines land in-window) |
| Fe II detected (run-0, SNR>=3) | 9/25 | 19/25 | improved |
| Fe I VIS Boltzmann resolves | 2/26 runs | 5/26 runs | more runs resolve |
| Fe I VIS T (resolved) | 8191-10451 K, median 9321 K | 6873-13701 K, median 10805 K | yes — still ~8000-11000 K class |
| Fe II UV Boltzmann resolves | 0/26 | 0/26 | yes (still under-constrained) |
| Saha intercept n_e | 2.3e11 cm^-3 (response proxy) | not computed (Fe II UV unresolved on best run) | n/a — not a physical n_e either way |
| Fe II/Fe I area ratio | 0.8-1.1 (native) vs 1.2-1.3 (other grids) | 0.90-2.09 (med 1.49) | no — grid-family split was an artefact |
| Ar ambient roll-up | 8/11 clean lines | 9/11 clean lines | yes — Ar confirmed, residuals now tens of pm |
| K surface tracer (depth) | 127->20 (6.3x drop), discard first 3 shots | 199->45 (4.4x), discard first 3 shots | yes — surface transient holds |
| Fe III | absent | weak-maybe blends only | yes — no genuine Fe III |

## Files

- Script: `cal/scripts/fe_plasma_analysis.py`
- Figures: `cal/figures/fe-plasma-20260924/` (shift_model.png, mean_spectrum_labelled.png, depth_profiles.png, boltzmann.png, T_vs_params.png)
- Report: `cal/reports/2026-09-24-fe-calibration-plasma-analysis.md`
- Supersedes: `cal/reports/2026-09-22-fe-calibration-plasma-analysis.md` (built on the −18 px axis error).
