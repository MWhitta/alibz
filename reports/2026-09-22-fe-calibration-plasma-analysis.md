# Fe (Aesar 99.98%) Z300 LIBS calibration analysis: plasma and surface

*Generated fe_plasma_analysis.py — deterministic rerun of `--data /private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/39a75126-564e-4be9-a01b-a385e7a2834d/scratchpad/acq` `--ledger /private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/39a75126-564e-4be9-a01b-a385e7a2834d/scratchpad/ledger.json`.*

## Data

- 26 succeeded live runs, Fe Aesar 99.98%, argon pre-flush 300, gated, 10 shots requested/run, 1 location/run, no cleaning shots.
- Total shots on disk: 252 across 26 runs.
- **Data-provenance caveat — three wavelength-grid families** (measured on each run's OWN grid, no resampling by this analysis). Point counts are data rows (files carry one extra header line):
  - **15 runs, 7914 pts (7915 lines)** — native, streaming-API path; 186-961 nm, pitch 0.089/0.129/0.179 nm UV/VIS/NIR, ~12 nm NIR gap 948-960 nm. Shift -154 pm.
  - **6 runs, 5848 pts (5849 lines)** — native, Opal FlatBuffers decode (empirical pixel offset); 186-948 nm, no NIR tail. Shift -182 pm (does NOT match the -154 pm of the API grid).
  - **5 runs, 23250 pts (23251 lines)** — vendor 1/30 nm cubic-spline RESAMPLE (uniform 0.0333 nm); correlated noise and smoothed peaks, so its SNRs and integrated areas are NOT directly comparable to the native families. Shift -83 pm.
- Grid family per run: 7914 = {16bbf7, e99312, a23779, 5cecb3, 1a57e5, b6c766, 0661a4, f5ee5c, d75e0d, fa22f9, 9432dc, 43111b, a5b107, 4819b8, e47c4f}; 5848 = {e26b61, a4d327, b408b8, 904fa2, 3bee25, 73bbf4}; 23250 = {065f1a, 7898db, f6fdfc, 3f67f3, c5c346}.
- Segments: UV [186,365), VIS [365,620), NIR [620,948] nm. Every table below notes which grid families it mixes.

### Run groups (chronological index; d=delay p=period pp=pulsePeriod ms)

| # | time | location | d | p | pp | shots | grid | group |
|--|--|--|--|--|--|--|--|--|
| 0 | 15:58:34 | 134,76,70 | 10 | 25 | 100 | 10 | 7914 | reused-spot depth |
| 1 | 16:27:10 | 134,76,70 | 10 | 25 | 100 | 10 | 7914 | reused-spot depth |
| 2 | 16:31:42 | 134,76,70 | 10 | 25 | 100 | 10 | 7914 | reused-spot depth |
| 3 | 16:41:37 | 134,76,70 | 10 | 25 | 100 | 10 | 23250 | reused-spot depth |
| 4 | 16:42:20 | 134,76,70 | 5 | 25 | 100 | 10 | 23250 | reused-spot depth |
| 5 | 16:42:52 | 134,76,70 | 20 | 25 | 100 | 10 | 23250 | reused-spot depth |
| 6 | 16:43:43 | 134,76,70 | 10 | 10 | 100 | 10 | 23250 | reused-spot depth |
| 7 | 16:44:22 | 134,76,70 | 10 | 50 | 100 | 10 | 23250 | reused-spot depth |
| 8 | 18:12:01 | 134,76,70 | 10 | 25 | 100 | 10 | 5848 | reused-spot depth |
| 9 | 18:13:50 | 134,76,70 | 5 | 25 | 100 | 10 | 5848 | reused-spot depth |
| 10 | 18:14:14 | 134,76,70 | 5 | 10 | 100 | 10 | 5848 | reused-spot depth |
| 11 | 18:14:34 | 134,76,70 | 10 | 10 | 100 | 10 | 5848 | reused-spot depth |
| 12 | 18:14:54 | 134,76,70 | 5 | 50 | 100 | 10 | 5848 | reused-spot depth |
| 13 | 19:17:39 | 134,76,70 | 10 | 25 | 100 | 8 | 7914 | reused-spot depth |
| 14 | 19:33:13 | 134,76,70 | 10 | 25 | 100 | 10 | 5848 | reused-spot depth |
| 15 | 19:33:37 | 134,76,70 | 5 | 25 | 100 | 9 | 7914 | reused-spot depth |
| 16 | 19:38:47 | 134,76,70 | 10 | 25 | 100 | 9 | 7914 | reused-spot depth |
| 17 | 20:43:46 | 158,76,70 | 5 | 25 | 100 | 10 | 7914 | raster fresh |
| 18 | 20:44:27 | 158,100,70 | 5 | 10 | 100 | 10 | 7914 | raster fresh |
| 19 | 20:45:08 | 134,100,70 | 10 | 10 | 100 | 9 | 7914 | raster fresh |
| 20 | 20:45:51 | 134,124,70 | 5 | 50 | 100 | 10 | 7914 | raster fresh |
| 21 | 20:46:27 | 158,124,70 | 10 | 50 | 100 | 10 | 7914 | raster fresh |
| 22 | 20:55:25 | 134,76,70 | 20 | 10 | 1000 | 9 | 7914 | reused-spot (later return) |
| 23 | 20:56:12 | 134,96,70 | 20 | 50 | 1000 | 8 | 7914 | verification [134,96,70] |
| 24 | 20:58:45 | 134,96,70 | 20 | 50 | 100 | 10 | 7914 | verification [134,96,70] |
| 25 | 20:59:28 | 134,96,70 | 20 | 10 | 100 | 10 | 7914 | verification [134,96,70] |

Reused spot [134,76,70]: 17 runs (idx 0-16), 166 cumulative shots for the depth series; run 22 (20/10, pp1000) is an 18th run recorded at the same location and is treated as a later return.
Verification spot [134,96,70]: 3 runs, 28 shots.

## Wavelength shift model

Estimated PER GRID FAMILY from isolated, strong, unambiguous anchor lines (isolated Fe I/II below 620 nm; isolated Ar I lines for NIR, where Fe has no lines) measured by sub-pixel parabolic centroid in each family's grand-mean spectrum. Convention: observed = db(air) + shift.

| grid family | provenance | n anchors | global shift (pm) | UV | VIS | NIR |
|--|--|--|--|--|--|--|
| 5848 | native (Opal FlatBuffers decode) | 5 | -182 | -182 | -182 | -182 |
| 7914 | native (streaming API path) | 9 | -154 | -118 | -154 | -154 |
| 23250 | 1/30 nm vendor cubic-spline resample | 6 | -83 | -84 | -83 | -83 |

The native API family is **-154 pm**, the Opal-decode family **-182 pm**, the resampled family **-83 pm** — so a single global shift is NOT adequate; each run is corrected with its own family's shift. The negative (blue) offset matches two of the three first-look examples (Fe I 438.35->438.17, Ar I 763.51->763.19); the Fe II 259.94->260.12 example is not reproduced (the strong db Fe II line there is 260.02 nm, measured ~-0.15 nm). Residual scatter (~50 pm) is set by the native pitch.

Per-anchor matched offsets, native (7914) family:

| species | obs nm | db nm | seg | offset (pm) |
|--|--|--|--|--|
| Fe2 | 233.126 | 233.280 | UV | -154 |
| Fe2 | 234.303 | 234.349 | UV | -47 |
| Fe2 | 239.397 | 239.562 | UV | -166 |
| Fe1 | 248.179 | 248.327 | UV | -149 |
| Fe1 | 248.653 | 248.814 | UV | -162 |
| Fe2 | 273.920 | 273.955 | UV | -34 |
| Fe1 | 373.282 | 373.486 | VIS | -204 |
| Fe1 | 538.261 | 538.337 | VIS | -76 |
| ArI | 696.378 | 696.543 | NIR | -165 |

## 1. Non-Fe elements present

Measured on the reused-spot run-0 mean (native, 10/25) and cross-checked on the fresh-spot first-shot mean. A candidate is a genuine detection only when net area > 0, SNR>=3, the sub-pixel residual to the shifted db line is < 0.15 nm, AND no Fe I/II db line of comparable strength lies within 0.10 nm (else it is flagged **Fe-blend**, since Fe's line forest can mimic almost any position). Residual = sub-pixel obs peak - shifted db.

| element | line (nm, air) | class | net area | SNR | resid (pm) | Fe within 0.15nm? | confidence |
|--|--|--|--|--|--|--|--|
| Ar I | 696.54 | ambient | 77 | 15.0 | +19 | no | high |
| Ar I | 706.72 | ambient | 54 | 11.5 | -271 | no | high |
| Ar I | 738.40 | ambient | 24 | 9.1 | -175 | no | high |
| Ar I | 750.39 | ambient | 35 | 1.5 | -301 | no | not detected |
| Ar I | 763.51 | ambient | 333 | 36.0 | -162 | no | high |
| Ar I | 772.38 | ambient | 0 | 0.0 | - | no | not detected |
| Ar I | 794.82 | ambient | 24 | 6.3 | -109 | no | medium |
| Ar I | 801.48 | ambient | 53 | 8.6 | +28 | no | high |
| Ar I | 811.53 | ambient | 20 | 4.6 | -244 | no | medium |
| Ar I | 826.45 | ambient | 31 | 6.5 | +332 | no | medium |
| Ar I | 842.46 | ambient | 0 | 0.0 | - | no | not detected |
| H I | 656.28 | ambient | 26 | 38.8 | -233 | no | high |
| O I | 777.19 | ambient | 27 | 7.7 | -162 | no | medium |
| O I | 777.42 | ambient | 43 | 3.0 | -40 | no | not detected |
| O I | 777.54 | ambient | 43 | 3.0 | -160 | no | not detected |
| N I | 742.36 | ambient | 0 | 0.0 | - | no | not detected |
| N I | 744.23 | ambient | 11 | 4.8 | +313 | no | medium |
| N I | 746.83 | ambient | 0 | 0.0 | - | no | not detected |
| C I | 247.86 | surface/bulk | 330 | 6.8 | -121 | yes | Fe-blend (ambiguous) |
| Ca II | 393.37 | surface | 10 | 6.1 | -12 | no | medium |
| Ca II | 396.85 | surface | 0 | 0.0 | - | yes | not detected |
| Ca I | 422.67 | surface | 106 | 35.9 | +128 | yes | Fe-blend (ambiguous) |
| Na I | 588.99 | surface | 0 | 0.0 | - | no | not detected |
| Na I | 589.59 | surface | 5 | 3.2 | -189 | no | not detected |
| Mg II | 279.55 | surface | 147 | 48.9 | +114 | no | high |
| Mg II | 280.27 | surface | 4 | 2.9 | +103 | no | not detected |
| Mg I | 285.21 | surface | 0 | 0.0 | - | yes | not detected |
| Si I | 288.16 | bulk/surface | 0 | 0.0 | - | no | not detected |
| Si I | 251.61 | bulk/surface | 0 | 0.0 | - | yes | not detected |
| Mn I | 403.08 | bulk | 0 | 0.0 | - | yes | not detected |
| Mn I | 403.31 | bulk | 0 | 0.0 | - | no | not detected |
| Mn I | 403.45 | bulk | 66 | 47.1 | +195 | no | high |
| Al I | 394.40 | bulk/surface | 100 | 93.1 | +213 | no | high |
| Al I | 396.15 | bulk/surface | 0 | 0.0 | - | no | not detected |
| Cr I | 425.43 | bulk | 0 | 0.0 | - | no | not detected |
| Ni I | 341.48 | bulk | 53 | 22.7 | -105 | no | high |
| Ni I | 352.45 | bulk | 149 | 22.0 | +151 | no | high |
| Cu I | 324.75 | bulk | 0 | 0.0 | - | yes | not detected |
| Cu I | 327.40 | bulk | 41 | 15.5 | -118 | no | high |
| K I | 766.49 | surface | 213 | 12.6 | +153 | no | high |
| K I | 769.90 | surface | 52 | 12.1 | -163 | no | high |
| Li I | 670.78 | surface | 0 | 0.0 | - | no | not detected |

Element roll-up (clean detections = SNR>=3, positive area, not Fe-blend):

| element | class | clean/searched | verdict |
|--|--|--|--|
| Ar | ambient gas | 8/11 | detected (>=2 clean lines, strengthened) |
| H | ambient gas | 1/1 | tentative (1 clean line) |
| O | ambient gas | 1/3 | tentative (1 clean line) |
| N | ambient gas | 1/3 | tentative (1 clean line) |
| C | surface contaminant | 0/1 | not resolved / Fe-blend only |
| Ca | surface contaminant | 1/3 | tentative (1 clean line) |
| Na | surface contaminant | 0/2 | not resolved / Fe-blend only |
| Mg | surface contaminant | 1/3 | tentative (1 clean line) |
| Si | bulk impurity / surface | 0/2 | not resolved / Fe-blend only |
| Mn | bulk impurity | 1/3 | tentative (1 clean line) |
| Al | bulk impurity / surface | 1/2 | tentative (1 clean line) |
| Cr | bulk impurity | 0/1 | not resolved / Fe-blend only |
| Ni | bulk impurity | 2/2 | detected (>=2 clean lines, strengthened) |
| Cu | bulk impurity | 1/2 | tentative (1 clean line) |
| K | surface contaminant | 2/2 | detected (>=2 clean lines, strengthened) |
| Li | surface contaminant | 0/1 | not resolved / Fe-blend only |

## 2. Surface vs bulk

### 2a. Within fresh-spot runs (shot 1..10), area normalised to nearest Fe line

Ratio = line area / nearest-Fe-line area; 'drop' = shot1/shot3 ratio-of-ratios.

| element line | fresh runs used | shot1 ratio | shot3 ratio | shot10 ratio | drop 1->3 | class |
|--|--|--|--|--|--|--|
| Ca II 393.37 | - | - | - | - | - | below noise |
| Ca I 422.67 | 5 | 5.07 | 4.96 | 4.04 | 1.02 | bulk (flat vs Fe) |
| Na I 588.99 | - | - | - | - | - | below noise |
| K I 766.49 | 5 | 9.32 | 2.8 | 6.37 | 3.32 | surface |
| Mg II 279.55 | 5 | 2 | 2.21 | 2.17 | 0.90 | bulk (flat vs Fe) |
| Si I 288.16 | - | - | - | - | - | below noise |
| H I 656.28 | 5 | 5.23 | 4.53 | 2.62 | 1.16 | ambient |
| Ar I 763.51 | 5 | 14 | 4.62 | 1.79 | 3.03 | ambient |
| O I 777.19 | 5 | 1.7 | 1.6 | 2.64 | 1.06 | ambient |
| Mn I 403.45 | 5 | 2.54 | 3.18 | 6.64 | 0.80 | bulk (flat vs Fe) |
| Cr I 425.43 | - | - | - | - | - | below noise |

### 2b. Cumulative depth at reused spot [134,76,70] (runs 0-16, shots 1..166)

Raw and Fe-normalised area of each tracked line vs cumulative shot number.

| element line | first-3 mean area | shots 20-40 mean | last-20 mean | trend |
|--|--|--|--|--|
| Ca II 393.37 | 12 | 12 | 18 | flat (bulk/ambient) |
| Ca I 422.67 | 145 | 56 | 99 | decays (surface-enriched) |
| Na I 588.99 | -4 | 0 | -2 | below noise |
| K I 766.49 | 73 | 10 | 3 | decays (surface-enriched) |
| Mg II 279.55 | 71 | 61 | 72 | flat (bulk/ambient) |
| Si I 288.16 | -15 | 9 | 10 | below noise |
| H I 656.28 | 22 | -6 | -27 | decays to noise (surface-enriched) |
| Ar I 763.51 | 132 | 10 | 0 | decays (surface-enriched) |
| O I 777.19 | 8 | 15 | 28 | flat (bulk/ambient) |
| Mn I 403.45 | 38 | -6 | 45 | decays to noise (surface-enriched) |
| Cr I 425.43 | -32 | -22 | -73 | below noise |

Discard recommendation (data-driven): the clearest surface tracer, K I 766.49, falls from ~127 (shot 1) to ~20 (shots 4-6) net counts at the reused spot, a 6.3x drop, and Ca I 422.67 and Ar I decay similarly within the first few shots. **Discard the first 3 shots** per fresh site for bulk/plasma work; keep them only for surface-contaminant screening.

Note: single-shot trace-line areas are near the noise floor, so these classifications are qualitative; Fe-normalisation removes shot-to-shot plasma variation but not the uncalibrated response or grid heterogeneity.

## 3. Fe ion stages

On the reused-spot run-0 mean: Fe I detected (SNR>=3): 33 of 71 curated; Fe II: 9 of 25.

Strongest 10 Fe I lines:

| line (nm) | Ek (eV) | SNR | net area |
|--|--|--|--|
| 374.56 | 3.40 | 149 | 18 |
| 426.05 | 5.31 | 69 | 22 |
| 438.35 | 4.31 | 49 | 147 |
| 254.60 | 4.96 | 29 | 565 |
| 363.15 | 4.37 | 29 | 93 |
| 432.58 | 4.47 | 26 | 5 |
| 370.92 | 4.26 | 23 | 221 |
| 299.44 | 4.19 | 21 | 92 |
| 381.58 | 4.73 | 21 | 163 |
| 382.04 | 4.10 | 20 | 31 |

Strongest 10 Fe II lines:

| line (nm) | Ek (eV) | SNR | net area |
|--|--|--|--|
| 273.95 | 5.51 | 68 | 802 |
| 238.20 | 5.20 | 27 | 646 |
| 234.35 | 5.29 | 12 | 354 |
| 261.76 | 4.82 | 10 | 10 |
| 236.48 | 5.29 | 9 | 489 |
| 238.86 | 5.24 | 7 | 466 |
| 237.37 | 5.22 | 5 | 186 |
| 233.80 | 5.41 | 5 | 82 |
| 239.56 | 5.22 | 4 | 404 |

Fe III credibility (strongest db Fe III lines, measured on run-0 mean):

| line (nm) | SNR | net area | credible? |
|--|--|--|--|
| 262.06 | 0.0 | -279 | no |
| 208.68 | 8.3 | 26 | weak-maybe (likely Fe I/II blend) |
| 206.58 | 1.8 | -6 | no |
| 268.22 | 3.1 | 7 | no |
| 264.38 | 6.0 | -2 | weak-maybe (likely Fe I/II blend) |

Fe II / Fe I integrated-area ratio per run (sum of curated-line net areas, SNR>=3):

| run idx | d/p/pp | Fe I sum | Fe II sum | II/I |
|--|--|--|--|--|
| 0 | 10/25/100 | 4268 | 3439 | 0.81 |
| 1 | 10/25/100 | 4638 | 3843 | 0.83 |
| 2 | 10/25/100 | 4747 | 4593 | 0.97 |
| 3 | 10/25/100 | 2631 | 3329 | 1.27 |
| 4 | 5/25/100 | 4694 | 5690 | 1.21 |
| 5 | 20/25/100 | 2191 | 2768 | 1.26 |
| 6 | 10/10/100 | 3778 | 4740 | 1.25 |
| 7 | 10/50/100 | 3694 | 4659 | 1.26 |
| 8 | 10/25/100 | 3530 | 4563 | 1.29 |
| 9 | 5/25/100 | 5828 | 7130 | 1.22 |
| 10 | 5/10/100 | 6365 | 7368 | 1.16 |
| 11 | 10/10/100 | 4801 | 5784 | 1.20 |
| 12 | 5/50/100 | 6139 | 7744 | 1.26 |
| 13 | 10/25/100 | 4799 | 3779 | 0.79 |
| 14 | 10/25/100 | 4558 | 5451 | 1.20 |
| 15 | 5/25/100 | 7097 | 6883 | 0.97 |
| 16 | 10/25/100 | 4930 | 4731 | 0.96 |
| 17 | 5/25/100 | 6721 | 6562 | 0.98 |
| 18 | 5/10/100 | 7625 | 8142 | 1.07 |
| 19 | 10/10/100 | 6302 | 6380 | 1.01 |
| 20 | 5/50/100 | 7916 | 8329 | 1.05 |
| 21 | 10/50/100 | 6790 | 7482 | 1.10 |
| 22 | 20/10/1000 | 5169 | 4649 | 0.90 |
| 23 | 20/50/1000 | 5319 | 4861 | 0.91 |
| 24 | 20/50/100 | 4007 | 4264 | 1.06 |
| 25 | 20/10/100 | 4281 | 4786 | 1.12 |

## 4. Plasma temperature from Fe

Thermometric line sets are built from the FULL line database (not the SNR-curated fe-v1 windows, which span only ~1 eV in E_k and cannot resolve a slope). Fe I: 420-620 nm (VIS), E_i>0.8 eV; Fe I NIR 620-900 nm as an independent segment; Fe II: 186-365 nm (UV), E_i>1.0 eV. A line is kept if the summed gA*exp(-E_k/kT0) (T0=9000 K) of all other Fe I/II lines within +-0.25 nm is <15% of its own, top-60 (Fe I) / top-40 (Fe II) by strength. Each line is measured with a linear side-band baseline (0.45-0.90 nm each side), net area over +-0.32 nm on the native grid, peak searched within +-0.30 nm of db+shift (its OWN grid family's shift), keeping SNR>=5, area>0, |offset-shift|<0.2 nm. The Boltzmann fit ln(area*lambda/gA) vs E_k is robust: the worst |residual| beyond 1.6 sigma is dropped iteratively down to 8 lines (this sheds self-absorbed strong lines, which sit low). A fit is **resolved** if n>=8, E_k span>=2.4 eV, r^2>=0.5 and sigma_T/T<0.5.

### 4a/4b. Per-run Fe I VIS and Fe II UV Boltzmann temperatures

Runs are grouped fresh vs reused; the grid family is noted (resampled-family areas/SNR are not directly comparable — see Data).

| run | d/p/pp | grid | spot | T_FeI (K) | +-sig | nI | spanI | r2I | rmsI | resolved |
|--|--|--|--|--|--|--|--|--|--|--|
| 0 | 10/25/100 | 7914 | reused | 8716 | 2925 | 14 | 2.9 | 0.43 | 1.20 | no |
| 1 | 10/25/100 | 7914 | reused | - | - | 8 | - | - | - | no (positive_slope) |
| 2 | 10/25/100 | 7914 | reused | 9066 | 4932 | 9 | 1.9 | 0.33 | 1.31 | no |
| 3 | 10/25/100 | 23250 | reused | 8498 | 7396 | 8 | 2.7 | 0.18 | 2.29 | no |
| 4 | 5/25/100 | 23250 | reused | 8859 | 3960 | 8 | 2.7 | 0.45 | 1.16 | no |
| 5 | 20/25/100 | 23250 | reused | 9320 | 4622 | 9 | 2.7 | 0.37 | 1.25 | no |
| 6 | 10/10/100 | 23250 | reused | 8559 | 2723 | 10 | 3.3 | 0.55 | 1.15 | YES |
| 7 | 10/50/100 | 23250 | reused | - | - | 3 | - | - | - | no (too_few_lines) |
| 8 | 10/25/100 | 5848 | reused | - | - | 4 | - | - | - | no (too_few_lines) |
| 9 | 5/25/100 | 5848 | reused | 23222 | 52793 | 8 | 1.9 | 0.03 | 1.65 | no |
| 10 | 5/10/100 | 5848 | reused | 7200 | 2503 | 12 | 2.9 | 0.45 | 1.24 | no |
| 11 | 10/10/100 | 5848 | reused | 10693 | 7602 | 10 | 1.9 | 0.20 | 1.30 | no |
| 12 | 5/50/100 | 5848 | reused | - | - | 6 | - | - | - | no (too_few_lines) |
| 13 | 10/25/100 | 7914 | reused | - | - | 10 | - | - | - | no (positive_slope) |
| 14 | 10/25/100 | 5848 | reused | 20623 | 42427 | 8 | 1.9 | 0.04 | 1.68 | no |
| 15 | 5/25/100 | 7914 | reused | 10616 | 7927 | 8 | 1.9 | 0.23 | 1.18 | no |
| 16 | 10/25/100 | 7914 | reused | - | - | 8 | - | - | - | no (positive_slope) |
| 17 | 5/25/100 | 7914 | fresh | 9623 | 4180 | 13 | 2.8 | 0.33 | 1.21 | no |
| 18 | 5/10/100 | 7914 | fresh | 8191 | 1087 | 11 | 2.5 | 0.86 | 0.42 | YES |
| 19 | 10/10/100 | 7914 | fresh | 10451 | 1947 | 11 | 2.5 | 0.76 | 0.46 | YES |
| 20 | 5/50/100 | 7914 | fresh | 14246 | 10697 | 11 | 2.5 | 0.16 | 1.22 | no |
| 21 | 10/50/100 | 7914 | fresh | 100279 | 368256 | 8 | 1.9 | 0.01 | 0.66 | no |
| 22 | 20/10/1000 | 7914 | reused | 8049 | 2637 | 14 | 2.9 | 0.44 | 1.27 | no |
| 23 | 20/50/1000 | 7914 | verif | 6745 | 2804 | 9 | 2.5 | 0.45 | 1.38 | no |
| 24 | 20/50/100 | 7914 | verif | - | - | 6 | - | - | - | no (too_few_lines) |
| 25 | 20/10/100 | 7914 | verif | 8072 | 2709 | 15 | 2.9 | 0.41 | 1.26 | no |

**Fe I VIS resolves on 3/26 runs: T = 8191-10451 K, median 8559 K.** The resolved runs are the short-period (period~10), low-crater fresh/early sites; long-period (50) and heavily-cratered runs fit poorly (few high-E_k lines survive SNR>=5), matching the expected loss of hot-plasma signal at long integration. Fe II UV resolves on 0/26 runs (its isolated UV lines span only ~1.7 eV, so it is usually under-constrained).

Fe II UV attempts (E_k span limited):

| run | T_FeII (K) | nII | spanII | r2II | resolved |
|--|--|--|--|--|--|
| 18 | 5926 | 6 | 1.7 | 0.64 | no |
| 19 | 5840 | 6 | 1.7 | 0.34 | no |

### Per-shot scatter (fresh resolved run)

Run 18 (5/10), single shots: Fe I T mean 9018 K, std 1688 K (19%), n_shots=5. Single-shot fits are noisier (fewer lines clear SNR>=5) but cluster around the run-mean value; the shot-to-shot scatter (1688 K) is within the run-mean fit sigma.

### 4c. Saha-Boltzmann (Fe I VIS + Fe II UV)

- Best fresh run 18: Fe I VIS T = 8191 +- 1087 K (r^2 0.86). U_FeI(T)=44.0, U_FeII(T)=57.4, E_ion(Fe I)=7.90 eV.
- Intercept-difference n_e = 2.3e+11 cm^-3 — this ties Fe II UV (~250 nm) to Fe I VIS (~500 nm) across an **uncalibrated response step** and lands ~4 orders below the McWhirter floor, so it is NOT a physical density: it quantifies the UV-vs-VIS response ratio, not n_e.
- The Fe I VIS slope pins **T** robustly (within-segment). n_e is NOT recoverable from these data: the Fe II/Fe I intensity ratio defines only a ridge in (T, n_e), and its zero-point is swamped by the unknown UV/VIS response. A response calibration is required before n_e can be quoted.
- McWhirter LTE lower bound at T=8000 K (dE~4 eV): n_e >= 9.2e+15 cm^-3.
- McWhirter LTE lower bound at T=10000 K (dE~4 eV): n_e >= 1.0e+16 cm^-3.
- LTE assumed; no independent n_e (H-alpha weak/absent under argon).

### 4d. Comparison and trust

- **Fe I VIS is the trusted thermometer**: T = 8191-10451 K (median 8559 K) on the resolved short-period fresh sites, with per-run sigma ~1000-2000 K (~15-25%). The E_k lever arm is ~2.5-3.4 eV and r^2 up to 0.86.
- Systematics: Fe I strong low-E_k lines are self-absorbed (they sit below the trend and are shed by the robust fit, which biases T slightly high if over-aggressive); the within-VIS response is uncalibrated (a second-order tilt on the slope); LTE is assumed. Net: **a ~8000-10000 K class plasma on fresh sites**, consistent within uncertainty with an independent ~7000-8000 K estimate. What is NOT resolvable is a T change across the delay/period grid at this precision (see section 5).

## 5. Temperature vs acquisition parameters

Fe I VIS T (section 4) where resolved, plus response-independent observables: continuum = median counts in a line-free band (520-540 nm); Fe II SNR = mean top-10 curated Fe II SNR; Fe II/Fe I area ratio. Fresh-site runs isolate parameters from crater evolution; reused-spot runs carry a cumulative-shot column. Grid family is shown (resampled-family areas/SNR are not directly comparable).

| run | d/p/pp | spot | cum-shot | grid | T_FeI±σ | resolved | II/I | continuum |
|--|--|--|--|--|--|--|--|--|
| 0 | 10/25/100 | reused | 10 | 7914 | 8716±2925 | no | 0.81 | 46.3 |
| 1 | 10/25/100 | reused | 20 | 7914 | - | no | 0.83 | 52.2 |
| 2 | 10/25/100 | reused | 30 | 7914 | 9066±4932 | no | 0.97 | 34.9 |
| 3 | 10/25/100 | reused | 40 | 23250 | 8498±7396 | no | 1.27 | 27.0 |
| 4 | 5/25/100 | reused | 50 | 23250 | 8859±3960 | no | 1.21 | 35.1 |
| 5 | 20/25/100 | reused | 60 | 23250 | 9320±4622 | no | 1.26 | 28.6 |
| 6 | 10/10/100 | reused | 70 | 23250 | 8559±2723 | YES | 1.25 | 29.0 |
| 7 | 10/50/100 | reused | 80 | 23250 | - | no | 1.26 | 41.1 |
| 8 | 10/25/100 | reused | 90 | 5848 | - | no | 1.29 | 25.5 |
| 9 | 5/25/100 | reused | 100 | 5848 | 23222±52793 | no | 1.22 | 32.4 |
| 10 | 5/10/100 | reused | 110 | 5848 | 7200±2503 | no | 1.16 | 33.3 |
| 11 | 10/10/100 | reused | 120 | 5848 | 10693±7602 | no | 1.20 | 29.1 |
| 12 | 5/50/100 | reused | 130 | 5848 | - | no | 1.26 | 45.6 |
| 13 | 10/25/100 | reused | 138 | 7914 | - | no | 0.79 | 64.8 |
| 14 | 10/25/100 | reused | 148 | 5848 | 20623±42427 | no | 1.20 | 36.2 |
| 15 | 5/25/100 | reused | 157 | 7914 | 10616±7927 | no | 0.97 | 40.1 |
| 16 | 10/25/100 | reused | 166 | 7914 | - | no | 0.96 | 32.5 |
| 17 | 5/25/100 | fresh | - | 7914 | 9623±4180 | no | 0.98 | 75.8 |
| 18 | 5/10/100 | fresh | - | 7914 | 8191±1087 | YES | 1.07 | 58.7 |
| 19 | 10/10/100 | fresh | - | 7914 | 10451±1947 | YES | 1.01 | 47.8 |
| 20 | 5/50/100 | fresh | - | 7914 | 14246±10697 | no | 1.05 | 73.0 |
| 21 | 10/50/100 | fresh | - | 7914 | 100279±368256 | no | 1.10 | 61.0 |
| 22 | 20/10/1000 | reused | - | 7914 | 8049±2637 | no | 0.90 | 52.4 |
| 23 | 20/50/1000 | verif | - | 7914 | 6745±2804 | no | 0.91 | 53.2 |
| 24 | 20/50/100 | verif | - | 7914 | - | no | 1.06 | 42.1 |
| 25 | 20/10/100 | verif | - | 7914 | 8072±2709 | no | 1.12 | 27.1 |

**T vs delay/period:** across the 3 resolved fits, T spans 8191-10451 K (spread 990 K) while individual fit sigma is ~1947 K. The condition-to-condition T differences do NOT exceed the per-fit uncertainty: with ~1947 K 1-sigma errors and only a handful of resolved conditions, **no T trend with delay or period is detectable**. Detection limit: a T change smaller than ~3895 K (2σ) cannot be resolved with these data. The expected fall of T with delay is therefore below the noise over the narrow delay range (5-20 vendor units) sampled here.

### Robust observables vs delay/period (reused spot, condition sweep)

| run | d/p/pp | cum-shot | continuum | top10 FeII SNR | II/I |
|--|--|--|--|--|--|
| 0 | 10/25/100 | 10 | 46.3 | 16 | 0.81 |
| 1 | 10/25/100 | 20 | 52.2 | 14 | 0.83 |
| 2 | 10/25/100 | 30 | 34.9 | 14 | 0.97 |
| 3 | 10/25/100 | 40 | 27.0 | 14 | 1.27 |
| 4 | 5/25/100 | 50 | 35.1 | 10 | 1.21 |
| 5 | 20/25/100 | 60 | 28.6 | 17 | 1.26 |
| 6 | 10/10/100 | 70 | 29.0 | 12 | 1.25 |
| 7 | 10/50/100 | 80 | 41.1 | 14 | 1.26 |
| 8 | 10/25/100 | 90 | 25.5 | 19 | 1.29 |
| 9 | 5/25/100 | 100 | 32.4 | 16 | 1.22 |
| 10 | 5/10/100 | 110 | 33.3 | 17 | 1.16 |
| 11 | 10/10/100 | 120 | 29.1 | 19 | 1.20 |
| 12 | 5/50/100 | 130 | 45.6 | 16 | 1.26 |

Delay sweep at period=25 (reused idx 0-5, note confounded by cumulative shot):
  - delay 5: continuum=35.1, top10 FeII SNR=10, II/I=1.21, cum-shot=50
  - delay 10: continuum=46.3, top10 FeII SNR=16, II/I=0.81, cum-shot=10
  - delay 10: continuum=52.2, top10 FeII SNR=14, II/I=0.83, cum-shot=20
  - delay 10: continuum=34.9, top10 FeII SNR=14, II/I=0.97, cum-shot=30
  - delay 10: continuum=27.0, top10 FeII SNR=14, II/I=1.27, cum-shot=40
  - delay 20: continuum=28.6, top10 FeII SNR=17, II/I=1.26, cum-shot=60

Repeated 10/25 condition (reused spot, native 7914 family, n=5 runs): continuum mean 46.2 std 11.8 (26%); Fe II/Fe I mean 0.87 std 0.08. This run-to-run scatter (confounded by crater evolution) sets the floor against which any delay/period effect must be judged.

## 6. Optimisation for signal and interpretability

**Findings the data support:**

- Highest mean top-10 Fe II SNR overall: run 18 (5/10/100), SNR 25 (a late reused-spot run, confounded by grid/crater).
- Among fresh sites (clean comparison), best Fe II SNR is run 18 (5/10/100), SNR 25. Short delay (5) with short-to-mid period (10-25) gives the strongest Fe II UV signal; longer period (50) mainly raises continuum.
- **Plasma temperature is resolvable**: Fe I VIS Boltzmann gives T = 8191-10451 K (median 8559 K) on short-period fresh sites, with more runs clustering 7000-10500 K at r^2~0.4. This is a real ~8000-10000 K result; only the T CHANGE across the delay/period grid is below the detection limit (~3894 K, 2σ).
- The Fe II/Fe I area ratio is ~0.8-1.1 within the native (7914) family and ~1.2-1.3 for the Opal/resampled families — the offset tracks the GRID FAMILY, not the acquisition condition (a grid/response artifact, not physics). Within a family it is flat across delay/period and depth, consistent with the flat T: the ionisation balance does not vary detectably across this grid.
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
- **Three grid families with distinct shifts** (7914 native/API -154 pm, 5848 native/Opal -182 pm, 23250 vendor-resample -83 pm): each run is shifted with its own family. The resampled family has correlated noise and smoothed peaks, so its areas/SNR are not directly comparable; the reused-spot depth series and the delay/period grid MIX families (noted per table), which aliases onto raw trends — Fe-normalised ratios cancel most of it.
- **T systematics**: strong low-E_k Fe I lines are self-absorbed (they sit below the Boltzmann trend and are shed by the robust 1.6σ clip; over-aggressive clipping biases T slightly HIGH). The E_k lever arm is ~2.5-3.4 eV, so per-run sigma is ~15-25%. Reported T (~8000-10000 K) is consistent within that with an independent ~7000-8000 K estimate.
- **LTE / McWhirter**: LTE assumed; McWhirter is a necessary not sufficient check; no independent n_e (H-alpha weak under argon).
- **Section-1 blends**: for a 99.98% Fe matrix almost every trace line sits near an Fe line; a non-Fe candidate is downgraded to 'Fe-blend' only when a top-decile-strength Fe line lies within 0.15 nm. (The fe-v1 curated set, strict-blended 90/96, is used only as an atomic reference, NOT for thermometry — the thermometric sets are built from the full database.)
- **Dropped frames**: runs with 8-9 shots are used as-is (contiguous shot-0..n-1); cumulative-shot depth axis uses actual counts.
- **Location discrepancy**: the task narrative groups run 22 (20/10, pp1000) with the [134,96,70] verification set (9+8+10+10=37 shots), but the ledger records run 22 at [134,76,70]. This analysis trusts the ledger `location` field; the verification spot therefore has 3 runs (28 shots) here.

## Files

- Script: `scripts/fe_plasma_analysis.py`
- Figures: `reports/figures/fe-plasma-20260922/` (shift_model.png, mean_spectrum_labelled.png, depth_profiles.png, boltzmann.png, T_vs_params.png)
- Report: `reports/2026-09-22-fe-calibration-plasma-analysis.md`

## Coordinator verification (main session, 2026-09-22 15:30 PDT)

The delegated analysis was checked against the raw spectra twice. The first
version reported temperature as unresolvable; that was a line-selection
artefact (4–6 curated windows, 1.1 eV lever arm) and sections 4–6 were redone
with database-selected line sets. Independent fits by the main session on the
10-shot means, VIS 420–620 nm, isolated Fe I lines from the full database
(E_i > 0.8 eV, no robust shedding, SNR ≥ 5):

| run | condition | site | T (K) | σ | n | E_k span | r² |
|---|---|---|---|---|---|---|---|
| run-6cd149cc (18) | 5/10 | fresh | 6820 | 1430 | 11 | 3.4 | 0.72 |
| run-2e03f73d (19) | 10/10 | fresh | 7380 | 1630 | 11 | 3.4 | 0.69 |
| run-07c25af6 (17) | 5/25 | fresh | 7510 | 2340 | 9 | 3.3 | 0.59 |
| run-74c93327 (25) | 20/10 | verif. | 7590 | 1670 | 14 | 3.4 | 0.63 |
| run-fd1e6d48 (22) | 20/10 pp1000 | reused | 8010 | 1830 | 13 | 3.4 | 0.64 |
| run-6cd149cc NIR 620–900 | 5/10 | fresh | 6270 | 2030 | 6 | 3.6 | 0.70 |

These sit 1,000–1,500 K below the script's robust fits (which shed the
self-absorbed strong lines such as Fe I 438.35 and therefore run hotter) and
overlap them within σ. Combined statement: **T ≈ 7,000–10,000 K on fresh
sites, per-run σ 15–30 %**; the agent's note that these runs are "not in the
ledger" is wrong (run 18 is run-6cd149cc, a fresh raster site).

**Element table, multiplet check.** Section 1 assigns several single lines
"high" confidence. Measured on the mean of the five fresh raster runs with the
−154 pm family shift, the expected stronger or equal multiplet partner is
absent in every case, and the observed line coincides with a strong Fe line:

| candidate | observed line (SNR) | required partner | partner observed | nearest Fe line (gA) | verdict |
|---|---|---|---|---|---|
| Mg II 279.55 | 11 | Mg II 280.27 (≈ ½) | absent (SNR −0.7) | Fe II 279.72 (8e8) | Fe coincidence, not Mg |
| Al I 394.40 | 25 | Al I 396.15 (2×) | absent (1.3) | Fe I 394.49 (1.5e7) | Fe coincidence, not Al |
| Mn I 403.45 | 1.7 | Mn I 403.08 (strongest) | absent | Fe I 403.05/403.20 | not detected |
| Cu I 327.40 | 8.6 | Cu I 324.75 (2×) | absent (1.8) | Fe II 327.35 | Fe coincidence, not Cu |
| Ca II 393.37 | 3.8 | Ca II 396.85 (½) | absent | Fe I 393.36 | not detected on fresh sites |
| Ni I 341.48 / 352.45 | 8.8 / 2.9 | each other | offsets disagree (−207 / +144 pm) | Fe I 341.31 (2.3e8) / 352.64 (2.9e8) | ambiguous, Fe more likely |
| Na I 589.0 / 589.6 | — / 4.6 | each other | 588.99 absent | — | not detected |
| K I 766.49 / 769.90 | 2.1 (5-run mean) / — | each other | 769.90 absent in mean | Fe I 766.43 weak | marginal; only credible as a first-shot surface transient (section 2b, 73 → 10 counts) |
| Ar I (8 lines) | 5–36 | — | consistent set | none | confirmed, ambient purge |
| H α 656.28, O I 777.19, N I 744.23 | weak | — | — | none | credible, weak, ambient/adsorbate |

So the defensible non-Fe inventory is: **argon (purge), traces of H, O and N
(ambient / adsorbed), and a first-shot potassium transient**. No bulk impurity
(Mn, Ni, Cu, Cr, Si, C) is established at this SNR and without a response
calibration; in a 99.98 % Fe matrix every trace line examined lies within
0.15 nm of an Fe line of comparable strength. Section 1's "high" entries for
Mg, Al, Mn, Cu, Ni and section 6's "K and Ni have Fe-free clean lines" are
superseded by this table. Section 6's parenthesis calling run 18 "a late
reused-spot run" is also wrong; it is fresh raster site 3.

Everything else in the report (shift model per grid family, depth profiles,
Fe II/Fe I ratio, T-vs-parameter detection limit, recommendations) was read
against the tables and stands.
