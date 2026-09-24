# Native vs "spline" A/B on the same Z300 shots (2026-09-23)

Author: specialist (Opus 4.8), for the Z300 acquisition-path quality-drop investigation.
Provider: Claude subscription (no fallback switch this session).
Data: 8 Fe runs from 2026-09-22, each with BOTH a 23,249/23,250-row "old" average CSV
(imported 16:40-16:44 UTC) and a 7,914-sample "native" average (refetched 17:01/22:12 UTC),
SAME analyzer `test_id` per pair. Pulled to scratchpad from moissanite
`~/.local/state/pantheum/alibz/{originals,acquisitions}` (numpy/scipy absent on moissanite;
all analysis run locally with the repo `.venv` + `alibz.utils.native_grid`/`peakfit`).
Script: `cal/scripts/native_vs_spline_ab.py` (deterministic, `--help`; subcommands
provenance/stitch/noise/sampling/peaks/profiles/all).

Pairs (run : OLD dataset : NATIVE dataset), all `test_id`-matched, 10 shots each:
e825d9f567:2679219a5c6d:c0c80c9313 · a50ac639cc:4f678d792ef0:9e26b7319b85 ·
eae9d54f22:0982aedefc2b:d7cb358901 · 126d0c96e1:6f0d21db2cc1:99c7e7afcb7b ·
5291e64308:27858788c9f2:c54077b06b73 · 11d07e557f:a37cd7368dea:b06a1b91cbc6 ·
6e38b11021:0fefcd7830c5:8864309665c2 · c52b7f2159:8e75d878e1c2:9d1aa9a17c87

---

## HEADLINE

The "old" 23,250-point file was **NOT the vendor cubic-spline export**. It is **Pantheum's own
`resample_uniform` — LINEAR interpolation of the SAME native pixels onto a uniform 1/30 nm grid**.
Old and native therefore come from *identical averaged pixels* and the *identical* pixToNm map.
There is **no wavelength-map, radiometry, dark, gain, or averaging difference** between them.
The switch changed only (a) grid DENSITY + interpolation, (b) NIR-gap/edge handling, (c) the dead
4th-channel placeholder, and (d) noise correlation. Native is the more faithful representation;
the quality drop is a **sampling/estimator effect plus two genuine native-path packaging defects**,
not a loss of measurement fidelity. Much of what made the old data "look" better (smoother peaks,
tighter centroids) was linear interpolation *adding correlated smoothing and no information*.

---

## 1. PROVENANCE (SETTLED)  — `native_vs_spline_ab.py provenance`

For all 8 runs: `old = linear_interp(native)` to rel-RMS **2.4e-4 – 3.1e-4** (at the 4-decimal CSV
rounding floor); `old = cubic_spline(native)` is rel-RMS **0.09 – 0.22** (up to 22% off). The data
is piecewise LINEAR between native samples: 2nd differences are frequently exact-zero (0.11 of a
quiet 400-500 nm span), 4th differences essentially never (<0.003) — the vendor cubic-export
signature (exact-zero 4th diffs between knots) is ABSENT.
Code: `pantheum-I pantheum/alibz/z300_calibration.py:190 resample_uniform` is LINEAR (`np.interp`
form, 1/30 nm, lo-knot→hi-knot); `pantheum/alibz/acquire.py:64 GRID='native'`, `:1018
_sorted_unique_csv` = current native path (`pixels_to_wavelength` on the pixel-averaged shot,
drop out-of-order dupes, no resample). Switch commit: `f466332` ("first live Z300 fires, native
grid, …"), deployed to moissanite between 16:44 and 17:01 UTC on 2026-09-22.

Consequence: the docs' "-70 pm grid-family" difference (native-API −154 pm vs vendor-resample
−83 pm vs Opal −182 pm, `docs/z300_device_notes.md`) does **not** apply here — the old file is the
API path, same map as native. No wavelength-map comparison is needed between these two (verified:
line centroids coincide because old is literally an interpolation of native).

## 2. WAVELENGTH MAP — identical
Both use the average shot's single `wlCalibrations` cubic pixToNm per segment. Within every run all
10 shots carry ONE identical calibration (max coef std ~1e-13). No per-segment map difference.

## 3. INTENSITY TRANSFORM — identical counts
old = linear(native) ⇒ identical counts (no additive/dark, gain, nonlinearity, clip or smoothing
difference). Any detector fixed-pattern (e.g. the V-study period-scaled NIR baseline) is present in
BOTH equally. These Fe runs are integrationPeriod=25 (not the period-50 V study).

## 4. AVERAGING & STITCHING — `stitch`
- **Averaging is a non-issue for these runs.** The stored average is analyzer shot index −1, which
  is the EXACT pixel-wise mean of the 10 shots (max|avg − mean(pixels)| = 0.000). Because all shots
  share one calibration, pixel-index averaging == wavelength-aligned averaging here. (The memory
  note of 2.0–3.5 px NIR per-shot drift is BETWEEN tests, not within a single test's shots.)
- **DEAD 4th CHANNEL (genuine native defect).** Segment 3 = 2066 pixels, **exactly 1 non-zero**,
  all crammed into knots [960,961] nm. That is **2066 samples = 26% of every 7,914-point native
  file that are all-zero placeholder** at 960–961 nm. `resample_uniform` collapsed that block to
  ~30 samples, so the old file did not carry it.
- **NIR gap real (12.2–12.3 nm, 947.9→960.2 nm)** between seg2 end and the placeholder. The old
  1/30 file spanned it with a **fake linear ramp**; native correctly leaves a gap.
- No dropped/duplicated real pixels: seg kept counts 2027+1985+1836+2066 = 7914 exactly.

## 5. NOISE — `noise`
Native high-freq noise is **white** (lag-1 autocorr median **0.12**); the old 1/30 file is
**correlated** (median **0.76**), induced purely by linear interpolation. Old peaks therefore
*look* cleaner/smoother; native is the honest noise.

## 6. SAMPLING (root cause) — `sampling`
Native pitch ≈ 90 pm UV / 129 pm VIS / 179 pm NIR. Measured line FWHM (half-max, e825): UV
180–280 pm, VIS ~275 pm, NIR ~350–400 pm ⇒ **native ≈ 1.3–2.2 samples/FWHM** (VIS/NIR at or below
the Nyquist limit of 2). Fit-based estimate (all runs): UV med 1.45, VIS med 0.84 samples/FWHM.
The 1/30 grid gives **~3.5–6 samples/FWHM** for the same lines. This under-sampling is INTRINSIC to
the hardware and is explicitly documented in `alibz/utils/peakfit.py` ("native FWHM ~1.3–1.8 px")
and `pantheum/alibz/optimization_metrics.py:67-74` ("a fixed nm window holds only one or two
samples"). Linear resampling did not fix it — it hid it behind 3× interpolated points.

## 7. PEAK LOCATIONS & PROFILE FITS  (owner priority)  — `peaks`, `profiles`
- **Peak-location scatter.** Shot-to-shot centre std (production 3-point parabola, guarded, n≈1690):
  native **4.2 pm**, linear-1/30 **2.0 pm**, cubic-1/30 4.4 pm. Interpolation halves the *apparent*
  jitter — because it smooths noise (§5), not because it adds information. Native peak positions are
  also quantized to ~pitch/√12 (≈26/37/52 pm UV/VIS/NIR) vs ~9.5 pm on 1/30.
- **Profile fits** (free pseudo-Voigt, 1237 lines): r² comparable (old 0.892 / native 0.888 /
  cub 0.875), BUT fitted **FWHM is grid-dependent and biased**: native **118 pm** (≈25% LOW,
  under-sampling), old/linear **159 pm** (biased HIGH — linear interp triangulates peaks), cubic
  **149 pm** (between, most physical). Lorentz fraction η is unconstrained on ALL grids
  (η pinned at bound 0.79–0.91 of fits) — free-width Voigt is ill-posed at ≤2 samples/FWHM
  regardless of grid. This is why the owner sees poor profiles on native: a free-width fit that the
  old dense grid propped up now under-resolves.
- **The correct fix already exists in production**: `alibz/utils/peakfit.py`
  `gaussian_subpixel_center`/`refine_peaks` fix the instrument width in PIXELS
  (`DEFAULT_PROFILE_PX` UV/VIS 1.4 px, NIR 1.65 px) and convert via local pitch, exactly to beat
  the under-sampled-parabola bias. `alibz/pipeline.py:634 refine_peaks` uses it. The DEGRADED path
  is the legacy free/parabolic estimators still used by the calibration scripts
  (`cal/scripts/fe_plasma_analysis.py:244 subpixel_center`, and its 3–5-sample nm windows).

## 8. ATTRIBUTION TABLE (same shots)

| Difference (old→native) | Metric impact | Verdict |
|---|---|---|
| Grid density 3.5–6 → 1.3–2.2 samples/FWHM | free-width FWHM 159→118 pm (~25% low); η unconstrained; centroid scatter 2.0→4.2 pm | ROOT CAUSE for peaks/profiles; fix = pixel-width fitter (exists) |
| Linear interp removed | noise correlation 0.76→0.12 (white) | native honest; old looked smoother |
| Dead 4th channel kept | 26% of file = 2066 zeros @960–961 nm | GENUINE native defect; drop it |
| NIR gap exposed | 12.2 nm real gap vs old fake ramp | native correct; must be masked downstream |
| Pantheum resample was LINEAR not CUBIC | old FWHM biased HIGH, noise correlated | if oversampling at all, use cubic |
| Wavelength map / dark / gain / averaging | none (identical) | ruled out |

## 9. RANKED ERRORS & MISSED OPPORTUNITIES (+ fixes, file:line)

1. **Legacy under-sampled peak/width estimators run on native** (biggest owner-visible effect).
   Free-width Voigt and 3-point parabola give ~25%-low widths, unconstrained η, 2× centroid scatter.
   FIX: route all peak/centroid/width work through `alibz/utils/peakfit.py:refine_peaks`
   (pixel-fixed instrument width + local pitch). Replace `cal/scripts/fe_plasma_analysis.py:244
   subpixel_center` and its `MIN_CENTRAL_PX/SIDEBAND_PX` (:53-54) nm windows with the peakfit path;
   audit any other consumer that fits a FREE width or a fixed-nm window.
2. **Dead 4th-channel placeholder inflates every native file by 26%** (2066 zeros @960–961 nm).
   Pollutes whole-spectrum normalization, peak-finding, and per-sample metrics; trips zero-guards
   (`optimization_metrics.py:102`). FIX: drop segment 3 (or the all-zero [960,961] block) in
   `pantheum/alibz/acquire.py:1018-1034 _sorted_unique_csv`.
3. **NIR gap must be masked, not fitted across.** peakfit clips at pitch jumps
   (`peakfit.py _GAP_FACTOR=2.5`) — good; ensure every other consumer (indexer windows,
   background/continuum, corpus response) masks 947.9–960.2 nm. Old's fake ramp silently fed real
   values there.
4. **Missed opportunity — if oversampling for legacy tools, use CUBIC not LINEAR.**
   `resample_uniform` (`z300_calibration.py:190`) is linear: it triangulates peaks (width bias
   high) and correlates noise. The vendor's own kernel is not-a-knot cubic
   (`alibz/utils/native_grid.py export_kernel_matrix`); `cubup` recovers a more physical width
   (149 vs 159/118 pm). Prefer native pixels + pixel-width fitter; use cubic only as a
   compatibility shim for un-migrated consumers.
5. **Missed opportunity — keep the white native noise for correct σ.** GLS/χ² weighting should use
   native (white) per-pixel noise; do NOT reintroduce interpolation (which correlated it 0.76).

---

## UNVERIFIED / OPEN
- **End-to-end `analyze_spectrum` (r²/qc_status) and Pantheum window-quality NOT run per pair.**
  The measured primitive drop is MODERATE (r² comparable; width bias ~25%; centroid scatter 2→4 pm).
  If the owner perceives a *substantial* end-to-end drop, it is most likely AMPLIFIED by the
  downstream indexer (PeakyIndexerV3) turning ~25% width / quantized-centroid errors into larger
  composition/T errors, and/or by the dead-channel + gap in whole-spectrum steps. Direct next step:
  build old-grid per-shot files by linear-resampling each native shot, then run
  `pantheum/alibz/optimization_metrics.analyze_batch` and `alibz/pipeline.analyze_spectrum` on both
  representations of the SAME run (feasible; ~20 min/run) to close the attribution numerically.
  This report establishes the mechanism, not the end-to-end score delta.
- Fit-based FWHM (§6 sampling cmd) is itself under-sampling-biased low; the robust half-max FWHM
  (§6) is the reliable sampling figure.
- η (Lorentz fraction) could not be constrained on any grid here; a Voigt decomposition needs the
  pixel-integrated model, not denser resampling.
- Only Fe integrationPeriod=25 runs examined; the period-scaled NIR fixed pattern (V study, period
  50) was not re-tested (it is identical in old and native by construction, so irrelevant to this A/B).
