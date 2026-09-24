# Native-grid method audit — estimators & code paths on undersampled data

Date: 2026-09-23
Author: mwhittaker (specialist agent)
Scope: METHOD side only. Which estimators/code paths are biased, noisy or
mis-tuned on the 7,914-sample native grid (UV 0.078–0.096, VIS 0.113–0.142,
NIR 0.156–0.198 nm/px; line FWHM ≈1.0–1.6 px). The parallel agent owns the
representation comparison (wavelength map, intensity, averaging, stitching);
not duplicated here.

Report written early and updated as work proceeded (per brief).

Coordinator re-prioritisation (mid-task): rank PEAK-LOCATION and PROFILE-FIT
findings first; prototype a pixel-integrated fit on real native lines and
report the gain in centroid scatter and fit residuals. Done in §A and §2.3.

---

## A. Ranked findings — peak location & profile fit (coordinator priority)

Impact (effect size on real data) × Confidence. `file:line` for each.

Verified context from the parallel A/B agent (`reports/2026-09-23-native-vs-spline-ab.md`):
the old 23,250-pt file is Pantheum's LINEAR resample of the SAME native pixels
(same map, same intensities) — the quality drop is sampling/estimator/packaging,
not lost fidelity; and every native file carries a dead 4th channel (2,066
samples at 960.18–961.0 nm, 2,065 zero) that drags `median(np.diff(x))` to
0.0955 nm vs 0.1277 nm without it (true per-segment 0.078–0.198). Both
independently reproduced here (§2.4).

### A0. Dead-channel-corrupted GLOBAL pitch feeds the PRODUCTION Voigt seed/cap/floor — HIGHEST impact / HIGH conf
The production profile fitter takes ONE pitch for the whole spectrum,
`inc = np.median(np.diff(x))`, at **`peaky_finder.py:871`** (`fit_all`), and the
identical pattern at **`peaky_finder.py:1046`** (`_fwhm_cap`), **:1078**
(`fit_residual_peaks`), **:1215** (`fit_global`), **:1403** (`analyze`), and
**:492** (`find_background` fallback `lam=(smooth_nm/pitch)**4`). That single
`inc` then:
- scales the per-peak seed Voigt FWHM and the sigma FLOOR:
  `_voigt_seed` **`peaky_finder.py:129-130`** `width = fwhm_idx*min_inc; width =
  max(width, min_inc)`, with `min_inc = inc` (871/876), and the seed amplitude
  `amp = height/V(0;σ,γ)` (135) inherits it;
- scales `median_fwhm = median(fwhm_array)*inc` (**:874**) and the width cap
  `fwhm_cap = FWHM_CAP_FACTOR(=2.0)*max(median_fwhm, min_inc)` (**:886-887**,
  `_fwhm_cap` **:1046-1049**).

On native data `inc = 0.0955 nm` (dead channel) while the true pitch is
0.078–0.096 (UV) / 0.113–0.142 (VIS) / 0.156–0.198 (NIR). The seed width, the
width cap and the σ floor are therefore mis-scaled per segment by
**+8% (UV) / −26% (VIS) / −47% (NIR)** (or +44/−1/−29% if the dead channel were
trimmed) — measured in §2.4. Exercising the ACTUAL `_voigt_seed` on a genuine
1.1-px NIR line: seed FWHM 0.105 nm = **0.59 px** (should be 1.10 px) and seed
amplitude 143 (should be 268) with the dead-channel pitch. The fit starts
~half-width and ~half-amplitude and the σ floor (0.0955 nm = 0.53 px in the NIR)
lets it stay sub-pixel, while the uniform-nm cap over-constrains the physically
wider NIR lines and under-constrains the UV. This is the leading method cause of
"poor peak locations and fit profiles" on native data and it is dormant on the
old uniform grid (there `median(diff)` = the true, constant pitch everywhere).
Fix (read-only proposal): compute pitch PER SEGMENT at the segment edges
(365/620 nm, already known) and pass the LOCAL pitch to `_voigt_seed`,
`median_fwhm` and the cap; and trim/exclude the dead 4th channel (x ≥ 960.18 nm,
all-zero) before any `np.diff`-based statistic. the good patterns to
replicate already exist nearby: `_segment_spans` (applied at
`peaky_finder.py:500`) and `valid_signal_mask` (`:498`, which already excludes
the dead pad from the background) — extend the same per-segment / dead-edge
handling to `fit_all`, `_fwhm_cap`, `fit_global`, `analyze`. NOTE
`find_background` itself is only PARTLY safe: it segments the ARPLS but its
stiffness `lam=(smooth_nm/pitch)**4` at `:492` still uses the GLOBAL pitch, so
the baseline smoothing length is also mis-scaled per segment (fix there too).

### A1. Point-sampled line shape everywhere (`voigt.py:84`) — HIGH impact / HIGH conf
`alibz/utils/voigt.py:83-85` evaluates `scipy.special.voigt_profile` AT the
sample abscissae (`voigt(x[:,None]-mu, sigma, gamma)`), never integrated over
the pixel. This `multi_voigt` is the SHARED line shape used by
`alibz/profiles.py` (analyze_peak_profiles, deblend_shoulders, recover_sa_areas),
`alibz/peaky_indexer_v3.py` (forward_model / whole-spectrum CF-LIBS fit,
overlaps at voigt(0,·)), and `alibz/minor_lines.py`. A point-sampled peak of
FWHM 1.0–1.6 px systematically MIS-STATES the per-pixel flux: near a pixel
centred on the peak it over-states, on a pixel straddling the peak it
under-states, so amplitude and area pick up a sub-pixel-phase-dependent bias
and the fitter compensates by broadening σ. On the old 4–5× oversampled 1/30 nm
grid the same code was near-exact (many points per FWHM), which is why quality
dropped at the switch. Quantified in §2.1/§2.2. Fix: pixel-integrated forward
model (erf for the Gaussian core; analytic or fine-oversample+bin for Voigt),
per-segment pitch. Prototype in §2.3.

### A2. Parabolic 3-point centroid drives the shift model (`fe_plasma_analysis.py:244-259`) — HIGH / HIGH
`subpixel_center` fits a 3-point parabola to raw `y[j-1],y[j],y[j+1]`
(lines 251-256) and is the peak-location estimator behind
`estimate_shift_model` (`line_physics_analysis.py:175-237`, docstring line 179
"sub-pixel parabolic centroid") for BOTH the Fe and V shift studies. The
parabolic estimator of a Gaussian sampled at ≈1.3 px FWHM has a well-known
systematic S-shaped bias vs sub-pixel phase; at critical sampling its amplitude
is a large fraction of the residual budget. This is a prime suspect for the
"~50 pm centroid scatter in the shift models." Quantified in §2.1. Fix:
pixel-integrated Gaussian/Voigt centroid (§2.3).

### A3. Peak location snapped to the sample grid (`fe_plasma_analysis.py:146,151`; `v_timing_study.py:113,138`) — HIGH / HIGH
`measure_line` returns `peak_nm = wl[argmax(heights)]` (fe_plasma:146,151):
the reported line position is SNAPPED to the nearest native sample. Uniform
±0.5 px snap error has std = pitch/√12 ≈ 0.038 nm in the VIS (38 pm) and
≈0.051 nm in the NIR — by itself most of the observed ~50 pm scatter. Every
`local_area`/`robust_height` recentre in `v_timing_study.py` (113,138) and
`find_peaks_simple` (fe_plasma:270) inherits it. Fix: sub-pixel center for any
position/window anchoring (§2.3).

### A4. Local-max HEIGHT as the ratio estimator (`v_timing_study.py:127-154`) — HIGH / HIGH
`robust_height` (127-145) takes the single largest baseline-subtracted sample
as the line height, and `pair_height_ratio`/`track_pair` build line RATIOS from
it. On an undersampled line the sampled max sits BELOW the true peak by a
phase-dependent amount (0 when a pixel is centred, up to ~1−exp(−(0.5px)²/2σ²)
when straddled). Two lines of a doublet sit at independent phases, so the bias
does NOT cancel in the ratio. This is a leading candidate for the measured
417.94/415.97 height ratio of 0.75 vs the optically-thin 2.05. Quantified in
§2.2 (how much of the 0.75 is estimator vs real self-absorption). Fix:
pixel-integrated amplitude, or area, not sampled height, for ratios.

### A5. Sub-pixel width floors in the profile fits (`peaky_indexer_v3.py:2371-2373`; `line_physics_analysis.py:278`; `peakfit.py` OK) — MED-HIGH / HIGH
- `peaky_indexer_v3.py:2371-2373`: default `sigma_bounds=(0.01, …)` and
  `gamma_bounds=(0.01, …)` in **nm**. 0.01 nm = 0.08 px (VIS) / 0.06 px (NIR):
  the fit can place essentially all the line inside one pixel — degenerate and
  unconstrained on native data, and σ/γ are mutually degenerate at 1–1.5 px
  FWHM (both bounds are nm constants tuned when a pixel was 0.033 nm). 
- `line_physics_analysis.py:278`: `measure_fwhm` sigma lower bound `0.2*pitch`
  (0.2 px) over only ±3 samples with a POINT-sampled Gaussian — admits
  sub-pixel spikes and is point-sampling-biased.
- `alibz/utils/peakfit.py` is the exception and mostly RIGHT: widths carried in
  px, per-segment σ, windows in px, local pitch (`_local_pitch`), `sigma_bounds`
  as multiples of the instrument σ. Its one residual gap is that its Gaussian is
  also point-sampled (`_gauss`, line 87; used at line 143) — same A1 bias, but
  bounded to 0.6–2.0× instrument σ so far less harmful. See §A7.
Fix: px-relative width bounds ≥ ~0.8 px; couple σ to the instrument LSF and fit
only γ (or a single effective width) at ≤1.5 px FWHM.

### A6. σ/γ degeneracy + Thompson FWHM at 1–1.5 px (`voigt.py:34-36`, `profiles.py`) — MED / MED-HIGH
`voigt_width` (Thompson pseudo-Voigt FWHM) and the Gaussian-fraction /
Stark-width bookkeeping in `profiles.py` (analyze_peak_profiles:174-192)
assume σ and γ are separately identifiable. At 1–1.5 px sampling they are not:
the data cannot distinguish Doppler(σ) from Stark(γ), so the reported
`gaussian_fraction`, Stark widths and any nₑ read from them are ill-posed on
native data. Quantified as fit-parameter scatter in §2.2. Fix: report one
effective width; get nₑ from a physical channel, not the fitted σ/γ split.

### A7. `peakfit.py` covariance uses fit-residual variance, not the noise model (`peakfit.py:166-176`) — MED / MED
`s2 = SSR/dof`, `cov = inv(JᵀJ)*s2` (166,172): on the OLD grid the
lag-1-correlated resample (ρ 0.76–0.91) makes SSR/dof under-state the true
variance, so `center_sigma_nm` was optimistic; on native (white) data it is
about right but the ±0.5 px snap fed to it as the seed still dominates unless
the fit moves off-grid. Lower priority than A1–A5.

---

## B. Other grid-assumption findings (whole-spectrum / detection paths)

### B1. Single global pitch via `median(np.diff(x))` — other call sites — MED / HIGH
Same root cause as A0, in non-fit modules that also take ONE whole-spectrum
pitch: `relative_profiles.py:279`, `minor_lines.py:274,684`, `profiles.py:451`,
`peak_window_pca.py:278`, `session_calibration.py:43`,
`wavelength_registration.py:174,665,862`, `gas_detection.py:115`,
`mw2_112.py:130`, `synthetic_calibration.py:121,132`. Each is mis-scaled per
segment by the factors in A0 / §2.4 (and dragged low by the dead channel). Any
px↔nm conversion, smoothing length or window built from this global pitch is
wrong by up to ~2× at the band ends. Fix: per-segment pitch (edges 365/620 nm)
and trim the dead channel first. NOTE: `pipeline.py:877` uses `median(np.diff)`
only to DETECT segment seams (`>max(0.5, 5*median)`) — benign.

### B2. Fixed-nm windows tuned on the 1/30 grid — MED / MED
`measure_line` central window `CENTRAL_HALF_NM=0.12`, sidebands `0.25–0.50`
(fe_plasma:51-52); `robust_height` `win=0.12`, `s_in/out=0.28/0.55`
(v_timing:127). ±0.12 nm is ≈1.8 px (VIS) but only ≈1.3 px (NIR) — the central
window is 5–7 samples on native vs ~7 on 1/30; the trapezoid area over a fixed
nm window truncates a different fraction of the line per segment. MIN_CENTRAL_PX=5
can force a fallback window on the NIR. Fix: windows in units of the local
instrument FWHM (px), per segment.

### B3. Noise via MAD of sidebands assumes white noise (`measure_line:143`, `peakfit.py:83`) — MED / MED
`noise = 1.4826*MAD(sideband residual)` and SNR from it. On the OLD spline grid
the noise was correlated (ρ(1) 0.76–0.91) so this MAD UNDER-estimated the true
per-line uncertainty by √(1+2Σρ) ≈ 2–3×; SNRs and detection thresholds
(`minor_lines` accept_snr, `detections` mean+2σ) were correspondingly
optimistic. On native (white) data the estimator is now correct — a case where
the switch IMPROVED an estimator, but downstream thresholds calibrated on the
old optimistic SNRs are now effectively stricter. Fix: re-baseline detection
thresholds on native SNRs.

### B4. BIC/χ² dof count uses raw sample count (`minor_lines.py:538,871`; `peakfit.py:164-176`) — MED / MED
BIC penalty `k*ln(n)` with `n` = samples in window (`minor_lines.py:538,871`
use `4*ln(n)`). On the old grid n was 4–5× inflated by correlated oversampling,
so ln(n) and χ² were both inflated (partly self-cancelling). On native data n
is the true independent count, changing the accept/reject balance of the
line-addition tests. Fix: verify accept thresholds on native n; use effective
dof, not raw n, when residuals are correlated.

### B5. Segment-response 620 nm gain step & background anchors — LOW-MED / MED (flagged, not yet verified here)
`detector.py` background is `np.interp` across anchors (395) and a median-filter
envelope (377-381) with window in samples (`max(win, n//20)`); on native data
`n//20` is a very different physical length than on 1/30. The 620 nm segment
gain step (segment-response) and negative baselines are owned by the parallel
representation agent (stitching/intensity); flagged here for coordination, not
re-derived. See Unverified.

---

## 2. Estimator-bias simulation

Script: `cal/scripts/native_grid_estimator_sim.py` (deterministic, `--help`).
Tables filled below after the run.

### 2.1 Peak-location bias vs sub-pixel phase
Instrument LSF: Gaussian, per-segment FWHM UV 1.35 / VIS 1.45 / NIR 1.10 px;
pitch 0.087 / 0.128 / 0.177 nm. Lines pixel-integrated with sub-pixel phase.

Noiseless systematic bias (peak-to-peak over phase), pm:

| seg | pitch | argmax-snap max\|e\| (phase-std) | parabolic max\|e\| (std) | point-Gauss | pixel-int-Gauss |
|-----|-------|------------------|-------------------|-------------|-----------------|
| UV  | 0.087 | 43.5 (25.1) | 9.3 (6.7)  | 0.2 | 1.3 |
| VIS | 0.128 | 64.0 (36.9) | 12.4 (8.9) | 0.2 | 0.7 |
| NIR | 0.177 | 88.5 (51.1) | 24.5 (17.5)| 1.2 | 6.2 |

The argmax-snap phase-std matches pitch/√12 exactly (VIS 0.128/3.46=37 pm,
NIR 0.177/3.46=51 pm) — pure grid-snapping (A3). The 3-point parabola (A2, the
SHIPPED shift-model centroid) has a real S-shaped systematic reaching 12 pm in
the VIS and **24 pm in the NIR** (1.1 px sampling). A Gaussian fit removes it to
<2 pm. So of the observed "~50 pm shift-model centroid scatter", the estimator
accounts for the bulk: argmax anchoring ~37–51 pm, parabolic refine ~9–18 pm.

MC scatter (std of location error, pm) over random phase + white noise:

| seg | SNR | argmax-snap | parabolic | point-Gauss | pixel-int-Gauss |
|-----|-----|-------------|-----------|-------------|-----------------|
| VIS | 20  | 34.7 | 10.7 | 5.4 | 5.4 |
| VIS | 50  | 36.5 | 9.2  | 2.2 | 2.3 |
| VIS | 100 | 32.8 | 8.5  | 1.0 | 1.1 |
| NIR | 20  | 50.4 | 17.5 | 10.1| 7.0 |
| NIR | 50  | 50.3 | 18.5 | 3.3 | 3.5 |
| NIR | 100 | 46.6 | 18.0 | 2.1 | 2.5 |

Parabolic floors at ~9 pm (VIS) / ~18 pm (NIR) even at SNR 100 — the phase bias,
not noise. A Gaussian fit (point OR pixel-int) reaches ~1–3 pm. **For CENTER
alone, point vs pixel-int are equivalent** (a symmetric line's center is not
biased by point-sampling); the pixel-int win is in amplitude/width/ratio (§2.2)
and it edges out point-Gauss only at low SNR in the NIR (7.0 vs 10.1 pm).

### 2.2 Amplitude / area / FWHM bias
Noiseless phase-dependent spread (%) or FWHM bias (%):

| seg | local-max height | trapz area | point-Gauss area | pixel-int area | point-Gauss FWHM bias | pixel-int FWHM bias |
|-----|------------------|-----------|------------------|----------------|-----------------------|---------------------|
| UV  | 28.0 | 12.3 | 0.2 | 0.0 | +12.6% | 0.0% |
| VIS | 25.3 | 1.9  | 0.2 | 0.0 | +11.0% | −0.0% |
| NIR | 36.6 | 0.7  | 0.9 | 2.5 | +17.9% | +7.0% |

Findings: (a) **local-max HEIGHT** swings 25–37% with sub-pixel phase — the
worst amplitude estimator, and it is what `robust_height`/`pair_height_ratio`
use (A4). (b) A **point-sampled Gaussian over-states FWHM by +11 to +18%**
(A1/A6) — the fitter widens to absorb the point-vs-integral mismatch; the
pixel-integrated fit removes it (0–7%). (c) The **trapz area** (`measure_line`)
is actually robust for an isolated line (0.7–2% VIS/NIR) but degrades to 12% in
the UV where the fixed ±0.12 nm window truncates a different fraction (B2); the
UV window is ~2.8 px wide. AREA is the phase-stable observable, HEIGHT is not.

### 2.2b Line-ratio recovery (same-upper-level pair, true 2.05)
Two lines at INDEPENDENT random sub-pixel phases, n_mc=200:

| grid   | estimator | median | scatter | err |
|--------|-----------|--------|---------|-----|
| native | height    | 2.049  | **0.219** | −0% |
| native | trapz     | 2.050  | 0.016   | −0% |
| native | pixel-int | 2.050  | 0.000   | −0% |
| vendor | height    | 2.033  | 0.106   | −1% |
| vendor | trapz     | 2.043  | 0.034   | −0% |
| vendor | pixel-int | 2.066  | 0.608   | +1% |

The local-max height ratio is **unbiased in the mean** but carries **±11%
per-measurement scatter** on native data (0.219/2.05) — and, for a FIXED pair at
fixed phases, up to ±14% single-shot bias. So the observed 417.94/415.97 = 0.75
vs 2.05 (a 63% deficit) is **NOT primarily an estimator artefact** — it is real
self-absorption/registration; the estimator contributes ~10% scatter and a
possible ±14% single-shot offset, not the factor of ~2.7. Crucially the OLD grid
**halved** the height-ratio scatter (0.106 vs 0.219): cubic-spline oversampling
recovered the peak better, so ratio noise doubled at the switch. Use AREA (trapz
or a fit), not height, for ratios. NOTE `pixel-int` on the VENDOR grid is WORSE
(scatter 0.61) — a pixel-integration model must only be applied to genuine native
pixels, never to resampled data.

### 2.3 Pixel-integrated fit prototype on REAL native V lines
Run `run-05541…` (auto-selected, highest signal), 9 shots, 6 auto-detected
strong isolated lines; centroid scatter across shots (pm) and fit-residual RMS
(% of peak), SAME 9-px window for point vs pixel-int:

| line nm | n | parabolic | point-Gauss | pixel-int | point RMS% | pixel-int RMS% |
|---------|---|-----------|-------------|-----------|-----------|----------------|
| 808.295 | 9 | 5.1 | 8.5 | 8.5 | 3.3 | 3.3 |
| 368.904 | 9 | 1.6 | 2.3 | 1.0 | 4.3 | 4.7 |
| 400.994 | 9 | 0.6 | 0.2 | 0.8 | 5.1 | 6.3 |
| 392.644 | 9 | 0.6 | 1.5 | 1.1 | 4.1 | 5.2 |
| 382.990 | 9 | 0.8 | 0.9 | 1.4 | 2.2 | 8.3 |
| 606.902 | 9 | 0.8 | 2.3 | 1.6 | 3.9 | 4.8 |

Honest result: within ONE run at ONE site the wavelength calibration and
sub-pixel phase are ~fixed across the 10 shots, so centroid scatter is small
(0.6–8.5 pm) and **estimator-insensitive** — parabolic, point-Gauss and
pixel-int all agree to ~1–2 pm. The parabolic vs pixel-int GAIN appears when the
sub-pixel phase VARIES across LINES — i.e. exactly the shift-MODEL case, where
§2.1 shows parabolic carries a 9 pm (VIS) / 18–24 pm (NIR) systematic that a fit
removes to <3 pm. The residual RMS floor (3–8%) is dominated by **Voigt
shape mismatch** (a Gaussian model on real Lorentzian-winged lines), NOT by
point-vs-integration — which is why point and pixel-int RMS are within ~1% here.
Implication: the demonstrable pixel-integration wins are in WIDTH and AREA/ratio
accuracy (§2.2) and in the shift model (§2.1); to also cut the residual floor the
forward model must be a pixel-integrated **Voigt**, not a Gaussian.

### 2.4 Dead-channel global-pitch effect on the PRODUCTION Voigt seed
From a real native shot (reproduces the A/B agent's facts):
- `median(np.diff(x))` = **0.0955 nm** (with dead channel) vs **0.1277 nm**
  (live only); dead channel = 2,065 zeros of 2,066 samples at ≥960.18 nm.
- true per-segment pitch UV 0.089 / VIS 0.129 / NIR 0.179 nm.
- seed-width mis-scale factor `inc/pitch_seg`: with dead channel **+8% / −26% /
  −47%** (UV/VIS/NIR); live-trim +44% / −1% / −29%.
- PRODUCTION `_voigt_seed`, genuine 1.1-px NIR line:

| pitch used | min_inc | seed FWHM | seed amp |
|------------|---------|-----------|----------|
| ALL+dead   | 0.0955  | 0.105 nm = **0.59 px** | 143 |
| live-trim  | 0.1277  | 0.140 nm = 0.78 px | 191 |
| NIR-local (correct) | 0.1794 | 0.197 nm = 1.10 px | 268 |

The seed is ~half-width and ~half-amplitude in the NIR — direct confirmation of
A0 using the shipped function.

---

## 3. Missed opportunities (estimated gains)

Ranked by expected gain on the real V/Fe native data.

1. **Per-segment pitch + dead-channel trim in the production fitter (A0/2.4).**
   Cheapest, largest. Removes the −47% NIR / −26% VIS seed-width and
   half-amplitude bias in `peaky_finder.fit_all`/`_fwhm_cap`/`fit_global`. Effort:
   a few lines (segment-local `np.diff` + drop x ≥ 960.18). Expected gain:
   correct NIR/VIS fitted widths and areas, the single biggest lever on "poor
   fit profiles." Do this first regardless of the rest.
2. **Pixel-integrated forward model (LSF convolved, integrated per native pixel,
   per-segment pitch) for line fitting AND the whole-spectrum CF-LIBS fit
   (A1).** Replace the point-sampled `voigt.py:84` with an erf-Gaussian /
   fine-oversample-Voigt integral over each pixel. Sim gain: removes the +11–18%
   FWHM bias and makes areas phase-exact (§2.2); the CF-LIBS whole-spectrum fit
   currently compares a point-sampled model to pixel-integrated data, a
   systematic per-line area bias that varies by segment. Effort: moderate
   (one shared function; keep the point path for GPU coarse generation). Prereq
   for cutting the 3–8% real-line residual — must be a Voigt, not a Gaussian
   (§2.3).
3. **Use AREA, not local-max HEIGHT, for line ratios (A4/§2.2b).** Switch
   `robust_height`/`pair_height_ratio` to a pixel-integrated area or fit. Cuts
   ratio noise from ±11% to <1% per measurement; removes the doubling of ratio
   scatter that came with the native switch. Effort: small.
4. **Joint fit of the 10 shots with per-shot wavelength calibration instead of
   pixel-index averaging.** The A/B agent shows averaging by pixel index; if the
   per-shot registration jitter is ~1 px, index-averaging broadens the mean line
   (adds ~ jitter in quadrature to the LSF) and biases widths high. Fitting each
   shot on its own recovered `x` (or a per-shot shift) and combining parameters
   avoids it. Estimated gain: recover the true instrument FWHM (removes the
   jitter-broadening, ~5–15% width depending on run stability) and √N centroid
   precision. Effort: moderate. NOTE: within a stable run (§2.3) the jitter is
   small, so the gain is run-dependent — quantify per run before committing.
5. **Per-pixel noise model (read + Poisson + the period-scaled fixed pattern) in
   the likelihood.** The current MAD/sideband noise (`measure_line:143`,
   `peakfit.py:83`) is a single scalar per line; a per-pixel σ lets the fit and
   the detection thresholds be correctly weighted and would let SNR/BIC be
   re-based for native (white) data (B3/B4). Estimated gain: correctly
   calibrated detection limits and χ²; modest effect on the bright-line
   parameters. Effort: moderate.
6. **Vendor spline for DISPLAY only.** §2.2b shows any estimator run on the
   vendor resample either loses information (height scatter) or breaks
   (pixel-int scatter 0.61). Keep the spline out of the analysis path entirely;
   use native pixels for every estimate. Effort: policy, ~zero.

---

## Unverified / caveats
- Segment-response gain step, stitching, negative baselines, and the
  wavelength-map/intensity/averaging comparison are the parallel agent's scope;
  B5 is flagged not proven here.
- Real-data numbers use the LOCAL copy of V run2 shots under the scratchpad
  (not moissanite); ledger `cal/data/v-pure-run2-ledger.json`.
- σ/γ degeneracy (A6) quantified as fit-parameter scatter, not propagated to
  an nₑ error bar.
- Simulation uses a Gaussian-dominated instrument LSF measured from the sim
  assumptions (FWHM per segment from the brief), not a per-line empirical LSF.
</content>
