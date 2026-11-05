# Signal-chain survey: grid-regime and design assumptions (finder → seeding → refinement → profiles)

Date: 2026-11-05. Agent: judgment (Opus 5.5). Repo read-only, apart from this report.
Scratch probes: /private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/7c311a86-0d91-4217-8224-5b49f94b239b/scratchpad/survey-a/probe{1..7}.py
(plus inline one-offs). The peaky_finder snapshot I audited is peaky_finder_snapshot.py in the same folder. The specialist's
diff at audit time was +16 lines: NATIVE_MIN_FWHM_FRACTION and an instrument_sigma_px import.

**Question.** Which steps on the production path of `analyze_spectrum` carry assumptions that fail on the Z300
native grid (pitch 0.089/0.129/0.179 nm, LSF FWHM about 1.4–1.65 px) or in another realistic regime?
The fit_peaks `len(x_window) > 5` gate and the fit_spectrum width/height filters are out of scope (the specialist owns them).

**Inputs.** Native: manuscript/figures/cache/fe_fresh_d5p25.pkl (5830 px) and fe_reused_fail.pkl. Export grid (1/30 nm):
user_spec1.pkl and mw2_1000.pkl. I also used the jc_*.pkl files for the segment-response check. Each "measured"
number comes from re-running the current code on the pkl's x, y, or from that pkl's stored analysis.

## Production chain actually executed (verified by reading code)
`pipeline.py:750` `PeakyFinder.__new__` → `fit_spectrum` (`peaky_finder.py:1359`) runs these steps in order:
- `valid_signal_mask`
- per-segment arPLS `find_background`
- `fourier_peaks` → `filter_peaks` → `find_peaks(window=5)` + `_noise_scale_local` 3σ
- `fit_peaks(fast=True)`. This path has no least-squares step: seeds come from `peak_parameter_guess`.
- `fit_shoulders` → `fit_all` → final width/height filters

After `fit_spectrum`:
- `refine_fit` 3a (`pipeline.py:779`) → `locate_records` (diagnostic only, `pipeline.py:788-797`) → segment shift / registration
- segment response (`pipeline.py:996-1036`) and `_amp_sigma` → `estimate_peak_uncertainties` (`pipeline.py:1055`)
- pass 1 → `refine_fit` 3b (`:1113`) → `seed_minor_lines` (`:1160`) → `recover_residual_lines` (`:1179`)
- `analyze_peak_profiles` / `deblend_shoulders` (`:1190+`) → pass 2 → deepening (`seed_minor_lines` / `recover_residual_lines` again, `:1307/1316`)

The following are not on the production path: `background_pca.py`, `detector.DetectorCorrection.correct*`, and the
cepstral output of `fourier_peaks` (computed, then discarded).

## 1. Ranked findings

Severity: **B** = blocks correctness, **Bi** = biases results, **L** = latent. Cost: S/M/L.
Measured counts are per segment, UV/VIS/NIR.

| id | file:line | class | assumption | tuned where/when | breaks in | evidence | sev | proposed fix (physical rule) | cost |
|---|---|---|---|---|---|---|---|---|---|
| F1 | minor_lines.py:431-433 (seed), :724,:790 (recover) | A | A seed/recovery window needs ≥ 12 samples, with span = max(3·wmed, 0.3 nm) | 1/30 nm grid, where 12 samples = 0.4 nm and always passes | native: 12 samples = 1.06/1.55/2.15 nm | fe_fresh seed records: **VIS 37/37 "no-data"**, UV 49 no-data; recover: **VIS 5/5 no-data**, no NIR candidates. Export: 0 no-data. Recomputed on fe_fresh final table: rows failing the gate UV 83/119, VIS 5/5, NIR 11/11. | **B** | Window = ±k·LSF_FWHM(segment) (k≈3–4, using the local pitch). Minimum sample count = 4·n_free + 2, not a fixed 12. | S |
| F2 | peaky_finder.py:1126-1129 (fit_shoulders) | C/A | A residual is a shoulder candidate if it exceeds mean + 2·std of the **whole-spectrum** residual | Inherited, pre-2026-07; never moved to the noise reference that filter_peaks adopted | Every regime. Worst on native, where the residual std is dominated by UV misfits. | At shoulder time the threshold sits at **8.6σ / 81σ / 24σ** of local noise on fe_fresh, 10.9/39/9.2σ on user_spec1 and 4.1/7.8/6.0σ on mw2_1000. On fe_fresh, shoulders added 55/4/2, while a 3σ noise-referenced test finds 121/102/35 residual maxima. The stage that should recover what fit_peaks dropped cannot do it in VIS. | **B** (native VIS/NIR), Bi (export) | Test against the per-segment `_noise_scale_local(residual, segment_indices)` plus a prominence test, i.e. the rule `recover_residual_lines` already uses. | S |
| F3 | refinement.py:313 (`if n < 15: return None`), window ±max(4·FWHM, 0.4 nm) from :281-286 | A | A blend/SA classification window needs ≥ 15 samples | 1/30 nm (0.8 nm ≥ 24 samples) | native: 0.8 nm = 9/6/4 px | fe_fresh final table: **UV 82/119** features below 15 samples (VIS 1/5, NIR 2/11). Export: 0/221 and 0/341. On native, 3a/3b are skipped silently with no record for most features. | **Bi** | Window ±4·LSF_FWHM. Minimum n = n_par(B) + 4 = 12, or scale with LSF px. Emit a "skipped: window" decision instead of returning None. | S |
| F4 | profiles.py:79 (`WINDOW_NM_RANGE`), :184 (`>= 9`), :229 (`band < 3`), :400 (deblend `< 9`), :211-212 (core = max(0.3·HWHM, 2 px)) | A | Profile shape QC needs ≥ 9 samples per window and ≥ 3 samples per shoulder band (0.8–3 HWHM) | 1/30 nm | native: HWHM ≈ 0.7 px, so a shoulder band holds ≈ 2 samples and the "core" ±2 px is the whole line | fe_fresh: **UV 52/119 `unresolved-window`**, shoulder metric computed for only 26/119 UV. Export: 0 unresolved, shoulder metric computed for 83/84, 59/59, 77/78. `element_shape_quality` runs on uninformative records. | **Bi** | Express windows/bands in LSF units with px minima. On native, report shape QC as "not resolvable" (flag) instead of feeding a classification. | S–M |
| F5 | pipeline.py:999-1016 + detector.py:525-530 | A/C/B | The 620 nm gain step counts as measurable only if **both flank medians exceed 5 × the global single-pixel MAD** of y−bg0 | 2026-07 export corpus; fallback 3.9 from 435 export spectra | Every regime. The floor compares an 8-nm median (≈60 native / 240 export samples) with a whole-spectrum, UV-dominated single-pixel σ. | **6/6 cached spectra use the 3.9 fallback.** Raw flank ratios: fe_fresh 4.83, fe_reused 8.10, user_spec1 7.22, mw2 3.69, alaskire 4.67, mdd017_29 9.30. Floors: 308/271 on native vs left flanks 47/17. corrections/detector.json records that the corpus is bimodal (≈9× secondary mode). If the raw ratios are right, NIR amplitudes are mis-scaled by 0.8–2.4× relative to VIS. | **Bi** (cross-segment Saha) | Gate on the ratio's own standard error (`ratio_unc`, already computed at detector.py:534-544), using **segment-local** noise/√n_flank. Keep the fallback per acquisition configuration (bimodal), not one corpus median. | S |
| F6 | peaky_finder.py:140-171 (`_noise_scale`, max over lags 1/2/4/8), :177 (`block=2048`) | A/E | (a) The noise is the lag plateau of a spline-upsampled export. (b) 2048 samples = a local block. | Designed for resampled data (docstring :145-150); block tuned on 1/30 nm, where 2048 samples = 68 nm | native: the data are near-white (quiet-block ρ1 = 0.38/0.23/−0.10, lag8/lag1 = 1.16/1.22/0.86), so the max-lag estimate only adds line/continuum structure. 2048 px > a native segment (≈1990 px), so the "local" noise is one scalar per segment. | Segment-wide max/lag-1 on fe_fresh y_bgsub = **1.75 / 1.39 / 1.11**; export 2.5–5.1 (correct there). >3σ detections: production 130/108/46, lag-1 170/123/52. Within-UV, 64-px block lag-1 noise spans p10–p90 61–848 under one scalar of 212. Smaller blocks with max-lag *reduced* detections (106/85/47), because line-forest structure enters as "noise". | **Bi** (detection completeness, every SNR gate downstream) | Choose the estimator by grid regime: lag-1 MAD (white) on native, lag plateau on export. Give the block length in nm (≈68 nm, i.e. 770/530/380 px). Longer term, use a measured read+shot noise model (darks) so line density cannot leak into σ. | S (estimator), M (noise model) |
| F7 | pipeline.py:1055 → inspection.py:84-86 (also :186, :261, :345) | E/A | `_noise_scale_local(y_bgsub)` is called **without** `segment_indices` | Pre-native default | native: 2048-sample blocks straddle 365/620, so VIS noise is interpolated between the UV and NIR values | VIS noise 41.1 vs 20.5 segment-aware (**×2.0**) on fe_fresh, ×2.18 on fe_reused; export ×0.86–1.12. Effect on amp_sigma (VIS median): ×1.01 fe_fresh, **×1.41 fe_reused**, because the χ² rescale masks part of it. | **Bi** | Pass segment-aware noise. All other production callers already do. | S |
| F8 | pipeline.py:999-1005 → peaky_indexer_v3.py:2053-2055 | D/B/C | The additive `amp_sigma_floor` = global **per-pixel** MAD (counts) is added in quadrature to **area** σ (counts·nm) | WNNLS fix 2026-07 (memory: "additive floor required") | Every regime. The unit mismatch makes the floor's weight depend on grid and segment. | Floor > statistical σ for 44/53/45 % (user_spec1) and 74/62/89 % (mw2) of lines: the floor, not data, sets most weights. On native fe_fresh the group-χ²-scaled σ_A is itself huge (UV median 4192 vs median area 582, i.e. area SNR < 1), so the weighted solve gets near-noise weights in UV. | **Bi** | Floor in area units, per segment: σ_floor = σ_noise,seg · pitch_seg · √(N_eff, LSF) (or the matched-filter form `inspection` already uses). Build σ from the noise model, not group χ²ν. Re-measure after the gate fix. | M |
| F9 | minor_lines.py:268-274 and :688-690 (global `sig_med`/`gam_med` and global `pitch`) | A/E | One spectrum-wide median σ/γ (nm) is "the instrument profile" for every seeded/recovered line, and one pitch serves all segments | MW2-112 export | Both grids: the LSF in nm differs ≈2× between UV and NIR | user_spec1: NIR median σ 0.100 vs global 0.070, and global γ = 0 forces pure Gaussians (clamp 0.85–1.2× of 0). fe_fresh: VIS 0.218 / NIR 0.140 vs global 0.054. The ±1.5·pitch centre bound uses the 0.129 nm global pitch: ±2.2 px UV, ±1.1 px NIR. **Global pitch previously reported (audit B1, minor_lines.py:274/684), still open.** | **Bi** (areas of seeded/recovered lines, NIR/VIS) | Use the per-segment instrument LSF, which already exists: `peakfit.DEFAULT_PROFILE_PX` (1.4/1.4/1.65 px) / `epsf-z300` json × local pitch. Make it the single source for seeds, clamps and windows. | S–M |
| F10 | peaky_finder.py:288 (`find_peaks(window=5)`), also used at :931, :1130, minor_lines recover | A | A line must be the maximum within ±2 samples | 1/30 nm (±0.067 nm) | native: ±2 px = ±0.18–0.36 nm, so close pairs merge | fe_fresh: **78** maxima significant at window 3 are suppressed at window 5, 58 of them >5σ (UV 26 / VIS 29 / NIR 3). user_spec1: 2. | **Bi** | Comparison half-width = ceil(0.5·LSF_FWHM_px) → window 3 on native, 5 on export (unchanged there). | S |
| F11 | peaky_finder.py:897-900, :1121-1124, :1272-1275, applied :1003-1004, :1181-1182, :1321-1322 | C/A/D | Width cap = 2 × **global** median detected FWHM (px) × local pitch, applied **per channel** (σ ≤ cap/2.355, γ ≤ cap/2) | MW2-112 "pancake" fix | Export: NIR lines are wider in px (median 9.8 vs global 6.4), so 21 % (user_spec1) / 14 % (mw2) of NIR seeds exceed the cap. Native: per-channel caps admit a combined Voigt FWHM of 1.64 × cap. | Combined FWHM > cap in NIR blind rows: 14/67 user_spec1, 6/11 fe_fresh. The fe_fresh blind table keeps VIS median FWHM 0.80 nm (6 px) and NIR 0.71 nm (4 px) against an LSF of 1.4–1.65 px: the surviving native VIS/NIR rows are the pancakes the cap was meant to stop. | **Bi** | Cap the **combined** Voigt FWHM at k × LSF_FWHM(segment) (per-segment instrument reference, not the population median). Relax only for Stark-broad species. | S |
| F12 | peaky_finder.py:1284-1300 (fit_all window = half-max crossings from `peak_parameter_guess`, cap ±50 samples) | A/D | The core between the half-max crossings is enough to fit 4 parameters per peak and extrapolate the area | 1/30 nm (window ≈ 6–12 samples) | native: window ≈ 4 samples | fe_fresh: windows with fewer samples than 4 × peaks-in-window: UV 24/87, **VIS 7/10**, NIR 4/16. Export: 0. σ/γ split and area are then set by the seed and bounds, not the data. Adjacent to the specialist's work: not part of the gate fix. | **Bi** | Window ±3·LSF_FWHM (wings included). On native, tie σ to the LSF and fit amp, μ, γ only. | M |
| F13 | utils/voigt.py:83-85 (point-sampled profile), peaky_finder.py:564 (height-based detection) | D | A Voigt evaluated at pixel centres represents pixel-integrated counts, and the peak-sample height is a fair detection statistic | 1/30 nm | native FWHM 1.4–1.65 px | **Previously reported (native-grid audit A1), still open.** New measurement (erf simulation): at 1.4/1.65 px FWHM the max-sample/area ratio varies **32 % / 24 %** with sub-pixel phase, so detection at threshold depends on phase. A matched-filter statistic gains 1.20–1.26× SNR. | Bi | Pixel-integrated (erf/binned) profile. Detection on a matched-filter (LSF-convolved) statistic. | M–L |
| F14 | detector.py:498-505, corrections/detector.json `notes`, pipeline.py:1036 (`edges=(620.0,)`) | B | The 365 nm junction step is **additive**, so it is not corrected | 2026-07 export corpus | native (untested there) | The two native shots give bg flank ratios 365: **0.36 (310→113) and 0.40 (125→50)**. One additive offset cannot produce both (−197 vs −75); a multiplicative k ≈ 0.38 fits both. Export shots range 0.5–3.6. **Two points only (unverified).** An uncorrected UV/VIS gain directly scales Fe II/Fe I, i.e. Saha T. | Bi→B if confirmed | Test additive vs multiplicative on ≥10 native shots spanning continuum level (regress R on L). If multiplicative, add 365 to `edges` with the same measured/fallback logic. | S (test) |
| F15 | pipeline.py:228 (`POISSON_GAIN_COUNTS = 1.0`) | B | 1 count = 1 photo-event for the shot-noise term | Flagged in the docstring as provisional for SciAps exports | native counts (different units from the vendor export, no spline) | Quiet native blocks: σ² vs background fits give 0.39·bg+69 (VIS), a negative slope (NIR) and nonsense (UV). The noise is not gain-1 Poisson; read/fixed-pattern noise dominates. | L/Bi | Photon-transfer measurement (darks + flat continuum at several exposures) per acquisition mode. | M |
| F16 | refinement.py:122-162 (opt-in `bic_effective_n`), pipeline.py:265/675 (default None, not in CLI); minor_lines.py σ_area (≈:353, :~830) white-noise matched filter | A/E | Samples are independent for BIC and for area SNR | — | export (ρ1 = 0.72–0.95 measured above) | **Previously reported (audit B3/B4), partially addressed.** The ρ correction exists but is off in production. minor_lines/profiles BIC and accept_snr have no correction. On export the 4σ accept bar is ≈2σ effective, so bars tuned on export are ~2× stricter on native. | L/Bi | Enable `"auto"` by default, or compute ρ1 from the residual. Re-baseline accept/deepen bars on native SNR. | S |
| F17 | minor_lines.py:488 (pin = 0.02 nm), tol_nm 0.06; refinement/recover ±0.03 nm bounds | A | Fixed-nm centre pins tuned at 1/30 nm | MW2-112 | native: 0.02 nm = 0.11–0.22 px, 0.06 nm = 0.3–0.7 px | Not measured. These are the bounds against which registration and centroid error compete. | L | Express in local pitch / LSF units, or in measured registration σ. | S |
| F18 | utils/native_grid.py:857-904 | A | One median pitch per segment | A0 fix 2026-09 | native | Within-segment pitch p5–p95: UV 0.0795–0.0956, VIS 0.1145–0.1406, NIR 0.1585–0.1966 (±10–12 %). Area per photon ∝ local pitch is not normalised within a segment (a small Boltzmann tilt). | L | Use the local `diff(x)`, smoothed, as the pitch array. Consider dividing y by the local pitch (per-nm units) before fitting on native. | S |

**E (structural, dual paths).** Production peak parameters come only from the finder's point-sampled Voigt. The
validated native tools, per-segment LSF (`peakfit.DEFAULT_PROFILE_PX`, `instrument_sigma_px`), ePSF
(`peak_location`, `alibz/data/epsf-z300-20260924.json`) and pixel-aware windows (peakfit `_window`), feed only
diagnostics (`pipeline.py:788-797`, "ADDITIVE ... intentionally NOT done") and wavelength registration. Every
self-referential width reference below rebuilds an "instrument width" from the fitted population:
- finder cap (F11)
- minor_lines `sig_med` (F9)
- profiles `segment_width_floor` (profiles.py:88-110, 20th percentile per segment)
- refinement `wmed` (refinement.py:241-242)

On native data the population is contaminated by the pancakes, so every one of those references is wrong at once.
**Recommendation:** one per-segment LSF object (px FWHM, shape), loaded from the packaged Z300 file (with a
measured fallback for other instruments). All windows, caps, seed shapes and min-sample tests become multiples of
it. This single change addresses F1, F3, F4, F9, F10, F11 and F12 together. **Strongest alternative:** patch each
constant to a native value behind `is_native`. It lost because it keeps the population-median self-reference and
doubles the code paths (class E) that caused this survey.

## 2. Checked and clean (or tuned-but-defensible)
- `fit_peaks` `cumulative_trapezoid(x_window, y_window)` "swap" (peaky_finder.py:925-929). I verified numerically
  that `intdiff[m] = x̄·(2y_m − y_{m−1} − y_{m+1})`: a negative discrete second difference times wavelength, i.e. a
  curvature (blend) detector. It works as intended. The λ factor is a harmless per-window constant. It is not the
  find_background bug, so nobody should "fix" the argument order. (After the gate fix it will run on 3–4-sample
  native windows, where F10's window 5 applies.)
- arPLS stiffness `lam = (smooth_nm / pitch_seg)^4` per segment (peaky_finder.py:504-512): grid-invariant. Audit B1
  is fixed here.
- `segment_pitch_array` per-segment median (native_grid.py:857): fixes A0. The dead-channel exclusion is correct.
  The residual issue is F18.
- Native segment seams coincide with the hard-coded 365/620 nm (pitch jumps measured at 364.957→365.082 and
  620.100→620.298 nm), so `DEFAULT_SEGMENT_EDGES` holds for the native layout. fe_fresh has no zero pads (ends at 947.94 nm).
- `valid_signal_mask` (peaky_finder.py:387): a no-op on native, handles export pads (626 zeros) correctly.
- `filter_peaks` noise-referenced 3σ with a monotone `n_sigma` (peaky_finder.py:560-565): sound concept. The
  estimator is the problem (F6).
- `fourier_peaks` cepstral block: output unused, exceptions contained. Wasted FFT only.
- `_voigt_seed` area = height / V(0; σ, γ) (peaky_finder.py:116-137): correct for unit-area profiles. The 50/50 G/L
  split is a seed only (refit in fit_all).
- fit_all / fit_peaks centre bounds ±1 **local** pitch (:1301, :1018): per segment, sound.
- Strongest-first pedestal subtraction (fit_peaks fast, fit_all): sound.
- `MIN_HEIGHT_SIGMA` gate (:1500-1503): dropped 0 peaks on all three spectra. Not a loss channel today.
- `100 × median_fwhm` gate with the `span/50` line-like guard (:1468-1476): robust.
- `recover_residual_lines` σ_area: the pitch cancels (σ_n/√Σv²), which is correct for white noise. Its prominence
  test uses segment-aware noise. Its BIC uses one loss on both sides.
- `refine_fit` χ² rescale by best-model χ²ν floored at 1 (refinement.py:368-375): defensible.
- `peak_location.is_z300_native` dispatch: export delegates to Gaussian; diagnostic only.
- `peakfit.py` (px windows, per-segment σ, bounds as multiples of the LSF): sound apart from A1 point-sampling (audit A7 noted).
- refinement/minor_lines/profiles `_noise_scale_local` calls pass `segment_indices` (refinement.py:617,
  minor_lines.py:263/682, profiles.py:158/450/595). Only the inspection.py callers do not (F7).
- `DEFAULT_KT_EV = 0.76` (minor_lines.py:46) and `_RANK_KT_EV = 0.75` (refinement.py:47): the production seeding
  passes pass-1 T (pipeline.py:1162). The refinement value only ranks db-line plausibility. Defensible.
- `background_pca.py` and `detector.DetectorCorrection`: not on the production path.

## 3. Unverified
- F14 (365 multiplicative) rests on two native shots. It needs ≥10 native shots across continuum levels; the raw
  corpus is on moissanite. It is also unclear whether the arPLS flank levels under the dense UV forest measure continuum.
- F5 assumes the arPLS flank medians are unbiased. Additive offsets could inflate the low-continuum raw ratios
  (e.g. fe_reused left flank = 17 counts).
- Whether quiet native noise is truly white. ρ1 of 0.38 (UV) and 0.23 (VIS) in the quietest blocks could be weak
  lines or detector cross-talk.
- All native counts for F2, F3, F4, F8, F11 and F12 were measured on the current, gate-broken blind table. They must
  be re-measured after the specialist's gate fix, which will change the residual std (F2), the window distribution
  (F3, F12) and the σ_A χ² scaling (F8).
- F17 (fixed-nm pins) and arPLS behaviour under the native UV forest were not measured.
- The shot-noise gain (F15) is not measured. The quiet-block regression is only suggestive.

## Suggested briefs
- **worker:** F1, F2, F7, F10, plus F5's gate (each a few lines; replay probe3/5/6/7 for before/after).
- **specialist:** the per-segment LSF object and its wiring through F3, F4, F9, F11 and F12.
- **judgment / owner decision:** F14 measurement plan and the F8 σ-model redesign.
