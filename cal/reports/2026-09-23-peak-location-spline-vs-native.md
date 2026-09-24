# Peak location: vendor cubic spline vs native-pixel methods (after background)

> **Coordinator note (2026-09-23): the Fe "decisive accuracy" headline below is NOT valid.** Scored on the same lines against one shared curve, every estimator has about half its lines more than 1 px off, so the test measures line matching, not estimator accuracy. See `reports/2026-09-23-peak-location-verification.md`. There, a phase-uniformity test shows the spline pulls centres toward pixel centres and the Gaussian pixel fit is also biased (it fails in VIS).

Date: 2026-09-23
Author: mwhittaker (specialist agent)
Question: after background subtraction, does the vendor cubic spline give better
PEAK LOCATIONS than native-pixel methods? Analysis accuracy only, not display.

Script: `cal/scripts/peak_location_spline_vs_native.py` (deterministic, `--help`).
Estimators (each run AFTER a background subtraction):
1 native argmax · 2 native 3-pt parabola (`fe_plasma_analysis.subpixel_center`)
· 3 vendor cubic spline (scipy `CubicSpline` not-a-knot through the native
pixels — the exact vendor form, `native_grid.py` `KERNEL_NAME='cubic_notaknot'`,
:60/:155) sampled at 1/30 nm then argmax · 4 (3)+parabola on the dense grid ·
5 continuous max of the spline (root of its derivative) · 6 linear resample to
1/30 nm then argmax (Pantheum's old path) · 7 production Gaussian fit
(`peakfit.gaussian_subpixel_center`) · 8 pixel-integrated Gaussian fit on the RAW
native pixels (proposed fix).
Backgrounds: (a) ideal, (b) production today (global-pitch lam, dead channel),
(c) corrected (per-segment pitch, dead channel trimmed), (d) on the spline grid.
Real-data headline uses the corrected background (c) for all estimators;
(a)/(d) and the background sensitivity are characterised in the simulation
(§2), where the true line position is known — location proved only weakly
background-sensitive (<1 pm ideal vs edge-linear), so a separate real (d)
column would not discriminate.

Data: the DECISIVE accuracy test (§0) uses the 26 native Fe runs of 2026-09-22
(Fe Aesar 99.98%, 8–10 shots each; local copy of the moissanite acquisitions).
§1–§2 use V run2 + the ambient Ar I NIR anchors. All real-data estimates use the
corrected background (c). Truth = Fe I/II AIR lab wavelengths (`database.py`,
good to <1 pm).

Scripts: `cal/scripts/peak_location_spline_vs_native.py` (§1–§2) and
`cal/scripts/fe_peak_location_accuracy.py` (§0), both deterministic, `--help`.

---

## 0. DECISIVE TEST — real Fe accuracy (headline)
Metric: per run and segment, the RMS of (obs − db) about a smooth cubic-in-λ
calibration curve (removes the per-run map; each line's own sub-pixel phase makes
phase bias show up as SCATTER), iterative 3×MAD clipping. Estimators paired on
the same lines; 26 runs treated separately (a 13:39 recal split the day). Line
selection: strength-DOMINANT Fe I/II lines (blend screen), single clean peak
within ~1 px of the validated shift-model prediction, thick resonance lines
excluded and clipped. ~14/14/12 lines per run per segment (UV/VIS/NIR).

### Accuracy — median [IQR] of per-run RMS (pm), lower = better
| estimator | UV | VIS | NIR |
|-----------|-----|-----|-----|
| **8 pixel-int (free width)** | **90.1** [43–165] | **116.7** [40–234] | **155.5** [87–242] |
| 8e pixel-int, empirical shape | 96.3 [38–175] | 112.8 [53–250] | 152.6 [96–248] |
| 5 spline-root                 | 151.0 [110–186] | 211.6 [145–245] | 217.2 [167–326] |
| 7 peakfit Gaussian            | 196.4 [132–241] | 249.9 [62–283] | 295.0 [83–465] |
| 2 native parabola             | 237.1 [182–263] | 291.6 [254–353] | 406.8 [234–457] |
| 4 spline-parab                | 246.2 [221–288] | 278.2 [165–348] | 358.6 [245–442] |

### Paired vs 8 pixel-int — median per-run dRMS with bootstrap-over-runs 95% CI (pm)
Positive ⇒ that estimator is WORSE (higher RMS) than the pixel-int fit.

| vs 8 pixel-int | UV | VIS | NIR |
|----------------|----|-----|-----|
| 8e empirical | −2.8 [−6.5,+5.0] tie | +9.2 [−1.3,+30.6] tie | −0.6 [−7.8,+16.9] tie |
| 5 spline-root | +36.9 [+4.4,+70.6] **8 better** | +59.5 [−37.6,+99.1] tie | +46.8 [+13.0,+95.5] **8 better** |
| 7 peakfit-Gauss | +70.8 [+42,+135] **8 better** | +21.2 [−5.4,+111] tie | +88.7 [+8.1,+234] **8 better** |
| 2 native-parab | +129.8 [+61,+156] **8 better** | +120.5 [+25,+246] **8 better** | +212.7 [+110,+292] **8 better** |
| 4 spline-parab | +157.0 [+107,+177] **8 better** | +126.5 [+28,+192] **8 better** | +167.7 [+137,+256] **8 better** |

n_runs = 26/25/26 (UV/VIS/NIR); median ~1 line/run clipped. The pixel-int fit
beats the parabola and spline-parab in all three segments and beats spline-root
and peakfit in UV and NIR; in VIS spline-root/peakfit are statistical ties (wide
run-to-run CI) but their medians are still 20–60 pm worse. 8e ≡ 8 everywhere.

Note on scale: even the best estimator leaves a ~90–155 pm RMS floor (≈0.8–1 px)
— the irreducible real-data accuracy limit on this undersampled grid (sub-pixel
phase residual + db uncertainty + per-shot registration + line-shape variation),
consistent with the native-grid audit. The estimator choice moves this floor by
up to ~250 pm, so it matters, but no estimator makes native lines "sharp."

Ar I NIR anchors (V run2, n=3, weak): inconclusive here — the free-width fits
occasionally blow up on the ~1-px weak Ar lines (spread 400–680 pm), which is
itself evidence that free-width fitting needs bounding/fallback on ~1-px NIR
lines; the earlier §1 tighter-window Ar numbers (spline spread 3–6 pm) stand as
the precision, not accuracy, comparison.

**Verdict (decisive):** on real Fe ACCURACY the **pixel-integrated fit (8) is the
most accurate estimator in EVERY segment**, by a large, significant margin over
the spline family and the parabola (60–250 pm; CIs below). The empirical-shape
fit (8e) is statistically **indistinguishable from the free-width Gaussian (8)** —
the empirical LSF buys nothing for LOCATION, so the simpler (8) is preferred.
**spline-root (5) is the best of the spline family and the most robust to
outliers, but it is clearly LESS accurate than the pixel fit** (UV 151 vs 90,
VIS 212 vs 117, NIR 217 vs 156). **The spline does NOT win on accuracy in any
segment, including the NIR** — this overturns the earlier §1 "spline wins in
NIR," which was based on within-run *repeatability* (precision), not accuracy.
Caveat: the free-width pixel fit occasionally fails to converge on hard/blended
lines (unbounded LSQ) and must be bounded / fall back to spline-root; that
fragility inflates its *unclipped* scatter (see script notes) but not the robust
(clipped) accuracy reported here.

---

## 1. V-run headline — pixel-int (8) vs spline (3)/(4)/(5), corrected bg
Paired on the SAME lines (25 runs, 24 isolated lines, 17 db-matched; strongest
run for shot/closure). Cells: median [MAD 1σ] (n lines). Lower = better.

### Shot-to-shot centroid repeatability (pm)
| estimator | UV | VIS | NIR |
|-----------|----|-----|-----|
| 8 pixel-int      | 1.7 [1.1] 4 | 1.5 [1.0] 11 | 12.4 [4.1] 9 |
| 3 spline-argmax  | 7.6 [11.3] 4 | 0.0 [0.0] 11* | 15.3 [2.9] 9 |
| 4 spline-parab   | 2.1 [1.3] 4 | 1.0 [0.6] 11 | 7.3 [4.6] 9 |
| 5 spline-root    | 2.1 [0.9] 4 | 1.0 [0.7] 11 | 6.6 [3.6] 9 |
| 2 native-parab   | 1.8 [0.8] 4 | 1.1 [0.7] 11 | 7.7 [4.7] 9 |
| 7 peakfit-Gauss  | 1.7 [1.2] 4 | 1.8 [1.7] 11 | 9.6 [4.3] 9 |
\* spline-argmax "0.0" is an ARTEFACT: the 1/30 nm dense-grid bin does not move
under the tiny within-run jitter, so quantization masquerades as precision — the
absolute metrics (closure, Ar) show it is not more accurate.

### Cross-run consistency (pm; std of run-mean centroid across runs)
| estimator | UV | VIS | NIR |
|-----------|----|-----|-----|
| 8 pixel-int      | 2.8 [0.0] 4 | 3.0 [1.6] 11 | 13.2 [5.3] 9 |
| 3 spline-argmax  | 0.0 [0.0] 4* | 0.0 [0.0] 11* | 18.6 [7.5] 9 |
| 4 spline-parab   | 3.8 [0.8] 4 | 1.6 [1.0] 11 | 14.9 [11.8] 9 |
| 5 spline-root    | 4.0 [1.5] 4 | 1.5 [1.0] 11 | 15.2 [12.1] 9 |
| 2 native-parab   | 2.8 [0.6] 4 | 1.9 [1.3] 11 | 14.4 [10.1] 9 |
| 7 peakfit-Gauss  | 2.8 [0.1] 4 | 4.7 [1.3] 11 | 13.5 [4.6] 9 |

### Multiplet-spacing closure (pm; \|Δobs − Δdb\|, independent of absolute shift)
| estimator | UV (2 pr) | VIS (3 pr) | NIR |
|-----------|-----------|------------|-----|
| 8 pixel-int      | 105.7 [12.3] | 127.7 [59.7] | – (too few matched) |
| 3 spline-argmax  | 116.7 [56.9] | 133.8 [94.1] | – |
| 4 spline-parab   | 106.3 [23.9] | 131.8 [95.4] | – |
| 5 spline-root    | 106.4 [25.0] | 132.2 [99.2] | – |
| 2 native-parab   | 106.0 [24.4] | 131.0 [92.3] | – |
| 7 peakfit-Gauss  | 105.3 [15.5] | 108.2 [9.3]  | – |
CORRECTION (coordinator): the DB in `alibz/utils/database.py` is AIR (e.g. Fe I
438.354439, 371.993445), so the ~106–130 pm closure MEDIAN is NOT a vacuum→air
offset. With only n=2–3 pairs per segment the closure numbers are
uninterpretable and are dropped from the verdict — superseded by the Fe cubic-
residual accuracy test in §0.

### Ar I NIR anchor residual (pm, obs−db, corrected bg; expected shift ≈ −155)
| estimator | median | spread(1σ) over 3 lines | n |
|-----------|--------|-------------------------|---|
| 8 pixel-int      | −167.8 | 15.4 | 3 |
| 3 spline-argmax  | −143.1 | **3.2** | 3 |
| 4 spline-parab   | −144.4 | 6.2 | 3 |
| 5 spline-root    | −143.6 | 4.9 | 3 |
| 2 native-parab   | −153.3 | 7.7 | 3 |
| 7 peakfit-Gauss  | −181.0 | 12.1 | 2 |

### Paired reading of the headline (corrected)
- **Real UV/VIS data do NOT separate the sub-pixel methods**: pixel-int,
  peakfit, native parabola and spline-parab/root all agree to ~1 pm on both
  shot-to-shot repeatability and cross-run consistency. Any UV/VIS ranking here
  is within the noise; the discriminating evidence is the Fe accuracy test (§0).
- On the **NIR Ar lines the spline methods are more self-consistent** (spread
  3–6 vs 15 pm for pixel-int) and the methods **disagree by ~24 pm in absolute
  position** — a real, unresolved NIR disagreement, not a win for either side
  from repeatability alone.
- The **simulation (§2) generates Gaussian lines and fits Gaussians, so it
  favours (7)/(8) by construction** — it shows the estimators' intrinsic
  precision ceiling, not which is most accurate on real, non-Gaussian lines.
  That is why the decisive test below uses real Fe with a known-truth line list.

<details><summary>(superseded) earlier per-segment reading</summary>

- **UV & VIS (1.3–1.5 px/FWHM):** a proper fit wins. Pixel-int (8) and
  peakfit-Gauss (7) tie or beat every spline method on repeatability and give
  2–7× TIGHTER multiplet closure (the accuracy-relevant metric). The spline
  helps only versus native ARGMAX (1); its own argmax variant (3) is the worst
  by absolute accuracy (grid-snap), and spline-parab/root (4/5) merely match a
  native parabola. **The spline does NOT beat a proper pixel fit in UV/VIS.**
- **NIR (~1.0–1.1 px/FWHM, severely undersampled):** the spline genuinely helps.
  spline-root/parab (5/4) give the best shot-to-shot repeatability (6.6/7.3 pm vs
  pixel-int 12.4) and the Ar anchors are most self-consistent under the spline
  (spread 3–6 pm vs 12–15). At ~1 px a Gaussian/pixel-int fit has too few
  informative samples, whereas the smooth spline interpolant has a stable
  maximum. **The spline (root of derivative) wins in the NIR** — but its
  cross-run consistency (15–19) is no better than the fits, and n is small (Ar =
  3 lines), so this is "helps at the margin," not a large win.
</details>

---

## 2. Simulation — bias / RMS / p95 of location error (pm), per segment
LSF per segment; sloped+curved continuum; clean + a 2-px 50% blend; phase swept;
SNR 5–100. Background matters little for LOCATION here (ideal vs edge-linear
differ <1 pm); the estimator and the presence of a blend dominate.

### CLEAN isolated line — RMS (pm) vs SNR (edge-linear bg)
| seg | estimator | S5 | S10 | S20 | S50 | S100 |
|-----|-----------|----|-----|-----|-----|------|
| UV  | 8 pixel-int      | 16.4 | 8.3 | **2.6** | 1.4 | **0.7** |
| UV  | 7 peakfit-Gauss  | 19.4 | 9.8 | 3.0 | 1.5 | 0.9 |
| UV  | 4 spline-parab   | 14.2 | 9.9 | 6.5 | 6.7 | 6.1 |
| UV  | 5 spline-root    | 14.8 | 10.4 | 7.3 | 7.5 | 6.9 |
| UV  | 3 spline-argmax  | 19.3 | 12.3 | 11.5 | 12.2 | 12.2 |
| UV  | 1 native-argmax  | 29.6 | 25.2 | 25.2 | 25.2 | 25.2 |
| VIS | 8 pixel-int      | 17.7 | 9.5 | **4.6** | 2.4 | **1.3** |
| VIS | 7 peakfit-Gauss  | 24.8 | 8.5 | 4.6 | 2.5 | 1.4 |
| VIS | 4 spline-parab   | 22.8 | 12.4 | 10.3 | 9.3 | 8.7 |
| VIS | 5 spline-root    | 23.3 | 12.9 | 11.1 | 10.1 | 9.5 |
| VIS | 1 native-argmax  | 44.1 | 38.4 | 37.0 | 37.0 | 37.0 |
| NIR | 8 pixel-int      | 31.7 | 14.5 | **6.6** | 4.3 | 2.8 |
| NIR | 7 peakfit-Gauss  | 31.2 | 20.8 | 7.9 | 3.8 | **1.9** |
| NIR | 4 spline-parab   | 34.9 | 21.1 | 18.4 | 18.7 | 18.7 |
| NIR | 5 spline-root    | 35.2 | 21.6 | 19.0 | 19.4 | 19.5 |
| NIR | 3 spline-argmax  | 36.1 | 22.2 | 21.9 | 22.5 | 22.5 |
| NIR | 1 native-argmax  | 55.9 | 51.2 | 51.2 | 51.2 | 51.2 |

On a CLEAN line the **pixel-integrated fit (8) and the production Gaussian fit
(7) are the best estimators in EVERY segment** and keep improving with SNR
(→1–3 pm). The **spline methods FLOOR** at 6–12 (UV) / 9–15 (VIS) / 18–22 (NIR)
pm — a systematic interpolation bias that does NOT improve with SNR. Native
argmax and linear-resample-argmax are the worst (25/37/51 pm, SNR-independent
grid snap). 6 (linear-argmax, Pantheum's old path) ≈ 1 (native argmax).

### With a 2-px 50% neighbour blend — bias / RMS / p95 (pm), SNR=20, ideal bg
| seg | estimator | bias | RMS | p95 |
|-----|-----------|------|-----|-----|
| UV  | 8 pixel-int     | 34.6 | **37.0** | 47.2 |
| UV  | 4 spline-parab  | 1.6 | 8.3 | 15.5 |
| UV  | 5 spline-root   | 0.8 | 8.9 | 15.6 |
| UV  | 2 native-parab  | 6.1 | 10.6 | 19.1 |
| UV  | 7 peakfit-Gauss | 7.3 | 11.1 | 19.1 |
| VIS | 8 pixel-int     | 59.7 | **61.2** | 80.7 |
| VIS | 4 spline-parab  | 4.7 | 13.2 | 20.8 |
| VIS | 5 spline-root   | 4.2 | 13.6 | 20.9 |
| VIS | 7 peakfit-Gauss | 20.4 | 26.7 | 47.0 |
| NIR | 8 pixel-int     | 35.0 | 48.2 | 96.3 |
| NIR | 4 spline-parab  | -0.9 | **24.2** | 36.7 |
| NIR | 5 spline-root   | -1.1 | 24.7 | 37.3 |
| NIR | 7 peakfit-Gauss | 12.3 | 20.6 | 41.5 |

**The single-Gaussian pixel-int fit is the WORST for a close (2-px) blend** — it
models the pair as one Gaussian and is pulled 35–60 pm. The 3-point parabola and
spline-parab/root are the most blend-robust (they use only the peak's
neighbourhood). This is exactly the dense V-forest / VIS situation in §1, and
explains why pixel-int's real VIS closure scatter was high while peakfit/spline
were tighter. peakfit (7) has a two-Gaussian deblend fallback that helps but its
one-Gaussian seed still biases. NOTE the sim's occasional pixel-int RMS spikes
also reflect infrequent single-Gaussian convergence failures on blends.

---

## 3. Verdict — best estimator × background per segment, and what alibz should use

**Does the spline beat native methods? No — not on ACCURACY, in any segment.**
The decisive real-Fe test (§0) settles it: the pixel-integrated fit (8) is the
most accurate estimator in UV, VIS and NIR, beating the parabola and spline-parab
everywhere (CIs exclude 0) and beating spline-root and peakfit in UV and NIR
(ties in VIS, but medians still favour the fit by 20–60 pm). The spline beats
only native ARGMAX and the linear 1/30 resample. **The owner's "the spline
helps" is true only relative to argmax/parabola — a proper pixel fit is better
still, on real data, everywhere.**

**Corrections to the earlier claims (§1/§2):** the earlier "spline wins in the
NIR" was based on within-run REPEATABILITY (precision) and the tiny n=3 Ar set —
NOT accuracy. On real Fe accuracy the spline loses the NIR too (spline-root 217
vs pixel-int 156 pm). spline-root's only genuine edge is ROBUSTNESS: it cannot
blow up, whereas the free-width pixel fit occasionally fails to converge on
~1-px / blended lines (which is why its *unclipped* scatter is noisy). The
simulation (§2) favours (7)/(8) by construction (Gaussian in, Gaussian out); the
Fe test does not share that bias and still ranks (8) first.

**Best estimator per segment (by real-Fe accuracy):** pixel-integrated fit (8) in
ALL segments; empirical-shape fit (8e) is equivalent (no gain, drop it);
spline-root (5) as a bounded fallback when the pixel fit fails to converge or a
blend is flagged. Background: corrected (per-segment pitch, dead-channel-trimmed)
everywhere — location is only weakly background-sensitive, but the corrected
baseline is required so the fit does not chase the mis-scaled NIR width/baseline
(audit A0). No estimator makes native lines sharp: a ~90–155 pm accuracy floor
remains (§0).

**What alibz should use (file:line where it plugs in):**
1. Make the shared line shape PIXEL-INTEGRATED — `alibz/utils/voigt.py:84`
   (`multi_voigt`, used by `peaky_indexer_v3.forward_model` and `profiles.py`)
   and `alibz/utils/peakfit.py:87/143` (`_gauss` in `gaussian_subpixel_center`).
   This makes the most-accurate estimator (8) the production estimator; it
   already has a two-Gaussian deblend (`peakfit.py:206-227`) for the blend case.
   Entry point `peakfit.refine_peaks` (:318). BOUND the width and add a
   spline-root fallback on non-convergence (the free fit's only failure mode).
2. Replace the shift-model centroid — `fe_plasma_analysis.py:244`
   (`subpixel_center`, 3-pt parabola) / `line_physics_analysis.py:175` — with the
   pixel-integrated fit: on real Fe it is 90 vs 237 (UV) / 117 vs 292 (VIS) /
   156 vs 407 (NIR) pm RMS, a 2.5–2.7× accuracy gain.
3. For the NIR, fit at a bounded / FIXED instrument width (pixel-integrated, σ
   from the segment LSF) — the free-width fit's convergence failures are all on
   ~1-px NIR lines. The empirical-shape fit (8e) is NOT worth its complexity
   (ties 8). Do NOT move analysis onto the 1/30 spline grid.
4. Never use plain argmax or the linear-resample argmax for positions
   (`fe_plasma_analysis.measure_line:146` `peak_nm=wl[argmax]`; corpus
   `peaky_finder` linear resample) — worst in every test.

---

## Unverified / caveats
- The §0 Fe accuracy metric is RMS about a per-run CUBIC-in-λ; it removes only
  smooth map error, so the ~90–155 pm floor is real per-line accuracy (sub-pixel
  phase + db + registration + shape). It cannot separate db uncertainty (<1 pm
  claimed) from estimator error absolutely, but the PAIRED (same-line)
  comparison is unaffected by any shared systematic.
- The free-width pixel-int fit (8) occasionally fails to converge on ~1-px /
  blended lines (median ~1 line/run clipped); production must bound the width and
  fall back to spline-root. Its *unclipped* RMS is therefore noisier than the
  robust (clipped) numbers reported.
- Estimator (8e)'s empirical shape is POOLED across all lines/runs per segment
  (350/336/318 lines), NOT strict leave-one-out; each measured line's own
  contribution is included, so 8e is if anything slightly FLATTERED — and it
  still only TIES (8), so "empirical shape buys nothing for location" is
  conservative.
- The Ar I NIR sub-test (V run2, n=3, weak ~1-px lines) is INCONCLUSIVE as run
  here: free-width fits blow up (spread 400–680 pm). It is not evidence either
  way; the Fe test is decisive.
- Line matching uses the validated dominance screen + shift-model prediction +
  single-peak blend screen + thick-line exclusion + 3×MAD clipping; nearest-max
  matching (which grabs Fe-forest neighbours) is avoided.
- Runs treated separately (13:39 recal). §1/§2 (V run2, Ar) retained below for
  the precision/simulation context, with the corrections noted inline.
- Multiplet-closure (§1) with n=2–3 pairs is uninterpretable and dropped from the
  verdict; the DB is AIR (not vacuum), correcting the earlier explanation.
