**Superseded** by cal/reports/2026-09-24-empirical-lsf.md (rebuilt and re-validated on the axis-corrected data, PIXEL_OFFSET = -18). Section 0 (the ~18.3 px offset) stands and is coordinator-verified in reports/2026-09-23-native-axis-18px-offset.md.

# Empirical line-spread function (ePSF) for the Z300 native grid — 2026-09-23

Status: COMPLETE (2026-09-23 23:10 PDT). No commit made.
Author: judgment agent (Opus 5.5). Provider: Claude subscription (no switch).

Question: can an empirical, pixel-integrated line-spread function (Anderson & King 2000 style),
built per detector segment from confidently identified thin lines, give phase-unbiased centres and
good areas on the critically sampled / undersampled native Z300 data, with genuine line-class
shape differences (ion vs neutral, delay/Stark, self-absorption) modelled as separate physical
kernels rather than folded into the instrument shape?

## 0. Blocking finding: the native wavelength axis is off by ~18.3 pixels

Pattern matching of the 40 strongest predicted db lines against every native run mean (score vs a
PIXEL offset -40..+40 px, `epsf_common.pattern_offset`) gives one unique maximum in every segment of
all 86 runs: observed u = u_vendor(db) + du with du = -18.0..-18.65 px (UV Fe median -18.4, V -18.6;
VIS Fe -18.1, V -18.25; NIR Fe -18.3, V -18.4), score z = 3-17 vs next-best z < 5. In nm that is
-1.4..-1.8 (UV), -2.1..-2.6 (VIS), -2.9..-3.6 nm (NIR): a constant PIXEL offset, not a nm offset.
Independent pattern checks: Na D 588.99/589.59 observed at 586.89/587.48 (spacing 0.59 nm); Ca II H&K
393.37/396.85 at 390.83/394.32 (spacing 3.49 nm); the V I 437.92/438.47/439.00/439.52/440.06 sequence
at 435.46/436.00/436.54/437.09/437.63 nm with the db intensity order; the whole Ar I NIR spectrum
(696.5 ... 922.4 nm, 20 lines, 811.5 and 763.5 strongest) within +-0.6 px of u_db - 18.3 px in every
run checked (9 runs). The native CSV equals the vendor pixToNm applied to the raw pixel index exactly (checked
on moissanite raw/shot-0.json, run-09f1642f: max VIS pixel 1505 -> 438.306 nm; coefficients
[625.4824, -0.111777, -9.35e-6, 6.53e-10]). Segments hold 2066 pixels; a 2048-pixel active array
plus 18 leading non-image pixels would explain it (HYPOTHESIS, not verified against vendor docs).

Consequence: every "-154 pm" shift model and every +-0.35 nm nearest-line match on native data
matched a WRONG line (a neighbour 1-3 nm away; e.g. the "Ar I 696.54 at -0.19 nm" of the
registration reports cannot be Ar I 696.54, which sits ~3.1 nm lower on this axis; what it is was
not checked here). This study registers each
run and segment by the pattern offset (refined to a linear du(u) from the identified lines) before
identifying anything.

## 1. Method

### 1.1 Data and coordinates
- 86 native run means (10-shot means; 26 Fe runs with shots in the Fe scratchpad acq tree and 60 V
  run2 runs), the same set as `cal/scripts/peak_phase_uniformity.py` (reproduced exactly: 141 UV /
  196 VIS / 344 NIR coordinator peaks, `epsf_common.coordinator_peaks`). The 4th channel (>= 955 nm)
  is excluded.
- Coordinate: u = index of a sample within its segment (consecutive native pixels, ascending
  wavelength). All shapes, centres and widths are in PIXELS of the local pitch; nm = u mapped
  through a per-run cubic fit to the file's wavelength column.
- Noise per pixel (`epsf_common.noise_model`): sigma^2 = s0^2 + a S + b^2 S^2 with S the signal above a
  41-px running median. s0 (first-difference scatter of the mean, ~15 counts) is ~4x the
  shot-to-shot noise of the mean in line-free pixels (~3.5 counts): it is the detector's static
  pixel pattern, which does not average down and acts as noise for a line fit. a, b from the
  shot-to-shot variance after removing each shot's common intensity scale.

### 1.2 Line identification (validated, not nearest-maximum)
Per run and segment: (i) pattern pixel offset du0 (section 0); (ii) local maxima with SNR >= 8, centre by
a pixel-integrated Gaussian (only for matching); (iii) a peak is identified with db line L if
|u_obs - (u_db(L) + du(u))| <= 0.45 px, L's predicted strength (per element, normalised in the
segment; Fe or V at 9000 K plus ambient Ar at full weight, Ca/Na/K/H/O/N at 5 %) dominates every
other db line within +-2 px by x3, L is >= 0.5 % of the segment's strongest predicted line of its
element, and no other observed maximum lies within +-2 px; (iv) du(u) is refitted as a line through
the identified set with 3-MAD clipping and the match repeated (3 passes). Result: 6,802 identified
windows; residual about the per-run registration MAD 0.18 / 0.13 / 0.12 px (UV / VIS / NIR) with a
Gaussian centroid, 6 / 3 / 2 % beyond 0.35 px (`build_epsf.catalog`). Isolation for fitting: no other maximum within +-2 px above 3 % of the line and
none taller than the line within +-4 px; maxima at 2-7.5 px are fitted simultaneously as neighbour
components (instrument ePSF, own area and centre).

### 1.3 The ePSF model and build (alibz/utils/epsf.py)
y_k = A Psi(u_k - c) + b0 + b1 (u_k - c) [+ neighbours]. Psi is a cubic B-spline, knots every 1/8 px,
compact support |d| < 3 px (fit window +-4.5 px so the background is set where Psi = 0). Constraints:
- partition of unity sum_k Psi(k + x) = 1 for all x (a property of any pixel-integrated LSF; imposed
  as linear equality constraints). It removes an exact degeneracy: a shape Psi(d)(1 + a(frac d)) with
  phase-dependent areas fits the samples equally well (verified on synthetic data: without it the
  built FWHM depended on the starting width, 1.32 vs 1.36 px for a 1.33 px truth);
- centroid within |d| <= 2 px = 0 (centring convention; a constant per segment, absorbed by the
  wavelength registration);
- second-difference smoothness, weight lam chosen per segment by 5-fold held-out chi2
  (UV 0.3, VIS 0.1, NIR 1.0 over the grid 0.03/0.1/0.3/1.0).

**The key design problem: phase-warp degeneracy.** For 1-D undersampled lines at unknown phases
the joint (shape, centres) problem is degenerate up to a periodic phase warp: a slightly wider
shape plus centres pulled by g(phi) fits the samples as well as the truth (synthetic, Gaussian
start: chi2/nu 0.9367 vs 0.9358 for the truth, while the centres carry a +-0.04 px phase bias; plain
Anderson-King alternation stalls there). Three ways to break it were implemented and tested on
synthetic data with known truth (`tests/test_epsf.py`):
1. `warp_mode="uniform"`: impose uniform fitted phases (rank-based Fourier warp). Works only for
   many DISTINCT lines: the residual bias is the sampling noise of the true phases,
   ~0.5/sqrt(N_distinct) (0.02-0.06 px p-p for N = 300-600).
2. `warp_mode="dither"`: estimate the warp from dither consistency (the same transition in many
   runs sits at different phases because each run's registration drifts): fit
   c = kappa_line + a_run + b_run u - g(phi). On synthetic recurring lines (40 transitions x 15 runs,
   drift +-0.3 px) held-out bias 0.001 px vs 0.03-0.07 px for the uniform prior. On the REAL data it
   was unstable (warp estimates wandering, some builds running away), because the significance of a
   Fourier warp is fragile with few lines per run.
3. `warp_mode="dither_struct"` (USED): each iteration replaces the free centres of recurring lines by
   the structured centres kappa_line + a_run + b_run u (robust LSQ, lines seen in >= 3 runs, runs with
   >= 5 such lines), refits area/background at those fixed centres, and builds the shape from them.
   Within a line the relative positions across runs are then set by the registration, which is
   estimated from all lines at once, so a phase bias cannot survive; no warp model or significance
   gate. Synthetic held-out bias 0.001-0.002 px p-p, FWHM within 1 %; on real data it converges
   (phase-binned correction ~0.005-0.01 px at the end).

This is a departure from the brief's "iterate shape <-> centres with centering constraints": those
constraints alone do not determine the answer on undersampled lines (demonstrated above), and the
coordinator's flat-histogram criterion needs many DISTINCT lines, which these data do not have
(section 3a).

### 1.4 Reference class (instrument shape)
Identified, isolated windows with: delay >= 5 us; SNR 15-3000; |obs - db - registration| < 0.35 px;
lower level E_i > 0.12 eV (no ground-term/resonance lines); optical-depth proxy
tau_rel = gA lambda^3 exp(-E_i/kT) / max(same element and ion) < 0.3 at 9000 K; not Na/Ca/K/H. Result:
UV 626 windows / 33 db lines / 68 runs (V II 461, Fe II 114, Fe I 51); VIS 679 / 44 / 65 (Fe I 356,
V I 219, V II 102, Ar I 2); NIR 70 / 17 / 10 (Ar I 54, V I 16). Departure from the brief: a
strict "thin, modest gA, high-E_i neutral" class leaves 0-16 windows per segment in these pure-metal
spectra (every detectable line is a strong line), so the class was widened and the class tests of
section 4 check what the widening costs (reference vs non-reference, thin vs mid vs thick).

## 2. Shapes (cal/data/epsf-z300-20260923.json, figures epsf_shapes.png, subband_fwhm.png)

| seg | windows / lines / runs | FWHM px (jackknife SD; 5 folds) | FWHM at median pitch | max abs(ePSF - pix-int Gaussian of equal FWHM) / peak |
|--|--|--|--|--|
| UV | 626 / 33 / 68 | 1.711 (0.154; 1.68-1.87) | 148 pm (0.0864 nm/px) | 0.117 |
| VIS | 677 / 44 / 65 | 1.258 (0.039; 1.24-1.29) | 174 pm (0.1380 nm/px) | 0.069 |
| NIR | 70 / 17 / 10 | 1.526 (0.035; 1.52-1.57) | 284 pm (0.1859 nm/px) | 0.117 |

Shape: not Gaussian. UV is flat-topped and skewed (core mass to the blue, a red shoulder at
d ~ +1.5 px); VIS has shoulders at d ~ -2.2 and +2.0 px at ~7 % of the peak; NIR has a red wing to
+2.5 px at ~7 % of the peak (d = u - c; u increases with wavelength). All three carry more mass on the
blue side of the core ((right - left)/total = -0.044 / -0.037 / -0.039). The coordinator's
"VIS Gaussian fits hit sigma = 0.25 px" is consistent with a 1.26 px FWHM shape whose core is narrower
than a Gaussian of the same FWHM.

**Pixel vs nm, and variation within a segment.** Three sub-bands per segment (equal numbers of
distinct reference lines; each with its own 5 CV folds, fold spread in brackets):

| seg | band (db nm) | windows / lines | FWHM px [fold range] | pitch nm/px | FWHM pm |
|--|--|--|--|--|--|
| UV | 242-275 | 153 / 11 | 1.62 [1.35-1.67] | 0.0906 | 147 |
| UV | 276-314 | 236 / 10 | 1.72 [1.67-1.82] | 0.0872 | 150 |
| UV | 324-365 | 224 / 11 | 2.10 [2.06-2.21] | 0.0805 | 169 |
| VIS | 373-404 | 247 / 14 | 1.22 [1.14-1.31] | 0.1395 | 170 |
| VIS | 406-440 | 226 / 14 | 1.25 [1.23-1.30] | 0.1366 | 171 |
| VIS | 442-612 | 168 / 15 | 1.51 [1.48-1.57] | 0.1216 | 183 |
| NIR | 625-697 | 20 / 7 | 1.41 [1.37-1.54] | 0.1974 | 279 |
| NIR | 707-750 | 23 / 4 | 1.52 [1.46-1.56] | 0.1866 | 283 |
| NIR | 751-852 | 27 / 6 | 1.54 [1.52-1.55] | 0.1789 | 275 |

Band-to-band coefficient of variation of the FWHM: in px 14 % (UV), 12 % (VIS), 4.5 % (NIR); in nm
7.6 %, 4.3 %, 1.3 %. The width varies within a segment by more than the fold scatter, and it is
closer to CONSTANT IN WAVELENGTH than in pixels (the pitch falls toward the red end of each segment
and the width in pixels rises there). The shape is therefore stored per sub-band in pixel units, and the
recommended shape at wavelength lambda is the coefficient-wise linear interpolation of the sub-band
shapes at their median wavelengths (`validate_epsf.Shapes.at`). Caveat: the bands also differ in line
mix (the red VIS band is mostly Fe I, the red UV band mostly V II), so part of the trend could be
class, not optics.

## 3. Validation (cal/data/epsf-validation-20260923.json; `cal/scripts/validate_epsf.py`)

Estimators on the SAME windows: spline-root, 3-point parabola and free-width pixel-integrated Gaussian
exactly as `peak_phase_uniformity.py` defines them (on the segment arrays), "ePSF seg" (one shape per
segment) and "ePSF lambda" (sub-band shapes interpolated at the line's db wavelength; the RECOMMENDED
form). Every reference line is fitted with shapes built WITHOUT its db line (5 folds by db line, for
the segment shape and for each sub-band); lines outside the reference class never entered a build.

**A property of these data that decides how the tests must be read:** the coordinator's 681 peaks are
only 9 (UV) / 18 (VIS) / 21 (NIR) distinct lines seen in many runs, and the reference set is 33 / 44 / 17
distinct lines. Their true sub-pixel phases are a lumpy sample of 9-44 values (each smeared by the
~0.1-0.3 px run-to-run drift), not a uniform distribution. A flat-histogram test that counts windows
as independent rejects EVERY estimator, including a perfect one. Two replacements are reported:
the cluster-aware p (null: one uniform phase per distinct line, observed within-line drift offsets
kept) and the dither test (fit c = kappa_line + a_run + b_run u per estimator, report the peak-to-peak
of the median residual vs phase with a bootstrap-over-lines 68 % interval). The dither test sees only
the part of a bias that varies within each line's drift range, so it UNDERSTATES absolute bias; it is
fair between estimators.

### 3a. Phase uniformity on the coordinator peak set
| seg | windows / distinct lines | estimator | 10-bin hist | chi2 p | KS p | cluster-aware p | dither ptp px [68%] (lines x runs) |
|--|--|--|--|--|--|--|--|
| UV | 141 / 9 | spline | 20 13 5 4 4 1 50 9 9 26 | 9e-26 | 5e-10 | 0.49 | too few recurring lines |
| UV | 141 / 9 | parabola | 16 18 12 7 3 0 44 16 7 18 | 8e-17 | 4e-06 | 0.78 | too few recurring lines |
| UV | 141 / 9 | pix-Gauss | 10 36 19 5 2 1 4 41 19 4 | 3e-24 | 0.0005 | 0.48 | too few recurring lines |
| UV | 141 / 9 | ePSF seg | 5 37 18 7 3 1 0 20 36 14 | 2e-21 | 2e-06 | 0.67 | too few recurring lines |
| UV | 141 / 9 | ePSF λ-interp | 10 7 43 19 4 1 0 39 16 2 | 1e-28 | 0.0007 | 0.50 | too few recurring lines |
| VIS | 196 / 18 | spline | 1 3 6 19 43 50 37 24 13 0 | 1e-28 | 5e-14 | 0.14 | 0.009 [0.009, 0.027] (6 x 6) |
| VIS | 196 / 18 | parabola | 1 4 7 32 27 46 32 31 16 0 | 2e-21 | 1e-11 | 0.30 | 0.014 [0.006, 0.018] (6 x 6) |
| VIS | 196 / 18 | pix-Gauss | 9 29 21 6 10 1 7 35 33 45 | 2e-18 | 2e-14 | 0.30 | 0.027 [0.009, 0.055] (6 x 6) |
| VIS | 196 / 18 | ePSF seg | 1 21 31 6 14 17 15 29 39 23 | 8e-10 | 1e-05 | 0.74 | 0.009 [0.007, 0.013] (6 x 6) |
| VIS | 196 / 18 | ePSF λ-interp | 1 14 27 20 9 25 6 32 42 20 | 6e-12 | 8e-07 | 0.60 | 0.008 [0.007, 0.017] (6 x 6) |
| NIR | 344 / 21 | spline | 18 12 12 21 49 84 51 38 39 20 | 1e-24 | 3e-16 | 0.14 | 0.060 [0.059, 0.072] (17 x 36) |
| NIR | 344 / 21 | parabola | 14 17 13 22 47 65 65 38 36 27 | 2e-17 | 5e-15 | 0.25 | 0.066 [0.052, 0.074] (17 x 36) |
| NIR | 344 / 21 | pix-Gauss | 30 28 19 22 30 34 44 46 41 50 | 0.0008 | 2e-05 | 0.81 | 0.013 [0.013, 0.022] (17 x 36) |
| NIR | 344 / 21 | ePSF seg | 30 20 35 24 28 45 34 46 31 51 | 0.002 | 0.0004 | 0.92 | 0.030 [0.025, 0.037] (17 x 36) |
| NIR | 344 / 21 | ePSF λ-interp | 44 26 24 19 29 46 28 44 33 51 | 0.0002 | 0.006 | 0.89 | 0.010 [0.012, 0.032] (17 x 36) |

### 3b. Cross-validation (held-out reference lines)
| seg | n ref | held-out chi2/nu median | fold FWHM px | held-out phase chi2 p / cluster p | dither ptp px [68%]: spline / parabola / pix-Gauss / ePSF seg / ePSF λ |
|--|--|--|--|--|--|
| UV | 624 | 9.86 | 1.68 1.87 1.69 1.79 1.85 | 3e-66 / 0.01 (33 lines) | 0.028 [0.028,0.060] / 0.027 [0.032,0.088] / 0.012 [0.015,0.030] / 0.013 [0.012,0.034] / 0.017 [0.014,0.031] |
| VIS | 675 | 9.15 | 1.24 1.29 1.27 1.26 1.24 | 1e-06 / 0.92 (44 lines) | 0.062 [0.058,0.097] / 0.049 [0.045,0.083] / 0.017 [0.014,0.032] / 0.008 [0.008,0.016] / 0.009 [0.009,0.017] |
| NIR | 69 | 4.28 | 1.52 1.54 1.52 1.52 1.56 | 0.1 / 0.39 (16 lines) | 0.106 [0.065,0.161] / 0.091 [0.051,0.147] / 0.026 [0.021,0.043] / 0.051 [0.032,0.057] / 0.032 [0.025,0.045] |


UV (b): the held-out ePSF phases also fail the cluster-aware test (p = 0.01, pile-up at +0.2..+0.3 px),
so a residual UV phase bias of the ePSF cannot be excluded.

### 3c. Accuracy on identified thin Fe lines (pm; obs - db about ONE per-run per-segment quadratic fitted to the median of the five estimators, 3.5-MAD clip common to all)
| seg | lines (kept) / runs / distinct | estimator | median abs [95% CI] | p90 abs | frac > 1 px | RMS kept [95% CI] |
|--|--|--|--|--|--|--|
| UV | 140 (133) / 13 / 17 | spline | 5.6 [4.3, 7.6] | 20.2 | 0.000 | 9.9 [8.5, 11.3] |
| UV | 140 (133) / 13 / 17 | parabola | 5.1 [3.8, 6.7] | 18.2 | 0.000 | 9.3 [8.0, 10.5] |
| UV | 140 (133) / 13 / 17 | pix-Gauss | 5.9 [4.6, 7.0] | 16.0 | 0.007 | 11.9 [8.2, 16.5] |
| UV | 140 (133) / 13 / 17 | ePSF seg | 8.1 [6.8, 10.0] | 18.7 | 0.000 | 10.3 [9.1, 11.5] |
| UV | 140 (133) / 13 / 17 | ePSF λ-interp | 7.3 [6.3, 8.5] | 19.6 | 0.000 | 10.2 [9.1, 11.2] |
| UV | paired | abs(spline) - abs(ePSF λ) | -1.63 [-3.05, +0.61] | ePSF better on 43 % | | |
| UV | paired | abs(parabola) - abs(ePSF λ) | -2.98 [-3.94, -1.28] | ePSF better on 37 % | | |
| UV | paired | abs(pix-Gauss) - abs(ePSF λ) | -1.65 [-3.56, +0.10] | ePSF better on 44 % | | |
| UV | paired | abs(ePSF seg) - abs(ePSF λ) | +0.25 [-1.61, +2.36] | ePSF better on 52 % | | |
| VIS | 357 (337) / 24 / 24 | spline | 13.0 [11.8, 14.4] | 24.5 | 0.000 | 14.5 [13.7, 15.1] |
| VIS | 357 (337) / 24 / 24 | parabola | 12.6 [11.2, 13.8] | 23.2 | 0.000 | 13.6 [12.9, 14.3] |
| VIS | 357 (337) / 24 / 24 | pix-Gauss | 8.1 [7.3, 9.5] | 24.6 | 0.000 | 12.4 [11.5, 13.2] |
| VIS | 357 (337) / 24 / 24 | ePSF seg | 5.2 [4.7, 5.7] | 14.9 | 0.000 | 8.3 [7.1, 9.5] |
| VIS | 357 (337) / 24 / 24 | ePSF λ-interp | 5.0 [4.6, 5.7] | 22.1 | 0.000 | 9.2 [8.2, 10.3] |
| VIS | paired | abs(spline) - abs(ePSF λ) | +6.16 [+4.48, +7.79] | ePSF better on 75 % | | |
| VIS | paired | abs(parabola) - abs(ePSF λ) | +5.33 [+4.47, +7.31] | ePSF better on 75 % | | |
| VIS | paired | abs(pix-Gauss) - abs(ePSF λ) | +2.29 [+1.52, +3.53] | ePSF better on 65 % | | |
| VIS | paired | abs(ePSF seg) - abs(ePSF λ) | -0.21 [-0.72, +0.48] | ePSF better on 49 % | | |


Robustness to the choice of common curve (median |r|, pm; curve fitted to one estimator or to the mean
of all five): VIS ePSF-lambda 5.0-7.1 and ePSF-seg 4.5-6.5 vs pix-Gauss 7.9-10.1, parabola 11.3-13.8,
spline 12.6-14.3 for EVERY choice (ePSF wins even on the spline's own curve). UV: every estimator wins
on its own curve (spline 6.3 on spline curve, ePSF-lambda 5.2 on its own, 6.4 on the mean-of-all curve
vs parabola 5.4, spline 6.1) -- the UV test cannot rank the estimators at the 1-3 pm level.
Only 13 Fe runs and 17 distinct lines contribute in UV; NIR has no usable Fe lines.

### 3c'. Ar I lines (pm; residual about a linear curve through the OTHER identified Ar I lines of the same run; 9 runs)
| line | estimator | n runs | median | MAD | median abs [95% CI] |
|--|--|--|--|--|--|
| 696.543 | spline | 9 | +8.8 | 19.7 | 17.8 [10.4, 48.1] |
| 696.543 | parabola | 9 | +8.1 | 20.1 | 15.8 [8.6, 44.2] |
| 696.543 | pix-Gauss | 9 | -4.2 | 6.9 | 4.2 [2.3, 17.1] |
| 696.543 | ePSF seg | 9 | -8.8 | 5.6 | 8.8 [5.0, 12.6] |
| 696.543 | ePSF λ-interp | 9 | -11.8 | 7.6 | 11.8 [7.8, 21.8] |
| 772.405 (772.376+772.421 blend) | spline | 9 | -6.4 | 14.5 | 9.2 [3.4, 22.9] |
| 772.405 (772.376+772.421 blend) | parabola | 9 | -4.7 | 11.7 | 8.1 [3.2, 20.3] |
| 772.405 (772.376+772.421 blend) | pix-Gauss | 9 | +2.4 | 2.8 | 2.8 [0.6, 5.1] |
| 772.405 (772.376+772.421 blend) | ePSF seg | 9 | +1.2 | 2.4 | 1.9 [1.2, 3.4] |
| 772.405 (772.376+772.421 blend) | ePSF λ-interp | 9 | +4.2 | 3.2 | 4.2 [0.6, 5.3] |
| 801.479 | spline | 9 | -27.2 | 4.1 | 27.2 [16.6, 29.2] |
| 801.479 | parabola | 9 | -25.1 | 3.1 | 25.1 [14.3, 26.7] |
| 801.479 | pix-Gauss | 9 | -13.8 | 7.5 | 13.8 [1.4, 18.9] |
| 801.479 | ePSF seg | 9 | -3.3 | 7.1 | 7.9 [3.3, 8.5] |
| 801.479 | ePSF λ-interp | 9 | -5.3 | 7.8 | 8.1 [3.8, 10.5] |


### 3d. Areas
#### V I 417.94/415.97, thin R0 = 2.047, 51 runs, median forced SNR 3.8 / 3.7
| estimator | n | median [95% CI] | robust SD of ln R | by delay 0/1/5/10/20/50 |
|--|--|--|--|--|
| forced | 51 | 2.103 [1.989, 2.182] | 0.116 | 2.378 / 2.187 / 2.124 / 2.176 / 1.911 / 1.586 |
| ep | 45 | 2.221 [2.045, 2.365] | 0.170 | n/a / n/a / 2.392 / 2.341 / 1.972 / 1.599 |
| p8 | 51 | 1.249 [1.158, 1.421] | 0.261 | 1.470 / 1.378 / 1.343 / 1.459 / 1.167 / 0.998 |
| trapz | 2 | 0.158 [n/a, n/a] | 0.271 | n/a / n/a / n/a / n/a / n/a / n/a |
| height | 51 | 5.373 [4.640, 5.746] | 0.277 | 6.478 / 5.955 / 6.156 / 5.268 / 4.129 / 2.839 |
pooled forced by delay: {'0': (1, 2.38, 1.12), '1': (12, 2.14, 0.32), '5': (12, 2.19, 0.3), '10': (11, 2.12, 0.28), '20': (9, 1.97, 0.21), '50': (6, 1.6, 0.19)}

#### Same-upper-level multiplets (ln Rmeas/R0; median, robust SD across runs)
| pair | runs | R0 | ePSF | pix-Gauss | ePSF+Lorentz | ePSF+selfabs |
|--|--|--|--|--|--|--|
| FeI 375.82/370.92 | 5 | 4.02 | -0.49 (0.013) | -0.92 (0.050) | -0.85 (0.047) | -0.84 (0.038) |
| FeI 406.36/396.93 | 7 | 2.88 | -0.15 (0.078) | -0.41 (0.198) | -0.62 (0.128) | -0.48 (0.053) |
| FeI 406.36/413.21 | 8 | 5.74 | -0.85 (0.123) | -0.58 (0.020) | -0.80 (0.054) | -0.81 (0.057) |
| FeI 407.17/400.52 | 4 | 3.68 | -0.26 (0.015) | -0.17 (0.003) | -0.26 (0.015) | -0.25 (0.012) |
| FeI 383.42/387.25 | 3 | 4.35 | -1.66 (0.150) | -1.71 (0.179) | -1.66 (0.150) | -1.65 (0.140) |
| FeII 233.80/232.74 | 5 | 1.73 | -0.12 (0.151) | -0.03 (0.020) | -0.12 (0.171) | -0.12 (0.144) |


"forced" = ePSF areas at the REGISTERED positions (registration from the run's ePSF centres; local maxima
2-4.5 px away fitted as fixed-centre neighbours; linear background) -- the use the ePSF makes possible for
weak lines. 417.94/415.97 share the upper level, so the thin ratio is (gA/lambda) = 2.047 independent of
T. The forced ePSF gives 2.10 [1.99, 2.18] (1-10 us: 2.12-2.19), i.e. consistent with OPTICALLY THIN, with
half the run-to-run scatter of the free Gaussian. The late-delay drop (1.97 +- 0.21 at 20 us, 1.61 +- 0.19
at 50 us, pooled) is at most 2 sigma and the lines are weak there. The formal forced SNR (~3.7) is
inflated by chi2/nu > 1 (the observed scatter implies ~12 % per run). The free Gaussian (1.25), the
local-max height (5.4) and the trapezoid (unusable: its 0.25-0.50 nm sidebands contain the neighbour
3 px away) all fail. This REFUTES the earlier "417.94/415.97 = 0.75 -> tau ~ 1.5 self-absorption"
(cal/reports/2026-09-22-line-matching-sa-broadening.md section 2a, vanadium-timing-study section 4.1):
those measured other features (section 0).

Multiplets: the ePSF lowers the run-to-run scatter for 3 of 6 pairs and raises it for 3; all Fe I pairs
sit 0.15-1.7 in ln below R0 (strong member self-absorbed and/or an uncorrected spectral response;
not separated here). No improvement is demonstrated.

### 3e. Fit residual vs sub-pixel phase (held-out reference lines; median residual/area in 5 phase bins x 9 offsets)
| seg | model | RMS of phase-bin medians | max abs | phase-dependent part | chi2/nu by phase bin (ePSF) |
|--|--|--|--|--|--|
| UV | ePSF λ | 0.0113 | 0.0357 | 0.0094 | -0.5:8.3 -0.3:8.5 -0.1:17.6 +0.1:10.1 +0.3:6.4 |
| UV | pix-Gauss free | 0.0257 | 0.0680 | 0.0181 |  |
| VIS | ePSF λ | 0.0103 | 0.0325 | 0.0081 | -0.5:7.2 -0.3:14.6 -0.1:10.3 +0.1:7.3 +0.3:10.4 |
| VIS | pix-Gauss free | 0.0215 | 0.0651 | 0.0145 |  |
| NIR | ePSF λ | 0.0166 | 0.0985 | 0.0135 | -0.5:2.0 -0.3:6.7 -0.1:6.8 +0.1:7.5 +0.3:2.3 |
| NIR | pix-Gauss free | 0.0112 | 0.0415 | 0.0085 |  |

## 4. Line classes: genuine shape differences (physical kernels on top of the instrument ePSF)

Every identified isolated window with SNR >= 15 (2,921) is fitted with the lambda-interpolated
instrument ePSF (held out where applicable) convolved with (i) a Lorentzian of HWHM gamma (px, >= 0) and
(ii) a self-absorbed slab 1 - exp(-tau phi), phi a unit-peak Lorentzian of fixed HWHM 0.02 px (a
narrow intrinsic line, so the kernel describes pure opacity broadening / core flattening). gamma is an
EXCESS width relative to the reference class (which carries its own small physical width).
Significance: per-run class medians, differences paired by run, t-test across runs (so run-to-run
scatter is the error). Classes: tau_rel thin < 0.1 <= mid < 0.3 <= thick; lower level res (E_i <= 0.12 eV,
ground term) / low (<= 1.2 eV) / high.

| seg | comparison (B - A) | runs | mean diff gamma px | SD across runs | t | p |
|--|--|--|--|--|--|--|
| UV | thick - thin | 67 | +0.280 | 0.184 | 12.5 | 5e-19 |
| UV | mid - thin | 67 | +0.118 | 0.110 | 8.8 | 1e-12 |
| UV | low-lying - high E_i | 66 | +0.116 | 0.186 | 5.1 | 4e-06 |
| UV | resonance - high E_i | 66 | +0.078 | 0.161 | 3.9 | 2e-04 |
| UV | ion - neutral | 71 | +0.078 | 0.177 | 3.7 | 4e-04 |
| UV | non-reference - reference | 68 | +0.057 | 0.092 | 5.1 | 3e-06 |
| VIS | thick - thin | 62 | +0.190 | 0.088 | 17.0 | 8e-25 |
| VIS | mid - thin | 76 | +0.030 | 0.063 | 4.1 | 1e-04 |
| VIS | low-lying - high E_i | 74 | +0.063 | 0.078 | 6.9 | 1e-09 |
| VIS | resonance - high E_i | 70 | +0.046 | 0.071 | 5.4 | 1e-06 |
| VIS | ion - neutral | 51 | -0.104 | 0.076 | -9.8 | 3e-13 |
| VIS | non-reference - reference | 62 | +0.117 | 0.091 | 10.1 | 1e-14 |
| NIR | thick - thin | 6 | +0.074 | 0.081 | 2.2 | 0.08 |
| NIR | non-reference - reference | 10 | +0.064 | 0.055 | 3.7 | 0.005 |

Class medians of gamma (px): reference 0.014 (UV) / 0.000 (VIS) / 0.000 (NIR); thick 0.23 / 0.20 / 0.17.
In nm at the median pitch the thick-line excess HWHM is ~20 pm (UV), ~28 pm (VIS), ~31 pm (NIR).

Findings:
1. **Optical depth is the dominant, real class effect.** Lines with a large optical-depth proxy and
   lines from low-lying lower levels are broader than the instrument shape by 0.06-0.28 px (t = 5-17
   against run-to-run scatter). They must be fitted as ePSF (x) kernel, not with the instrument shape.
2. **Delay (Stark / density) is real for V I and V II.** Excess gamma of non-thick lines vs delay
   (median over runs, SD across runs): VIS V I 0.087 (1 us), 0.065 (5), 0.037 (10), 0.013 (20), 0.000 (50),
   Spearman rho = -0.28, p = 2e-10; UV V II 0.157 / 0.186 / 0.140 / 0.074 / 0.036, rho = -0.13, p = 3e-4.
   VIS V II (4 lines) and UV V I (3 lines): flat at 0. The width decays with delay as n_e (and column
   density) falls. The reference build admits delays >= 5 us, where the reference class itself shows
   gamma ~ 0 (VIS 0.000, UV 0.014), so the instrument shape is not visibly Stark-contaminated; lines at
   0-5 us need the kernel.
3. **Ion vs neutral: no clean statement.** UV ions are broader (+0.08 px) but VIS ions are narrower
   (-0.10 px); in both segments the comparison is confounded with optical depth (VIS neutrals include
   the strong V I a6D lines; UV has only 14 neutral lines) and with element. Not separable with these data.
4. **Core flattening / self-reversal: not detected.** The slab kernel beats the Lorentzian (by
   delta chi2 > 0.5) in only 0-7 % of windows of any class; broadened lines look Lorentz-winged, not
   flat-topped, at 1.3-1.7 px per FWHM. Stacked residuals of thick lines about the instrument fit
   (median residual/area vs offset) show a central deficit of 0.04-0.07 with shoulders of +0.03-0.07 at
   +-1.1 px, i.e. broadening, with a RED-side excess (UV +0.068 at +1.1 px vs +0.041 at -0.9; VIS +0.067 vs
   +0.014; VIS resonance lines -0.04 at -1.4 px and +0.073 at +1.6 px). A red asymmetry of strong lines is
   what a Stark shift/asymmetry or absorption in a cooler, receding layer would produce; which one is not
   decided here. No central dip deep enough to indicate reversal.
5. The widened reference is justified by the numbers for VIS (reference gamma 0.000 [0.000, 0.011];
   thin class 0.013) but in UV the reference is 0.057 px narrower than the rest and mid-tau lines already
   carry +0.12 px: the UV instrument shape is the least certain of the three.

Figures: class_gamma_vs_delay.png (gamma vs delay by species), v_417_415_ratio.png.

## 5. Acceptance criteria: plain verdicts

| criterion | UV | VIS | NIR |
|--|--|--|--|
| (a) flat phase histogram on the coordinator peaks | FAIL as specified for all 5 estimators (naive p <= 7e-4); the test has no power here (9 distinct lines; cluster-aware p 0.48-0.78 for all) | FAIL as specified for all (naive p <= 1e-5); cluster-aware p 0.14 spline / 0.30 parabola / 0.30 pix-Gauss / 0.60 ePSF; dither residual 0.008 px ePSF vs 0.009-0.027 others (6 lines x 6 runs only) | FAIL as specified for all (naive p <= 6e-3); cluster-aware p 0.14-0.92; dither 0.010 px ePSF vs 0.013 pix-Gauss, 0.060 spline, 0.066 parabola |
| (b) cross-validated shape (held-out lines) | FAIL (marginal): held-out cluster-aware p = 0.01; dither 0.017 px vs pix-Gauss 0.012 | PASS: cluster-aware p = 0.92; dither 0.009 px vs 0.017 pix-Gauss, 0.049 parabola, 0.062 spline | NOT SHOWN: 69 windows / 16 lines; dither 0.032 vs 0.026 pix-Gauss |
| (c) Fe accuracy, common curve, paired | FAIL / inconclusive: 7.3 pm vs 5.1-5.9; ranking flips with the curve | PASS: 5.0 [4.6, 5.7] pm vs 8.1 pix-Gauss, 12.6 parabola, 13.0 spline; paired CIs exclude 0; robust to curve choice | no Fe lines |
| (c) Ar I 696.54 / 772.40 / 801.48 | - | - | mixed (9 runs): 801.48 ePSF 8 pm vs 14-27; 772.40 4.2 vs 2.8 pix-Gauss; 696.54 11.8 vs 4.2 pix-Gauss |
| (d) V 417.94/415.97 | - | PASS: forced ePSF 2.10 [1.99, 2.18] vs thin 2.05; ln-scatter 0.12 vs 0.26 pix-Gauss, 0.28 height | - |
| (d) multiplets | no gain (1 pair, worse) | no gain (3 better, 2 worse of 5) | - |
| (e) residual shape vs phase | PASS: phase-dependent part 0.009 vs 0.018 pix-Gauss | PASS: 0.008 vs 0.015 | FAIL: 0.014 vs 0.009 |

Bottom line: the ePSF is the better estimator in VIS on every test that has power (centres ~2.5x more
accurate than spline/parabola, ~1.6x than the Gaussian; forced areas recover the thin branching ratio).
In UV and NIR it is not demonstrated: UV because the shape varies strongly within the segment and the
reference is ion-dominated and partly broadened, NIR because only 17 reference lines (mostly Ar I in
10 runs) exist. The coordinator's flat-histogram criterion cannot be met or failed with 9-44 distinct
lines; it should be replaced by the dither test (or data with many more distinct lines).

## 6. Integration (later; no existing module was edited)

- Model/fit API: `alibz/utils/epsf.py` -- `EPSF.from_dict`, `fit_line(epsf, u, y, sigma, c0, neighbors=...,
  kernel=None|'lorentz'|'selfabs')`, `convolve_kernel`, `build_epsf(..., warp_mode='dither_struct')`.
  Shapes: `cal/data/epsf-z300-20260923.json` -> `segments[seg].sub_bands[b].epsf` (+ `cv_folds`);
  interpolate coefficients at the line wavelength as `cal/scripts/validate_epsf.py` `Shapes.at` does.
  Inputs must be GENUINE native pixels (u = consecutive sample index within a segment); never apply it
  to resampled data (the native-grid audit showed pixel-integrated models are worse on resampled grids).
- `alibz/utils/peakfit.py:90` `gaussian_subpixel_center` and `:318` `refine_peaks`: add an `epsf=` path
  (centre from `fit_line`, width no longer free; convert u -> nm with the local pitch, `_local_pitch` at
  `:59`); keep the Gaussian as the documented fallback (fit_line already falls back to it).
- `alibz/utils/voigt.py:47` `multi_voigt` (point-sampled profile, audit A1 at `voigt.py:84`): for native
  data replace by sum_j A_j (Psi (x) Lorentz(gamma_j))(u - c_j) evaluated per pixel (`convolve_kernel`).
- `alibz/peaky_finder.py:845` `fit_peaks` / `:115` `_voigt_seed` / `:1044` `_fwhm_cap`: seed widths from
  the ePSF FWHM of the sub-band (px) instead of a global pitch, fit gamma as the only free width.
- Wavelength registration (`alibz/wavelength_registration.py`, `alibz/utils/wavelength.py`) must first
  absorb the ~18.3 px pixel offset (section 0); until then every db-matched line is suspect.
- Per-class use: thick / low-E_i / early-delay lines -> `kernel='lorentz'` (gamma bounds [0, 3] px); the
  slab kernel adds nothing measurable at this resolution.

## 7. Files

- `alibz/utils/epsf.py` (new): shape, fitter, kernels, build with `uniform` / `dither` /
  `dither_struct` phase handling, partition-of-unity constraint.
- `tests/test_epsf.py` (new): 17 synthetic-truth tests (phases, SNR 30/300, flat-top vs Gaussian bias,
  build recovery from wrong starts, warp degeneracy, recurring-line dither, neighbours/blends,
  thick-line slab kernel, fallback, round trip). `.venv/bin/python -m pytest -q tests/test_epsf.py`:
  17 passed (93 s).
- `cal/scripts/epsf_common.py` (new): loading, noise model, coordinator estimators, pattern registration,
  identification.
- `cal/scripts/build_epsf.py` (new, `--help`, deterministic; stages catalog / lamcv / build / write):
  writes `cal/data/epsf-z300-20260923.json` (shapes per segment, 5 CV folds, 3 sub-bands x 5 folds,
  per-run registration, lam CV table, provenance incl. git HEAD and a SHA-256 over all line windows).
  Rebuilding reproduced every FWHM to 3 decimals.
- `cal/scripts/validate_epsf.py` (new, `--help`, deterministic): writes
  `cal/data/epsf-validation-20260923.json` and `cal/figures/epsf-20260923/*.png` (epsf_shapes,
  subband_fwhm, phase_histograms, dither_test, class_gamma_vs_delay, v_417_415_ratio). Two runs gave
  byte-identical summary tables.
- Regenerate: `.venv/bin/python cal/scripts/build_epsf.py --workdir DIR --stage catalog`, then
  `--stage lamcv` (optional, ~13 min on 10 cores), `--stage build` (~15 min), then
  `.venv/bin/python cal/scripts/validate_epsf.py --workdir DIR` (~12 min).

## 8. Unverified / limits

- The ~18.3 px offset is established by pattern matching on 86 runs (one calibration epoch, 13:39
  2026-09-22 cubics) and checked against raw pixels for one run; its cause (18 non-image leading pixels
  in a 2066-sample readout?) and whether the vendor software applies it are NOT verified. Other epochs,
  the Opal-decode path and the old 1/30 nm exports were not checked. Every prior shift model and
  line-identification result on native data should be re-examined by the coordinator.
- Line identification uses predicted strengths at 9000 K with per-element normalisation; misidentified
  windows are limited by the 0.35 px residual cut but not excluded. tau_rel is a proxy (no column
  densities).
- chi2/nu of held-out fits is 4-10: the noise model (static pixel pattern + shot noise) underestimates
  the effective noise or the shapes vary line to line; formal centre and area errors are understated.
  The static detector pattern was not subtracted.
- UV shape: fold FWHM 1.68-1.87 px and within-segment variation 1.62-2.10 px; the reference is 74 %
  V II. NIR shape rests on 70 windows (17 lines, 10 runs). Sub-band interpolation of B-spline
  coefficients does not re-impose the centroid convention exactly.
- The dither test understates absolute bias (it sees only bias variation within each line's drift range)
  and needs >= 5 recurring lines per run; UV coordinator peaks could not be tested.
- The common-curve accuracy test uses the median of five estimators, three of which are classic; the
  robustness check (section 3c) covers this for VIS, not UV.
- Ar I tests: 9 runs with >= 4 other identified Ar I lines; the 772.40 target is a blend whose centroid
  depends on T (+-~5 pm).
- Multiplet ratios assume a flat spectral response across each pair; V 417.94/415.97 at 20-50 us relies
  on SNR ~2-4 lines.
- Only the 86 local runs were used (moissanite holds the same runs plus 2); more distinct lines (other
  matrices) would tighten every test, the UV and NIR shapes in particular.
