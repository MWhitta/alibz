# Empirical LSF (ePSF) on the axis-corrected Z300 native data — 2026-09-24

Supersedes cal/reports/2026-09-23-empirical-lsf.md (method, model and code unchanged except as noted;
read that report for the method, sections 1.1-1.4). Judgment agent (Opus 5.5); Claude subscription, no
provider switch. No commit, no hardware.

## 0. Data and what changed
- Corrected tree (pantheum PIXEL_OFFSET = -18, re-derived 2026-09-24):
  scratchpad/corrected/acq/<run>/shots/*.csv, 5,848 samples from 187.78 nm, ledger.json with 113 runs.
  Used: 26 Fe runs (20 labelled Fe in the ledger + 6 unlabelled early runs that are in the 2026-09-22
  Fe ledger) and 87 V runs (60 V run2 + 27 V run1, which were not available before). All 26 Fe runs were
  already in the 2026-09-23 build; the extra data are V run1.
- `epsf_common.DATASETS` presets: `native-20260923` (old, uncorrected) and `corrected-20260924`
  (default). With the corrected preset the pattern offset is ONLY A CHECK: registration starts from
  0 px and is refined to a per-run linear du(u) from the identified lines (the genuine sub-pixel
  calibration shift). The old workaround (starting from the pattern offset) is kept for the old preset.
- Estimators added for the owner's question: vendor cubic spline emulated as a not-a-knot cubic
  spline through the native samples (+-8 px), read two ways: `vr` = maximum of the continuous spline,
  `vp` = 3-point parabola on the spline resampled to the vendor 1/30 nm grid. Also kept: local 9-point
  spline root (coordinator), native 3-point parabola, free-width pixel-integrated Gaussian, ePSF
  segment-average ("ePSF seg") and ePSF interpolated in wavelength from 3 sub-bands ("ePSF lambda").
- Smoothness re-chosen by 5-fold held-out chi2 on the corrected data: UV 1.0, VIS 0.3, NIR 1.0.

### Registration check (pattern offset, which found -18.0..-18.65 px before; now per segment)
| seg | pattern offset px, Fe runs median [range] | V runs median [range] | refined du at u=1000, all runs median [range] | manifest axis_check px |
|--|--|--|--|--|
| UV | -0.38 [-0.60, 0.00] | -0.55 [-0.65, -0.35] | -0.46 [-0.70, 0.00] | -0.5 / -0.25 / 0 / +0.25 |
| VIS | -0.10 [-0.20, 0.00] | -0.25 [-0.45, -0.15] | -0.31 [-0.48, +0.12] | same set |
| NIR | -0.30 [-0.65, 0.00] | -0.45 [-0.60, -0.25] | -0.22 [-0.48, 0.03] | same set |
The offset is ~0: what remains (-0.1..-0.55 px) is the sub-pixel calibration shift the coordinator
also found (axis_check ~ -0.25 px). 9,328 identified windows (6,802 before); identification residual
MAD 0.22 / 0.14 / 0.13 px (UV/VIS/NIR, Gaussian centroid).

## 1. Shapes (cal/data/epsf-z300-20260924.json; cal/figures/epsf-20260924/epsf_shapes.png, subband_fwhm.png)
| seg | reference windows / db lines / runs | species | FWHM px (jackknife SD) | FWHM at median pitch |
|--|--|--|--|--|
| UV | 901 / 31 / 95 | V II 745, Fe II 108, Fe I 48 | 1.745 (0.044) | 150 pm |
| VIS | 844 / 44 / 89 | Fe I 349, V I 318, V II 175 | 1.263 (0.041) | 174 pm |
| NIR | 268 / 21 / 41 | Ar I 168, V I 100 | 1.670 (0.052) | 317 pm |

Sub-bands (FWHM px [fold range], pm): UV 242-274 nm 1.62 [1.43-1.65] 147; 275-314 1.61 [1.58-1.83] 140;
324-362 2.09 [2.00-2.16] 168. VIS 373-404 1.25 [1.20-1.30] 174; 406-441 1.27 [1.25-1.30] 173; 442-612
1.43 [1.39-1.48] 174. NIR 622-645 2.14 [1.42-2.14] 424 (UNSTABLE: 7 V I lines, folds disagree);
653-750 1.54 [1.52-1.62] 287; 752-852 1.55 [1.53-1.57] 277. VIS is now constant in nm to 1 %; UV and
VIS widths in px grow toward each segment's red end. The NIR first band should not be used.

## 2. Acceptance tests (a)-(e) (cal/data/epsf-validation-20260924.json)
How to read (a) and (b): the coordinator-style peaks are only 8 / 17 / 21 distinct lines. A naive
flat-histogram test therefore rejects every estimator. "Cluster-aware p" draws one uniform phase per
line. The dither test measures phase bias left after per-run registration; it understates absolute bias.

### 2a. Coordinator-style peak set (same selection on the corrected files)
| seg | windows / distinct lines | estimator | 10-bin hist | chi2 p | KS p | cluster-aware p | dither ptp px [68%] (lines x runs) |
|--|--|--|--|--|--|--|--|
| UV | 186 / 8 | vendor spline root | 32 25 5 0 4 1 72 12 9 26 | 6e-45 | 1e-10 | 0.47 | too few recurring lines |
| UV | 186 / 8 | vendor spline 1/30 parabola | 37 25 5 1 3 1 65 18 7 24 | 2e-38 | 4e-09 | 0.63 | too few recurring lines |
| UV | 186 / 8 | local spline root | 32 25 5 0 4 1 72 12 9 26 | 6e-45 | 1e-10 | 0.46 | too few recurring lines |
| UV | 186 / 8 | native parabola | 18 32 20 3 3 0 58 27 7 18 | 1e-27 | 2e-07 | 0.84 | too few recurring lines |
| UV | 186 / 8 | pix-Gauss | 10 45 34 3 0 1 4 60 25 4 | 2e-42 | 2e-06 | 0.44 | too few recurring lines |
| UV | 186 / 8 | ePSF seg | 3 37 45 7 0 0 1 36 44 13 | 2e-34 | 3e-09 | 0.91 | too few recurring lines |
| UV | 186 / 8 | ePSF λ | 7 17 49 24 0 0 1 46 34 8 | 3e-32 | 3e-06 | 0.94 | too few recurring lines |
| VIS | 227 / 17 | vendor spline root | 1 3 7 18 51 69 41 24 13 0 | 1e-42 | 5e-17 | 0.08 | 0.009 [0.009, 0.027] (6 x 6) |
| VIS | 227 / 17 | vendor spline 1/30 parabola | 1 3 8 18 51 64 43 26 13 0 | 4e-39 | 5e-16 | 0.11 | 0.015 [0.013, 0.018] (6 x 6) |
| VIS | 227 / 17 | local spline root | 1 3 6 19 51 69 41 24 13 0 | 8e-43 | 4e-17 | 0.08 | 0.009 [0.009, 0.028] (6 x 6) |
| VIS | 227 / 17 | native parabola | 1 4 7 32 41 58 41 27 16 0 | 5e-30 | 2e-14 | 0.23 | 0.014 [0.006, 0.018] (6 x 6) |
| VIS | 227 / 17 | pix-Gauss | 9 26 25 7 18 3 7 41 46 45 | 2e-19 | 2e-17 | 0.35 | 0.027 [0.009, 0.055] (6 x 6) |
| VIS | 227 / 17 | ePSF seg | 3 22 34 8 22 23 24 34 35 22 | 1e-06 | 0.0003 | 0.93 | 0.012 [0.007, 0.018] (6 x 6) |
| VIS | 227 / 17 | ePSF λ | 2 15 34 23 14 26 17 20 49 27 | 2e-10 | 0.0002 | 0.77 | 0.009 [0.005, 0.017] (6 x 6) |
| NIR | 473 / 21 | vendor spline root | 26 14 12 28 63 122 60 51 58 39 | 9e-38 | 3e-26 | 0.20 | 0.049 [0.042, 0.058] (17 x 51) |
| NIR | 473 / 21 | vendor spline 1/30 parabola | 27 14 14 27 61 120 63 46 61 40 | 3e-36 | 1e-24 | 0.22 | 0.046 [0.041, 0.053] (17 x 51) |
| NIR | 473 / 21 | local spline root | 27 14 12 28 64 121 61 51 58 37 | 2e-37 | 5e-26 | 0.21 | 0.049 [0.043, 0.058] (17 x 51) |
| NIR | 473 / 21 | native parabola | 23 20 15 27 61 91 86 49 57 44 | 4e-25 | 3e-23 | 0.39 | 0.042 [0.041, 0.055] (17 x 51) |
| NIR | 473 / 21 | pix-Gauss | 51 32 23 26 43 47 60 65 49 77 | 6e-09 | 2e-08 | 0.78 | 0.012 [0.012, 0.019] (17 x 51) |
| NIR | 473 / 21 | ePSF seg | 66 59 22 27 30 58 46 70 36 59 | 4e-09 | 0.008 | 0.85 | 0.016 [0.015, 0.025] (17 x 51) |
| NIR | 473 / 21 | ePSF λ | 69 40 30 30 32 55 47 68 34 68 | 5e-08 | 0.001 | 0.87 | 0.013 [0.013, 0.022] (17 x 51) |

### 2b. Cross-validation: reference lines fitted with shapes built without their db line
| seg | n ref | held-out chi2/nu median | fold FWHM px | held-out phase chi2 p / cluster p | dither ptp px [68%]: vendor spline root / vendor spline 1/30 parabola / local spline root / native parabola / pix-Gauss / ePSF seg / ePSF λ |
|--|--|--|--|--|--|
| UV | 898 | 6.52 | 1.76 1.75 1.74 1.76 1.80 | 1e-64 / 0.11 (31 lines) | 0.032 [0.037,0.090] / 0.031 [0.027,0.080] / 0.030 [0.037,0.094] / 0.029 [0.031,0.086] / 0.015 [0.016,0.039] / 0.011 [0.013,0.043] / 0.022 [0.019,0.047] |
| VIS | 842 | 9.37 | 1.28 1.27 1.25 1.23 1.28 | 2e-16 / 0.52 (44 lines) | 0.052 [0.054,0.099] / 0.069 [0.062,0.099] / 0.052 [0.051,0.097] / 0.054 [0.038,0.082] / 0.018 [0.018,0.033] / 0.009 [0.009,0.020] / 0.018 [0.017,0.031] |
| NIR | 268 | 5.47 | 1.61 1.67 1.67 1.67 1.69 | 0.0005 / 0.64 (20 lines) | 0.127 [0.102,0.154] / 0.121 [0.099,0.150] / 0.130 [0.102,0.155] / 0.061 [0.032,0.084] / 0.021 [0.024,0.047] / 0.042 [0.035,0.059] / 0.015 [0.020,0.039] |

### 2c/2e and areas: see section 3 (Fe accuracy) and below.

**(d) V I 417.94/415.97** (same upper level; thin ratio gA/lambda = 2.047; 76 V runs; forced ePSF areas
at registered positions): 2.00 [1.96, 2.10], robust SD of ln R 0.11, vs free pixel-Gaussian 1.17
(0.21), local-max height 5.0 (0.29). By delay (pooled): 1 us 2.16 +- 0.32, 5 us 2.11 +- 0.24, 10 us 1.99 +- 0.21,
20 us 1.94 +- 0.19, 50 us 1.61 +- 0.19. Consistent with optically thin up to 20 us. Same-upper-level Fe
multiplets: no consistent gain over the Gaussian (details in the json `d_multiplets`).

### 2e. Fit residual vs sub-pixel phase (held-out reference lines)
| seg | model | RMS of phase-bin medians | max abs | phase-dependent part | chi2/nu by phase bin (ePSF) |
|--|--|--|--|--|--|
| UV | ePSF λ | 0.0087 | 0.0261 | 0.0069 | -0.5:3.4 -0.3:6.1 -0.1:8.6 +0.1:4.6 +0.3:13.7 |
| UV | pix-Gauss free | 0.0269 | 0.0783 | 0.0191 |  |
| VIS | ePSF λ | 0.0111 | 0.0336 | 0.0085 | -0.5:4.3 -0.3:12.0 -0.1:9.8 +0.1:8.6 +0.3:9.9 |
| VIS | pix-Gauss free | 0.0184 | 0.0609 | 0.0143 |  |
| NIR | ePSF λ | 0.0205 | 0.1200 | 0.0158 | -0.5:5.2 -0.3:53.4 -0.1:10.1 +0.1:2.5 +0.3:3.1 |
| NIR | pix-Gauss free | 0.0244 | 0.1489 | 0.0223 |  |

### Verdicts
| test | UV | VIS | NIR |
|--|--|--|--|
| (a) flat phase histogram, as specified | FAIL for all 7 estimators (8 lines: no power); cluster-aware p: ePSF 0.91-0.94, spline 0.46-0.63 | FAIL for all (naive); cluster-aware ePSF 0.77-0.93 vs spline 0.08-0.11; dither 0.009 ePSF lambda vs 0.009-0.015 spline (6 lines x 6 runs) | FAIL for all (naive); cluster-aware ePSF 0.85-0.87 vs spline 0.20-0.22; dither 0.013 ePSF lambda vs 0.046-0.049 spline, 0.012 Gaussian — PASS vs spline |
| (b) cross-validation | PASS on phases (cluster p 0.11; was 0.01); dither 0.022 vs Gaussian 0.015, spline 0.031: not better than the Gaussian | PASS: cluster p 0.52; dither ePSF seg 0.009, lambda 0.018, Gaussian 0.018, spline 0.052-0.069 | PASS: cluster p 0.64; dither 0.015 vs Gaussian 0.021, spline 0.12-0.13 |
| (c) Fe accuracy | FAIL: spline/parabola 4.7-4.8 pm beat ePSF lambda 7.6 (paired -4.0 [-6.6, -1.4] pm) | PASS: ePSF seg 5.3, lambda 6.8 vs spline 11.6-12.3, native parabola 11.2, Gaussian 10.2 | no Fe lines; Ar I: PASS (696.54: 4.8 vs 14-15 spline; 801.48: 5-6 vs 27; 772.40: 2.5-2.7 vs 6.5) |
| (d) V 417.94/415.97 | - | PASS: 2.00 [1.96, 2.10] vs 2.05 | - |
| (e) residual vs phase | PASS 0.007 vs 0.019 | PASS 0.0085 vs 0.014 | PASS (marginal) 0.016 vs 0.022; one phase bin chi2/nu 53 |

UV and NIR, which failed or were inconclusive on 2026-09-23: NIR now passes (b), (e) and the Ar I
accuracy test with 29 runs (was 9), and beats the splines on the dither test. UV passes the phase tests
(b, e) but still loses on Fe accuracy (section 3).

## 3. The owner's question: which peak-location method, on correctly identified thin Fe lines?
Lines: identified Fe I/II in Fe runs; tau_rel < 0.3; lower level > 0.12 eV; identification residual
< 0.35 px; isolated. Per run and segment, ONE quadratic in pixels is fitted to the median of all seven
estimators, with a 3.5-MAD clip mask shared by all. |obs - db - curve| is reported in pm. Paired
differences are per line, with 2,000-sample bootstrap 95 % CIs.

### 3a. Fe accuracy
| seg | lines (kept) / runs / distinct | estimator | median abs [95% CI] | p90 abs | frac > 1 px | RMS kept [95% CI] |
|--|--|--|--|--|--|--|
| UV | 118 (117) / 12 / 15 | vendor spline root | 4.8 [3.5, 6.4] | 11.8 | 0.000 | 7.7 [6.6, 8.7] |
| UV | 118 (117) / 12 / 15 | vendor spline 1/30 parabola | 4.8 [3.6, 6.2] | 10.2 | 0.000 | 6.9 [6.0, 7.8] |
| UV | 118 (117) / 12 / 15 | local spline root | 4.8 [3.3, 6.4] | 12.0 | 0.000 | 7.6 [6.5, 8.7] |
| UV | 118 (117) / 12 / 15 | native parabola | 4.7 [3.7, 6.5] | 14.5 | 0.000 | 7.7 [6.7, 8.6] |
| UV | 118 (117) / 12 / 15 | pix-Gauss | 8.1 [7.1, 8.7] | 14.3 | 0.008 | 12.5 [8.6, 17.6] |
| UV | 118 (117) / 12 / 15 | ePSF seg | 10.1 [8.6, 11.6] | 17.7 | 0.000 | 12.4 [11.0, 13.7] |
| UV | 118 (117) / 12 / 15 | ePSF λ | 7.6 [6.6, 9.3] | 19.2 | 0.000 | 11.1 [9.8, 12.4] |
| UV | paired | abs(vendor spline root) - abs(ePSF λ) | -4.03 [-6.59, -1.36] | ePSF better on 35 % | | |
| UV | paired | abs(vendor spline 1/30 parabola) - abs(ePSF λ) | -4.28 [-6.74, -1.91] | ePSF better on 32 % | | |
| UV | paired | abs(local spline root) - abs(ePSF λ) | -4.14 [-6.68, -1.34] | ePSF better on 35 % | | |
| UV | paired | abs(native parabola) - abs(ePSF λ) | -4.49 [-6.08, -2.32] | ePSF better on 35 % | | |
| UV | paired | abs(pix-Gauss) - abs(ePSF λ) | -0.35 [-2.32, +1.50] | ePSF better on 46 % | | |
| UV | paired | abs(ePSF seg) - abs(ePSF λ) | +2.70 [+1.85, +3.95] | ePSF better on 65 % | | |
| VIS | 330 (328) / 24 / 24 | vendor spline root | 12.2 [11.2, 13.3] | 21.7 | 0.000 | 14.2 [13.5, 15.0] |
| VIS | 330 (328) / 24 / 24 | vendor spline 1/30 parabola | 11.6 [10.7, 12.7] | 21.3 | 0.000 | 13.7 [13.0, 14.5] |
| VIS | 330 (328) / 24 / 24 | local spline root | 12.3 [11.3, 13.3] | 21.8 | 0.000 | 14.3 [13.6, 15.1] |
| VIS | 330 (328) / 24 / 24 | native parabola | 11.2 [10.3, 12.4] | 21.3 | 0.000 | 14.0 [13.2, 14.8] |
| VIS | 330 (328) / 24 / 24 | pix-Gauss | 10.2 [8.2, 11.6] | 24.5 | 0.000 | 15.4 [13.9, 16.8] |
| VIS | 330 (328) / 24 / 24 | ePSF seg | 5.3 [4.8, 5.9] | 16.1 | 0.000 | 10.1 [8.9, 11.4] |
| VIS | 330 (328) / 24 / 24 | ePSF λ | 6.8 [5.9, 7.9] | 18.6 | 0.000 | 11.6 [10.4, 12.8] |
| VIS | paired | abs(vendor spline root) - abs(ePSF λ) | +3.40 [+2.32, +4.53] | ePSF better on 68 % | | |
| VIS | paired | abs(vendor spline 1/30 parabola) - abs(ePSF λ) | +2.94 [+1.88, +3.84] | ePSF better on 67 % | | |
| VIS | paired | abs(local spline root) - abs(ePSF λ) | +3.45 [+2.42, +4.61] | ePSF better on 70 % | | |
| VIS | paired | abs(native parabola) - abs(ePSF λ) | +3.90 [+2.72, +5.14] | ePSF better on 70 % | | |
| VIS | paired | abs(pix-Gauss) - abs(ePSF λ) | +2.82 [+1.65, +3.87] | ePSF better on 62 % | | |
| VIS | paired | abs(ePSF seg) - abs(ePSF λ) | -0.59 [-1.50, +0.07] | ePSF better on 45 % | | |

### 3b. Vendor spline paired against each other method (pm; > 0 = the other method is more accurate)
| UV | vendor spline root | vendor spline 1/30 parabola +0.54 [+0.33,+0.68] | local spline root +0.02 [+0.00,+0.04] | native parabola +0.10 [-0.47,+0.57] | pix-Gauss -2.51 [-4.46,-1.00] | ePSF seg -4.72 [-6.82,-2.59] | ePSF λ -4.03 [-6.59,-1.36] |
| UV | vendor spline 1/30 parabola | vendor spline root -0.54 [-0.68,-0.33] | local spline root -0.48 [-0.64,-0.27] | native parabola -0.18 [-0.42,-0.07] | pix-Gauss -2.84 [-4.35,-1.98] | ePSF seg -5.46 [-7.02,-3.02] | ePSF λ -4.28 [-6.74,-1.91] |
| VIS | vendor spline root | vendor spline 1/30 parabola +0.56 [+0.51,+0.60] | local spline root -0.11 [-0.14,-0.10] | native parabola +0.77 [+0.40,+1.22] | pix-Gauss +0.31 [-0.83,+3.03] | ePSF seg +6.19 [+4.90,+7.15] | ePSF λ +3.40 [+2.32,+4.53] |
| VIS | vendor spline 1/30 parabola | vendor spline root -0.56 [-0.60,-0.51] | local spline root -0.67 [-0.73,-0.63] | native parabola +0.12 [-0.11,+0.38] | pix-Gauss +0.08 [-1.35,+2.46] | ePSF seg +5.55 [+4.09,+6.69] | ePSF λ +2.94 [+1.88,+3.84] |


### 3c. Robustness: median |r| (pm) when the common curve is fitted to ONE estimator (rows) or the mean
| seg | curve from | vendor root | vendor 1/30 parabola | local spline | native parabola | pix-Gauss | ePSF seg | ePSF lambda |
|--|--|--|--|--|--|--|--|--|
| UV | vr | 5.2 | 4.9 | 5.2 | 4.9 | 8.8 | 11.3 | 8.3 |
| UV | vp | 5.1 | 4.9 | 5.1 | 4.8 | 8.0 | 10.6 | 8.3 |
| UV | sp | 5.1 | 4.9 | 5.1 | 4.8 | 8.8 | 11.1 | 8.4 |
| UV | pa | 4.7 | 4.7 | 4.6 | 4.6 | 7.8 | 9.9 | 7.8 |
| UV | p8 | 6.6 | 6.0 | 6.2 | 5.5 | 6.2 | 9.2 | 6.8 |
| UV | es | 9.6 | 9.4 | 9.4 | 9.1 | 4.9 | 4.1 | 7.2 |
| UV | ep | 7.9 | 7.2 | 7.8 | 7.1 | 5.4 | 4.8 | 4.6 |
| UV | mean | 4.6 | 4.4 | 4.6 | 4.6 | 7.2 | 7.8 | 6.9 |
| VIS | vr | 12.2 | 11.7 | 12.2 | 11.2 | 10.5 | 5.7 | 7.2 |
| VIS | vp | 12.2 | 11.8 | 12.3 | 11.3 | 10.5 | 5.4 | 7.0 |
| VIS | sp | 12.2 | 11.6 | 12.4 | 11.1 | 10.5 | 5.8 | 7.3 |
| VIS | pa | 12.2 | 11.6 | 12.4 | 11.3 | 9.7 | 5.5 | 6.8 |
| VIS | p8 | 13.1 | 12.6 | 13.3 | 12.3 | 7.8 | 5.0 | 5.9 |
| VIS | es | 13.6 | 12.9 | 13.7 | 12.9 | 9.1 | 4.3 | 5.6 |
| VIS | ep | 13.6 | 13.1 | 13.8 | 13.1 | 8.9 | 6.7 | 5.1 |
| VIS | mean | 12.7 | 12.2 | 12.8 | 11.5 | 9.4 | 4.8 | 5.7 |

### 3d. Same test on V lines (secondary; V db wavelengths are less certain) — median |r| pm [95 % CI]
| UV | 878 / 72 / 16 | vendor spline root 7.3 [6.9,7.8] | vendor spline 1/30 parabola 6.4 [6.0,7.1] | local spline root 7.3 [6.8,7.8] | native parabola 7.3 [6.8,7.8] | pix-Gauss 6.5 [6.0,7.1] | ePSF seg 12.8 [11.9,13.3] | ePSF λ 9.1 [8.1,9.9] |
| VIS | 481 / 47 / 19 | vendor spline root 10.3 [8.9,11.8] | vendor spline 1/30 parabola 9.3 [8.1,10.9] | local spline root 10.4 [8.9,11.9] | native parabola 8.5 [7.4,10.2] | pix-Gauss 15.0 [13.5,15.9] | ePSF seg 9.1 [8.5,10.4] | ePSF λ 8.9 [8.2,9.9] |
| NIR | 29 / 4 / 9 | vendor spline root 8.8 [6.5,21.0] | vendor spline 1/30 parabola 8.6 [6.2,20.9] | local spline root 9.1 [6.7,21.0] | native parabola 13.6 [7.3,22.2] | pix-Gauss 33.0 [16.1,42.5] | ePSF seg 29.3 [22.0,47.0] | ePSF λ 41.5 [23.9,59.9] |

### Answer (per segment)
- **VIS: the ePSF wins clearly and robustly.** ePSF seg 5.3 pm, ePSF lambda 6.8, vs vendor-spline root 12.2,
  vendor-spline 1/30 parabola 11.6, native parabola 11.2 and Gaussian pixel fit 10.2. The paired CIs
  exclude 0, and the ePSF wins whichever estimator the common curve is fitted to. The vendor spline
  is ~2x worse than the ePSF. It is no better than the native parabola (vendor root vs native parabola
  +0.77 [+0.40, +1.22] pm in the parabola's favour). On V lines, the ePSF and native parabola tie (8.5-9.1)
  and the Gaussian is worst (15.0).
- **UV: THE SPLINE WINS.** It is tied with the native parabola. Median |r|: vendor root 4.8, vendor 1/30
  parabola 4.8, native parabola 4.7, vs Gaussian 8.1 and ePSF lambda 7.6. ePSF lambda minus vendor
  spline is +4.0 [+1.4, +6.6] pm in the spline's favour. This holds on every common curve except the
  ePSF's and Gaussian's own, and on V lines (vendor parabola 6.4, Gaussian 6.5, ePSF 9.1). Within the
  spline family the 1/30 nm parabola beats the continuous root by 0.54 [0.33, 0.68] pm. Likely reason:
  UV lines are the best sampled (~1.75 px FWHM), so the classic estimators' phase bias is small, while
  the UV ePSF is uncertain (width 1.6-2.1 px across the segment, reference 83 % V II). Only 12 Fe runs
  and 15 distinct Fe lines enter.
- **NIR: no Fe lines.** On Ar I (29 runs, leave-one-out Ar curves) the ePSF wins or ties:
  - 696.54: 4.8 pm vs 14-15 spline;
  - 801.48: 5-6 vs 27;
  - 772.40 blend: 2.5-2.7 vs 6.5 spline, tied with the Gaussian 2.9.
  On 29 V I NIR windows (4 runs) the spline is better (8.8 vs 41.5). That sample is tiny and uses the
  unstable 622-645 nm band. Weight it low.
- Recommendation: ePSF for VIS and NIR; native parabola (or the vendor spline, which is equivalent) for
  UV until a better UV reference exists. The coordinator's earlier "spline pixel-locks up to +-0.25 px in
  VIS" is confirmed in accuracy terms: 12 pm vs 5 pm.

## 4. Line classes (excess Lorentzian HWHM gamma, px; paired by run)
| seg | comparison | runs | diff | SD | t | p |
|--|--|--|--|--|--|--|
| UV | thick - thin | 92 | +0.298 | 0.167 | 17.2 | 3e-30 |
| UV | mid - thin | 92 | +0.102 | 0.078 | 12.6 | 1e-21 |
| UV | low-lying - high E_i | 91 | +0.117 | 0.157 | 7.1 | 3e-10 |
| UV | resonance - high E_i | 91 | +0.071 | 0.143 | 4.7 | 8e-06 |
| UV | ion - neutral | 98 | +0.018 | 0.262 | 0.7 | 0.49 |
| VIS | thick - thin | 86 | +0.231 | 0.118 | 18.2 | 5e-31 |
| VIS | low-lying - high E_i | 98 | +0.119 | 0.098 | 12.0 | 6e-21 |
| VIS | resonance - high E_i | 94 | +0.060 | 0.072 | 8.1 | 2e-12 |
| VIS | ion - neutral | 76 | -0.134 | 0.072 | -16.2 | 4e-26 |
| NIR | thick - thin | 40 | +0.076 | 0.123 | 3.9 | 4e-04 |
| NIR | non-reference - reference | 41 | +0.088 | 0.059 | 9.5 | 9e-12 |
Delay: VIS V I gamma 0.113 (1 us) -> 0.082 (5) -> 0.054 (10) -> 0.051 (20) -> 0.000 (50), rho -0.23,
p 1e-9; UV V II 0.111 -> 0.094 -> 0.106 -> 0.034 -> 0.046, p 4e-6; V II VIS flat at 0. The conclusions of
2026-09-23 section 4 hold. Optical depth and low-lying lower levels broaden lines, and this is real
against run-to-run scatter. Early delays add Stark broadening for V I and V II. Ion vs neutral cannot
be separated from optical depth: UV shows no difference, and VIS ions are narrower. The slab
(self-absorption) kernel beats the Lorentzian in <= 12 % of windows in any class. Thick-line stacks
show broadening with a red-side excess, and no reversal.

## 5. Files
- New: cal/reports/2026-09-24-empirical-lsf.md (this), cal/data/epsf-z300-20260924.json,
  cal/data/epsf-validation-20260924.json, cal/figures/epsf-20260924/ (6 figures).
- Edited (own files): cal/scripts/epsf_common.py (DATASETS presets, ledger loader, vendor-spline
  estimators), cal/scripts/build_epsf.py (`--dataset`, registration from 0 with pattern check,
  per-dataset smoothness), cal/scripts/validate_epsf.py (`--dataset`, 7 estimators, vendor-spline paired
  tables, curve robustness, V accuracy, registration check). alibz/utils/epsf.py unchanged since 09-23.
- tests/test_epsf.py re-run: 17 passed (95 s).
- Regenerate: `.venv/bin/python cal/scripts/build_epsf.py --dataset corrected-20260924 --workdir DIR`
  (catalog + build + write; add `--stage lamcv` first to redo the smoothness CV), then
  `.venv/bin/python cal/scripts/validate_epsf.py --dataset corrected-20260924 --workdir DIR`.
- cal/reports/2026-09-23-empirical-lsf.md carries a "Superseded" line.

## 6. Unverified / limits
- "Vendor cubic spline" is an EMULATION (not-a-knot spline through native samples, 1/30 nm grid). The
  vendor's exact spline (boundary conditions, knots, any smoothing) was not reproduced from vendor
  exports here.
- UV Fe accuracy rests on 12 runs and 15 distinct lines. The UV ePSF shape varies strongly within the
  segment. A per-class UV reference (Fe II vs V II) was not tried.
- NIR band 622-645 nm is unstable (folds 1.42-2.14 px). It affects ePSF lambda at the blue end of NIR
  and the tiny NIR V-line test.
- The common curve is the median of seven estimators, four of which are spline/parabola variants.
  Section 3c shows the VIS conclusion does not depend on this. In UV each estimator family wins on its
  own curve, but the spline/parabola lead also holds on the mean curve (4.4-4.6 vs 6.9 pm).
- Held-out chi2/nu is 5-9, so formal errors are understated. The static detector pattern is not
  subtracted. Multiplet ratios assume a flat response. The 772.40 Ar target is a T-dependent blend.
