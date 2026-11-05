# Manuscript Workflow section + parameter appendix (2026-11-03)

Files: manuscript/sections/04-workflow.tex (~4,100 prose words by texcount incl. ~250 table words; target was 3,000-4,000, slightly over),
manuscript/sections/A2-pipeline-parameters.tex (4 booktabs tables, ~60 rows; columns Stage | Module | Parameter | Default | Meaning).
Build: `make` FAILS at the time of writing, NOT from these files: manuscript/sections/05-datasets.tex:20 uses `\SI{1/30}{\nano\metre}` (siunitx: Invalid number '1/30'). Verified by compiling a temporary main with only 04 and A2 included: compiles, 12 pages, no errors; the only warnings are unresolved cross-refs to excluded sections. (Temp files deleted.)
Provider: subscription, no switch. Tests not applicable (no code changed).

## Source map for quantitative claims (workflow)
| Claim | Source |
|---|---|
| 3 segments, junctions 365/620, pitches 0.096->0.079 / 0.142->0.113 / 0.198->0.156 nm, 7,914 samples | task brief; reports/2026-09-22-native-grid-processing.md (7,914; UV 2,027 / VIS 1,985 / NIR 1,836) |
| 2,066-sample dead channel at 960-961 nm; 5,848 usable samples from 187.78 nm; 12 nm NIR gap | reports/2026-09-23-native-vs-spline-ab.md sec 4; cal/reports/2026-09-24-empirical-lsf.md sec 0; reports/2026-09-24-wavelength-registration.md (5,848, 187.8-950.8) |
| noise lag-1 autocorr native 0.12 vs linear 1/30 file 0.76; 1.3-2.2 samples/FWHM | reports/2026-09-23-native-vs-spline-ab.md sec 5, 6. NOTE: that "old" file was Pantheum LINEAR resampling, not the vendor cubic export; the text says so |
| -18 px offset; UV/VIS/NIR -1.6/-2.3/-3.3 nm; residual -0.25 px | docs/wavelength_registration.md "Why"; reports/2026-09-24-wavelength-registration.md |
| db vacuum vs air offset 0.11-0.24 nm | memory note sim-to-real-gaps-mw2112 (not re-verified in a repo file); fix doc: docs/fit_pipeline.md sec 6 (Database converts at load) |
| ~5x gain step at 620 nm, edges=(620.0,) only | alibz/pipeline.py comment block "detector segment throughput" |
| arPLS smooth_nm 1.9, max_iter 20, lam=(smooth_nm/pitch)^4, +54-count ledge | alibz/peaky_finder.py find_background; docs/fit_pipeline.md sec 1 |
| strict local max (4 of 5 neighbours), 3.0+max(n_sigma,0) sigma, lags 1/2/4/8, block 2048 | peaky_finder.py (filter_peaks, _noise_scale_local); docs/fit_pipeline.md sec 2 |
| fit_shoulders mean+2 std; fit_all +/-50; 100x median FWHM; MIN_HEIGHT_SIGMA 2.0 | peaky_finder.py (fit_shoulders, fit_all, fit_spectrum, class attr MIN_HEIGHT_SIGMA) |
| sigma=289 nm absorbed 89% area; 303 peaks | docs/fit_pipeline.md sec 4, header |
| 303 -> 288 -> 299 peaks; model A residuals 2-8x lower; emission/observed 1.24-7.0 over 15 lines; 383.0 nm 549 vs 76 | docs/fit_pipeline.md sec 5a, header |
| per-segment location: VIS ePSF 5.3 pm vs spline 11.6-12.3; UV spline 4.7-4.8 vs ePSF 7.6; Ar I 696.54 4.8 vs 14-15; 26 Fe + 87 V runs; NIR ePSF gated for seam bias | cal/reports/2026-09-24-empirical-lsf.md sec 0, verdict (c); alibz/utils/peak_location.py docstring |
| location recorded, inversion keeps blind Voigt centres | alibz/pipeline.py comment at peak_refinement |
| BIC margins 10/30, TAU_MAX 8, SA_ROBUST_F_SCALE 8, soft_l1, s2_action_max 50, Ei<=0.2 eV | alibz/refinement.py constants, _fit, classify_feature |
| Fe I 248.814/249.064 tau~2.7 fix (sentence later removed in trim) | docs/development_guide.md Component 2 |
| 3a/3b staging; merge zones before seeding; 21-93% erosion | docs/development_guide.md "Staged refinement" |
| non-destructive sa-tag: +19 sigma, 40 sigma, chi2 32821 -> 51701 -> 28262, residuals 61 -> 28 | docs/development_guide.md "Non-destructive self-absorption tags" |
| deblend SNR 4, BIC 6, max 12; SA recovery BIC 10, amp cap 5; 26/38 and 24/38 | alibz/profiles.py constants; dev guide Component 2 |
| registration: Ar I anchors list, +/-30 pm, 10-12 lines, >=3 line gate; golden lines +/-0.5 nm; Huber offset; slope diagnostic only | docs/wavelength_registration.md; alibz/wavelength_registration.py:863 docstring, pipeline.py n_lines>=3 gate |
| residual: -0.25 px (manifest) / -0.26 (ambient), 107 runs, -0.5..+0.25; Fe->V VIS 40 vs 597 pm; SD <=12 pm | reports/2026-09-24-wavelength-registration.md; docs/wavelength_registration.md |
| default mode ambient (NIR only); element registration recorded not applied; gas engine >=3 groups; 3 sigma flag | docs/wavelength_registration.md "Pipeline integration"; AnalysisConfig |
| per-segment shift gate 2x SE, MIN_SHIFT_MATCHES 5, MAD 26-71 pm, 25% phantom Ti | alibz/utils/wavelength.py:69,305; dev guide "Per-segment wavelength shift" |
| seeding gates (10% contamination, tol 0.06, min_ref 3, factor 5, kT 0.76 eV) | alibz/minor_lines.py:46,184-187; docs/fit_pipeline.md sec 6 |
| deepening: CONFIDENT_MIN_REFS 4, DEEPEN_BARS, 41 lines (23+18), 3 rounds, ~264 s, 4 sigma elsewhere | alibz/pipeline.py:170,178; dev guide "Iterative line deepening" |
| WNNLS opt-in, default off; Ca/Mg inversion | alibz/pipeline.py AnalysisConfig, cli.py --weighted-solve help |
| 15x7 grid, DEFAULT_SEARCH grid, NE prior (17.0,0.5), +/-1 sigma, synthetic Ca/Mg 0.52/0.48 at 8.9 kK vs 2-sigma box 8.2 kK | alibz/pipeline.py:206-217 comments; peaky_indexer_v3.py:2435 |
| STAGE_CONSISTENCY_WEIGHT 1.0 | peaky_indexer_v3.py:104 |
| collapse guard 0.9/0.2; Hg 1.000, Fe 0.991, r2 0.87; A/B outcomes | pipeline.py:164-165; dev guide Component 2 |
| detectability: z>=3, Bi z=2.2 vs >=4.6, Bi 0.94, Hg 0.27 z=0.5, silicate fractions | peaky_indexer_v3.py:49; dev guide "Non-destructive..." |
| contest grids, cover 0.5, build state (9000,17.0); Mn 53->0, Mg 8->31, Be/Fe 42% phantom | alibz/detections.py:36-64; dev guide "Spectral confounders" |
| SA recovery re-solve; Si +3.1 / K -3.5 pp; Al I 396.1, Mg I 383.8 tau~0.5 | dev guide Component 2 |
| 32 draws | alibz/detections.py DEFAULT_DRAWS / pipeline.py:DEFAULT_DRAWS |
| outputs, manifest, strict provenance | README "Directory Analysis"; cli.py --no-provenance/--strict-provenance |
| runtime 225 s -> 106 s (median 410-peak spectrum) | TASK BRIEF ONLY; no repo report found by grep. Unverified |

## Discrepancies found (flag for owner)
1. Brief says DEEPEN_BARS (3, 2.5, 2); code (pipeline.py:178) is (3.0, 2.0). Table uses code; text states the 3-round measurement was under an earlier schedule.
2. Brief calls the 1/30 nm grid a vendor cubic-spline export. The A/B report found the stored "old" Pantheum file was LINEAR resampling; text states both and attributes the 0.76 autocorrelation to the linear file.
3. Brief lists TAU_MAX "8 / 50"; 50 is the indexer doublet cap (docs/fit_pipeline.md sec 7), not in the table because the doublet channel is off by default and the 50 was not re-grepped.
4. docs/wavelength_registration.md says NIR registration needs >=3 lines; wavelength_registration.py `_DEFAULTS` has segment_min_lines 5. Table lists the 3-line golden-line minimum only.
5. Memory says vote/pipeline default weighted_solve False; confirmed False (AnalysisConfig, CLI). Theory section frames WNNLS as the model; workflow says it is opt-in.

## Appendix defaults grep-verified (module : constant = value)
peaky_finder: DEFAULT_SEGMENT_EDGES (365.0, 620.0), smooth_nm 1.9, max_iter 20, MIN_HEIGHT_SIGMA 2.0, block 2048. refinement: BIC_MARGIN 10, BIC_MARGIN_UNSUPPORTED 30, TAU_MAX 8, SA_ROBUST_F_SCALE 8, s2_action_max 50. profiles: DEBLEND_ACCEPT_SNR 4, DEBLEND_BIC_MARGIN 6, DEBLEND_MAX 12, SA_BIC_MARGIN 10, SA_AMPLIFICATION_CAP 5. minor_lines: DEFAULT_KT_EV 0.76, tol_nm 0.06, min_ref_lines 3, max_scale_spread 0.6, min_ref_snr 10, consistency_factor 5, bic_margin 2, recover snr_min 4 / bic 6 / max_new 40. pipeline: CONFIDENT_MIN_REFS 4, DEEPEN_BARS (3.0,2.0), SUPPORT_GA_FLOOR 1e6, SUPPORT_TOL_NM 0.06, ESTABLISHED_MIN_FRACTION 0.002, COLLAPSE 0.9/0.2, STAGE_FLAG_THRESHOLD 0.5, PLASMA_T_BOUNDS (4000,25000), PLASMA_LOG_NE_BOUNDS (14,19), NE_PRIOR_DEFAULT (17.0,0.5), NE_PRIOR_BOUND_SIGMAS 1.0, DEFAULT_SEARCH grid, DEFAULT_N_CALLS 40, DEFAULT_GP_SEED 42, POISSON_GAIN_COUNTS 1.0, DEFAULT_TIMEOUT_S 900, weighted_solve False, physical_triage off, DEFAULT_STIMULATED_EMISSION False, wavelength_registration ambient, subpixel gaussian, gas_wavelength_calibration apply. peaky_indexer_v3: ELEMENT_DETECT_Z_MIN 3.0, STAGE_CONSISTENCY_WEIGHT 1.0. detections: DETECT_Z 3, MARGINAL_Z 2, MIN_SUPPORT_FRACTION 0.3, RIVAL_COVER_FRACTION 0.5, RIVAL_MIN_RESPONSE 0.05, CONTEST_T_GRID, CONTEST_LOGNE_GRID, CONTEST_BUILD_STATE, DEFAULT_DRAWS 32. wavelength_registration: SEGMENT_EDGES, isolation_window_nm 0.30, kT_eV 8.617e-5*9000=0.776 eV. utils.wavelength: MIN_SHIFT_MATCHES 5, gate 2.0*se (line 305). Literals (marked "literal" in table): fit_all +/-50, 100x median FWHM, Ei<=0.2 eV, 2 SE.
Note: DEFAULT_DRAWS is defined in detections.py:38 and pipeline.py; the table cites detections.

## Citations
Used (all exist in bib/): baek2015arpls, olivero1977voigt, thompson1987voigt, anderson2000epsf, schwarz1978bic, huber1964robust, lawson1995nnls, jones1998ego, efron1979bootstrap, edlen1966refractive, ciddor1996refractive, gigosos2003halpha.
Missing: Eilers 2003 perfect smoother (`\todo{cite: Eilers 2003 perfect smoother}` in 04-workflow.tex; key eilers2003smoother is NOT in bib/). Soft-L1 robust loss is only cited loosely via huber1964robust ("cf.").
Cross-refs used: sec:theory, sec:datasets, eq:emissivity, eq:saha, eq:escape, eq:sa-voigt, eq:tie-cost, eq:wnnls, eq:sigma-eff, eq:detect, eq:contest. eq:doublet dropped in trimming. Added labels: sec:wf-instrument, -background, -fit, -refinement, -registration, -seeding, -solve, -outputs; tab:stages, tab:params-*.
