# Manuscript sections 05-datasets, 06-metrics, 07-results (2026-11-03)

Worker: Sonnet 5.5, Claude subscription, no provider switch. Files rewritten (the `\section`/`\label` lines kept):
`manuscript/sections/05-datasets.tex`, `06-metrics.tex`, `07-results.tex`. Every number carries a `% src:` LaTeX
comment on the same or the following line (comment placed after the sentence it supports). No bib files edited.

## Build and tests
- `cd manuscript && make` succeeds (main.pdf, 44 pages). Fixed: siunitx rejects `\SI{1/30}{nm}` (now `$1/30$\,\si{nm}`); unescaped underscores inside `\todo{}`.
- Remaining Overfull hboxes (largest ~55 pt, in the long `\todo` paragraph of 06 and some prose lines) are cosmetic; not fixed.
- No code changed, so no pytest run (repo test counts not affected). Word counts (comments stripped, markup and tables included): 05 ~1465, 06 ~2076, 07 ~2529 (07 is above the 1500-2200 brief range because of the failure-modes subsection and markup).

## Table of numbers -> source (section of the cited file)
### 05 datasets
| number | source |
|---|---|
| 5 suite-S cases and their compositions | bench/synth_cases.py CASES |
| NIST benchmark: 370-600 nm, 0.02 nm, 3 cases, 3-7 active elements, T 7000-13000 K, log ne 16-17.3, 92-element db | bench/benchmark.py (run_benchmark defaults, sample_composition, sample_theta, docstring) |
| MW2-112: 929 shots, IDs 989-1917, 23,431 pts, 180-961 nm, 1/30 nm, 190-910 nm active, 100 um step, ~50 um spot | docs/mw2_112/provenance.md "Raw observations"; analysis_plan.md sec. 2 and G1 |
| ID 1632 all-zero; breaks 4 and 176 min; 927 vendor summaries | docs/mw2_112/analysis_plan.md sec. 2 |
| JChristensen 38 spectra, 23,431 rows | docs/development_guide.md ("Measured on JChristensen (38 spectra)"); docs/transformer_plan.md l.142 |
| remote_samples 5 files, 23,431 rows each | data/remote_samples/*.csv (wc -l = 23432 incl. header) |
| 4,909 corpus spectra; 435 with strong continuum; 620 nm step median 3.9, IQR 3.0-5.2 | corrections/detector.json segment_response_fallback.provenance |
| 8,213 archive spectra; 97.9 % on 1/30 nm grid | reports/2026-09-20-alibz-review.md "Moissanite archive runs" |
| 1,022,092 (paper) vs 1,023,490 (pinned) whole-rock samples | docs/whole_rock_prior.md |
| Fe: 26 runs, 252 shots, 5,848-pt native grid, 187.8-950.7 nm, Aesar 99.98 % | cal/reports/2026-09-24-fe-calibration-plasma-analysis.md Data; cal/README.md |
| V run2: 59 batches, 5x4 grid, 3 replicates | cal/reports/2026-09-24-vanadium-timing-study.md header |
| 18-px offset, 1.6/2.3/3.3 nm too blue | reports/2026-09-23-native-axis-18px-offset.md |
| Z300-0915 identity | docs/z300_device_notes.md sec. 1 |
### 06 metrics
| number | source |
|---|---|
| r2 warn 0.50, fail 0.30; T bounds 4000-25000 K; top fraction 0.9; stage flag 0.5; ne prior 17.0+-0.5, 1 sigma; at-bound tol 0.01 dex | alibz/pipeline.py (FIT_R2_*, PLASMA_T_BOUNDS, COLLAPSE_TOP_FRACTION, STAGE_FLAG_THRESHOLD, NE_PRIOR_*, qc block) |
| collapse r2 0.87 at ~5200 K | docs/development_guide.md (JChristensen caveat) |
| BIC = C + k ln N, k = nnz+4; ambiguity 6.0 | alibz/peaky_indexer_v3.py _select_basin_bic, BIC_AMBIGUITY |
| s2 = max(1, min chi2/(n-k)); params 4/6/8; s2_action_max 50 | alibz/refinement.py |
| z >= 3 | alibz/peaky_indexer_v3.py ELEMENT_DETECT_Z_MIN |
| status thresholds (z>=3, 2<=z<3, z<2), 95th percentile, 32 draws | alibz/detections.py classify_detections, DEFAULT_DRAWS |
| benchmark det threshold 0.03; suite-S major floor 0.05 (missed if < 0.025) | bench/benchmark.py score; bench/synth_cases.py score_recovery |
| multiplet test R0 in [1.2,4], SNR>=12; Boltzmann ref Ei>=0.8 eV | cal/scripts/line_physics_analysis.py; cal/reports/2026-09-24-line-matching-sa-broadening.md 2a/2b |
| resolved Boltzmann fit: n>=8, span>=2.4 eV, r2>=0.5, sigT/T<0.5; 1.6 sigma clip | cal/scripts/fe_plasma_analysis.py fit_resolved; cal report sec. 4 note |
| gas cross-check 20 pm floor, 3 sigma flag; target 30 pm | DECISIONS.md 2026-09-23 (two argon engines); reports/2026-09-24-wavelength-registration.md; tests/test_wavelength_registration.py |
| peak-location metric: common curve, 3.5-MAD clip | cal/reports/2026-09-24-empirical-lsf.md sec. 3 |
| round-trip gates: 0.30 abs, spurious < 0.25, T in [0.55,1.45] x truth | tests/test_roundtrip_synthetic.py |
| peak-finder gates: area 2 %, centre 0.003 nm, sigma/gamma 0.002 nm | tests/test_peaky_finder_accuracy.py |
| Stark ne gate 0.3 dex | tests/test_stark_ne.py |
| pilot gate G5 | docs/mw2_112/analysis_plan.md G5 |
### 07 results
| number | source |
|---|---|
| baseline Ca 0.764/Mg 0.236, 7000 K, r2 0.509 | tests/test_roundtrip_synthetic.py docstring |
| Si missed, Be 0.23, ne railed 14.0 (quartz-carbonate) | bench/synth_cases.py docstring |
| flat misfit in ne (14-18), <1 %; tie-cost 360 vs 2.6e6; Table (7000->8908 etc., median rel err 0.33->0.17, 0.66->0.53, 0.80->0.18); 314 passed/3 skipped; real-sample T 4.6-6.1 -> 7.3-8.6 kK, tie cost 10-60x, scan9x9 r2 0.875->0.779; REE_01, argon_noAr (12191->8563 K) | reports/2026-09-21-stage-consistency-thermometer.md secs. 1,3,4 |
| Mn 53->0 %, Mg 8->31 %, Be 3.8 % / Fe 42 % phantom, resp. ratio ~12, Bi 0.94 | docs/development_guide.md (confounders; fit-fidelity review) |
| chi2/sigma2 32821 -> 51701 -> 28262; >3 sigma 61 -> 28; +19/40 sigma | docs/development_guide.md |
| merged-line RMS 227/59, 101/19, 141/22, 640/549 | docs/fit_pipeline.md |
| pilot 19 nonzero; 3 pass/4 warn/12 fail; 255 multiplets; Spearman 0.934, 0.883, 0.705, 0.657, 0.598, 0.030; ten supported elements | docs/mw2_112/analysis_plan.md secs. 9,10; provenance.md |
| stimulated emission A/B: 14/38, 3/38 control, median dr2 0 | docs/development_guide.md Appendix item 1 |
| native-grid recovery 99.1/96.6/97.3 %, 7,821 spectra, 7e-11 | reports/2026-09-20-alibz-review.md |
| offsets -18.5/-18.25/-18.5 px; -22 pm vs -154 pm; -0.25 px (-0.50..+0.25, n=107); 107 runs (20 Fe, 87 V); 1 pm/h; 17-32 pm; VIS slope ~0 | reports/2026-09-23-native-axis-18px-offset.md; cal Fe report; reports/2026-09-24-wavelength-registration.md |
| peak-location pm figures (5.3, 11.2-12.3, 4.7-4.8, 7.6, 4.8 vs 14-15; 330 and 118 lines) | cal/reports/2026-09-24-empirical-lsf.md; alibz/data/epsf-z300-20260924.json |
| 86/95 % Fe, 78/88 % V matching | cal/reports/2026-09-24-line-matching-sa-broadening.md 1.4 |
| Fe I VIS T 6873-13701 K, median 10805 K, 5/26; Fe II 0/26; 421 K; ratio 0.90-2.09 (1.49); old 8191-10451/9321/2 of 26; McWhirter 1e16; K 4.4x | cal/reports/2026-09-24-fe-calibration-plasma-analysis.md |
| period means 1.234/1.182/1.119/0.993; V II -0.0252+-0.0049 (5.1 sigma); 417/415 ~2.05 thin | cal/reports/2026-09-24-vanadium-timing-study.md sec. 8 |
| V pairs tau 3.5, 1.4, 1.4; 8/14; Fe 589.80/589.53 tau 8.5 | cal/reports/2026-09-24-line-matching-sa-broadening.md 2a, 2e |
| Hg 1.000/Fe 0.991/Ca 0.990 at ~5200 K | docs/development_guide.md; alibz/pipeline.py comment |
| 926 of 928 K pairs outside +-20 % | DECISIONS.md 2026-09-22 |
| axis check warns at >2 px, never rejects | reports/2026-09-24-pixel-offset-fix.md |

## `\todo{source:}` placeholders
1. 07 Engine benchmark: searched reports/, docs/, notebooks/, DECISIONS.md, bench/ for recorded engine head-to-head scores. None. `notebooks/track_ab_results.ipynb` does not exist. bench/benchmark.py `generate_case` needs the removed cflibs prototype, so the harness cannot run as written. Needs a re-run with the in-repo generator.
2. 06 Acceptance gates: no numeric relative-accuracy target ("<20 % on majors") and no CRM gates found in DECISIONS.md, docs/transformer_plan.md, docs/mw2_112/analysis_plan.md. The brief's 20 % figure was therefore not used.

## Items from the brief not supported by the repo (stated differently in text)
- "Li pegmatite": repo says finely stratified lithium mudrock (provenance.md).
- "suites S/M/R": only suite S exists in bench/synth_cases.py; M and R not found.
- "V doublet ratio near thin": only the 417.94/415.97 pair is near thin (tau 2.10 vs 2.05 thin); other V pairs show tau up to 3.5 (cal line-matching report). Text says so.
- "BHVO-2, AGV-2, GSP-2, K-feldspar, spodumene": grep found no repo file naming them (BHVO, AGV-2, spodumene), so text says no CRMs measured, nothing cited.
- Argon-vs-air: no measured pair; only synthetic `binary_argon_ambient`. Prerequisite noted in DECISIONS.md 2026-09-23.
- Z300 native grid point count differs between reports (7,914 in 2026-09-22 native-grid-processing before the offset fix; 5,848 in corrected 2026-09-24 reports). Only 5,848 used.
- Superseded (not used): 2026-09-22/23 cal reports and 2026-09-23 registration report; DECISIONS.md 2026-09-23 entries predate the axis fix (text says so). The DECISIONS "only 696.5/772.4/801.5 nm are argon" statement is not used as a result.

## Not verified
- Numbers were transcribed from reports; I did not re-run any analysis. Data behind runs/ (gitignored) and the Z300 shots on the lab server were not inspected.
- JChristensen has "MDD" drill-core names in scripts/fit_inspector.py and flagged.json; sample provenance beyond that is not documented in repo.
- Round-trip test tolerances in tests/ may have been tightened after the dates cited; I read the file as it stands.

## Citations
Used (exist in manuscript/bib): `gard2019wholerock` (local.bib), `schwarz1978bic` (ml.bib). Missing/not added: none requested via `\todo{cite:}`; candidate keys that do NOT exist in the bib: McWhirter criterion (no key), `kass1995bayesfactors`, `sciaps2021` (appears in listing but was not found by the key check), `aguilera2007sahaboltzmann`. Existing but unused: `anderson2000epsf`, `lawson1995nnls`, `ralchenko2020nist`, `cristoforetti2010lte`, `elsherbini2006halpha`.
