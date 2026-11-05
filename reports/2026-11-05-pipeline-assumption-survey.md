# Pipeline assumption survey — synthesis

Date: 2026-11-05. Trigger: owner, after the Fe hot-basin root cause
(reports/2026-11-04-fe-hot-basin-root-cause.md): "there are poor or at least
incongruent design choices carried through from a previous patchwork of
development attempts. Survey the whole pipeline for issues such as the
native-grid window that quietly inject bad assumptions into the model."

Companion work in flight: the native-grid gate fix + Fe panel re-run
(reports/2026-11-05-native-gate-fix.md). Survey parts:
reports/2026-11-05-survey-finder.md (signal chain),
reports/2026-11-05-survey-indexer-physics.md (inverse model + physics),
reports/2026-11-05-survey-pipeline-glue.md (orchestration, scoring, fixtures).
This file is the coordinator's verified synthesis; the parts hold the evidence.

## 1. Why the exemplar happened (history from git)

| period | commits | what the code was built against |
|---|---|---|
| 2025-06 → 2025-11 | ~120 | original maker/finder/fitter/indexer scripts; codex refactors of the finder; speed-ups to the peak fit (2025-10) — all on the vendor 0.0333 nm export grid |
| 2026-03 | 74 | "packaged" (2780eef, 2026-03-20): the `len(x_window) > 5` gate enters the packaged module unchanged from the scripts; new indexing pipeline, profile PCA, GPU |
| 2026-07 | 110 | finder repair, noise/background/response fixes, tracks A–C, pipeline review — tuned and verified on MW2-112 and the 2026-07 corpus, both exports |
| 2026-09 | 90 | native-grid recovery, Pantheum acquires native pixels, −18 px axis fix, per-segment pitch added to the width caps and peak location — but not to the seeding gate |
| 2026-11 | (uncommitted) | A/B waves 1–3 and manuscript figures scored on the fit-quality panel: synthetic + 5 exports + 26 Fe native means |

Reading: three sampling regimes (export spline grid, native pixel grid, synthetic
renders) now flow through code whose constants were fixed in the first one.
The gate is the instance that surfaced; the survey looks for the rest.

## 2. Classes of assumption hunted

A. sample-count / grid-regime constants (samples instead of nm or LSF units)
B. dataset-tuned constants applied universally without a guard
C. structural: hypotheses pruned before the data can vote; order dependence;
   discard-instead-of-flag; self-referential gates
D. physics / convention mismatches (partition extrapolation, Saha chain,
   point-sampled vs pixel-integrated, energy convention)
E. dual or drifted code paths and default sets
F. scoring / fixture provenance (what the benchmarks can and cannot see)

## 3. Findings (verified by the coordinator)

Legend: **V** = coordinator re-read the cited lines and/or re-measured; **M** = agent-measured, lines verified,
number not independently re-run. Severity: B blocks correctness, Bi biases results, L latent.

### 3.1 Signal chain (from reports/2026-11-05-survey-finder.md; 18 findings there, top ones here)

| id | where | class | assumption → failure | evidence | sev | V/M |
|---|---|---|---|---|---|---|
| F1 | alibz/minor_lines.py:433, :790 | A | seed/recovery window must hold ≥ 12 samples (span = max(3·wmed, 0.3 nm)); on native 12 samples = 1.1–2.2 nm | fe_fresh seed records: 86/200 "no-data" (UV 49, VIS 37/37), 48 "missing"; recovery 5/5 VIS "no-data"; exports 0 | B | V (counts from the cached run) |
| F2 | peaky_finder.py `fit_shoulders` (`residual_mean + 2*residual_std` of the whole-spectrum residual) | C/A | shoulder threshold referenced to the global residual std, not local noise; worth 8.6/81/24 σ of local noise on fe_fresh, 39 σ in VIS on user_spec1 | adds 55/4/2 shoulders where a 3σ local test finds 121/102/35 | B | V (code), M (counts) |
| F5 | pipeline.py:999–1016 + detector.py:525–531 | A/C/B | the 620 nm gain step is "measurable" only if both 8-nm flank medians exceed 5× the global single-pixel MAD; otherwise the corpus fallback 3.9 is used silently | **all 7 cached showcase runs (2 native, 5 export) carry source=fallback, ratio 3.9, raw_ratio None**; agent's flank ratios 3.7–9.3 | Bi | V |
| F6 | peaky_finder.py `_noise_scale` (max over lags 1/2/4/8), `_noise_scale_local(block=2048)` | A/E | estimator built for the spline-upsampled export; native noise is near-white and a 2048-sample block exceeds a native segment (~1935/1960/1812 px) | native noise inflated ×1.75/1.39/1.11; detections 130/108/46 vs 170/123/52 with lag-1 | Bi | V (design), M (numbers) |
| F7 | pipeline.py:1055 → inspection.py:86 | E/A | `estimate_peak_uncertainties` calls `_noise_scale_local(y_bgsub)` **without** segment boundaries (all other callers pass them) | VIS noise ×2.0 on fe_fresh, ×2.18 on fe_reused; VIS amp σ up to ×1.41 | Bi | V (code), M (numbers) |
| F8 | pipeline.py:1000–1005 → peaky_indexer_v3.py:2053–2055 | D/B | `amp_sigma_floor` = global per-pixel MAD (counts) added in quadrature to an **area** σ (counts·nm); the floor's weight depends on grid pitch and segment | floor exceeds the statistical σ for 44/53/45 % (user_spec1) and 74/62/89 % (fe_fresh) of peaks | Bi | V (units from code) |
| F3 | refinement.py:313 `if n < 15: return None` | A | 3a/3b blend/SA classification needs ≥ 15 samples; silently skipped, no record | fe_fresh: UV 82/119 features skipped; exports 0 | Bi | V (code), M (counts) |
| F4 | profiles.py:184 (≥ 9), :229 (band < 3) | A | profile QC needs ≥ 9 samples/window and ≥ 3 per shoulder band; native HWHM ≈ 0.7 px | fe_fresh: UV 52/119 "unresolved-window" | Bi | V (code), M (counts) |
| F10 | peaky_finder.py:288 `find_peaks(window=5)` | A | a line must be the maximum within ±2 samples (±0.07 nm on export, ±0.18–0.36 nm native) | fe_fresh: 78 significant maxima suppressed (58 > 5σ); user_spec1: 2 | Bi | V (default), M (counts) |
| F12 | peaky_finder.py `fit_all` window = half-max crossings | A/D | the core between half-max crossings suffices for 4 params/peak; native windows ≈ 4 samples | fe_fresh: fewer samples than parameters in UV 24/87, VIS 7/10, NIR 4/16 windows | Bi | M |
| F11 | peaky_finder.py `_fwhm_cap` (2× global median FWHM in px × local pitch, per channel) | C | the width cap references the median of the population it filters | export NIR seeds over the cap 14–21 %; native population contaminated by pancakes | Bi | M |
| F9 | minor_lines.py:268–274, :688–690 | A/E | one spectrum-wide median σ/γ and one pitch stand in for the instrument profile in every segment | user_spec1 NIR σ 0.100 vs global 0.070; global γ = 0 forces Gaussians | Bi | M |
| F13 | utils/voigt.py:83–85, finder height detection | D | point-sampled Voigt at pixel centres = pixel-integrated counts | previously reported (native-grid audit A1), still open; at 1.4–1.65 px FWHM the sampled height is biased | Bi | M |
| F14 | detector.py:498–505 | B | the 365 nm junction step is additive (export corpus), so it is never corrected | two native shots give flank ratios 0.36 and 0.40 → consistent with multiplicative, not additive; needs ≥ 10 native shots | Bi (unverified) | M |
| F15 | pipeline.py:228 `POISSON_GAIN_COUNTS = 1.0` | B | 1 count = 1 photo-event | native quiet blocks: σ² ≈ 0.39·bg + 69 (VIS); not gain-1 Poisson | L | M |
| F16–F18 | refinement `bic_effective_n` off by default; fixed-nm centre pins 0.02/0.06 nm; one median pitch per segment (±10–12 % within) | A/E | sample independence on the ρ ≈ 0.7–0.95 export grid; sub-pixel pins on native; within-segment pitch tilt | previously reported / latent | L | M |

Agent's checked-and-clean list (coordinator agrees on the items re-read): the `cumulative_trapezoid(x_window, y_window)`
"swap" in fit_peaks is a discrete-curvature detector, not the find_background bug — **do not "fix" the argument order**;
arPLS stiffness is per-segment and grid-invariant; native seams coincide with the 365/620 edges; `MIN_HEIGHT_SIGMA`
drops 0 peaks on all three test spectra; the seeding kT passes pass-1 T (the 0.76 eV constant is only a ranking default).

Structural reading (agent, coordinator concurs): every width reference in the chain — the finder cap, minor_lines
`sig_med`, profiles `segment_width_floor`, refinement `wmed` — is rebuilt from the fitted-peak population, which on
native data is contaminated by the over-wide components the gate problem produces. The validated per-segment LSF
(`peakfit.DEFAULT_PROFILE_PX`, `alibz/data/epsf-z300-20260924.json`) feeds only diagnostics and registration.
Candidate design: one per-segment instrument-width object as the single reference for windows, caps, seed shapes and
minimum-sample tests (addresses F1, F3, F4, F9, F10, F11, F12 together); the rejected alternative is patching each
constant behind `is_native`, which keeps the self-reference and doubles the code paths.

### 3.2 Inverse model and physics (from reports/2026-11-05-survey-indexer-physics.md; 16 findings there)

| id | where | class | assumption → failure | evidence | sev | V/M |
|---|---|---|---|---|---|---|
| I1 | peaky_indexer_v3.py:837 (`evidence_max_missing_lines=6`), :1488, :1388–1471 | C/B | a species is impossible if more than 6 of its 20 strongest predicted lines have no peak within 0.1 nm **in the peak table** (never the raw spectrum); the cap is an absolute count, so line-rich species fail and line-poor ones pass (Mn I, Pd I, As I kept with 2/5) | rejected by this rule alone: Fe I (12/20 supported, native Fe), Ti I (9/20, both export runs with Ti), Y and La on Alaskire; constant dates from 2bb9540 (2026-03-30), never re-examined | B | V (code), M (species lists) |
| I2 | peaky_indexer_v3.py:991–998; pipeline.py:1219, 1347–1350 | C | the candidate set (both prefilters, pseudo-lines, Stark and SA freezes) built at T_init is valid for every state the outer search visits; pass 2 inits at the pass-1 T and deepening rebuilds at the pass-2 T | same Fe table: 7 species at 8–10 kK init vs 3 at 22 kK; user_spec1 loses Al I, Ca I, Mg I at a 22 kK init; the pure-Fe table at a 10 kK init solves to Ir 0.44 / Cu 0.27 | B | V (code), M (numbers) |
| I3 | peaky_indexer_v3.py:2670 (`len(supported) < 2 → continue`); pipeline.py:1350 | C | the stage-tie thermometer votes only for elements with two surviving stages; T is never re-optimised after deepening adds peaks; no diagnostic marks a thermometer-silent solve | Fe table: tie cost 0 at every T, data cost falls monotonically toward hot → 22.6 kK; with Fe I restored the Fe tie minimum is 11.5 kK (= the VIS Fe I thermometer 11.47 kK) but the data term still outweighs it on the current table | Bi | V (code), M (curves) |
| I5 | peaky_indexer_v3.py:451 (`ci[0, ion - 1]`); sahaboltzmann.py:344–383 | D | stage labels are contiguous 1..N and every physical stage has db lines; the ion fraction is read by **column position** while the chain is built from the **stage labels present in the line db** | coordinator re-measured at 10 kK / log nₑ 17: Pr db stages [2, 59] → Pr II gets 0.0013; Tb [2, 65] → 0.0008; Gd [1, 64] → Gd I 0.72 with 0.28 in a "hydrogen-like" stage (physically Gd is ≈ 90 % Gd II); Zr [3, 6, 40] → Zr III gets 0.0; H single stage never ionizes. 59/92 elements carry stage labels > 30; 11 elements (Zr, Nb, Pr, Pm, Tb, Re, Os, Po, At, Rn, Pa) have no stage-I lines; all 59 are in `supported_elements`. Audit 2026-07 item 4 reported "H never ionizes" and "Cs bridged" — previously reported, still open, wider than recorded. Also a db-coverage question: no Gd II / Pr I lines at all in a NIST-derived db is implausible (build filter?) | B for REE/heavy elements (the project's critical-mineral targets) | V |
| I4/I6 | peaky_indexer_v3.py:2167–2228 (ridge rows), 1532–1621 (pseudo rows), :825 (`shift_tolerance=0.1`) | C/A | a strong predicted line absent from the peak table is an observed zero; ±0.1 nm around a centre contains every contributing db line | ridge + pseudo rows are 0–44 % of the outer cost and move user_spec1's grid-best T 8.5 → 7.0 kK; 0.1 nm = 0.13–0.14 FWHM on native VIS/NIR (0.56–0.77 px); peaks wider than 0.2 nm carry 40–77 % of Σ amp² the solve fits; one 0.74 nm blend carries 48 % of the Fe residual | Bi | V (code, defaults), M (fractions) |
| I7 | peaky_indexer_v3.py:2962–2966 (σ, γ bounds = (0.01, clip(2·median, 0.02, 0.3))) | A/B | the overlap-kernel width is identifiable from the amplitude objective | coordinator re-checked all 7 cached runs: σ = 2·median table width (0.106–0.146) and γ = 0.02 = the bound in 7/7 | Bi | V |
| I12 | pipeline.py:189–209, 940–983; peaky_indexer_v3.py:2834–2852 | B | the H-alpha nₑ box bounds the search; the prior acts only in weighted mode | coordinator: nₑ at a box edge in 6/7 cached runs (fe_fresh and jc_mdd017_29_2c upper; Alaskire, mdd017_24_15, mw2_1000, user_spec1 lower; fe_reused inside) — previously reported, still open | Bi | V |
| I8 | peaky_indexer_v3.py:3329–3345, :3182 | B/E | the outer cost is a χ² (BIC `cost + k·ln n`, posterior `exp(−rel/2)`) | production cost is unweighted SSE in counts² (1e6–9e7) vs k·ln n ≈ 1e2; posterior collapses to one grid node (spread 0.0 in 4/6 runs); `basin_ambiguous` is vacuous | L (diagnostics meaningless) | V (code) |
| I9 | sahaboltzmann.py:219–245 | D | ln U linear above the 10 kK Barklem table edge | U_Barklem/U_lines at 25 kK: Ca I 57×, Na I 44×, Fe I 2.2× — but the Saha ratio uses U_II/U_I so within-element fractions and the tie cost are unaffected; element totals shift ≤ 1.4× at 25 kK; **not** what makes hot states cheap | L | M |
| I10 | minor_lines.py:75–76, :87; pipeline.py:184–185 | E/A | the seeder's Boltzmann prior (`gA·exp(−Ek/kT)`, no hc/λ) matches the indexer's energy convention; 0.06 nm is a tight match | audit 2026-07 item 3 fixed hc/λ in the indexer only; 0.06 nm = 0.3–0.7 native px | Bi | M |
| I11/I14 | stark.py:103–138; wavelength.py:36 | A/D | the fitted H-alpha Lorentzian is pure Stark (instrumental Lorentzian ignored); db lines below 200 nm are vacuum, above air (0.065 nm step inside the UV segment if the instrument is air throughout) | reasoning only | L (unverified) | M |
| I15/I16 | synthetic.py:592; peaky_indexer_v3.py:3634 vs pipeline.py:189; alibz/peaky_indexer_v3.py.orig | E | one default set | a **third** default set (generator: line-sum, no IPD) besides library and pipeline; T box duplicated as a run() default and `PLASMA_T_BOUNDS`; untracked `.orig` backup (3 653 lines, 80 diff lines, not importable) | L | V (.orig exists) |

Mechanism chain for the Fe case as the indexer sees it (agent, coordinator concurs with 2026-11-04 §3): finder drops the VIS
peaks → I1 rejects Fe I at every T (8 of 20 strong lines "missing") → I2 leaves {Fe II, Ni II, H I} plus init-dependent
junk → I3 no element has two stages, tie cost 0, the Fe II forest cost falls monotonically with T → 22.6 kK; nothing
re-optimises T afterwards; I6 adds that most of the objective sits in broad blend envelopes the ±0.1 nm design cannot book.

Agent's checked-and-clean list (coordinator agrees where re-read): overlap conserve vs legacy changes no prefilter outcome;
the initial-strength ladder is not the Fe culprit; column-max un-normalisation fixed (audit item 1); hc/λ shared by
forward and inverse via `line_emissivity` (audit item 3, seeder excepted); Debye IPD uses (z+1) correctly; all wave-1–3
options are OFF on the production path (no default-path leak found); `solve_at` reproduces the tail of `fit()`;
Barklem vs line-sum partitions agree within 2 % for Fe I/II below 10 kK.
### 3.3 Orchestration, registration, scoring and fixtures (from reports/2026-11-05-survey-pipeline-glue.md; 13 findings + call-chain inventory there)

| id | where | class | assumption → failure | evidence | sev | V/M |
|---|---|---|---|---|---|---|
| G1 | scripts/fit_quality_panel.py:37–56, 212–276; bench/synth_cases.py:98–109; alibz/synthetic.py:55 | F | the fit-quality panel scores the production regime | only native items: 26 means of one pure-Fe file; all 5 synthetic + 5 real items are 0.0333 nm exports (`NATIVE_PITCH_NM = 1/30` — "native" is the export grid, misnamed); panel r² is the indexer's peak-table r², so missing peaks cannot lower it; `git_head: None` in every JSON. Coordinator re-read the production JSONs: baseline → adopted defaults kept **23/26 Fe runs above 15 kK** (mean T 21.7 → 21.4 kK), nₑ at bound 18 → 16 of 26, pipeline `qc_status` **fail on 31/31 → 30/31** (+1 warn), while the headline `fe_fraction` rose 0.398 → 0.562 and scored as "Fe improved" (the agent quoted 23 → 25 and 22.8 kK; not reproduced on the -prod files) | blind spot: every 2026-11 A/B verdict is provisional for native data | V |
| G6 | minor_lines.py:433, 790 (≥ 12), :592 (`max_new=40`); profiles.py:184, 400 (≥ 9) | A | (same gates as F1/F4) | chain-level: fe_fresh per-segment table UV/VIS/NIR = blind 72/5/11 → pass-2 105/5/11 → final 119/5/11, i.e. the whole seeding/recovery/deblend chain adds **0 VIS and 0 NIR peaks**; the 40-candidate recovery cap is consumed by UV; 40 % of peaks "unresolved-window" so SA recovery never runs on native. Fixing the finder gate alone will not restore VIS | B | V (counts match 2026-11-04 §1/§4) |
| G2 | wavelength_registration.py:318–333; utils/peakfit.py:31–33, 110–113, 150–157; pipeline.py:301/696 (`subpixel="gaussian"` default) | A (native → export direction) | the registration's Gaussian sub-pixel fitter uses an instrument σ in **native pixels** (FWHM 1.4/1.65 px) on any grid | on user_spec1 (export) registration `applied = False`, `gas_cross_check = None`; agent: gaussian finds 0 ambient lines, parabolic finds 13 (quality ok, −0.028 nm) | Bi (exports) | V (applied=False), M (0 vs 13) |
| G3 | pipeline.py:1000–1021; detector.py:516–551; corrections/detector.json | B/A | the 620 nm gain is measured per spectrum, else the corpus constant 3.9 (from 435 export AverageSpectrum files, bimodal with a second mode ≈ 9×) | = F5; agent: fe_fresh raw flank ratio 227/47 = 4.8 rejected because flanks < 5·`_noise` (floor 308); synthetic `iron_rich_low_t` has true gain 1.0 and gets 3.9 | Bi | V (7/7 fallback) |
| G4 | pipeline.py:1160–1164, 1328–1350, 1417–1419, 1088–1092, 1580–1582 | C | T, nₑ, σ, γ are fixed at pass 2; deepening and SA recovery re-solve linearly; pass-1 output gates seeding unguarded | fe_fresh: pass-2 T 22 601 K carried unchanged through 14 added peaks; **fe_reused_fail pass-1 r² = −1.28** (coordinator re-read) and the run continues; nₑ at a box edge 6/7 cached and 16–18/26 panel Fe runs | Bi | V |
| G5 | detections.py:359–363; scripts/fit_inspector.py (≈ 8 settings); pipeline.py:960–974 vs 1202–1221 vs 1328–1347 | E | every consumer that builds an indexer uses the pipeline's physics | the **production confounder-contest indexer** (detections.py:359) is built with only dbpath / T_init / nₑ_init / overlap_mode → library defaults (line-sum partition, no IPD), not the adopted barklem2016/debye; deepening drops `stark_width_weight`; fit_inspector drifted | Bi (contest decisions under different physics than the solve) | V (code) |
| G7 | pipeline.py:659–752, 1466–1491; scripts/z300_fb_decode.py:204–206 (`export_grid=True` default, linear 0.1 nm); scripts/fit_quality_panel.py:173–196 | A/D | the input grid is whatever arrives; no regime detection or record; preprocessing lives in loaders | a **third regime** exists (linear-interpolated 0.1 nm decode) besides spline export and native; nothing records which one a run saw | L → blind spot | V (default) |
| G8/G9 | minor_lines.py:268–274, 688–690; pipeline.py:1163/1277/1312 (`segment_edges=(620.0,)`) vs minor_lines.py:676 / profiles.py:156 (365, 620) | A/E | one global narrow-line σ/γ; two segment definitions in one run | NIR narrow-line median FWHM 0.26–0.32 nm vs global seed 0.14–0.17 nm → seeded NIR lines capped at ~60 % of the profile; seeding pools UV+VIS for noise and `ref_segments` | Bi / L | M |
| G10 | pipeline.py:1005–1006, 1055–1056; peaky_indexer_v3.py:2053–2058 | A/E | floor and Poisson terms share units with response-corrected amplitudes | `amp_sigma` is divided by the segment response, the floor (raw counts) and `|amp|/gain` are not → NIR floor 3.9× too large relative to amplitudes (matters only in weighted mode) | L | V (code, = F8) |
| G11 | pipeline.py:466–512 | B/A | coarse-composition bootstrap: top 60 peaks within 0.12 nm of gA ≥ 1e6 lines, ≥ 2 hits | on pure Fe the bootstrap includes U, Th, Ne, Co, Hg, Cl (diagnostic use only) | L | M |
| G12/G13 | docs/development_guide.md:630–633; alibz/peaky_indexer_v3.py.orig | E | docs say the export "is not a resampling" and "no 1/30 nm assumption in the analysis path" | falsified by G2/G6 and the gate; `.orig` leftover | L | V |

Call-chain inventory: section 1 of the glue report lists every analyze_spectrum step with inputs, mutated arrays and
constants consumed — use it as the map for the fixes below.

### 3.4 Gate-fix outcome (reports/2026-11-05-native-gate-fix.md; coordinator-verified)

Fix: `alibz/peaky_finder.py` only (+90/−12; `tests/test_native_gate.py`, fixture `tests/fixtures/z300_fe_native_trimmed.json`;
patch `reports/patches/2026-11-05-native-gate.patch`). Two native defects, both gated on `segment_pitch_array(x).is_native`:
(A) narrow (≤ 5-sample) windows now seed directly with the FWHM floored at the instrument LSF; (B) the per-channel
`< key_inc/2` width filter, which zeroed BOTH Voigt channels of a resolution-limited line after the equal G/L seed split,
now tests the combined width against 0.5 × instrument FWHM on native grids. Verified: export blind tables byte-identical
(user_spec1, mw2_1000: `np.array_equal`), production synthetic + real panels exactly unchanged on every item/metric
(coordinator re-read `reports/fit-panel/2026-11-05-gate-compare-prod.md`), new test 4/4 (coordinator re-ran), full suite
695 passed / 10 skipped / 2 xfailed (agent; baseline 691 + 4 new).

Native effect on fe_fresh_d5p25: blind table 88 (72/5/11) → 243 (125/85/33); all six strong Fe I VIS lines present;
final table 135 → 271 (151/87/33); Fe I assigned 0 → 140, Fe II 112 → 108. **But the solve stays hot and flips to a
sodium basin:** T 22 601 → 20 929 K, r² 0.346 → 0.421, element fractions Na 0.81 / Fe 0.12 / Ga 0.05 / H 0.02 (was
Fe 0.93 / H 0.07). Fe panel headline regressed (fe_fraction mean 0.563 → 0.287, 8 better / 18 worse; |T − 10 kK|
18 better / 7 worse).

Coordinator fixed-state re-solves on the NEW 271-peak table (same indexer build as the pipeline's deepening re-solve):

| state | r² | Fe | Na | Fe stage disagreement | note |
|---|---|---|---|---|---|
| 20 929 K, 17.655 (pipeline) | 0.421 | 0.12 | 0.81 | 0.97 | reproduced |
| 11 467 K, 17.4 / 17.655 | 0.320 | 0.61 / 0.73 | 0.36 / 0.25 | 0.37 / 0.11 | |
| 10 636 K, 17.4 | 0.302 | 0.76 | 0.22 | 0.05 | stage-tie minimum ≈ here (= VIS Fe I thermometer 10.6–11.5 kK) |
| 8 036 K, 17.4 | 0.248 | 0.91 | 0.06 | 0.91 | |
| 11 467 K, evidence prefilter off | 0.400 | 0.02 | 0.01 | — | 66 species, Ba 0.30 / Sb 0.15 / Si 0.10 junk |

Reading. (1) Na is a hot-basin artefact, not an assignment error: the same 4 Na I peaks (D doublet 589 nm, 818/819 nm)
are assigned at every state; at 21 kK sodium is almost fully ionised and Na II has no lines, so those 4 peaks demand a
huge Na concentration — the fraction swings 0.06 → 0.81 purely through the Saha factor. The old Fe 0.93 was the same
artefact hidden by a single-element design. (2) The stage-tie thermometer now has a live Fe vote and its minimum sits at
≈ 10.6–11.5 kK, agreeing with the Fe I VIS Boltzmann temperature; the unweighted data cost (r² 0.42 vs 0.32) still outvotes
it. (3) Switching the prefilter off does not help — the data term then buys 66 junk species. So the finder fix removed a
real information-destroying defect and exposed the next link exactly as the surveys predicted: the whole-pattern,
unweighted amplitude objective prefers the hot state on a pure-Fe spectrum (the wave-3 lesson "objective minimum ≠
truth"), and nothing in the outer search lets the thermometer, a T prior, or a basin plausibility test override it.

## 4. Checked and clean (coverage)

Union of the three agents' clean lists, re-read by the coordinator where marked (✓):
- fit_peaks `cumulative_trapezoid(x_window, y_window)` argument "swap" is a discrete-curvature (blend) detector, not the
  find_background bug — do not change the argument order (✓ derivation re-read).
- arPLS stiffness per segment and grid-invariant; native seams coincide with the 365/620 nm edges; `valid_signal_mask`
  handles export pads; `MIN_HEIGHT_SIGMA` drops 0 peaks on the three test spectra; `_voigt_seed` area convention; ±1
  local-pitch centre bounds; strongest-first pedestal subtraction.
- Seeding passes pass-1 T (the 0.76 eV constant only ranks db plausibility).
- Indexer: conserve vs legacy changes no prefilter outcome; column-max un-normalisation fixed (audit-2026-07 item 1);
  hc/λ shared via `line_emissivity` (item 3; seeder excepted, I10); Debye IPD (z+1) correct; `solve_at` reproduces
  `fit()`'s tail (✓ L3444–3450 vs L3075–3084); all wave-1–3 options OFF on the production path (✓ pipeline.py:960–983,
  1202–1221, 1328–1349); Barklem vs line-sum within 2 % for Fe I/II below 10 kK.
- Glue: segment response is applied to every peak array handed to every indexer pass and to the SA re-solve (the
  2026-07 "never called" finding is fixed; the problem is now that it never measures, G3); vacuum→air at Database load
  (above 200 nm); the −18 px native axis offset is applied on the acquisition path the showcase data came through;
  `capture_stages` is byte-identical when off.

Not covered by this survey: gas_calibration/gas_detection internals, triage.py, whole_rock priors, GPU paths,
peaky_corpus/peaky_pca, the Pantheum acquisition code, and every script under scripts/ other than the panel, the
inspector, the decoder and manuscript_figures.

## 5. Recommended order of fixes (coordinator; owner decides)

The pattern across all three surveys is one defect class, not many: **constants and references fixed in the
export-grid era, applied to the native grid without a regime record, and scored by a panel that cannot see the
native regime.** Four of them chain into the Fe hot basin (gate → F1/G6 seeding gates → I1 evidence prefilter →
I2/I3 candidate set and silent thermometer → G4 frozen T). Ordering below is by "what unblocks verification of
everything else", then by severity.

0. **Make the regime visible before fixing anything else (G1, G7).** Record the input grid (pitch per segment,
   native/export/0.1-decode) in `analysis` and in the panel JSON; add a native panel tier (the Fe means plus native
   showcase shots, and a synthetic case rendered on the true native pitch — rename `ChannelGrid.native()`); score
   `qc_status`, a whole-spectrum r², per-segment peak counts and |T − T_FeI| as headline metrics; record `git_head`.
   Until this exists no A/B verdict on native behaviour is defensible, including the gate fix in flight.
1. **Gate fix (DONE, in the working tree, uncommitted; ship recommended) + the sample-count gates behind it (F1/G6, F3, F4, F10, F12), one physical rule.**
   One per-segment instrument-width object (px FWHM from `peakfit.DEFAULT_PROFILE_PX` / the ePSF file, measured
   fallback for other instruments) as the single reference for windows, caps, seed shapes and minimum-sample tests;
   remove population-median self-references (F9, F11, G8). Specialist, M.
2. **Stop pruning on peak-table absence (I1, I4).** Replace `_species_line_evidence`'s table presence with a
   raw-spectrum presence vector (side-band SNR at db positions, the cal-campaign method); drop the absolute
   `evidence_max_missing_lines=6` cap for a fraction with a noise-aware threshold; pseudo-observation rows only where
   the raw spectrum is genuinely empty. Specialist, M. Panel-score with the native tier.
3. **Let the thermometers outvote the amplitude cost (I2, I3, G4; now the measured blocker, §3.4).** On the
   post-fix Fe table the stage-tie minimum (10.6–11.5 kK) and the VIS Fe I Boltzmann T agree, and r² still picks
   20.9 kK with Na 0.81. Options, in rising cost: (a) the line-selected VIS Boltzmann T as a prior/box in the outer
   search when ≥ 8 lines over ≥ 2.4 eV resolve (owner's F5 proposal; S–M); (b) a basin-plausibility guard: an element
   whose fraction rests on ≤ 4 neutral peaks at a T where its neutral fraction is < 1 % is an artefact (the existing
   basin-collapse guard, hot direction; S); (c) stage-tie weight on a comparable scale to the data cost (currently
   counts² vs data units; M); (d) candidate set as a union over the T box + one T/nₑ re-optimisation on the final
   table after deepening, and a pass-1 failure guard (r² < 0) (M–L). Specialist, scored on the native tier from step 0.
4. **Saha stage lookup by label + chain from the ionization table (I5).** Index by stage label at
   peaky_indexer_v3.py:451; build the chain from stages I→III regardless of which have db lines; H ionizes; unit
   tests for Pr II, Tb II, Gd I/II, H. Worker, S. Separately audit the db build: Pr I, Gd II, Tb I lines absent from a
   NIST-derived db (REE are the project's targets). Worker, S–M.
5. **Measure instead of assume at the instrument layer (F5/G3, F6, F7, F8/G10, F14, F15).** Segment-blind noise call
   in inspection.py:86 (S); lag-1 noise on native / regime-aware estimator (S); response floor relative to the flank's
   own noise, log the measured ratio even when rejected, and test the 365 nm step on ≥ 10 native shots (M); area-floor
   in area units per segment (S); gain-aware Poisson term (S).
6. **One physics for every consumer (G5, I15, I10).** The confounder-contest indexer and the seeder take the
   pipeline's SahaBoltzmann/emissivity; deepening keeps every option; fit_inspector either calls `capture_stages` or is
   retired (S each).
7. **Registration on exports (G2)**: pixel-aware σ from the grid's own pitch, or parabolic on non-native grids (S).
8. Housekeeping: delete `alibz/peaky_indexer_v3.py.orig`; correct docs/development_guide.md:630–633 and add a
   "grid regimes" section; make BIC/posterior unavailable in unweighted mode (I8).

Owner decision points: (a) whether to build the native panel tier before or alongside the gate fix (recommended:
before any further adoption); (b) whether the VIS Fe I thermometer enters as a prior (step 3) or as a check only;
(c) scope of the db rebuild (step 4).

## Unverified — PENDING
