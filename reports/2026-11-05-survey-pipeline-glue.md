# Survey: pipeline glue, ordering, configuration, scoring and fixtures (2026-11-05)

Agent: judgment (Opus 5.5). Read-only on the repo; scratch scripts in
`/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/7c311a86-0d91-4217-8224-5b49f94b239b/scratchpad/survey-c/`
(pitch.py, segresp.py, pk.py, st.py, prof.py, shape.py, reg.py, amb.py, seam3.py). No provider switch.
Line numbers are for the working tree as of 2026-11-05 (pipeline.py is 2626 lines; the brief's ~line numbers are ~10 lines low).

Question: which design choices in the orchestration layer (everything that connects, orders,
configures, scores and feeds the finder and indexer) inject regime- or dataset-specific
assumptions, and would the benchmarks notice?

Evidence base: code reads (cited file:line), the seven `manuscript/figures/cache/*.pkl` full runs
(2 native: fe_fresh_d5p25, fe_reused_fail; 5 export: user_spec1, mw2_1000, jc_*), panel JSONs in
`reports/fit-panel/`, and cheap re-computations (synthetic render, segment response on the 5
synthetic cases, ambient registration on user_spec1 with both subpixel modes, +18 axis-shift error).
Overlap: survey-finder (F1, F4, F9, F10, F16, F17) already covers the sample-count gates in
`minor_lines.py` and `profiles.py`. Here they appear only as G6 with chain-level counts that survey-finder
does not have.

---------------------------------------------------------------------------------------------------

## 1. Call-chain inventory: `analyze_spectrum` (alibz/pipeline.py:659)

Columns: inputs → outputs / mutated state; constants consumed.

1. **Validate modes** (:723-733). Fresh `telemetry`, `stages` only if `capture_stages`.
2. **DB singleton** `_get_db(dbpath)` (:440, :738).
3. **Blind fit** (:739-752). `PeakyFinder.__new__` bypasses `__init__`, which only sets `data_dir`/`data`, so the bypass is benign (peaky_finder.py:70-71).
   `fit_spectrum(x, y, subtract_background=True, n_sigma=0)` → `fit` (`sorted_parameter_array`
   [amp, center, sigma, gamma], `background`). All finder constants live there (out of scope).
4. **Gas calibration** (:765-769). `calibrate_background_gases(x, y, db, peak_array=peaks)` runs on
   the raw blind table. Its windows are pitch-aware: `tolerance=max(0.015, 0.75·pitch)` (gas_calibration.py:200).
5. **Global shift** `estimate_wavelength_shift(peaks, db)` (:772). tol 0.15 nm, n_peaks 40, gA ≥ 1e6
   (utils/wavelength.py:158-161).
6. **Refine 3a (data only)** `refine_fit(..., shift_nm=shift0, asymmetric="defer", bic_effective_n)`
   (:779) → `refined`, `dec_data`. BIC rho is `None`→0 by default (refinement.py:142-143).
7. **Peak-location diagnostics** `locate_records` (:804). This step is additive. The native/export dispatch is
   `segment_pitch_array(...).is_native`: per-segment medians must differ by more than 1.05×, and
   samples at or above 955 nm are excluded (utils/native_grid.py:845-852, peak_location.py:65).
8. **Per-segment shift** `estimate_wavelength_shift_segments(rpeaks, db)` (:822). Edges (365, 620),
   tol 0.15 nm, significance gate tuned on MW2-112 exports (comment :814-818).
9. **Shift prior/offset overrides** (:827-861). These are optional.
10. **Wavelength registration** (:868-922). `_coarse_composition(rpeaks)` (:466) uses tol 0.12 nm, top 60 peaks, and
    ≥ 2 hits. `ambient_registration` / `element_registration` / `combined_registration` follow. The default
    mode "ambient" replaces the NIR shift only when NIR has ≥ 3 lines and quality ok/weak (:888-893).
    Constants: wavelength_registration.py:70-115 (`search_window_nm` 0.5, `min_snr` 5) and
    `instrument_sigma_px` in **native pixels** (peakfit.py:31-33).
11. **Gas calibration applied** `apply_gas_calibrations(baseline_shift, ...)` (:928) → final `shift`.
12. **H-alpha nₑ** `_halpha_ne(rpeaks)` (:940). This uses the instrument frame, ±0.2 nm, gamma ≥ 0.05 nm, ±0.3 dex
    (utils/stark.py:103-138). `ne_prior` comes from H-alpha, or `NE_PRIOR_DEFAULT` (17.0, 0.5) with a 1σ box
    (:208-209, :944-956).
13. **SahaBoltzmann singleton** (:958). Defaults are barklem2016 + debye.
14. **Indexer kwargs** `idx_kwargs` (:960-974) and `run_kwargs` (:975-983). These are used verbatim only by pass 1.
15. **Noise and segment response** (:985-1027). `_noise` = 1.4826·MAD(y − bg0) over the whole array
    (:1000). `amp_sigma_floor=_noise` and `POISSON_GAIN_COUNTS=1.0` (:1005-1006).
    `segment_response_fallback(edges=(620.0,))` reads corrections/detector.json (3.9, 2026-07 corpus).
    `estimate_segment_response(x, bg0, edges=(620,), noise_scale=_noise)` measures flank medians 2-10 nm,
    and both flanks must exceed 5·`_noise` (detector.py:516-527), else the fallback is used.
16. **Closures** `_db_frame` (:1029-1037) removes the per-segment shift and divides amplitudes by the
    response. `_amp_sigma` (:1040-1058) computes `estimate_peak_uncertainties(x, y−bg0)` divided by the response,
    memoised per table.
17. **Background gas detection** `detect_background_gases(x, y, db, shift_nm=shift, peak_array=_db_frame(rpeaks))`
    (:1065). Peaks are in the database frame and compared against database lines, which is consistent. Windows
    are pitch-aware (gas_detection.py:108-118).
18. **Triage coverage** (:1071-1077). Only runs when `physical_triage != "off"` (default off).
19. **Pass 1** (:1079-1087). `PeakyIndexerV3(_db_frame(rpeaks), amp_sigma, **idx_kwargs).run(n_calls=min(n, 24))`.
    No `temperature_init` is passed, so the indexer default applies. Output `established` = fractions ≥ 0.002
    (:1088-1092). No guard on a failed or empty pass 1.
20. **Refine 3b** (:1113) on `posterior = species retained by pass 1`, with the final `shift`.
21. **SA zones/merges** from 3b decisions (:1127-1158).
22. **Seeding** `seed_minor_lines(..., established, kT_ev=KB·res1.T, segment_edges=(620.0,))` (:1160-1164).
    Uses the pass-1 T **without** `_warm_start_temperature`.
23. **Residual recovery** `recover_residual_lines(x, y, final, exclude=sa_zones)` (:1179). Defaults: 4σ, `max_new=40`,
    segment edges (365, 620).
24. **Profiles + shoulder deblends** (:1190-1196).
25. **Pass 2** (:1202-1222). This is a hand-written copy of the pass-1 kwargs (drops `wavelength_registration`), with
    `temperature_init=_warm_start_temperature(res1.T)`, `ne_init=res1.ne`, and `run_kwargs` (full n_calls).
26. **Basin info** captured (:1228-1236).
27. **Deepening** (:1262-1385). `confirmed` = pass-2 fractions ≥ 0.002. `match_and_scale(..., kT=res2.T,
    segment_edges=(620.0,))` (:1275), and `confident` = n_ref ≥ 4 (calibrated on MW2-112 #1000, :168-172).
    For bars (3, 2): seed (:1307) and recover with support tol 0.06 nm (:1316) run, then a **third hand copy** of the
    indexer kwargs (:1328-1346) and `build_candidate_matrix(sa_doublets, sa_stimulated_emission)` with no
    `stark_width_weight` (:1347). The re-solve is `solve_at(res2.T, res2.ne, res2.sigma, res2.gamma)` (:1350), i.e. **T, nₑ, σ, γ fixed**.
    A round that newly collapses (top ≥ 0.9 or a jump, :166-167) is rejected.
28. **SA area recovery** `analyze_peak_profiles` + `recover_sa_areas(fidx, result, ...)` (:1417-1419). This is a linear
    re-solve at the fixed state.
29. **Detections** (:1430-1448). `area_sigma=_amp_sigma(fpeaks)` is SA-scaled for recovered rows (:1436-1441).
    `analyze_detections(fidx, ...)` then rebuilds a contest indexer (detections.py:359) with only
    `dbpath`/`overlap_mode`, which means the default SahaBoltzmann (lines, no IPD). It scans T 6-12 kK × log nₑ 16.5/17.5
    (detections.py:60-61).
30. **Shape QC** `element_shape_quality` (:1451).
31. **Output dict** (:1466-1491). It has no grid/pitch/regime field. `_summary_row` derives qc (:1580-1600).

Default sets (three): the library default `SahaBoltzmann(partition_source="lines", ipd=None)`
(sahaboltzmann.py:90) and `PeakyIndexerV3(overlap_mode="legacy")` (peaky_indexer_v3.py:614, 638), versus the pipeline,
AnalysisConfig, analyze_directory and CLI default (barklem2016, debye, conserve) (pipeline.py:255-277, 676-681;
cli.py:128). CLI → analyze_directory → AnalysisConfig → `_analyze_file` passes all 30 knobs and these defaults agree.

---------------------------------------------------------------------------------------------------

## 2. Ranked findings

Severity: **blocks** (blocks correctness) / **biases** / **latent** / **blind** (scoring blind spot).

| id | file:line | class | assumption | origin | breaks in | evidence | severity | fix (cost) |
|---|---|---|---|---|---|---|---|---|
| G1 | scripts/fit_quality_panel.py:37-56 (HEADLINE/SIGN), :212-276 (verdict), :219-225 (Fe metrics); bench/synth_cases.py:98-109; alibz/synthetic.py:53-55, 211-214 | D | The panel scores the production regime | Panel built 2026-11-03 on export-era fixtures | native | **Native inputs:** 26 of 36 panel items, all one pure-Fe npz; synthetic (5) and real (5) are 1/30 nm exports (measured pitch 0.0333 for all 10). `ChannelGrid.native()` is the 1/30 nm grid (`NATIVE_PITCH_NM = 1/30`, misnamed). **Fe headline:** `fe_fraction` is rewarded and `non_fe_mass = 1 − fe_fraction` counts it twice. The adopted defaults (baseline-prod → cpi-prod2) raised hot Fe runs (> 18 kK) from 23 to 25 of 26 and mean T from 21 692 to 22 830 K, yet the round was recorded as "Fe 0.398→0.563 improved". **QC ignored:** the pipeline's own `qc_status` is "fail" on 31/31 real+Fe items in baseline-prod, cpi-prod2, cpi-s3 and w3* (Fe: physical-pattern-r2 21/26, ne-at-bound 19/26). **r²** is peak-table amplitude r² (peaky_indexer_v3.py:3474-3478), so dropped peaks cannot lower it. **Verdict** counts better vs worse item-metrics; it is not "all means improve". **Provenance:** `config: {}`, `git_head: None` in cpi-s3/cpi-prod2/baseline-prod. **Physics mismatch:** synthetic truth is rendered with partition "lines", no IPD (synth_cases.py:103) while the inverse runs barklem2016+debye. | **blind** | Add a native panel tier: real native multi-element runs, plus synthetic rendered on the 0.089/0.129/0.179 nm ChannelGrid with a 620 nm gain step. Add per-segment peak-count recall, an Fe I/Fe II stage-presence metric and `qc_status` as headline. Drop `non_fe_mass`. Record resolved config + git HEAD. Rename `ChannelGrid.native` → `export_grid`. (M) |
| G2 | alibz/wavelength_registration.py:318-333 (gaussian path), utils/peakfit.py:31-33, 110-113, 150-157; pipeline.py:290 (`subpixel="gaussian"` default) | A (native→export direction) | Instrument σ in **native pixels** (FWHM 1.4/1.65 px) applies to any grid | Registration designed on native data 2026-09-23/24 | export (all JC/MW2 production data) | On 1/30 nm exports the Gaussian is clamped to 0.6-2.0 × 0.59 export px with a ±3-sample window. Ar I 763.5 on user_spec1: σ railed at 1.40 px, SNR 3.04 (< `min_snr` 5), `blend=True`. Ambient registration finds **0 lines** with gaussian and **13 lines, quality ok, −0.028 nm** with parabolic (reg.py). The cache confirms 0 ambient lines and empty `gas_cross_check` on all 4 export runs, while both native runs register NIR (7 lines). Registration is silently inert on exports and nothing flags it. | **biases** (exports keep the legacy anchor shift and the advertised independent registration and Ar cross-check never run) | Convert `instrument_sigma_px` from nm per segment using the local pitch (σ_nm / pitch), or fall back to parabolic when `is_native` is False. Record per-segment line counts in the summary row. (S) |
| G3 | pipeline.py:1000, 1007-1021; detector.py:516-551; corrections/detector.json `segment_response_fallback` | B/A | The 620 nm gain is measured per spectrum, else a corpus constant 3.9 | 2026-07 corpus of 435 AverageSpectrum **exports** (bimodal, second mode ~9×) | all; native and synthetic especially | 7/7 cached runs use `source=fallback` 3.9, including native fe_fresh. Its raw flank ratio of 227/47 = 4.8 was rejected because the flanks were < 5·`_noise` (floor 308). `_noise` is a whole-array MAD that lines inflate. On the synthetic suite (true `segment_gain` 1.0, synthetic.py:365), `iron_rich_low_t` gets the 3.9 fallback (flanks 0.015/0.008 < floor 0.085), so its NIR amplitudes are divided by 3.9 when no step exists. The other 4 cases measure ≈ 1.0. Whether native API spectra carry the same step as the vendor export is unverified. The qc warning "detector-response-fallback" fires on 26/26 Fe items. | **biases** (cross-segment intensities feed T) | Use a flank-local noise for the floor, not global MAD. Key the fallback on the grid regime/acquisition configuration, with a native-measured value. Make the synthetic panel render a known step and assert recovery. Flag rather than apply when the regime does not match the calibration. (M) |
| G4 | pipeline.py:1160-1164, 1328-1350, 1417-1419; 1088-1092; stark.py:103-138; pipeline.py:1580-1582 | C | T, nₑ (and σ, γ) are fixed at pass 2. Deepening and SA recovery re-solve linearly. The final table is never re-optimised. Pass-1 output gates seeding unguarded. | 2026-07 basin-drift fix (solve_at replaced pass-3 GP) | all; worst when pass 2 is in a wrong basin | fe_fresh: pass-2 T 22 601 K is carried unchanged through 14 added peaks to the final answer. **nₑ at the box edge:** 6/7 cached runs and 20/26 Fe + 5/5 real in cpi-prod2. The H-alpha ±0.3 dex box, not the data, sets nₑ, and qc "electron-density-at-bound" fails nearly every run. **Pass-1 failure unguarded:** on fe_reused_fail pass 1 returned r² −1.28 with empty fractions, so seeding was skipped silently. Pass 2 then reports B 0.73/Fe 0.15 on pure Fe. **Railed-T seeding:** seeding uses `res1.temperature` raw (:1161), while `_warm_start_temperature` is applied only to the pass-2 init (:1219). | **biases** | After deepening, run one bounded re-optimisation (or a profile check of T over the final table) with the basin guard. Flag pass-1 r² < 0 / empty and fall back to a default-T seed. Apply `_warm_start_temperature` to the seed kT. Treat "nₑ at box edge" as unmeasured nₑ, not a QC fail. (M) |
| G5 | detections.py:359-363, 60-61; scripts/fit_inspector.py:61, 176-181, 221-222, 246, 269-272, 120; pipeline.py:960-974 vs 1202-1221 vs 1328-1347; pipeline.py:266, 268, 273 | E | Every consumer that builds an indexer uses the pipeline's physics | Defaults flipped at pipeline level only (2026-11-04) | all | The **production contest indexer** (detections.py:359) gets only `dbpath`, `temperature_init`, `ne_init` and `overlap_mode`. It therefore uses the default SahaBoltzmann (partition "lines", no IPD; sahaboltzmann.py:90, peaky_indexer_v3.py:638) and no `ne_prior`, and scans 6-12 kK while the solve sits at 22 kK on native Fe. `resolved_fractions` and confounder verdicts thus use different physics from the composition they correct. **fit_inspector** uses `overlap_mode="legacy"`, lines partition, `search` default (gp, not grid), no ne_prior, n_calls 40 for pass 1, `seed_minor_lines` without `kT_ev`/`segment_edges` (so 365/620 instead of 620 and the default kT), and `shift_at` without `frame="database"` (:120 vs pipeline :1291). **Pipeline self-duplication:** pass-2 and deepen copy the kwargs by hand. Deepen's `build_candidate_matrix` omits `stark_width_weight` (:1347), so with `stark_width_weight ≠ 0` deepening re-solves without it. Pass 2/deepen drop `wavelength_registration` (diagnostic only). AnalysisConfig comments still call "lines", None and "legacy" the defaults (:266, :268, :273). | **biases** (contest), **latent** (rest) | Move the defaults into one `PhysicsConfig` consumed by SahaBoltzmann, PeakyIndexerV3, detections and scripts. Have the pipeline build `idx_kwargs` once and pass it to all three passes and the contest (pass `sb`). Make fit_inspector call `analyze_spectrum(capture_stages=True)` instead of re-implementing it. (M) |
| G6 | minor_lines.py:431-436, 724, 790 (`< 12`), :592 (`max_new=40`); profiles.py:184 (`>= 9`), :400 (`< 9`) | A | Seed, recover, profile and deblend windows need ≥ 12 / ≥ 9 samples | export era (12 samples = 0.4 nm at 1/30) | native VIS/NIR | Gates already listed by survey-finder F1/F4. **Chain-level evidence added here:** on fe_fresh the per-segment table counts (UV/VIS/NIR) are blind 72/5/11, pass-2 105/5/11 and final 119/5/11, so the whole chain adds **0 VIS and 0 NIR peaks**. The windows hold 16/11/8 samples (UV/VIS/NIR) for span = 0.687 nm, against 75 on exports. Fe records show VIS 37 "no-data" (seed) + 17 rejected "no-data" (recover). Recovery's 40-candidate cap is saturated by UV on both native runs (fe_fresh 40/40 UV or VIS-rejected; fe_reused 40/40 UV). Profiles mark 54/135 (fe_fresh) and 40/113 (fe_reused) peaks "unresolved-window" (0 on exports). Those peaks are excluded from SA recovery (`sa_used` False on both natives) and from clean-anchor QC. | **blocks** (native) | Express windows in instrument FWHM and the per-segment pitch (`segment_pitch_array`), with gates in samples ≥ 5 (fit dof). Make `max_new` per segment. Note that fixing the finder gate alone does not restore VIS seeding or recovery. (M) |
| G7 | pipeline.py:659-752, 1466-1491; scripts/z300_fb_decode.py:182-189, 204-206, 274, 299; reports/2026-09-22-native-grid-processing.md:24-27; scripts/fit_quality_panel.py:173-196 | A/D | The input grid is whatever arrives. There is no regime detection or record. Preprocessing lives in the loaders. | 2026-09-22 switch to native | native raw, 0.1 nm decode | **Third regime:** `z300_fb_decode.py` writes a **linear-interpolated 0.1 nm grid by default** (`export_grid=True`, native only with `--native-grid`) with 0.0 fill. The analysis then treats it as uniform non-native (is_native False, bic rho 0; amb.py). **Dead channel:** production native arrays carry 2066 zero samples at 960.18-961.0 nm (26 % of 7914; the 2026-09-22 report lists the "960-961 nm stub"). The panel loader trims them (fit_quality_panel.py:187-195) but analyze_spectrum does not, so the panel never sees production-shaped native input. Global-median consumers are dragged: global pitch 0.0955 raw vs 0.1278 trimmed. **+18 axis shift:** interior label error ≤ 6 pm, but the last 18 samples of UV and VIS are mislabelled by up to **1.12 nm and 1.55 nm** (the comment says < 1 nm), at exactly the 365/620 junctions the response estimator reads. The last 18 NIR samples are dropped. The output and summary carry no pitch or regime field. | **latent** (native raw, 0.1 nm), **blind** (panel) | One `ingest_grid(x, y)` in alibz that trims dead pads and zero fills, classifies the regime (native / export-1/30 / uniform-other) and returns it into the output and summary. Refuse or flag uniform grids that are not the vendor 1/30. Re-derive the Fe npz on the true per-segment axis (z300_opal_ingest path) instead of +18. (M) |
| G8 | minor_lines.py:268-274, 688-690 (shape clamps 0.9-1.15 × / 0.85-1.2 ×) | A/C | One spectrum-wide narrow-line σ/γ is the instrument profile in every segment | pre-2026-07 | all (NIR) | Covered by survey-finder F9. Measured here: the global seed FWHM is 0.139-0.167 nm, while the per-segment narrow-line median FWHM in NIR is 0.26-0.32 nm on both exports and natives (shape.py). Seeded and recovered NIR lines are capped at ~0.19 nm, about 60 % of the NIR profile, which biases their areas. | **biases** | Per-segment σ/γ medians (S) |
| G9 | pipeline.py:1163, 1277, 1312 vs minor_lines.py:676-677, profiles.py:156 | E/A | `segment_edges=(620.0,)` for seeding and match_and_scale; (365, 620) elsewhere | 620 step handling 2026-07 | native | Seeding pools UV and VIS for local noise segmentation and `ref_segments`, so a UV-only Fe scale may predict into VIS (minor_lines.py:347-349 allows it). Recovery and profiles segment at 365 as well, so two segment definitions coexist in one run. | **latent** | Pass the response edges and the noise/pitch edges separately (S) |
| G10 | pipeline.py:1005-1006, 1055-1056; peaky_indexer_v3.py:2053-2058 | A/E | Additive floor and Poisson term share units with the response-corrected amplitudes | 2026-07 WNNLS | weighted_solve (opt-in) | `amp_sigma` is divided by `seg_response`, but `amp_sigma_floor=_noise` (raw counts) and `|amp_corr|/gain` are not. In NIR the floor is 3.9× too large relative to the amplitudes, and the Poisson term is off by a factor of `resp`. | **latent** (default off) | Scale the floor and Poisson term per segment (S) |
| G11 | pipeline.py:466-512 | B/A | Coarse-composition bootstrap: top 60 peaks within 0.12 nm of gA ≥ 1e6 lines, ≥ 2 hits | 2026-09 registration | native | Bootstraps include U, Th, Ne, Co, Hg, Cl on pure Fe (fe_fresh: Fe, Co, U, Mn, Ar, Ne, Th). Used only to remove ambient anchors and select element lines (diagnostic). | **latent** | Exclude no-line and actinide elements from the bootstrap and scale tol to pitch (S) |
| G12 | docs/development_guide.md:630-633; reports/2026-09-22-native-grid-processing.md:20 | E | Docs: the export "is not a resampling", and "alibz has no 1/30 nm assumption in the analysis path" | previously reported (memory, export-grid-spline-resampled) | — | development_guide.md:630-633 still states it. The guide has no section on the native grid. The 2026-09-22 claim is falsified by G2/G6 and the finder gate. | **latent** | Correct the docs; add a "grid regimes" section (S) |
| G13 | alibz/peaky_indexer_v3.py.orig | D | — | wave-3 patch leftover (Nov 4 09:58) | — | 80 changed lines vs the working file (pre-outer-polish). Untracked and not imported. | **latent** | Delete (S) |

---------------------------------------------------------------------------------------------------

## 3. Checked and clean (or previously reported, now fixed)

- Response correction reaches every indexer pass: `_db_frame` feeds pass 1 (:1079), pass 2 (:1202) and deepening
  (:1329), detections use `fidx` built from `_db_frame`, and `_amp_sigma` divides by the same response (:1055).
  Previously reported 2026-07 ("never called"): **fixed at pipeline.py:1036**.
- Previously reported "area_sigma not SA-scaled": **fixed at pipeline.py:1436-1441**.
- `PeakyFinder.__new__` bypass: `__init__` sets only `data_dir`/`data` (peaky_finder.py:70-71). Benign but fragile.
- gas_detection windows are pitch-aware (gas_detection.py:108-118). The gas_calibration match tolerance is pitch-aware
  (gas_calibration.py:200). Background gas detection compares database-frame peaks with database lines, which is consistent.
- `resolve_bic_rho("auto")` classifies 1/30 → 0.92 and native/0.1/0.05 → 0 correctly (refinement.py:134-158). It is opt-in.
- `segment_pitch_array` excludes the dead channel and classifies native correctly with or without the stub (amb.py).
  Global-median-pitch B1 (2026-09-23 audit) is fixed in relative_profiles, session_calibration, peak_window_pca and
  wavelength_registration. It is **still open** at minor_lines.py:274, 684, profiles.py:451 and mw2_112.py:130 (the last is off the
  production path).
- mw2_112.py is not imported on the production path. triage runs only when `physical_triage != "off"`.
- CLI → analyze_directory → AnalysisConfig → `_analyze_file` (pipeline.py:1742-1774) passes all knobs, and the defaults agree
  across the four layers.
- manuscript_figures `build_thermo_set`/`measure_thermo`/`robust_boltzmann` (scripts/manuscript_figures.py:911-1008) are
  verbatim copies of cal/scripts/fe_plasma_analysis.py:392-520 (renames and docstrings only). The 420-620 nm range in
  `_f5_feI_fit` (:1020) differs from the root-cause report's recommended 365-900 nm set (a choice, not drift).
- bench/nist_forward.py:48-62 dev/dev1 path insert is guarded, and `dev/` is absent. bench/synth_cases.py does not use it.
- tests: test_roundtrip_synthetic.py renders binary_ca_mg on the 1/30 grid with loose bands (±0.30 fraction,
  T ±45 %). test_oracle_roundtrip.py bypasses the finder entirely (peak oracle, merge radius 0.25 × synthetic FWHM).
  Neither touches native. The only analyze_spectrum test on a native-like grid
  (tests/test_wavelength_registration.py:21-24, 318-345) asserts plumbing/schema only. **Native-grid inputs across
  tests + panel + fixtures: 1 plumbing test + 26 runs of one pure-Fe npz**, and there are no native multi-element items.
- manuscript/data/showcase.json: 5 export items (mw2_1000, jc×3, user_spec1) and 2 native (fe_fresh_d5p25, fe_reused_fail),
  labelled correctly with `grid`.

## 4. Unverified

- Whether the Z300 native API spectra carry the same 620 nm gain step as the vendor export (G3). No native-measured ratio exists.
  fe_fresh's raw flank ratio of 4.8 is one data point.
- Whether Pantheum trims the 960-961 nm stub before calling alibz (G7). I relied on the 2026-09-22 report's statement that native
  arrays are 7914 points with the stub.
- G6 VIS/NIR recovery after the finder-gate fix in the working tree was not re-run. The window-sample counts are computed from cached
  wmed. If the gate fix changes wmed, the counts move. At wmed 0.229 nm the NIR still fails.
- The G2 effect on composition: the registration only moves the NIR shift by ~tens of pm. The magnitude of the bias was not measured.
- G1 numbers come from panel JSONs that carry no git HEAD, so they cannot be tied to an exact tree.
- G4: whether a final re-optimisation would move T on fe_fresh was not tested (the root-cause report showed a forced 11.5 kK makes
  r² worse on the current table, so the gain depends on G6 and the finder fix first).
- The units of `amp` (area vs height) relative to `_noise` in the floor (G10) were not traced.

## Recommendation (one)

Make the grid regime a first-class, recorded input before any further A/B work. That means one `ingest_grid`
(trim pads, classify native / export-1/30 / other, record it), every window and gate in the glue in
instrument-FWHM and per-segment-pitch units (G6, G8, G2, G3 floor), and a native panel tier (G1) that
includes multi-element native spectra and a native-grid synthetic render with a known 620 nm step, plus
per-segment peak recall and `qc_status` as headline metrics. Rationale: G2, G3, G6 and G7 each have the same shape
as the exemplar: a constant tuned in one regime, silently wrong in the other, and invisible to the current panel
(G1). Fixing them one at a time without the regime record and native panel repeats the 2026-11-04 pattern of
adopting defaults that worsened the native Fe hot-basin count (23 → 25/26) while scoring as an improvement.

Strongest alternative: patch the individual gates now (G6 minor_lines/profiles, G2 sigma conversion,
G5 contest physics) and re-run the existing Fe panel. It is cheaper, but it lost because the existing panel
cannot tell whether the patches work. Its only native items are pure Fe, its Fe metric rewards the hot
single-stage basin, and its r² cannot see missing peaks.

Brief for follow-up (specialist): (a) G6 + G8 + G9 in minor_lines.py/profiles.py using
`segment_pitch_array`, coordinated with the finder gate patch. (b) G2 `instrument_sigma_px` from nm
on non-native grids. (c) G5 single physics config passed to detections' contest. Brief for worker: G1 panel
additions (native synthetic ChannelGrid from `tests/test_wavelength_registration.py:_native_grid`, per-segment
recall, qc headline, git HEAD + resolved config), G7 `ingest_grid`, G13 delete the `.orig`, G12 docs.
