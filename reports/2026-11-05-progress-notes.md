# Progress notes, 2026-11-05

Audience: the repository owner, and any session or agent resuming this work. Everything
below is in the repository; nothing lives only in a chat transcript. Branch
`fix/z300-pixel-offset`, pushed to origin at commits 7309e76, 77ffdb9, 3216485 (+ the commit
that adds this file and the plan).

## 1. Where things stand in one paragraph

The alibz pipeline was built and tuned on the vendor's 0.0333 nm export grid. Since
September 2026 production data arrive on the native Z300 pixel grid. Many rules written as
"number of samples" silently changed meaning, the most damaging one being a 5-sample window
gate in the peak finder that discarded most native detections. That gate is now fixed and
export behaviour is unchanged, but the pure-Fe standard still solves to a wrong, hot
temperature (20.9 kK) because the pipeline chooses temperature by whole-pattern amplitude fit,
and that fit is a poor thermometer. The two direct thermometers already in the code (Fe I
Boltzmann slope in the visible, Fe I/Fe II stage balance) agree at 10.6 to 11.5 kK. The next
step is to make the pipeline know which grid it is on and test on it (Piece 1), and to let the
direct thermometers set the temperature when they resolve (Piece 3). The plan is in
`reports/2026-11-05-plan-regime-and-thermometer.md`.

## 2. Timeline of the last three days

| date | work | evidence |
|---|---|---|
| 2026-11-03 | fit-quality panel built (deterministic, matching-code baseline); literature A/B wave 1 (BO/BIC, FDR, partition+IPD, sparse/robust solve); manuscript initialised; literature search, Crossref verification, lra fixes | reports/2026-11-03-* |
| 2026-11-04 | wave 2: emission-conserving design matrix; defaults adopted (conserve / barklem2016 / debye); two unflagged default changes found by bisection and gated OFF; wave 3 (width kernel, outer polish, two-T thermometer) scored and NOT adopted; stage capture; manuscript figures F1–F5 from real data; Fe hot-basin root cause found (the finder gate) | reports/2026-11-04-*, DECISIONS.md 2026-11-04 |
| 2026-11-05 | finder gate fixed (export byte-identical, suite 695 green) and Fe panel re-run; three-part pipeline assumption survey (finder, indexer/physics, glue/scoring) verified and synthesised; everything committed and pushed | reports/2026-11-05-*, DECISIONS.md 2026-11-05 |

## 3. State of each thread

### 3.1 Pipeline defaults (A/B round) — closed, committed (7309e76)
Adopted at the pipeline/CLI level: `overlap_mode="conserve"`, `partition_source="barklem2016"`,
`ipd="debye"`. Library class defaults stay legacy so round-trip tests keep their meaning.
Everything else from waves 1–3 is behind OFF flags. Caveat recorded in the survey: the panel
that produced these verdicts scores only export-grid synthetic and real items plus 26 Fe
means on an approximate axis, so the verdicts are provisional for native data (survey G1).

### 3.2 Manuscript — drafted, committed (77ffdb9), two sections stale
`manuscript/` builds (54 pp at last check). Figures F1–F5 come from the production chain via
`scripts/manuscript_figures.py` with `capture_stages=True`; the figure cache is regenerable and
gitignored. Stale after today: the F5 caption and the Fe paragraph in
`manuscript/sections/07-results.tex` still describe the Fe II-only 22.6 kK solve. The F5
figure files were re-rendered from the post-fix cache (now 20.9 kK, Na-dominated), so figure
and text disagree until the solve-side fix lands. Do not re-derive the prose before Piece 3.

### 3.3 Native-grid finder gate — fixed, committed (3216485)
`alibz/peaky_finder.py`: narrow (≤ 5-sample) windows now seed directly with the FWHM floored
at the instrument line width; the per-channel width filter tests the combined Voigt width on
native grids. Both gated on `segment_pitch_array(x).is_native`. Verified: export blind tables
byte-identical on user_spec1 and MW2-112; synthetic and real panels unchanged on every
metric; `tests/test_native_gate.py` 4/4; full suite 695 passed / 10 skipped / 2 xfailed.
Patch kept at `reports/patches/2026-11-05-native-gate.patch`. Panel JSONs
`reports/fit-panel/gate-{base,fix}-prod.json`.

### 3.4 The Fe standard and the temperature problem — diagnosed, not fixed
Post-fix fe_fresh_d5p25: 271 peaks (151/87/33 UV/VIS/NIR), Fe I 140 assigned, Fe II 108,
T 20 929 K, r² 0.42, fractions Na 0.81 / Fe 0.12. Fixed-state re-solves on the same table
(reports/2026-11-05-pipeline-assumption-survey.md §3.4):

| state | r² | Fe | Na | Fe stage disagreement |
|---|---|---|---|---|
| 20.9 kK (pipeline) | 0.42 | 0.12 | 0.81 | 0.97 |
| 11.5 kK | 0.32 | 0.73 | 0.25 | 0.11 |
| 10.6 kK | 0.30 | 0.76 | 0.22 | 0.05 |
| 8.0 kK | 0.25 | 0.91 | 0.06 | 0.91 |

Reading: Na is a Saha artefact of the hot state (the same four Na I peaks at every T). The
stage-tie minimum and the VIS Fe I Boltzmann thermometer (10.6 ± 0.4 kK on the 16-line
365–900 nm set; 11.5 kK on the F5 10-line set; 7.7–11.5 kK across clip choices) agree; the
unweighted amplitude cost outvotes them. Switching the evidence prefilter off makes it worse
(66 junk species). This is the measured blocker for Piece 3.

### 3.5 Pipeline assumption survey — done, verified, committed (3216485)
Synthesis `reports/2026-11-05-pipeline-assumption-survey.md` (§3 findings with a V/M column:
V = coordinator re-read the lines and/or re-measured; M = agent-measured, lines verified).
Parts: `-survey-finder.md` (18 findings), `-survey-indexer-physics.md` (16),
`-survey-pipeline-glue.md` (13 + a call-chain inventory of `analyze_spectrum`). Classes:
sample-count constants, dataset-tuned constants, hypotheses pruned before the data vote,
physics conventions, dual code paths, scoring blind spots. Fix order in §5.

Independent bugs found and NOT fixed:
- Saha ion fraction read by column position while the chain is built from the stage labels
  present in the line db (`peaky_indexer_v3.py:451`): Pr II 0.0013, Tb II 0.0008, Gd I 0.72
  chained to a hydrogen-like ion, H never ionises; 11 elements have no stage-I lines, 59 carry
  stage labels > 30. Rare earths are affected.
- 620 nm segment response is never measured: all 7 showcase runs use the corpus fallback 3.9
  because the significance floor compares 8-nm flank medians with a whole-spectrum noise.
- Confounder-contest indexer (`detections.py:359`) is built with library physics (line-sum
  partition, no IPD), not the adopted defaults.
- `evidence_max_missing_lines=6`: a species is dropped when more than 6 of its 20 strongest
  predicted lines have no peak in the table (never checks the raw spectrum).
- Seed/recovery/profile windows need ≥ 12 / ≥ 9 samples (export-era), so on native data the
  whole seeding chain adds 0 VIS and 0 NIR peaks.
- Kernel widths sit at their bounds in 7/7 cached runs; nₑ sits at the H-alpha box edge in
  6/7 cached and 16–18/26 panel Fe runs.
- Registration never applies on exports (its sub-pixel fitter assumes native pixel widths).

### 3.6 Infrastructure and data provenance
- Panel: `scripts/fit_quality_panel.py` on moissanite (`~/alibz`, 32 workers, ≈ 11 min fast /
  ≈ 50 min production per variant). Remote trees `~/alibz-gate-base` and `~/alibz-gate`
  still exist (JSONs verified local; delete when convenient).
- Fe panel input `data/triage-real/fe_aesar_9998_means_20260922.npz` carries an approximate
  +18-sample axis shift (predates the −18 px fix). Re-deriving the 26 means from the archived
  native shots is an open item (needed for a trustworthy native tier).
- Showcase data: `manuscript/data/showcase.json` + `scripts/showcase_data.py`; staged copies
  in `data/showcase/` (gitignored). The Corpus_One drive was not attached on 2026-11-04.
- Caches: `manuscript/figures/cache/*.pkl` hold full post-fix runs with stage captures; the
  scratch harness `fixed_t_resolve.py` (session scratchpad, not in the repo) rebuilds the
  indexer from a cache for fixed-state re-solves. Worth porting into `scripts/` (see plan).
- Untracked leftovers: `alibz/peaky_indexer_v3.py.orig` (delete), `reports/fit-panel/ab-done.flag`.

## 4. What is verified and what is not

Verified by the coordinator: every line citation in the synthesis §3 marked V; the fallback
3.9 on 7/7 runs; the Saha lookup results for Pr/Tb/Gd/Zr/H; kernel-at-bound and nₑ-at-bound
counts; panel JSON hot-Fe counts (23/26 in both baseline and adopted defaults) and
`qc_status` fail counts (31/31 → 30/31); the gate fix's export byte-identity (via the
unchanged panel) and the new test; the fixed-state re-solve table above.

Agent-measured only (lines verified, numbers not re-run): shoulder-threshold σ ratios,
noise-estimator inflation factors, refinement/profile skip counts, overlap-window FWHM
fractions, ridge/pseudo cost fractions, registration 0-vs-13 ambient lines on user_spec1,
the 365 nm step on two native shots.

Not run anywhere: an end-to-end pipeline with any indexer-side change; any native rock
spectrum other than the Fe standards; the full suite after today's report-only edits
(code unchanged since the 695-green run).

## 5. Open items and owner decisions

1. Piece 1 + Piece 3 (plan file): approve scope, box width, and whether to re-derive the Fe
   means from the native archive as part of Piece 1.
2. Piece 2 (one per-segment line-width object for all sample-count gates) after Piece 1.
3. Saha lookup by label + db stage coverage audit (worker, small).
4. Response measurement floor and the 365 nm step on native data (needs ≥ 10 native shots).
5. Confounder contest physics; seeder convention; fit_inspector retirement or alignment.
6. Manuscript: re-derive F5 caption and 07-results Fe prose after Piece 3; the "thermometer
   sets T, fit sets composition" story replaces the "two-thermometer discrepancy".
7. Housekeeping: delete `.orig`, moissanite trees, `ab-done.flag`; correct
   docs/development_guide.md:630–633 ("not a resampling") and add a grid-regimes section.

## 6. How to resume

Read, in order: `reports/STATUS.md` (head block), this file, the plan,
`reports/2026-11-05-pipeline-assumption-survey.md` §3.4 and §5, then
`DECISIONS.md` 2026-11-05. For a fixed-state experiment on the Fe table, regenerate the
cache (`python -m scripts.manuscript_figures analyze --keys fe_fresh_d5p25`) and rebuild the
indexer as `analyze_spectrum` does for the deepening re-solve (the recipe is in
reports/2026-11-04-fe-hot-basin-root-cause.md §2 and the plan's appendix). Panel runs: see
the memory note "fit-quality-panel" or `scripts/fit_quality_panel.py --help`.
