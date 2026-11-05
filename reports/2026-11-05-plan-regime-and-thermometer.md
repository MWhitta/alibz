# Plan: Piece 1 (grid regime + native panel tier) and Piece 3 (thermometer-first temperature)

Date: 2026-11-05. Audience: the owner (to approve scope and the decision points in §6) and
the implementing agents (specialist for the pipeline changes, worker for the panel and
tests). Background: `reports/2026-11-05-pipeline-assumption-survey.md` §3.4 and §5;
`reports/2026-11-05-native-gate-fix.md`; `reports/2026-11-04-fe-hot-basin-root-cause.md`.

## 0. Goal and the one pass/fail criterion

Goal: the pipeline reports which sampling grid it analysed, the benchmark scores that grid,
and when a direct spectroscopic thermometer resolves, it sets the plasma temperature instead
of the whole-pattern amplitude fit.

Pass/fail, stated up front so the work cannot drift:

| check | today | pass |
|---|---|---|
| Fe standard (26 panel means + 2 native showcase runs): T within 10 ± 2.5 kK | 3/26 | ≥ 20/26 and both showcase runs |
| Fe standard: Fe element fraction ≥ 0.8 | ≈ 0/26 after the gate fix (mean 0.29) | ≥ 20/26 |
| export synthetic suite (5): median relative error, |T error| | 0.170 / 0.077 | not worse by more than 0.01 / 0.01 on the mean, no case worse by > 0.05 |
| export real samples (5): peak-table r², stage disagreement | 0.489 / 0.489 | not worse on the mean, ≤ 1/5 worse by > 0.02 |
| JChristensen 38-spectrum survey: Mn→Mg confound count, median r² | 6/38, 0.782 | not worse |
| native tier (new): qc_status pass rate, whole-spectrum r², per-segment peak counts | unknown | recorded; no regression once a baseline exists |
| every result carries `analysis["grid"]` and every panel JSON carries `git_head` and grid | no | yes |

If the Fe criterion fails with the thermometer box applied, the plan stops and the result is
reported as such; the box is the hypothesis under test.

## 1. Background in ten lines

The pipeline's outer search picks T and nₑ by minimising an unweighted amplitude cost whose
dependence on T is weak and biased by the model's approximations. On the pure-Fe standard the
minimum sits at 20.9 kK; the Fe I Boltzmann slope in the visible and the Fe I/Fe II stage-tie
both put it at 10.6–11.5 kK, and fixed-state re-solves at those temperatures give Fe 0.73–0.91
instead of 0.12. The direct thermometers are already implemented: the stage-tie inside the
indexer (`stage_consistency_cost`, votes only when two stages of an element survive) and the
line-selected Boltzmann fit in `scripts/manuscript_figures.py` (`build_thermo_set`,
`robust_boltzmann`, Fe-only, used for figure F5) and `cal/scripts/fe_plasma_analysis.py`.
Neither can override the amplitude cost today. Separately, nothing records which grid a run
saw, and the fit-quality panel's synthetic and real items are all exports, so native behaviour
is unscored. Piece 1 makes native behaviour visible; Piece 3 changes who decides T.

## 2. Piece 1 — grid regime record and a native panel tier

### 2.1 Regime record (package)
- New function `grid_regime(x, y=None)` in `alibz/utils/native_grid.py`, built on the existing
  `segment_pitch_array(x)` (returns `(pitch, inc, is_native)`). Returns a plain dict:
  `regime` ∈ {`native`, `export-spline`, `uniform`, `unknown`}; per segment (UV/VIS/NIR
  by the 365/620 nm edges) `n_samples`, `pitch_median_nm`, `pitch_p5_nm`, `pitch_p95_nm`;
  `uniform_pitch_nm` when the whole axis is uniform; `dead_channel` (samples past a > 5 nm
  gap, as `fit_quality_panel.load_fe` already detects); `x_min_nm`, `x_max_nm`.
  Classification: per-segment pitches differ → `native`; uniform 1/30 nm → `export-spline`;
  uniform other (e.g. 0.1 nm from `z300_fb_decode.py` default) → `uniform`; else `unknown`.
- `analyze_spectrum` stores it as `analysis["grid"]` right after the axis is validated
  (the native/export dispatch near `pipeline.py:797`), and the summary-row builder used by
  `qc_status` adds `grid_regime` and the three pitches. One `logging.info` line per run.
- The CLI (`alibz-analyze`) prints the regime with the run header. No behaviour change.

### 2.2 Panel changes (`scripts/fit_quality_panel.py`)
- `git_head` is `None` in every existing JSON (the panel runs from a copied tree without
  `.git`). Record it at launch on the launching machine and pass it through
  `--git-head`; also record `tree_sha` = sha256 over `alibz/*.py` so copied trees are
  identifiable without git.
- Each item records `grid` (from `grid_regime`) next to its metrics.
- New tier `--panel native` with these items:
  1. the 26 Fe means (existing loader; keep the approximate +18-sample axis caveat in the
     JSON as `axis: "approx+18"`), plus
  2. the two native showcase runs (`fe_fresh_d5p25`, `fe_reused_fail`) via
     `scripts/showcase_data.py`, and
  3. a synthetic case rendered on the true native pitch: add `ChannelGrid.z300_native()`
     to `alibz/synthetic.py` using the per-segment cubic calibration in
     `alibz/utils/native_grid.py` (`INSTRUMENT_CALIBRATION`), and rename the current
     `ChannelGrid.native()` to `ChannelGrid.export_spline()` with a deprecation alias so
     nothing breaks. Render `iron_rich_low_t` and `silicate_4el` on it.
  4. optional, if accessible on moissanite: five native rock shots from the Pantheum archive
     (`~/.local/state/pantheum/alibz/archive/...`), truth unknown, QC metrics only.
- Native-tier headline metrics: `qc_status` pass rate; whole-spectrum r² (model profile vs
  y − background over all live samples, not the peak-table r²); final-table peak counts per
  segment; `|T − T_thermo|` when a thermometer resolves (Piece 3 adds `T_thermo`);
  `ne_at_bound`; for the Fe items `fe_fraction` and `|T − 10 kK|`; for the synthetic native
  cases the usual recovery metrics.
- Verdict rule extended: no default adoption unless the native tier is not worse.

### 2.3 Tests
- `tests/test_native_grid.py`: `grid_regime` on the trimmed native fixture
  (`tests/fixtures/z300_fe_native_trimmed.json`) → `native` with the three pitches within
  5 % of 0.089/0.129/0.179; on the trimmed export fixture → `export-spline`; on a synthetic
  0.1 nm axis → `uniform`.
- `tests/test_fit_quality_panel.py`: the JSON schema carries `git_head`, `tree_sha`, per-item
  `grid`; `--panel native` lists the expected item ids.
- `tests/test_synthetic_generator.py`: `ChannelGrid.z300_native()` has three pitch regimes
  and the deprecation alias still returns the 1/30 nm grid.

### 2.4 Effort and agents
Worker (Sonnet): 2.1 + 2.3 (half a day). Worker: 2.2 (half a day, plus one native baseline
run on moissanite at production settings, ≈ 1 h). Report file:
`reports/2026-11-06-piece1-regime-record.md`. Decision needed on §6 item 3 (re-derive the
Fe means) before the baseline run, because it changes the Fe items.

## 3. Piece 3 — thermometer-first temperature

### 3.1 The thermometer module (`alibz/thermometer.py`, new)
Port and generalise the cal-campaign method that produced F5:
- Inputs: `x`, `y`, `background`, the registration shift (`shift_at`), segment edges, the
  `Database`, a candidate species list, and the local noise array.
- Candidate species: elements with ≥ 15 database lines of one stage inside one segment with
  gA above the support floor; in practice Fe I, Ti I, Cr I, Ni I, V I, Mn I, Ca I, and Fe II
  in the UV. The coarse-composition bootstrap (`pipeline.py:466–512`) proposes elements; the
  thermometer only runs species the bootstrap found.
- Per species and segment: select the top-N predicted lines at a reference 10 kK with an
  isolation test against the rest of the db (as `build_thermo_set` does), measure side-band
  areas at the registered db positions (not from the peak table), fit ln(I λ / gA) vs E_k by
  robust least squares with iterative clipping; repeat at clips 1.6, 2.5 and 3.5 σ.
- Resolution criterion (unchanged from the campaign): n ≥ 8, E_k span ≥ 2.4 eV, r² ≥ 0.5,
  σ_T/T < 0.5. Systematic uncertainty = half the spread of T across the three clips, floored
  at 1 000 K. Exclude lines with E_i below `ei_min` (resonance-connected, self-absorption
  prone), default 0.8 eV as in F5.
- Combine resolved estimates across species and segments by inverse variance
  (statistical + systematic in quadrature); never combine across segments within one fit
  (the cross-segment response is uncalibrated).
- Output `ThermometerResult`: `T`, `sigma_stat`, `sigma_sys`, `resolved`, per-fit records
  (species, segment, n, span, r², T, lines used with λ, E_k, area, SNR, clip survivors).
  Stored as `analysis["thermometer"]` and captured in `capture_stages`.

### 3.2 Pipeline integration (`alibz/pipeline.py`)
- Run after background, registration and segment response, before pass 1.
- New `AnalysisConfig` / `analyze_spectrum` / CLI options:
  `temperature_source` ∈ {`auto` (default after scoring), `search` (today's behaviour),
  `thermometer`}, `thermometer_box_k` (default 2.0), `thermometer_box_min_k` (default
  1 500 K).
- When `resolved` and source is `auto` or `thermometer`:
  - T box = [T − max(k·σ_sys, box_min), T + max(k·σ_sys, box_min)], clipped to
    `PLASMA_T_BOUNDS`; passed as `T_bounds` to `run()` for pass 1 and pass 2 and to the
    deepening re-solves (`solve_at` already takes a fixed T; keep the pass-2 optimum).
  - `temperature_init = T` for pass 1, pass 2 and the deepening builds, so the candidate
    set and the frozen Stark/SA references are built at the right state (survey I2).
  - `result.temperature_source = "thermometer"`, and the box is recorded.
- When not resolved: unchanged behaviour; set `temperature_source = "search"`; if in
  addition no element has two supported stages after pass 2, add the flag
  `thermometer-silent` to `qc_reasons` (survey I3).
- The stage-tie thermometer keeps working inside the box.
- Not in this piece: a soft prior term. The outer cost is an unweighted sum of squares in
  counts², so a prior cannot be scaled sensibly until the cost is a χ² (survey I8). A hard
  box is the honest version today.

### 3.3 Tests
- Unit: the thermometer recovers T within 10 % on `iron_rich_low_t` rendered on
  `ChannelGrid.z300_native()` (Piece 1) and on the export grid; on the trimmed native Fe
  fixture it resolves between 9 and 13 kK; on a Ca/Mg binary it reports `resolved=False`
  without error.
- Pipeline: with `temperature_source="thermometer"` on the native Fe fixture the result's T
  lies inside the box and `temperature_source` is recorded; with `"search"` the result is
  byte-identical to today (regression guard, same pattern as `capture_stages`).
- Export regression: user_spec1 and MW2-112 blind tables unchanged (the thermometer does not
  touch the finder); results may change only through the box.

### 3.4 Scoring
Production settings on moissanite, both trees from the same base: `search` (= today) vs
`auto`. Panels: synthetic, real, fe, native (Piece 1), plus the JChristensen 38-spectrum CLI
survey (`reports/jchristensen-20261104/` is the current product). Apply the §0 table. Report
per Fe run: thermometer T, box, final T, Fe fraction, Fe I/Fe II assignment counts. Expected
from the fixed-state re-solves: Fe 0.75–0.9 at 10.6–11.5 kK with residual Na 0.2; if Na
stays above 0.3 the remaining artefact is the unweighted cost inside the box and is reported,
not patched here.

### 3.5 Risks and mitigations
- Self-absorbed strong lines tilt the slope cold: `ei_min` cut, robust clip, clip-sensitivity
  as σ_sys. Report the three-clip spread per run.
- Within-segment response tilt: per-segment fits only; the Fe I VIS fit is the least exposed.
- Rocks with only Ca/Mg/Si/Al/K/Na lines: no thermometer resolves → unchanged behaviour; the
  flag makes this visible in the panel.
- Wrong species from the bootstrap: the r² and n gates reject it; log the rejection.
- Registration error: positions are registered; the side-band window tolerates ± one local
  pixel.
- The box hides a model problem instead of fixing it: by design for this piece; the amplitude
  cost's hot preference stays on the list (survey §5 steps 2, 3c, 3d).
- Cost: a few hundred ms per spectrum.

### 3.6 Effort and agents
Specialist (Opus 4.8): 3.1 + 3.2 (one to two days). Worker: 3.3 tests, 3.4 runs. Coordinator
verifies the diff, re-runs the fixed-state harness on the new cache, checks the §0 table
against the JSONs. Report files: `reports/2026-11-0X-piece3-thermometer.md`,
`reports/fit-panel/thermo-{search,auto}-prod.json`.

## 4. Sequencing and deliverables

1. Piece 1 (2.1–2.3), native baseline run, `reports/2026-11-06-piece1-regime-record.md`.
2. Piece 3 (3.1–3.3), then 3.4 scored against that baseline.
3. If §0 passes: adopt `temperature_source="auto"` as the pipeline/CLI default (DECISIONS
   entry), re-derive the F5 caption and 07-results Fe prose from the new cache, update
   `docs/development_guide.md` (grid regimes section; remove the "not a resampling" claim).
4. If it fails: record which criterion failed and the per-run thermometer/box/T table; the
   next lever is survey §5 step 3b–3d.

Deliverable checklist for each piece: diff under `reports/patches/`, report with file:line of
every change, test counts verbatim, panel JSONs copied local, STATUS head updated, nothing
committed by agents (owner commits).

## 5. What this plan does not do

Piece 2 (one per-segment line-width object for the 12/15/9/5-sample gates in seeding,
refinement and profiles), the Saha lookup-by-label fix, the response measurement floor, the
confounder-contest physics, the seeder convention, kernel-at-bound, nₑ-at-bound. Each is in
survey §5 with an owner and a cost. Piece 3 should still move the Fe standard without them,
because the visible Fe I peaks already reach the table after the gate fix.

## 6. Decisions needed from the owner

1. Box width: `k = 2` on the systematic σ with a 1 500 K floor (recommended), or a fixed
   ± 2 000 K.
2. Default after scoring: `temperature_source="auto"` for the pipeline and CLI (recommended),
   or opt-in only.
3. Fe panel axis: keep the approximate +18-sample npz for this round (fast) or re-derive the
   26 means from the archived native shots first (right; adds about a day if the shots are on
   moissanite). Recommended: keep for Piece 1's baseline, re-derive in parallel, and re-score
   once.
4. Whether the thermometer also applies on export data when it resolves (recommended: yes;
   that is how the manuscript's showcase exports would be affected).

## Appendix: fixed-state re-solve recipe

Rebuild the indexer as `analyze_spectrum` does for the deepening re-solve: database
`Database("db")`, `SahaBoltzmann` via `_get_sb("db", "barklem2016", "debye")`; peak frame =
final `sorted_parameter_array` with the registration shift removed and
`correct_segment_response(edges=(620.0,))` applied; `amp_sigma` from
`estimate_peak_uncertainties(...)[:, 0]` divided by the segment response; `amp_sigma_floor`
= 1.4826·MAD of y − background; `overlap_mode="conserve"`, `overlap_kernel="median"`,
`weighted_solve=False`, `stage_consistency_weight=STAGE_CONSISTENCY_WEIGHT`, `ne_prior`
from the H-alpha init; then `build_candidate_matrix(sa_doublets=True, ...)` and
`solve_at(T, log_ne, sigma, gamma)` with the pipeline's kernel widths. Port this into
`scripts/resolve_at_state.py` as part of Piece 3 so the owner can run it in one command.
