# The stage-consistency thermometer

Date: 2026-09-21. Provider: Claude subscription (no switch). Follows
`reports/2026-09-20-alibz-review.md` §2.4, which left the plasma temperature
"weakly determined by the amplitude objective" as the next physics step.

## 1. The problem, measured

The concentration solve keeps every (element, ion stage) column an independent
unknown. Each stage's lines then estimate the element's density on their own,
and the amplitude objective is nearly flat in the plasma state: the Saha stage
fractions only scale the columns, a free per-stage concentration absorbs any
scale, so the outer search sees T only through within-stage Boltzmann ratios
and nₑ hardly at all. `scripts/thermometer_scan.py` tabulates the data misfit
over a 15 × 6 (T, log nₑ) grid at the fitted kernel widths, on the pipeline's
final peak table:

| spectrum | data misfit, 5.5–7 kK | at 10 kK | flat in nₑ? |
|---|---:|---:|---|
| Ca/Mg synthetic (truth 10 kK, 17.0) | 1.178–1.182e6 | 1.189e6 (+0.6 %) | identical to 4 digits, log nₑ 14–18 |
| silicate synthetic (truth 10 kK, 17.0) | 3.964–3.970e5 | 3.979e5 (+0.4 %) | identical |
| Profile Builder export (`user_spec1`) | 1.054e6 | 1.202e6 (+14 %) | identical |
| REE_44 | 6.44e6 | 6.84e6 (+6 %) | identical |
| scan9x9 | 7.65e6 | 1.175e7 (+54 %) | identical |

On both synthetic scenes the data prefers 5.5–7 kK over the true 10 kK, by
less than 1 %, and the composition moves with T (Ca 0.76 / Mg 0.24 at the
7 kK state the search chose, 0.65 / 0.35 at the truth node, truth 0.6 / 0.4).
The cold preference is systematic: it is what self-absorbed or under-measured
strong lines look like to a Boltzmann fit.

## 2. What the thermometer is

`PeakyIndexerV3._stage_tie_cost` (alibz/peaky_indexer_v3.py). For each element
with at least two ion stages that have observed support, take the element's
own residual (observations minus every other species' fitted contribution)
and fit it twice: FREE, one non-negative density per stage, and TIED, one
density through the summed stage columns (the estimate `_aggregate_elements`
already reports). The thermometer charges

    sum over elements of max(0, misfit_tied − misfit_free)

in the same squared units as the data misfit. It is exactly the misfit the
data would pay if the Saha stage ratios were enforced, so the default weight
`STAGE_CONSISTENCY_WEIGHT = 1.0` trusts the physics as much as the data and
no more; `0` disables it (`--stage-consistency-weight` on the CLI,
`stage_consistency_weight` on `analyze_spectrum`). It enters only the outer
search over (T, nₑ, σ, γ) (`_outer_objective`, and the BIC basin scoring in
grid mode). The reported composition is still the free solve at the chosen
state, with its `stage_disagreement` diagnostic (the phase-heterogeneity
proxy) unchanged; per-element terms are in `convergence_info`
(`stage_tie_cost`, `stage_tie_by_element`).

Two details that matter:

- The free fit uses the design matrix BEFORE the per-trial relative-emissivity
  gate (`_last_A_all`). Without that, a stage the gate removed at a cold trial
  state simply did not vote, the thermometer was zero by absence over most of
  the cold/low-nₑ region, and one spectrum (REE_01) was pushed to the 4000 K
  floor where it was silent. With the ungated columns the free fit can only
  explain the observed ion lines with an enormous density on the tiny column,
  the tied fit cannot explain them at all, and that difference is the charge.
- Stages with no observed support (all-zero column) carry no measurement and
  do not vote, so a feldspar-like spectrum with only neutral lines (the
  `feldspar_k_na` synthetic) gets no thermometer at all; there T remains a
  data-only choice.

## 3. What it does on the (T, nₑ) surface

Tie cost (same grids as above; rows T in K, columns log nₑ 14 … 19):

Ca/Mg synthetic, truth (10 000, 17.0):

| T | 14 | 15 | 16 | 17 | 18 | 19 |
|---:|---:|---:|---:|---:|---:|---:|
| 5500 | 1.3e5 | 8.4e5 | 2.8e6 | 3.1e6 | 3.1e6 | 3.1e6 |
| 7000 | 3.2e4 | **3.7e3** | 4.1e5 | 2.0e6 | 3.1e6 | 3.1e6 |
| 8500 | 3.7e4 | 3.4e4 | **7.6e3** | 3.3e5 | 2.2e6 | 3.1e6 |
| 10000 | 3.8e4 | 3.7e4 | 3.1e4 | **3.6e2** | 6.2e5 | 2.8e6 |
| 13000 | 3.8e4 | 3.8e4 | 3.8e4 | 3.2e4 | **1.7e3** | 6.4e5 |
| 17500 | 3.7e4 | 3.7e4 | 3.7e4 | 3.7e4 | 3.1e4 | **1.0e3** |

The global minimum is the truth node (360, against 2.6e6 at the state the
amplitude objective alone had chosen), and the free composition there is Ca
0.65 / Mg 0.35 with stage disagreement 0.05 / 0.07. Around it runs the Saha
degeneracy ridge (bold): a hotter plasma at higher nₑ gives the same ion
fractions, and two-element thermometry separates T from nₑ only weakly (the
silicate synthetic's ridge is flat to within 2×: 700 at (7000, 15), 755 at
(8500, 16), 690 at the truth, 356 at (13000, 18)). Along the ridge the data
term prefers the cold end, so with nothing else the combined objective settles
at (7 kK, log nₑ 15) on every spectrum scanned.

That is why nₑ must be bounded. The amplitude objective is exactly flat in nₑ
at fixed T, so a bound costs no data fidelity. When an H-alpha Stark width is
present its bounds already apply (scan9x9: the search then lands at 10.2 kK).
Otherwise the pipeline now bounds the search by the default prior
(`NE_PRIOR_DEFAULT = (17.0, 0.5)` dex, ± `NE_PRIOR_BOUND_SIGMAS = 2`, i.e.
log nₑ 16–18, typical LIBS plasmas in the emission window); the QC
`electron-density-at-bound` flag tests the bounds actually used
(`analysis["ne_bounds"]`, source `"halpha"` or `"prior"`). Read off the
surfaces, the expected states inside those bounds are: Ca/Mg synthetic →
(10 000, 17) (the truth); silicate synthetic → (8 500, 16) over the truth by
0.1 %; `user_spec1` → (7 000, 16); REE_44 → (10 000, 18).

## 4. Pipeline results

`analyze_spectrum` defaults before (`ee475e4`: no thermometer, cold-start GP,
log nₑ free in 14–19) and after (`5b634e3`: thermometer weight 1, grid-seeded
search, nₑ bounded by H-alpha or the 1σ prior box). Laptop, `ALIBZ_WORKERS=3`,
three batches sharing the machine, so the times are relative only. Synthetic
cases use the round-trip test's fast settings (`n_calls=8`, seed 11).

| spectrum | before: T / log nₑ, r², top fractions | after: T / log nₑ, r², top fractions |
|---|---|---|
| Ca/Mg synthetic (truth 10 kK, 17.0; Ca 0.6 / Mg 0.4) | 7000 / 17.25, 0.72, Ca 0.76 Mg 0.24 (median rel. err. 0.33) | 8908 / 16.50 (prior edge), 0.63, Ca 0.52 Mg 0.48 (0.17) |
| silicate synthetic (10 kK; Si 0.4 Ca 0.3 Fe 0.2 Mg 0.1) | 7000 / 17.25, 0.59, Si 0.71 Ca 0.14 Fe 0.12 Mg 0.02 (0.66) | 9043 / 16.50, 0.60, Si 0.51 Fe 0.33 Ca 0.11 Mg 0.06 (0.53) |
| feldspar synthetic (9 kK; Si 0.5 Al 0.2 K 0.2 Na 0.1) | 7000 / 17.25, 0.75, Si 0.86 Al 0.10 K 0.03 Na 0.01 (0.80) | 9242 / 16.88, 0.85, Si 0.42 K 0.32 Al 0.18 Na 0.08 (0.18) |
| Profile Builder export (`user_spec1`, H-alpha bounds) | 6138 / 16.79, 0.892, Si 0.82 K 0.12 Al 0.04 | 7527 / 16.31 (H-alpha edge), 0.900, K 0.74 Si 0.12 Rb 0.08 Al 0.03 |
| scan9x9 (H-alpha bounds) | 5074 / 17.39, 0.875, Si 0.80 Al 0.12 Fe 0.05 | 8599 / 17.03, 0.779, K 0.30 Al 0.25 Si 0.12 Na 0.09 |
| REE_44 | 4639 / 16.71, 0.711, Fe 0.47 Al 0.41 Mn 0.06 | 7317 / 16.61, 0.716, K 0.23 Fe 0.20 Si 0.20 Al 0.19 |
| REE_01 (r² < 0 before and after) | 5325 / 17.09, −0.33, Si 0.61 Al 0.31 | 7687 / 16.99, −0.29, C 0.95 |
| argon_noAr (r² < 0.3 before and after) | 12191 / 18.31, 0.28, K 0.40 Na 0.28 Ca 0.28 | 8563 / 17.50, 0.26, Ca 0.28 K 0.25 C 0.22 Na 0.17 |

What the table says:

- On every synthetic scene the plasma temperature moves from the 7 kK the
  cold-start search picked to within 10 % of the truth, and the composition
  error falls (median relative error 0.33 → 0.17, 0.66 → 0.53, 0.80 → 0.18).
  The feldspar scene has no two-stage element, so its whole improvement is
  the grid-seeded search; the Ca/Mg scene's is the thermometer plus the nₑ
  box (with the 2σ box it settled at 8242 K / 16.06 with Mg 0.53 / Ca 0.47).
- Separating the effects: bounding nₑ alone (thermometer off) reproduced the
  before-state on all eight spectra to the third decimal, as the flat nₑ
  profile predicts. Thermometer on with the cold-start GP landed off the
  ridge (13 362 K / 16.20, Ca 0.89) on the Ca/Mg scene; the grid seeds fix
  that.
- On the real samples the fitted temperature rises from 4.6–6.1 kK to
  7.3–8.6 kK and the stage-tie cost falls by 10–60× (scan9x9 3.8e7 → 6.6e5,
  REE_44 1.5e7 → 9.2e4, `user_spec1` 6.2e4 → 9.0e3). r² is unchanged or
  slightly higher except on scan9x9 (0.875 → 0.779): at the cold state its
  data misfit was 5× smaller than the misfit the Saha ratios would add, i.e.
  that fit was only good because the stages were free.
- The real samples all come out potassium-rich. K's support is the
  self-absorbed 766/770 nm resonance doublet (QC `dominant-weak-shape`) and
  the NIR amplitudes on two of them rest on the 620 nm detector-response
  FALLBACK (QC `detector-response-fallback`); the K fraction at a given T is
  therefore limited by those two things, not by the thermometer, and the QC
  flags say so. The synthetic scenes, which have neither problem, are the
  evidence that the state selection itself is right.
- Round-trip suite: 314 passed, 3 skipped (234 s), including the fast
  Ca/Mg gate at the new defaults.

Deployed on Moissanite (engine clone at `5b634e3`, portal config
`options.search` moved from `gp` to `grid`, services restarted, `/health`
ok): the reported export runs in 112 s (was 60 s at `5ae2de1`; the two
grid scans add ~105 solves per pass on a core ~5× slower than the laptop)
and gives K 0.75 / Si 0.12 / Rb 0.09 at 7.6 kK, r² 0.894, QC fail
(`electron-density-at-bound` at the H-alpha edge, `dominant-weak-shape`,
`detector-response-fallback`).

## 5. Files

- `alibz/peaky_indexer_v3.py`: `STAGE_CONSISTENCY_WEIGHT`, `_stage_tie_cost`,
  `_stage_consistency_cost`, `_last_A_all`, constructor parameter
  `stage_consistency_weight`, convergence-info keys.
- `alibz/pipeline.py`: `stage_consistency_weight` threaded through
  `AnalysisConfig`, `analyze_spectrum`, `analyze_directory` and the three
  indexer constructions; `NE_PRIOR_DEFAULT`, `NE_PRIOR_BOUND_SIGMAS`; QC uses
  the bounds actually searched.
- `alibz/cli.py`: `--stage-consistency-weight`.
- `scripts/thermometer_scan.py`: the (T, nₑ) surface scan used here
  (`--save` writes the surfaces as `.npz`).
- `tests/test_peaky_indexer.py::TestStageConsistencyThermometer`.

## What remains unverified

- Ground truth for the real spectra: `user_spec1`, scan9x9 and REE_44 have no
  reference composition; only the synthetic scenes test correctness.
- The runtime cost of the grid seeds on Moissanite (112 s vs 60 s on the
  reported export) was measured once, on an otherwise idle host; a smaller
  grid (11 × 5) would cut it and has not been tried.
- The `iron_rich_low_t` synthetic case (7 kK, the low-T thermometer test)
  cannot be rendered: `bench/synth_cases.py` builds stage abundances that sum
  to 0.99999998884 and `alibz/synthetic.py` rejects them (pre-existing).
