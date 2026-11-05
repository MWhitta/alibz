# Wave-2: overlap-conserving design matrix (`overlap_mode`)

Date 2026-11-03. Agent: specialist (Opus 4.8). Provider: Claude subscription (no switch).
`.venv/bin/python`, db `db`. Builds on the airtight cause in
`reports/2026-11-03-debug-composition-bias-at-truth.md`. The fit-quality panel was NOT run
(coordinator runs it on moissanite).

## Option

`overlap_mode: str = "legacy"` on `PeakyIndexerV3`, plumbed through `analyze_spectrum`,
`AnalysisConfig`, `analyze_directory`, the directory driver, every indexer construction
(pass-1, pass-2, deepening `solve_at`), the detection contest, the CLI, and `fit_inspector.py`.
Values: `"legacy"` (historic, non-conserving), `"conserve"` (cap each line's summed overlap at
`min(1, Σ_i raw)` — divide by the raw column sum only when it exceeds 1; never scale up),
`"nearest"` (hard-assign each database line to its single best-matching peak). Default `"legacy"`
→ byte-identical to today (test 6 proves it against a hand-rolled raw map).

## Changes (file:line)

peaky_indexer_v3.py
- :613 `__init__` gains `overlap_mode="legacy"`; :736-737 validate + store `self._overlap_mode`.
- :1002 new `_apply_overlap_mode(rows, cols, vals, n_cols)` — the shared conserve/nearest kernel,
  docstring carries the replication numbers (Ca 1.04 / Mg 1.97 binary; Ca 2.24 / Mg 4.65 /
  Si 4.98 / Fe 5.27 silicate).
- :1058 `_build_overlap_map` gains `apply_mode=False`; :1098-1100 applies the kernel when set;
  `len(rows)` guard replaces `if rows:` (identical matrix).
- :1142 `_build_observed_overlap_map` always applies the kernel (the resting per-fitted-width map).
- :2627 `_rebuild_overlap` calls `_build_overlap_map(self._obs_wl, σ, γ, apply_mode=True)` for the
  SOLVE-path observed map; the pseudo map call just below stays `apply_mode=False` (left raw).

pipeline.py
- :268-272 `AnalysisConfig.overlap_mode`; :564 `analyze_spectrum` param; :824/:1039/:1151 the three
  indexer constructions (idx_kwargs, idx2, idxN); :1520 cfg→analyze_spectrum; :1581 + :1640
  `analyze_directory` param + `AnalysisConfig(...)`.

detections.py
- :342 `_run_contest(..., overlap_mode="legacy")`; :362 passes it to the contest indexer `idx_c`;
  :679-680 `analyze_detections` reads `indexer._overlap_mode` and forwards it, so the contest's
  per-peak templates `T_j` and concentration caps use the SAME conserved map.

cli.py
- :121 `--overlap-mode {legacy,conserve,nearest}`; :164 forwarded to `analyze_directory`.

scripts/fit_inspector.py
- :60 module `OVERLAP_MODE="legacy"`; :178/:235/:283/:330 all four indexer builds use it;
  :866/:872-873 `--overlap-mode` sets the global in `main()`.

tests/test_overlap_conservation.py — new, 6 groups / 9 cases (see below).

## Oracle at truth (solve_at), by mode

Forward model's own per-line areas at the exact true state → fresh indexer → `solve_at`.

binary_ca_mg (truth Ca 0.60 / Mg 0.40):

| mode | Ca | Mg |
|---|---|---|
| legacy | 0.730 | 0.268 |
| conserve | 0.587 | 0.410 |
| nearest | 0.600 | 0.400 |

Both conserve and nearest restore Ca to within ±0.02 of 0.60; legacy reproduces the documented
0.73 defect exactly.

silicate_4el (truth Si 0.40 / Ca 0.30 / Fe 0.20 / Mg 0.10):

| mode | Si | Ca | Fe | Mg |
|---|---|---|---|---|
| legacy | 0.232 | 0.396 | 0.160 | 0.062 |
| conserve | 0.426 | 0.273 | **0.000** | **0.000** |
| nearest | 0.330 | 0.248 | 0.247 | 0.019 |

Si and Ca land within ±0.05 under conserve. Fe and Mg COLLAPSE to 0 under conserve — not a
success. I did NOT loosen the ±0.05 bound silently: test 3 asserts Si/Ca within ±0.05 and
explicitly asserts Fe<0.05, Mg<0.05 with a loud comment documenting the regression, and prints
the full three-mode table.

Real-spectrum smoke (user_spec1.csv, n_calls=8, draws=2, conserve), end to end, no error:
Li 0.017, Na 0.018, Al 0.047, Si 0.335, K 0.516, Rb 0.061; r² 0.9104.

## Downstream consumers — changed vs left

CHANGED (now read the conserved observed map, transparently):
- `_build_design_matrix` / the NNLS design A (quantification) — the target of the fix.
- Candidate prefilters (`_prefilter_species_by_initial_strength`, `_prefilter_species_by_line_evidence`),
  `_line_presence`, `_species_line_evidence` — all read `peak_line_map` / `_build_design_matrix_from_map`.
- `_freeze_stark_assignments`, `_freeze_sa_reference`, `_freeze_doublet_taus` / SA κ reference —
  the frozen per-peak argmax is over the weighted conserved map; conserve rescales but preserves
  the dominant-line ordering at each peak, so the reference-line identity still resolves (verified:
  the binary/silicate oracles and the smoke run all freeze and solve without error).
- Deepening / `solve_at` / SA re-solve — `idxN` is constructed with `overlap_mode`, so it inherits.
- Detection contest templates `T_j(E')` and caps `c_max` — `idx_c` now built with the same mode.

LEFT RAW (deliberately), with reason:
- The pseudo-observation (missing-line) map `pseudo_line_map` (`_build_overlap_map` apply_mode=False,
  and the `_select_pseudo_wavelengths` build): its rows are pseudo-WAVELENGTHS of strong UNMATCHED
  lines, not observed peaks. Per-line conservation across pseudo rows would re-weight the
  missing-mass / missing-count penalty, a different quantity from the quantification bias the debug
  localised. The debug pinned the bias to the observed design A; the A_pseudo ridge is a separate
  predicted-amplitude path and is left on the historic construction.

## Key design decision (needs judgment / panel scoring before any default change)

conserve dilutes dense-forest lines; the candidate prefilter is a per-peak ABSOLUTE-strength
detectability gate, so the diluted forest majors (silicate Fe, Mg) fall below the bar and are
pruned at `build_candidate_matrix` (verified: Fe/Mg present under legacy+nearest, absent under
conserve). Probe (NOT shipped): building candidates on the RAW map and solving conserve still
yields Fe≈0 / Mg≈0 with hallucinated U 0.178 / Pb 0.055 absorbing their flux — so the forest-major
loss is intrinsic to conserve's capping on this at-truth oracle, not only a prefilter artifact.
nearest keeps all four majors present (Fe 0.247) but under-recovers Si/Ca/Mg. Neither conserve nor
nearest is a clean multi-element win at truth; both fix the binary case. RECOMMENDATION: keep
`legacy` the default (done); score all three modes on the fit-quality panel (and the suite-S
round-trip) before promoting one. Do not adopt conserve as default on the binary result alone.

## Tests (tests/test_overlap_conservation.py)

1. `test_conserve_caps_line_replication[binary,silicate]` — max column sum ≤ 1+1e-9 on both the
   resting observed map and the shared-width solve map.
2. `test_binary_oracle_at_truth` — legacy Ca≈0.730; conserve/nearest Ca within ±0.02 of 0.60.
3. `test_silicate_oracle_at_truth` — Si/Ca within ±0.05 under conserve; Fe/Mg collapse documented;
   nearest Fe>0.05; prints the full table.
4. `test_clean_control_returns_truth[legacy,conserve,nearest]` — obs=A·c_true → Ca 0.60 / Mg 0.40
   in every mode (solver + aggregation faithful under each A).
5. `test_real_spectrum_smoke_conserve` — full pipeline on user_spec1 in conserve; no value assertion.
6. `test_legacy_bit_identical_to_raw_map` — legacy map == hand-rolled raw Voigt-height-ratio map
   (bitwise), and legacy's max column sum > 1 (so it genuinely differs from conserve).

New file result: `9 passed` (ALIBZ_SKIP_ROUNDTRIP=1).

## Test suite before / after

Command: `ALIBZ_SKIP_ROUNDTRIP=1 .venv/bin/python -m pytest -q`
- BEFORE (new test file absent, production changes present but inert at legacy):
  `651 passed, 10 skipped, 71 subtests passed in 681.56s` (exit 0, 0 failed).
- AFTER (incl. tests/test_overlap_conservation.py):
  `660 passed, 10 skipped, 71 subtests passed in 762.22s` (exit 0). +9 = the new cases; no new failures.

## What I could not verify
- The fit-quality panel (coordinator runs it) — the silicate regression and the mode choice must be
  scored there; the at-truth oracle is not the panel.
- Only the at-truth oracle was exercised here (the pipeline's basin landing is the round-trip's job).
- CLI `--overlap-mode` and fit_inspector `--overlap-mode` wired but only smoke-reasoned, not driven
  end to end through the CLI.

## Follow-up (coordinator): why `nearest` != truth on silicate_4el at the oracle

Each rendered db line IS its own peak at its own wavelength with its exact area (one peak row
per line). Isolation (10 kK / 17.0, solve_at, Si/Ca/Fe/Mg):
- Full candidate set (spurious species in): nearest .330/.248/.247/.019; legacy .232/.396/.160/.062.
- Candidates restricted to the 4 truth elements (drops U/Pb/... hallucinations): nearest
  .390/.293/.294/.023 — Si/Ca recover, Fe-over / Mg-under persist.
- Line + species prefilters OFF (min_init_relative_intensity=0, min_gA=0): unchanged
  (.390/.293/.294/.023). NOT the prefilter.
- CLEAN self-consistent control, obs = A.c_true, 4-element: **nearest -> EXACT .40/.30/.20/.10**
  (A 14701x8, full rank 8); **legacy -> .363/.274/.272/.091 (NOT truth)**; conserve collapses to
  .51/.49/0/0 (Fe/Mg pruned at candidate build, rank 3). So the solver AND the nearest design are
  faithful; legacy's replication is self-INCONSISTENT even on its own synthetic data. (Note: an
  earlier draft of this control mis-reported nearest as .363 — it had stale pseudo-rows from
  skipping `_select_pseudo_wavelengths` after the species filter; re-running it correctly gives
  exact truth.)
- Therefore with REAL obs, obs != A_nearest.c_true. Misrouting map (flux by rendered element ->
  nearest-assigned element): of Mg's rendered flux, **54% lands on peaks with NO candidate Mg line
  within shift_tolerance (0.1 nm)** — "orphaned" (assigned NONE) — vs ~0% orphan for Si/Ca/Fe. The
  orphaned Mg flux is Mg II (stage 2); it persists with all line filters off AND when the oracle
  peaks are clipped to the detector band (190–910 nm). It is a candidate-line COVERAGE gap (Mg II
  lines the inverse candidate builder never matches to a peak), NOT overlap accounting, NOT merged
  peaks (each line is its own peak), NOT argmax ties.

Verdict: `nearest` IS the principled mode — each fitted component = the sum of the db lines mapped
under it, and its self-consistent control is EXACT. The silicate miss is two SEPARATE defects the
overlap change merely exposes: (1) candidate-line coverage for Mg II, and (2) spurious-species
hallucination (U/Pb). Both live in the candidate builder / prefilter / contest, not in the overlap.
Those are what need re-tuning; legacy's apparently-better Mg is an artifact of self-inconsistent
replication, not a correct answer. Recommend adopting `nearest` as the overlap semantics and
redirecting the silicate work to Mg II candidate coverage + the spurious-species gate (panel still
the arbiter).
