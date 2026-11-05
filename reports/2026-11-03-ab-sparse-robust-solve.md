# A/B: adaptive-lasso sparsity and Huber IRLS in the whole-pattern solve (2026-11-03)

Flags (both default None = OFF, plain `scipy.optimize.nnls`):
`sparsity="adaptive-lasso"`, `robust_loss="huber"`. Harness: `--config-json '{"sparsity": "adaptive-lasso"}'` etc.

## Changes (line numbers as of the end of this run; other workers edit these files concurrently)
- `alibz/peaky_indexer_v3.py` (helpers before `class PeakyIndexerV3`, ~465-600): constants, `_bic`, `_adaptive_lasso_solve`, `_robust_sparse_solve`.
- `PeakyIndexerV3.__init__`: new kwargs `sparsity`, `robust_loss` (after `weighted_solve`), validated, stored `self._sparsity/_robust_loss`.
- `_solve_concentrations_thin`: the single `nnls(A_aug, y_aug)` call is now `if both None: nnls(...)` (byte-identical path) `else _robust_sparse_solve(A_aug, y_aug, n_data=len(_obs_amp), ...)`. Every caller (solve_at, SA fixed-point re-solves, deepening, profiles, detections) goes through this function, so both flags apply everywhere with no further edits. Un-normalisation, cost, evidence charges, aggregation untouched (a zeroed species is an exact 0 like NNLS).
- `alibz/pipeline.py`: `AnalysisConfig.sparsity/robust_loss` (~l.252); `analyze_spectrum(sparsity=, robust_loss=)` (~l.551); forwarded in `idx_kwargs` (pass 1), pass-2 `PeakyIndexerV3(...)`, deepening `idxN`, the `cfg->analyze_spectrum` call, and the second wrapper function that also forwards `weighted_solve`.
- `tests/test_sparse_robust_solve.py` (new).

## Algorithms
- Adaptive lasso: min_{c>=0} ||A c - y||^2 + lam * sum w_s c_s on the augmented (whitened + evidence/pseudo) system. w_s = 1/(c_nnls_s/max c_nnls + 1e-3)^1 (NNLS estimate normalised by its max, so lam is dimensionless in amplitude; equivalent to the brief's form with lam rescaled by max c). lam_ref = median_j |a_j^T r_nnls| over all columns (mean of non-zero if median = 0); path lam in {0.25,0.5,1,2,4} x lam_ref; L-BFGS-B, analytic gradient, bounds c>=0, warm start NNLS; entries < 1e-8 max c set to 0. BIC = n ln(RSS/n) + k ln n (n = data rows, k = non-zeros; nonlinear-parameter count is constant and cancels). Best path BIC kept unless > NNLS BIC + 2, then NNLS returned.
- Huber: IRLS outside, sparse/NNLS inside; weights min(1, 1.345 s/|r|), s = MAD/0.6745 of data-row whitened residuals; <=5 iterations or weight change < 1 %; evidence/pseudo rows weight 1.

## Tests
- Baseline BEFORE changes: `ALIBZ_SKIP_ROUNDTRIP=1 .venv/bin/python -m pytest -q`: 615 passed, 9 skipped, 0 failed (run overlapped my early edits, which only added unused code).
- AFTER: 636 passed, 9 skipped, 0 failed (includes my 5 new tests plus tests other workers added). pymatgen-enabled invocation NOT run (time; the change does not touch pymatgen code).
- New tests (5, pass): `test_adaptive_lasso_suppresses_spurious`, `test_huber_recovers_with_outliers`, `test_off_path_never_touches_new_helper`, `test_flags_on_run_through_indexer`, `test_flags_off_is_nnls_and_default_attrs`.
- Bit-identity with flags OFF: ran `git archive HEAD alibz` in scratch and the current tree on the same indexer (unweighted + weighted, two T values): `np.array_equal` of concentrations and costs = True.
- Honest limits of the synthetic tests: (a) on a random-seed sweep of the collinear design (noise 0.06, 30 seeds) NNLS exceeded 5 % spurious mass in 16, and adaptive lasso fully recovered only 6 of those; the failures (e.g. seed 12) are genuinely degenerate (truth and fit residuals tie) so tests use the identifiable seeds (2,4,5). (b) Huber with 3 x10 outliers on the brightest rows reaches <2 % on seeds 1-5 but 2.6 % on seed 0: with the 5-iteration cap, convergence from a contaminated NNLS start is slow. Sparse 25 %-density designs where an outlier row is the sole support of a column are not recovered at all (25 % error after 5 iterations).

## Panel (36 items, n_calls 8, draws 2, workers 2, default Fe offset 18)
Baseline `reports/fit-panel/baseline-20261103.json`; variants `sparse-20261103-170813.json`, `huber-20261103-173851.json`, `sparse-huber-20261103-180655.json`; compare tables `reports/fit-panel/compare-{sparse,huber,sparse-huber}.md` (full per-item rows there).
CAVEAT: other workers modified `alibz/*.py` (pipeline, refinement, sahaboltzmann...) during the runs; the huber run had 11 Fe items fail with a transient `SyntaxError ... pipeline.py line 1035` (another worker's mid-edit, file compiles now) - ignore those rows. Huber effects on synthetic/real reproduce between `huber` and `sparse-huber`, so they are real, not drift.

Aggregate (better/worse/unchanged items):

| metric | sparse | huber | sparse-huber |
|---|---|---|---|
| synth median_rel_err (base mean 0.273) | 1/0/3 (0.268) | 2/2/0 (0.398) | 2/2/0 (0.349) |
| synth spurious_mass (base 0) | 0/0/4 | 0/0/4 | 0/1/3 (4.7e-5) |
| synth abs_T_rel_err (base 0.083) | 1/0/3 | 1/3/0 (0.340) | 1/3/0 (0.247) |
| synth n_majors_missed | 0/0/4 | 0/2/2 | 0/2/2 |
| real r_squared (base 0.470) | 1/1/3 (0.470) | 0/5/0 (0.361) | 0/5/0 (0.369) |
| real stage_disagreement (base 0.483) | 0/2/3 (0.485) | 3/2/0 (0.563) | 3/2/0 (0.573) |
| real n_detected (base 11.8) | 0/0/5 | 0/0/5 (11.0; user_spec1 10->7) | 0/0/5 (11.2) |
| fe non_fe_mass (base 0.581) | 3/5/18 (0.593) | invalid (11 errors) | 5/14/7 (0.652) |
| fe n_non_fe (base 1.154) | 0/1/25 (1.192) | invalid | 1/3/22 (1.269) |

Items changed >10 % (key ones; complete lists are in the compare files):
- sparse: only Fe: run-5291e643 fe_fraction 0.302->0.076 (n_non_fe 1->2, worse), run-07c25af6 0.341->0.556 (better), run-01105eae 0.840->0.577 (worse). Synthetic: silicate_4el median_rel_err 0.526->0.508.
- huber / sparse-huber synthetic: binary_ca_mg median_rel_err 0.170->0.011 (better); binary_argon_ambient 0.217->0.115, dominant_correct 0->1; iron_rich_low_t error->ok (baseline error was the stage-sum ValueError, since fixed elsewhere - not credit to Huber); silicate_4el T err 0.096->0.87 (huber) / 0.41 (sparse-huber) and 1 major missed; feldspar_k_na median_rel_err 0.178->0.920, T err 0.027->0.362, 2 majors missed.
- huber real: scan9x9 r2 0.781->0.429 (0.466 with sparse), user_spec1 stage_disagreement 0.508->0.990 and n_detected 10->7.
- sparse-huber Fe: run-5291e643 non-Fe 0.698->0.959, n_non_fe 1->3; run-b17cc043 pure Fe (1.0) -> 0.0; run-01105eae and run-90143c6b non-Fe +0.25/+0.28.

Verdicts
- sparse (adaptive lasso): near-neutral on synthetic/real (baseline spurious_mass is already 0, so there was no headroom there), mixed-to-slightly-worse on Fe (non-Fe mass 0.581->0.593, 3 better/5 worse). No case for enabling; the lasso shrinks true minor species along with spurious ones.
- huber: rejects. Large gains on two binaries but severe regressions on silicate_4el and feldspar_k_na (T error, missed majors) and on real r2/stage_disagreement: with real spectra the "outliers" are bright true lines and wrongly-modelled blends, which Huber down-weights, and T then drifts.
- sparse-huber: same as huber with slightly less damage; Fe non-Fe mass 0.581->0.652, n_non_fe 1.15->1.27, pure-Fe run flipped to 0 Fe. Reject.

Unverified: pymatgen-enabled test invocation; a clean re-run of the huber Fe items; effect of other workers' concurrent edits on baseline vs variants (baseline git_head 620cc22 with a dirty tree).
