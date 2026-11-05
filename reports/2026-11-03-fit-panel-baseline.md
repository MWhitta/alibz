# Fit-quality panel baseline (2026-11-03)

JSON: `reports/fit-panel/baseline-20261103.json` (log `.log` beside it). git_head 620cc22 (working tree dirty with unrelated files), variant `baseline`, fast settings n_calls=8, draws=2, seed=11, 4 workers, wall 902 s (15 min; 36 items). Config `{}` (pipeline defaults). Fe axis corrected by +18 samples (see harness report).

## Aggregates (mean / median)

| panel | metric | mean | median |
|---|---|---|---|
| synthetic (4 ok, 1 error) | median_rel_err | 0.273 | 0.198 |
| | spurious_mass | 0.000 | 0.000 |
| | abs T rel err | 0.083 | 0.098 |
| | majors missed (count) | 0 | 0 |
| | dominant_correct rate | 0.75 | 1 |
| real (5) | r_squared | 0.470 | 0.721 |
| | median stage_disagreement | 0.483 | 0.431 |
| | n_detected | 11.8 | 10 |
| | ne at bound (fraction) | 1.00 | 1 |
| fe (26) | Fe fraction | 0.419 | 0.312 |
| | non-Fe mass | 0.581 | 0.688 |
| | n non-Fe >0.5% | 1.15 | 1 |
| | abs(T-10000)/10000 | 1.31 | 1.43 |

## Notable rows

Synthetic (T_rec/T_truth, rec log ne):
- binary_ca_mg: Ca 0.518 / Mg 0.482 (truth 0.6/0.4), T 8908/10000, ne 16.5 (railed low vs truth 17.0), r2 0.63.
- silicate_4el: Si 0.506 Fe 0.327 Ca 0.109 Mg 0.058 (truth 0.4/0.2/0.3/0.1); median rel err 0.53 (worst), T 9043, r2 0.60. No major missed, no spurious.
- feldspar_k_na: Si 0.42 K 0.32 Al 0.18 Na 0.08 (truth .5/.2/.2/.1); K 60% high; T 9242 vs 9000; r2 0.85.
- binary_argon_ambient: Mg 0.504 Ca 0.496 -> dominant element WRONG (truth Ca 0.6); T 9000.
- iron_rich_low_t: ERROR (render): `ValueError: target stage abundances must sum to one, got 0.99999998884` (tolerance in scene builder, as anticipated).
- No synthetic case misses a major. ne rails to 16.5 in 3/4.

Real (all 5 have ne at a bound and qc_status=fail):
- REE_01: r2 -0.29, C = 0.953 of mass (composition collapse, qc: physical-pattern-r2; electron-density-at-bound; composition-collapse; dominant-weak-shape), T 7916.
- REE_44: r2 0.72, T 7296, ne 16.61 (halpha bounds), 14 elements.
- argon_noAr: r2 0.25, T 8511, ne 17.48 (prior bounds).
- scan9x9: r2 0.78, 16 detected, median stage_disagreement 0.68 (highest), K 0.31.
- user_spec1: r2 0.89, T 7000.0 (railed), K 0.61, Rb 0.07 (suspicious), ne 16.31.
- Caveat: `ne_at_bound` is my 0.05-dex proximity test of log_ne to `ne_bounds`; it agrees with the pipeline's own "electron-density-at-bound" qc reason in REE_01 but all-5 = 1 could partly reflect bound-tolerance; verify one by hand before relying on it.

Fe (target 99.98 % Fe; 26 runs):
- Fe fraction per run: 10 runs below 0.08 (two at exactly 0), 8 in 0.26-0.56, two at 0.79 and 0.84, six at exactly 1.0 (pure Fe, correct). 20 of 26 have non-Fe mass > 0.05.
- Median T 24 286 K, many railed at 25 000 K; median r2 0.28. Spurious majors across runs (count of runs with >0.5 %): H 14, C 6, B 5, P 3, Cd 1, Na 1 (H/C/B are plasma/ambient contaminants, not sample).
- The uncorrected stored axis (`baseline-20261103-rawfeaxis.json`) is far worse: Fe fraction 0.000 in all 26 (Fe never detected), r2 about -0.9, confirming the known 18 px offset invalidates this npz; compare mode shows fe_fraction worse in 24/26 raw vs corrected.
- The +18 sample shift is an approximation (seams at 365/620 nm mislabel ~18 samples each); a properly re-derived corrected npz would be better.

## Reproducibility
Synthetic and real panels were run twice (first run with raw Fe axis, then final): all headline metrics bit-identical across the two runs (compare shows 0 better/worse/changed for 9 metrics x items), so the harness is deterministic at fixed seed.
