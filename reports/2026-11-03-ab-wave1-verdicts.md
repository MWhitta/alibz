# Wave-1 A/B verdicts (clean series on moissanite, 2026-11-03)

All variants scored on the same machine against `reports/fit-panel/baseline-moissanite.json`
(fast settings n_calls=8, draws=2, seed 11; 5 synthetic suite-S cases, 5 real sample spectra,
26 Fe-standard run means; 32 workers). Per-item compare tables: `fit-quality-panel compare
reports/fit-panel/baseline-moissanite.json reports/fit-panel/<variant>-moissanite.json`.

| variant | syn median rel err | syn |T err| | syn dominant ok | real r² | real stage dis. | real n det | Fe fraction | Fe non-Fe mass | Fe n non-Fe |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.207 | 0.082 | 1.00 | 0.471 | 0.515 | 11.6 | 0.404 | 0.596 | 1.15 |
| sparse (adaptive NN-LASSO) | 0.204 | 0.081 | 1.00 | 0.469 | 0.467 | 11.4 | 0.441 | 0.559 | 1.08 |
| huber (IRLS) | 0.375 | 0.197 | 0.80 | 0.371 | 0.496 | 11.2 | 0.334 | 0.666 | 1.15 |
| sparse-huber | 0.314 | 0.216 | 0.80 | 0.386 | 0.491 | 11.6 | 0.328 | 0.672 | 1.19 |
| ipd (Debye) | 0.237 | 0.078 | 0.80 | 0.471 | 0.487 | 12.0 | 0.385 | 0.615 | 1.19 |
| partition (Barklem–Collet) | 0.218 | 0.085 | 1.00 | 0.466 | 0.491 | 11.8 | **0.559** | **0.441** | 1.08 |
| partition-ipd | 0.231 | 0.081 | 0.80 | **0.473 (+5/−0)** | 0.509 | 11.6 | 0.533 | 0.467 | 1.04 |
| fdr05 (BH) | 0.207 | 0.082 | 1.00 | 0.471 | 0.515 | 11.8 | 0.186 | 0.814 | 0.04 |
| fdr05-by | 0.207 | 0.082 | 1.00 | 0.471 | 0.515 | 11.6 | 0.186 | 0.814 | 0.04 |

Paired counts (better/worse) are in the coordinator log (reports/2026-11-03-manuscript-init.md, 23:30).

## Verdicts
- **huber, sparse-huber: REJECT** — worse on every panel.
- **fdr05 / fdr05-by: REJECT as implemented** — no effect on synthetic/real, but on the pure-Fe
  standard the Fe fraction collapses (0.40 → 0.19, 19/26 worse) while spurious elements vanish:
  FDR over m ≈ hundreds of candidates raises the threshold above Fe's own best-line z on runs where
  the design-matrix replication defect (below) already starves Fe. Re-test after wave 2.
- **sparse: NOT ADOPTED** — neutral where it was meant to act (spurious mass is already 0); the small
  Fe gain (+6/−2) is an interaction with the replication defect. Re-test after wave 2.
- **ipd: DEFER** — mixed (real stage disagreement 4/1 better, r² 3/2; synthetic worse); physically
  correct (0.07 eV per unit charge at 1e17 / 8 kK); basin effects dominate. Re-test after wave 2.
- **partition: ADOPT AFTER WAVE 2** — physically correct tables; Fe standard +15/−8 (Fe 0.40 → 0.56);
  synthetic slightly worse ONLY because the tables put the thermometer on the true state where the
  design-matrix bias is fully expressed (reports/2026-11-03-debug-partition-roundtrip.md).
- **partition-ipd: strongest real-data signal** (r² better on 5/5) — re-score after wave 2.

## The finding that reorders everything
`reports/2026-11-03-debug-composition-bias-at-truth.md`: the design matrix replicates each database
line over every fitted peak within the 0.1 nm shift tolerance (Voigt height ratio), so
Σ_i overlap(i, j) = 1.04 for Ca but 1.97 for Mg (binary) and 2.2–5.3 on the silicate; dense-forest
elements are over-predicted per unit concentration and shrunk by NNLS (oracle areas at exact truth →
Ca 0.73 vs 0.60; control A·c_true → 0.600). The fix (`overlap_mode="conserve"`, partition of unity
per database line) is wave 2; every variant above is re-scored on top of it.
