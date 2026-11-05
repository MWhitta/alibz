
### 2026-11-04 — Wave 3 (model options after the adoption): nothing adopted; three options kept behind flags

Scored on the fit-quality panel against a same-tree baseline (each agent's `base` run reproduced
`reports/fit-panel/cpi-s3.json` on 36/36 items before any variant ran). Verdicts (coordinator-verified
from the JSONs):

- **Per-peak / per-line Stark width overlap kernel** (`overlap_kernel="peak+stark"`, conserve) — the
  model target named in the forward/design parity report. Fast settings: lifts every panel's primary
  headline (syn 0.156→0.127, real r² 0.478→0.482, stage dis. 0.500→0.445, Fe 0.546→0.563, n_non_fe
  1.15→1.00) with two small mean regressions. Production settings vs `cpi-prod2`: MIXED — real r²
  0.489→0.509 (4/1) and the binary dominant flip repaired, but syn median 0.170→0.174 (3/2) and Fe
  0.563→0.545 (10/14, beyond the 11/26 limit). NOT adopted; option kept. Plain `peak` mixed; both
  `nearest` variants worse. Finding: the merged oracle cannot arbitrate a width kernel (peaks sit at
  Δλ≈0 where every width gives overlap 1), so the parity report's "28 % → <5 %" target was not a
  well-posed oracle quantity; the panel is the arbiter. reports/2026-11-04-w3-width-kernel.md.
- **Outer-state polish after the GP** (`outer_polish="powell"|"nelder-mead"`, maxfev 40) — REJECT.
  Powell at this budget is a near no-op (budget spent in the first line search, ΔT ≤ 13 K) that
  perturbs results chaotically; Nelder–Mead lowers the objective 10–60 % on synthetic scenes yet moves
  T away from truth (ΔT up to +1800 K) and degrades Fe (16/26 worse). Lesson: the outer objective's
  minimum is not at the true state; better optimisation cannot help until the model does.
  reports/2026-11-04-w3-outer-polish.md.
- **Two-temperature stage thermometer** (`stage_temperature_offset_max_k=2000`) — REJECT: δ rails at
  the bound on 3/5 single-T synthetic scenes (absorbs outer-state error), syn |T err| 0.069→0.138,
  Fe 0.546→0.523 (8/26 worse), real stage dis. 0.500→0.685 (5/5 worse). **McWhirter/Fujimoto LTE gap
  flag** (`lte_gap_flag`) — NO-OP on this panel (36/36 identical; flags only single-stage gas species;
  the smallest-nonzero-level ΔE proxy lands on fine-structure levels for Fe). Revise the proxy before
  any re-test. reports/2026-11-04-w3-thermometer.md.

All three implementations are applied to the working tree behind OFF-by-default flags (same policy as
huber/sparse/fdr), with tests (overlap_kernel 6 + 2 xfail, thermometer 5, polish 5 + 1 relaxed).
Deferred without testing, with reasons in reports/2026-11-03-manuscript-init.md (09:30 entry): oxide
closure, one-point calibration, SA Cσ/branching gates, LOD/LOQ reporting, fit_all speed, p,T air-index
audit. Merged-tree full suite and a merged base-equality run on moissanite: see the STATUS head.
No provider switch.
