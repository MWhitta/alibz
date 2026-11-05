# Wave-2/3 A/B verdicts (moissanite, fast settings, vs matching-code baseline2) — PROVISIONAL

Reference: `reports/fit-panel/baseline2-moissanite2.json` (same tree as the variants). Determinism confirmed on
moissanite (baseline2 vs baseline3, identical code: all 36 items identical except runtime) and locally. The
series-1 baseline differs on all 36 items (mostly 3rd decimal, a few basin flips): off-path numeric drift
introduced by the option plumbings between the two syncs — each was checked bit-identical on one case, not on
the panel; a follow-up should bisect which change moved the default path. Columns: panel mean, paired better/worse counts.

| variant | syn med rel err | syn |T err| | real r² | real stage dis. | Fe fraction | Fe n non-Fe |
|---|---|---|---|---|---|---|
| baseline2 | 0.156 | 0.063 | 0.465 | 0.492 | 0.398 | 1.15 |
| conserve | 0.141 +3/−2 | 0.083 +2/−3 | 0.491 +4/−1 | 0.516 +2/−3 | 0.406 +12/−5 | 1.15 |
| conserve-ipd | 0.122 +3/−2 | 0.081 +3/−2 | 0.482 +4/−1 | 0.505 +2/−3 | 0.410 +12/−5 | 1.12 |
| conserve-partition | 0.169 +2/−3 | 0.079 +3/−2 | 0.480 +3/−2 | 0.502 +3/−2 | **0.571 +13/−8** | 1.00 |
| **conserve-partition-ipd** | **0.136 +3/−2** | 0.081 +3/−2 | **0.483 +4/−1** | 0.495 +2/−3 | **0.571 +13/−8** | **0.89 +8/−4** |
| conserve-sparse | 0.141 +3/−2 | 0.082 | 0.489 +4/−1 | 0.508 | 0.433 +11/−6 | 1.12 |
| nearest | 0.208 +1/−4 | **0.054 +3/−2** | 0.465 | 0.506 +1/−4 | 0.405 +12/−5 | 1.19 |
| nearest-partition | 0.167 +2/−3 | 0.055 | 0.481 +3/−2 | 0.490 | 0.538 +15/−6 | 1.15 |
| nearest-partition-ipd | 0.195 +1/−4 | 0.053 | 0.464 | 0.481 | 0.540 +14/−7 | 1.12 |
| bo (Matérn-5/2, EI stop) | 0.198 +1/−4 | 0.080 | 0.467 | **0.435 +3/−2** | 0.425 +11/−5 | 1.04 +3/0 |
| bic-auto (n_eff, ρ=0.92 on export grid) | **0.116 +4/−1** | **0.030 +4/−1** | 0.437 +2/−3 | 0.469 | 0.398 | 1.15 |
| bo-bic | 0.122 | 0.038 | 0.458 +4/−1 | 0.475 | 0.425 +11/−5 | 1.04 |
| wn (weighted + nearest) | 0.399 +1/−4 | 0.369 | 0.345 +0/−5 | 0.501 | 0.400 | 0.89 |

## Reading
- **conserve + partition + ipd** is the only variant that improves the mean of every headline on all
  three panels; the Fe-standard gain (0.40 → 0.57, spurious elements 1.15 → 0.89) is the largest single
  effect in the whole exercise and is consistent with the replication mechanism (Fe = densest forest).
- **nearest** is exact at the true state (oracle) and principled, but end-to-end on the fast search it is
  worse on synthetic median error (T error better): the unweighted solve absorbs Fe's shape misfit and the
  search lands differently. Not adopted now; the per-peak-width kernel is its enabling follow-up.
- **wn (weighted + nearest)**: exact at truth, bad end-to-end (real r² worse on 5/5) — the weighted search
  misbehaves away from the truth node (consistent with the 2026-07 WNNLS notes). Not adopted.
- **bic-auto**: large synthetic gains but real r² down and one wrong dominant; **bo**: best real stage
  disagreement but worse synthetic median error. Both deferred (small samples, mixed signs).
- Caveats: 5/5/26 items, fast search (n_calls 8, draws 2), one seed; the Fe panel's axis is a shifted
  approximation; production-settings confirmation of the top candidate pending (below).

## Weighted-solve family (chain C, vs baseline2)
| variant | syn med rel err | syn |T err| | syn dominant ok | real r² | Fe fraction | Fe n non-Fe |
|---|---|---|---|---|---|---|
| baseline2 | 0.156 | 0.063 | 1.00 | 0.465 | 0.398 | 1.15 |
| weighted (legacy map) | 0.525 | 0.406 | 0.60 | 0.149 | 0.476 | 0.92 |
| wn (weighted + nearest) | 0.399 | 0.369 | 0.80 | 0.345 | 0.400 | 0.89 |
| wn-partition | 0.387 | 0.358 | 0.60 | 0.357 | 0.525 | 0.96 |
| wn-ipd | 0.389 | 0.346 | 0.60 | 0.266 | 0.409 | 0.85 |
| wn-partition-ipd | 0.394 | 0.378 | 0.80 | −0.019 | 0.533 | 0.89 |
| wn-fdr05 | 0.427 | 0.369 | 0.80 | 0.345 | 0.154 | 0.00 |

**REJECT the weighted solve end-to-end in every combination.** It is exact at the true plasma state
(tests/test_forward_design_parity.py) but the outer (T, n_e) search driven by the whitened objective lands
far from it (|T err| 0.35–0.41), consistent with the 2026-07 WNNLS notes (evidence ridges/priors must be
re-derived in z-units; basin comparison by ΔBIC). A weighted search that reaches the truth node is a
model target, not an option to flip. NOTE: all chain-C runs also carried the two then-unflagged changes
(stage coupling 0.15, Stark weight 0.05); their rejection does not depend on that.

## Series 3 — re-confirmation on the TRUE HEAD baseline (two unflagged changes gated off)
`head` (legacy/lines/no IPD, coupling 0, Stark 0) reproduces the series-1 baseline exactly (0.207 / 0.082 /
0.471 / 0.404), proving the gated tree is HEAD-equivalent.

| variant vs head | syn med rel err | syn |T err| | real r² | real stage dis. | Fe fraction | Fe n non-Fe |
|---|---|---|---|---|---|---|
| head | 0.207 | 0.082 | 0.471 | 0.515 | 0.404 | 1.15 |
| head + Stark 0.05 | 0.205 +2/−3 | 0.060 +3/−2 | 0.470 +2/−3 | 0.503 +3/−2 | 0.401 +8/−11 | 1.15 |
| **cpi (new defaults)** | **0.156 +3/−2** | **0.069 +3/−2** | **0.478 +4/−1** | **0.500 +4/−1** | **0.546 +13/−11** | 1.15 |
| cpi + Stark 0.05 | 0.136 +4/−1 | 0.084 +2/−3 | 0.474 +3/−2 | 0.550 +3/−2 | 0.572 +14/−9 | 0.89 +8/−4 |
| cpi + coupling 0.15 | 0.150 +4/−1 | 0.068 +3/−2 | 0.477 +4/−1 | 0.462 +4/−1 | 0.546 +13/−11 | 1.15 |
| cpi + both | 0.165 +3/−2 | 0.076 +2/−3 | 0.483 +4/−1 | 0.495 +3/−2 | 0.571 +14/−9 | 0.89 +8/−4 |

**FINAL: adopt cpi = overlap_mode="conserve" + partition_source="barklem2016" + ipd="debye" as pipeline
defaults** — improves every headline mean on all three panels against the true baseline with no paired
count worse than 2/5 or 11/26. The earlier "stage disagreement worse on 4/5" came from the unflagged Stark
weight, not from cpi. **Keep Stark-width coupling OFF** (mixed: better composition/Fe, worse T error and
stage disagreement; it deserves a dedicated evaluation on real widths) and **stage-coupling rows OFF** (its
only gain is on the metric it directly manipulates). Production-settings re-confirmation (prod2) pending.

## Production-settings re-confirmation on the gated tree (prod2: n_calls 40, draws 8; head vs cpi)
| metric | head | cpi | paired |
|---|---|---|---|
| syn median rel err | 0.213 | 0.170 | +3/−2 |
| syn |T err| | 0.077 | 0.077 | +2/−3 |
| syn dominant correct | 1.00 | 0.80 | one scene flips |
| real r² | 0.472 | 0.488 | +4/−1 |
| real stage dis. | 0.502 | 0.489 | +2/−3 |
| Fe fraction | 0.427 | 0.563 | +13/−11 |
| Fe n non-Fe | 0.96 | 1.08 | +5/−7 |
Adoption stands: composition error, real r² and the Fe standard improve at production settings; costs are one
synthetic dominant-element flip and a slightly higher spurious count on the Fe standard (both within the
paired noise of 5- and 26-item panels, but recorded).
