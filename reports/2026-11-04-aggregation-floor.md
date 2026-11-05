# Aggregation "3-stage Fe floor" — HALTED, premise falsified, needs judgment

Date 2026-11-04. Agent: specialist (Opus 4.8). Provider: Claude subscription (no switch).
`.venv/bin/python`, db `db`. Fit-quality panel NOT run (moissanite on series 2).
**No production code changed.** Stop-and-escalate: the brief's stated defect — that
`PeakyIndexerV3._aggregate_elements` over-recovers Fe *even when every stage concentration
is exact* — is not reproducible. The clean control recovers Fe exactly, and the prescribed
`aggregate_mode="exact"` fix would be a no-op. Evidence below; design question at the end.

## Bottom line (three brief claims checked, all false as stated)

1. **"CLEAN control obs = A·c_true still aggregates to Fe ≈ 0.27."** FALSE.
   With the *true equal-stage* concentration vector `c_true[s] = N_element` (every stage of an
   element set to the element density — correct because the columns already carry the Saha
   fraction `f_z`), `obs := A·c_true`, re-solve and aggregate gives element fractions
   **Si .4000 / Ca .3000 / Fe .2000 / Mg .1000** (exact to ~1e-7). Aggregating `c_true` directly
   (no re-solve) gives the same. The aggregation already maps exact per-stage concentrations back
   to the true element fractions.
2. **"Fe, the one element with three observable stages."** FALSE for this scene. The
   `silicate_4el` restricted oracle `line_table.species` is exactly
   `[Mg I, Mg II, Si I, Si II, Ca I, Ca II, Fe I, Fe II]` — **Fe has only two stages** (Fe III has
   negligible Saha population at 10 kK and is not a candidate column; `_last_A_all` confirms no
   gated Fe III either). There is no 3-stage element in the reproduction.
3. **"`_aggregate_elements` … over-recovers iron … even when every stage concentration is exact."**
   FALSE. The over-recovery (Fe 0.271) appears only against the **real** merged-oracle areas, where
   the solved per-stage concentrations are **not** exact: Fe I c=2.942e-2, Fe II c=2.971e-2 (truth
   density 2.0e-2 → inflated ~1.47×), while Si/Ca/Mg stages land on 4.0e-2/3.0e-2/1.0e-2 (truth).
   The aggregate faithfully averages the inflated Fe stages; the inflation is upstream in the
   **concentration solve**, not in the aggregation.

## The invariant the brief's fix should establish ALREADY HOLDS in legacy

Hand-built `A` (3 elements, {1,2,3} stages, disjoint support, stage weights ‖A‖² =
1.0e2 / 1.0e4,2.5e1 / 2.5e3,4.0,1e-6 incl. a near-zero Saha-faint 3rd stage), `c` = true
stage populations:

| element | stages | expected frac | legacy `_aggregate_elements` frac |
|---|---|---|---|
| E1 | 1 | 0.500 | 0.500 |
| E2 | 2 | 0.300 | 0.300 |
| E3 | 3 | 0.200 | 0.200 |

Exact to 1e-6. Mechanism: for disjoint support the tied non-negative 1-D re-solve reduces to the
‖A[:,s]‖²-weighted **average** of the per-stage estimates (not a sum); when every stage estimate
equals the element density the weighted average is that density, for any stage count and any
weights including a vanishing one. For blended support the same holds algebraically: `own =
Σ_s A[:,s]·c_s = N_el·a_tied` when `c_s≡N_el`, so `element_conc = N_el·⟨a_tied,a_tied⟩/⟨a_tied,a_tied⟩
= N_el`. **So `aggregate_mode="exact"` built to this invariant would be byte-for-byte identical to
`legacy`.** The candidate list of mechanisms in the brief (stage-column summing breaking the map,
a per-stage floor/clip, Fe III ‖A‖² weight, `stage_disagreement` max-vs-tie, cross-element double
count) are all ruled out: none fires, because the legacy map is already exact.

## Where Fe 0.271 actually comes from (so judgment has the real target)

Per-element real oracle area vs design-predicted area at truth (peaks nearest-assigned by the
design argmax at `c_true`):

| element | real obs area | design@truth | ratio real/design |
|---|---|---|---|
| Si | 1.057e4 | 1.056e5 | **0.100** |
| Ca | 5.458e4 | 5.452e5 | **0.100** |
| Mg | 1.764e4 | 1.762e5 | **0.100** |
| Fe | 1.761e4 | 1.426e5 | **0.123** |

The design over-predicts every element by a uniform **10×** global scale (ratio 0.100) that the NNLS
absorbs into the absolute concentration — harmless for fractions — **except Fe, whose real area is
0.123/0.100 ≈ 1.23× above that uniform scale**, so the solve lifts both Fe stage concentrations
(measured Fe-stage c ≈ 1.47× the proportional truth) and Fe's fraction rises to 0.271. Si/Ca/Mg have
no such forward↔design gap and solve to truth. (`‖obs−A·c_true‖/‖obs‖=8.98`, `Σ(A·c_true)=9.70e5`
vs `Σ(obs)=1.00e5` is this same 10× scale plus the Fe excess.) This is a forward-model
(`synthetic._component_lines`) vs indexer-design (`_build_design_matrix`/`_line_weights`)
emissivity/line-set parity gap specific to Fe — **not** an aggregation defect, and not reachable by
any `_aggregate_elements` change.

The candidate-coverage report (this file's predecessor, line 151/164-168) concluded "the fully
self-consistent control obs := A·c_true gives the SAME Fe .273 … so no peak-table change can move
it … It is the stage-aggregation floor." That control is **circular**: it set `obs := A·c_solved`
(the *solved* vector, a fixed point of the solve), not `A·c_true` with the true stage populations.
Re-solving a fixed point trivially returns the same c and the same Fe .271 — I reproduced it
(obs:=A·c_solved → Fe .2707). It proves the solve is self-consistent, **not** that the aggregation
over-recovers. Substituting the genuine true-population c_true (as above) gives Fe .2000. The
"3-stage aggregation floor" diagnosis is an artifact of the circular control.

## The design question for judgment (why I did not implement)

The brief asks me to add `aggregate_mode="exact"` plumbed like `overlap_mode`, enforce the invariant,
and assert Fe within ±0.01 of 0.20 under `"exact"`. But (a) legacy already satisfies that invariant
to 1e-7, so `"exact"` is a no-op and the A/B on the panel would show zero delta; (b) no aggregation
mode can bring Fe to 0.20, because Fe 0.271 is set by the concentration solve fitting real areas the
design can't reproduce for Fe — a test asserting Fe≈0.20 under `"exact"` would fail or require
changing the solve/forward model, which is a different, open debugging task (forward `_component_lines`
vs inverse `_line_weights` Fe emissivity/line-set parity). Recommend judgment decide:
  (1) Confirm the aggregation is sound (invariant evidence above) and **close** the aggregate_mode
      item as a no-op; then
  (2) Re-brief the real defect as a **forward-vs-design Fe area-parity** investigation (quantify the
      1.47× per-line, find the missing/extra Fe lines or the emissivity/partition difference between
      `synthetic._component_lines` and `_build_design_matrix`). This touches the forward model and
      wants the panel the brief told me to avoid — a judgment/second-brief call, not a one-liner.
I did not want to ship a no-op option + a test asserting a number the aggregation cannot control.

## Tests
No production code and no test code changed → "after" == "before"; only this report + scratchpad
scripts were written. `ALIBZ_SKIP_ROUNDTRIP=1 pytest --collect-only` → **673 tests collected**
(unchanged from the predecessor report's 673). The directly relevant file
`ALIBZ_SKIP_ROUNDTRIP=1 pytest tests/test_oracle_roundtrip.py -q` → **3 passed (16.3s)**. A full
`pytest -q` run completed this session (exit 0, 797.8s): **663 passed / 10 skipped / 71 subtests**,
673 collected (no production/test files changed → after == before). `tests/test_aggregation.py`
NOT added (would assert a fiction). The invariant evidence lives in scratchpad `invariant.py`
(reusable if judgment wants it added as a pure legacy-correctness test — it passes unchanged).

## Artifacts (scratchpad, not in repo)
`repro.py` (per-stage table, clean control), `repro2.py` (species list, residual, self-consistent
control), `invariant.py` (hand-built 1/2/3-stage invariant). All numbers above are from these.
