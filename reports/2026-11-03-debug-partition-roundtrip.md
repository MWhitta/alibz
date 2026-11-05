# Debug: Barklem-2016 partition round-trip regression (binary_ca_mg)

Date 2026-11-03. Agent: specialist (Opus 4.8). No repo files modified (debug + verdict only;
cause is identifiability/physics, not a plumbing bug, so no fix). All work in session scratchpad.
Provider: Claude subscription (no switch).

## Known reproduction (confirmed)
`render_case(binary_ca_mg, seed=11, partition_source=src)` -> `analyze_spectrum(..., n_calls=8,
draws=2, gp_seed=11, partition_source=src)`, truth Ca 0.60 / Mg 0.40, 10 kK, log n_e 17.0.

| pair | found T | found log n_e | r^2 | median_rel_err | Ca | Mg |
|---|---|---|---|---|---|---|
| lines / lines | 8922 | 16.50 | 0.709 | **0.084** | 0.560 | 0.440 |
| barklem / barklem | 9873 | 17.01 | 0.929 | **0.216** | 0.704 | 0.296 |

Reproduced the brief's 0.084 vs 0.216 exactly. Note barklem lands *at* truth (T,n_e) with much
higher r^2, yet recovers worse composition.

## Measurements

### 1. Forward model is clean and consistent (scratchpad/tau.py)
Rendered the target component for both sources and compared summed line area per (element,stage)
to the optically-thin prediction (escape_factor):
- tau = 0 for every Ca/Mg line in both scenes (effective_column ~ 0); self-absorption suppression
  = 1.000 everywhere. The scenes are optically thin, so self-absorption is NOT a confounder.
- Element total area ratio Ca/Mg = 1.547 (lines) vs 1.538 (barklem); truth nuclei ratio 0.6/0.4 =
  1.5. Forward emission tracks truth for both sources; barklem barely changes the areas.
- Analytic consequence (escape=1, same sb on both sides): forward_area_j =
  (frac_el * emission_scale) * line_emissivity(wl,gA,Ek,T,Z_s,f_s). So **per database line the
  forward and inverse agree exactly at the true state for both sources** — any composition error is
  produced in peak/amplitude space, not in line_emissivity / partition / stage-fraction.

### 2. Partition / Saha at truth (10 kK, log n_e 17)
Ca I U: lines 3.929 vs barklem 5.696 (+45%); Ca I fraction 0.0264 -> 0.0379. Mg I U: 1.546 ->
1.644 (+6%). Ca II / Mg II partitions essentially unchanged. The barklem change is concentrated in
the neutral-stage partition of Ca near 10 kK.

### 3. Decomposition: solve at a (T, n_e) grid on the pipeline's final peak table
(scratchpad/scan.py, adapted from scripts/thermometer_scan.py but threaded through
`partition_source` on render, `analyze_spectrum`, and `_get_sb`). Composition **at the exact truth
node (10000, 17.0)**, same stage_consistency_weight=1:

| source | data misfit | tie cost | maxdis | Ca | Mg | median_rel_err |
|---|---|---|---|---|---|---|
| lines | 1.280e6 | 5.7e2 | 0.16 | 0.627 | 0.373 | 0.056 |
| barklem | 3.04e5 | 2.4e2 | 0.04 | 0.711 | 0.289 | 0.232 |

Both over-recover Ca at the *true* state (truth 0.60): lines 0.627, barklem 0.711. The peak-based
inverse therefore carries a pre-existing Ca-over-Mg bias at truth; barklem amplifies it by ~0.08.

Thermometer: for barklem the stage-tie surface minimum sits **exactly at the truth node**
(tie 239, maxdis 0.04) and the data misfit minimum is also ~there; for lines the tie minimum is at
(9000, 16.5) and the pipeline landed at (8922, 16.5) — cold-biased. Across the grid the data
objective is nearly flat (barklem data ~3.0-3.4e5 over 9-12 kK; lines ~1.28e6 over 9-11 kK), so
composition is set by *where the thermometer lands* and the peak-inverse bias at that state, not by
the amplitude fit.

Composition vs state (barklem): cold basin ~8900/16.5 -> Ca ~0.62-0.66; near-truth 9873/17.0 ->
Ca ~0.71. The cold (wrong-T) basin gives the *more accurate* composition because the Ca bias is
smaller there.

### 4. Seeds 3, 7 (scratchpad/seeds.py) — not a single-seed fluke, but bimodal
| seed | lines (T, Ca, medErr) | barklem (T, Ca, medErr) |
|---|---|---|
| 11 | 8922, 0.560, 0.084 | 9873, 0.704, 0.216 |
| 3  | 8943, 0.538, 0.129 | 9873, 0.704, 0.216 |
| 7  | 8922, 0.677, 0.160 | 8902, 0.630, **0.062** |

The thermometer is bimodal: a cold basin (~8900/16.5) and a near-truth basin (~9873/17.0). barklem
lands in the near-truth basin on seeds 11 & 3 (-> Ca ~0.70, worse metric); on seed 7 it lands cold
and is the **best** of all four (Ca 0.630, medErr 0.062), while lines is worst (0.160). So barklem
is not uniformly worse — composition accuracy is governed by basin, not partition correctness.

## Cause (verdict: identifiability / inverse-model, NOT a plumbing bug)
1. partition_source is correctly wired on BOTH sides: forward (`synthetic._stage_partition` +
   `bench.synth_cases.saha_stage_map` via `sb`) and inverse (`PeakyIndexerV3` uses the injected
   `sb`; pipeline.py:810-812). `minor_lines` uses only within-stage gA*exp(-Ek/kT) ratios
   (partition cancels) and the `_level_cache` stores partition-source-independent raw levels with
   barklem applied on top in `partition()` — no cache mis-keying. Verified forward is optically
   thin and reproduces the truth area ratio for both sources.
2. The composition is wrong at the TRUE state for *both* sources (Ca 0.627 / 0.711 vs 0.60), so the
   forward/inverse disagreement lives in the peak-amplitude inverse (blind-fit amplitudes + single
   per-segment kernel width + NNLS + stage aggregation / detection coverage: Mg has ~700 lines vs
   Ca ~234, its area spread thinner), independent of partition in structure.
3. The data/amplitude objective is nearly flat in (T, n_e) (as the stage-tie docstring states), so
   composition is decided by the thermometer landing. barklem's +45% Ca I U near 10 kK sharpens and
   relocates the stage-tie minimum onto the truth node, where the peak-inverse Ca bias is fully
   expressed. lines lands in a cold basin where the bias partially cancels toward 0.60. The
   round-trip metric thus rewards lines' compensating cold bias and penalizes barklem for having the
   better thermometer.

## Recommendation
**Reject adopting `partition_source="barklem2016"` as a default now**, and do NOT read the
round-trip xfail as evidence of a partition bug. Evidence:
- The barklem tables are the more physically correct partition and *improve* the (T, n_e)
  thermometer (minimum lands at truth), a genuine gain.
- But with the current peak-inverse they do not improve composition; on this scene they relocate
  the thermometer to a state where a pre-existing Ca-over-Mg bias costs ~0.08 in Ca fraction. The
  effect is bimodal/seed-dependent (barklem best on seed 7), under shallow optimisation (n_calls=8).
- The real defect to fix is the peak-inverse composition bias at the true state (Ca over Mg at
  thin, correct T/n_e) — out of scope here.
Condition for future adoption: re-evaluate once that inverse-composition bias is characterised and
corrected; barklem should then help, since it already pins the plasma state better.

Suggested follow-up (not done, to respect the partition-ipd worker's uncommitted test file): the
`xfail` reason in `tests/test_partition_ipd.py::test_roundtrip_tables_not_worse_than_lines`
currently implies "cause not yet isolated"; it should be re-annotated to: tables/tables is worse
because barklem's partition lands the thermometer at truth where the peak-inverse over-recovers Ca
(0.71 vs truth 0.60), while lines lands cold and compensates — an identifiability artifact of the
flat data objective, not a partition inconsistency. I did not edit that worker's file.

## What I could not verify
- Mechanism of the peak-inverse Ca-over-Mg bias at truth narrowed to peak-amplitude space (single
  kernel width / blending / detection coverage / stage aggregation) but not isolated to one line of
  code; the Doppler-width difference alone is ~2% (inst sigma 0.010 nm dominates the ~0.002 nm
  Doppler term) so that is not the dominant driver.
- Only binary_ca_mg exercised (the known reproduction); other CASES not scanned.
- Full pytest suite not re-run (no code change); `tests/test_partition_ipd.py` fast subset: 18
  passed, 1 skipped (slow roundtrip opt-in) in 8.0 s.

## Artifacts (session scratchpad, not in repo)
scratchpad/repro.py (+repro.json, xy_*.npy), scratchpad/tau.py, scratchpad/scan.py,
scratchpad/seeds.py, and the three task output logs.
