# Debug: composition bias at the TRUE plasma state (binary_ca_mg / oracle)

Date 2026-11-03. Agent: specialist (Opus 4.8). Tier: isolating a composition bias inside a
multi-stage inverse with a fast exact reproduction, to decide the wave-2 fix. No repo files
modified (debug only; the fix is a design-matrix redefinition with regression risk — described
for a follow-up worker, not applied). Provider: Claude subscription (no switch). All work in the
session scratchpad. `.venv/bin/python`, db `db`. The fit-quality panel was NOT run.

## Verdict (one paragraph)

The Ca-over / Mg-under bias at the true state is produced in **`PeakyIndexerV3`'s design matrix**,
specifically `_build_observed_overlap_map` + `_build_design_matrix_from_map`
(`alibz/peaky_indexer_v3.py:1037` and `:1086`). The overlap weight is a Voigt **height ratio**
`voigt(Δλ,σ,γ)/voigt(0,σ,γ) ∈ [0,1]` collected over **every** database line within the
`shift_tolerance` window (default 0.1 nm), and the design column is `A[i,s]=Σ_j lw_j·overlap(i,j)`.
This construction is **not area/replication-conserving**: a database line's weight is replicated onto
every observed peak it overlaps, and every observed peak sums all database lines near it. The
per-line replication factor `Σ_i overlap(i,j)` (emissivity-weighted) is **1.04 for Ca** (isolated
resonance lines) but **1.97 for Mg** (dense line forest). So at a unit concentration the design
over-predicts Mg's total emission ~1.9× relative to Ca; NNLS compensates by shrinking Mg
(0.04→0.022) while Ca is recovered correctly (0.060), giving Ca fraction 0.73 instead of 0.60.
Partition source, the stage-tie thermometer, evidence/pseudo rows, column normalisation, the
weighted solve and `_aggregate_elements` are all shown innocent. Minimal fix: make the design
matrix conserve each line's emission across the observed peaks (normalise the overlap map per
database line, or assign each line to one best peak) — a redefinition that changes every solve, so
it needs its own regression suite; NOT a one-liner.

## Decomposition table — Ca fraction at truth (10 kK, log nₑ 17.0), binary_ca_mg

| step | seed 11 | seed 3 | seed 7 | notes |
|---|---|---|---|---|
| truth | 0.600 | 0.600 | 0.600 | Ca 0.60 / Mg 0.40 |
| 1. ORACLE areas (lines) | 0.731 | 0.731 | 0.731 | forward's own per-line areas at truth; seed-independent (noise-free) |
| 1. ORACLE areas (barklem) | 0.731 | 0.731 | 0.731 | identical — partition cancels (see below) |
| 3. FITTED @ truth (lines) | 0.627 | 0.604 | 0.735 | reproduces prior report; fewer merged peaks soften/vary the bias |
| 3. FITTED @ truth (barklem) | 0.711 | 0.711 | 0.694 | near the raw-oracle value |
| 4a. oracle, evidence+pseudo rows zeroed | 0.731 | — | — | no change |
| 4a. +inner stage-coupling off | 0.731 | — | — | no change |
| 4b. oracle, weighted_solve=True | 1.000 | — | — | worse (bright Ca lines dominate χ²); known-bad opt-in |
| 4c. oracle, SA channels off (thin, no doublets) | 0.731 | — | — | scenes are τ=0; no change |
| 4d. oracle, raw NNLS (no col-norm, no aug rows) | 0.731 | — | — | no change |
| 5. CLEAN control: obs = A·c_true exactly | 0.600 | — | — | NNLS+aggregation recover 0.60 exactly → solver faithful |

Ca stays pinned at 0.731 through every solver-term toggle; the ONLY thing that restores 0.60 is
replacing the real areas with design-consistent ones (`obs = A·c_true`). That localises the bias to
A itself, not any downstream term.

## Evidence, step by step

### 1. Oracle areas (`scratchpad/decomp.py oracle`)
Fed the forward model's own `_component_lines(scene.target)` table (every rendered db line, true
area, same wavelength, true σ/γ) into a fresh `PeakyIndexerV3`, `build_candidate_matrix(sa_doublets=True)`,
`_rebuild_overlap(median σ, median γ)`, `_solve_concentrations(10000,17)`. Result Ca 0.730 / Mg 0.268
(**worse** than the fitted 0.627). Raw per-stage densities: Ca I 0.0593, Ca II 0.0600 (truth
0.06=0.6·emission_scale 0.1 — correct); Mg I 0.0229, Mg II 0.0220 (truth 0.04 — ~55 % low). Oracle
summed area ratio Ca/Mg = 1.547 (matches the forward emissivity). So the areas are right; the bias is
in the solver. Per the brief's branch: Ca ≠ 0.60 at oracle → go to step 4 (solver), not peak
extraction.

### 2/line-matching (`scratchpad/diag.py`)
Isolated bright lines reproduce **exactly**: for Ca II 393.37/396.85 and Mg II 279.55/280.27,
`A_col·c_expected == forward_area` to ratio 1.000 (c_expected = frac·0.1 = Ca 0.06, Mg 0.04).
Inverse per-species saha fractions equal the forward stage fractions (Mg II 0.8986=0.8986, Ca II
0.9600=0.9600) and partitions match (Ca I 3.929, Mg I 1.546 for `lines`). So emissivity convention,
hc/λ factor, Saha fractions and partition are consistent forward↔inverse at truth. The error is NOT
per-line and NOT partition.

### Mechanism — overlap replication (`scratchpad/resid.py`, `scratchpad/repl.py`)
At c_true (Mg 0.04) the design predicts **1.97× more area** on Mg-dominated peaks than observed
(pred 1.387e5 vs obs 7.050e4); narrowing the kernel to ~δ lowers it to 1.66× and lifts Mg to 0.030,
confirming blend width drives part of it. Emissivity-weighted per-line replication
`Σ_i overlap(i,j)`: **Ca 1.040, Mg 1.967** (ratio 1.89). Predicted Mg density =
0.04·(1.04/1.97)=0.021 ≈ the observed 0.022; Ca fraction = 0.06/(0.06+0.021)=0.74 ≈ 0.731. The
replication ratio fully explains the oracle bias. Sub-mechanism: median 3 database lines overlap
each Mg observed peak (max 12), and 74 % of Mg peaks have >1 overlapping line — the dense forest, not
a kernel artefact. The fitted pipeline peak table softens this (merges blended lines into fewer
peaks), so FITTED-@-truth varies by seed: lines 0.627/0.604/0.735, barklem 0.711/0.711/0.694 for
seeds 11/3/7 (seed 11 matches the prior round-trip report exactly).

### 4. Solver terms (`scratchpad/terms.py`, `scratchpad/ctrl.py`)
On the oracle table at truth: zeroing evidence missing-mass/missing-count + pseudo weight → 0.731;
also zeroing inner stage-coupling → 0.731; raw NNLS with no column normalisation and no augmented
rows → 0.731; column-normalised NNLS with no aug rows → 0.731; SA off (scenes τ=0) → 0.731. Weighted
solve → 1.000 (worse). The single biggest mover is the weighted solve, in the wrong direction; every
other term is inert. The design matrix is the lever.

### 5. Aggregation faithful (`scratchpad/ctrl.py`)
CLEAN control: set `obs = A·c_true` (self-consistent with the design), re-solve and aggregate →
recovered Ca 0.0600 / Mg 0.0400, **fraction 0.6000 exactly**. `_aggregate_elements`
(`:2129`, precision-weighted ‖A‖² stage tie) and the NNLS reproduce truth when the areas are
design-consistent. Aggregation is not the defect; the stage densities it receives are already biased.

## silicate_4el at truth (one seed, `scratchpad/sil.py`)
Same mechanism, present and strong, but the SIGN is set by *relative* replication, not "majors
over-recover". Oracle at (10 kK, 17.0): replication Ca 2.24 (lowest), Mg 4.65, Si 4.98, Fe 5.27.
Recovered Si 0.231 (truth 0.40, **under**), Ca 0.395 (0.30, over), Fe 0.159 (0.20), Mg 0.062 (0.10).
The spectrally-isolated major (Ca) is over-recovered; the dense forests (Si/Fe/Mg) under-recovered.
(barklem near-identical: Si 0.223, Ca 0.381, Fe 0.154, Mg 0.060 — partition again cancels.)
NOTE: this at-truth oracle does **not** reproduce the baseline panel's Si *over*-recovery (0.51 vs
0.40) — at truth Si is under-recovered. So the panel's Si-over is a compounded/different effect
(fitted peak set + thermometer basin + Si resonance-line dominance of the detected peaks), only
partially sharing this overlap-replication cause. The follow-up fix must be re-scored against the
panel, not assumed to move Si the same way.

## Recommended minimal fix (for a follow-up worker — do NOT ship blind)
Make `_build_observed_overlap_map` emission-conserving so each database line contributes its weight
once across the observed peaks. Two candidate forms:
- normalise each line's overlap column so `Σ_i overlap(i,j) = 1` (soft assignment / partition of
  unity over peaks), or
- hard-assign each database line to its single best-matching observed peak within `shift_tolerance`.
Either removes the density-dependent replication. Expected: Mg density → ~0.04, Ca fraction → ~0.60
on binary_ca_mg at truth. This changes **every** solve (real data included), so it needs a dedicated
regression (suite-S round-trip + the fit-quality panel + a real-spectrum sanity set) before adoption.
It also interacts with `_aggregate_elements`' ‖A‖² weights and the evidence ridges, which assume the
current A — re-verify those after the change.

## What I could not verify
- Only the at-truth state was decomposed (the brief's scope). The thermometer/basin interaction that
  governs the *pipeline's* landing is covered by the prior round-trip report, not re-done here.
- Weighted-solve numbers used a crude σ proxy (√area + floor) on the oracle; the 1.000 collapse is
  directionally right (bright-line dominance) but not the pipeline's real σ.
- Fitted-@-truth confirmed for all three seeds (table above); seed 11 reproduces the prior report
  (lines 0.627 / barklem 0.711) exactly, validating the standalone rebuild path.
- Full pytest suite not re-run (no repo code changed). The fit-quality panel was deliberately not run.

## Artifacts (scratchpad, not in repo)
`scratchpad/decomp.py`, `diag.py`, `terms.py`, `resid.py`, `repl.py`, `sil.py`, `ctrl.py`, `final.py`.
