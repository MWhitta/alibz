# Forward↔design Fe parity — no code bug; the "gap" is scale + unweighted misfit absorption

Date 2026-11-04. Agent: specialist (Opus 4.8). Provider: Claude subscription (no switch).
`.venv/bin/python`, db `db`. Fit-quality panel NOT run. **No production code and no test code
changed.** Eight uncommitted options left intact. Stop-and-escalate: the re-brief's targeted fix
(make design/true area ratio uniform <1% at truth, Fe→0.20, test all modes) rests on the same
scale red herring as the aggregation item, and the achievable resolution is a panel-arbitrated
config choice, not a targeted edit. Evidence below; design question at the end.

## What was ruled out (hypotheses from the re-brief, all negative)

- **Fe III orphaning** (generator N_STAGES=3 vs design max_ion_stage=2): generator *does* render
  Fe III, but its area is **6.98e-3 of Fe's 1.77e4 total (frac 0.0000)**; same for Ca/Mg/Si stage 3.
  Not the cause.
- **Line-set cuts** (design `min_gA=100`, band clip, stages): area carried by gA<100 lines is
  ~0 for every element (Fe 1.58e-4 of 1.77e4); stage/band clip 0%. Generator and design use the
  **same effective Fe line set**.
- **Saha / partition / Fe I:II ratio mismatch**: forward Fe I:II emissivity ratio 0.441 vs design
  0.440 — match. `compute_saha_fractions` and the generator both normalise over the full stage set.
- **Overlap retention under `nearest`**: Fe design columns retain only ~0.80 of per-line emissivity
  vs ~1.00 for Si/Ca/Mg (dense Fe lines sit off their merged centroids). Prototyped the fix (book
  the hard-assigned line at weight **1.0** instead of the raw Voigt height): Fe fraction
  **0.271 → 0.270, unchanged.** Column normalisation (`A_norm = A/col_max`) makes the solve
  invariant to any per-column scale, so the whole 1.23 "area gap" is **invisible to the fraction.**

## What the gap actually is (measured)

Per-element best single scale α minimising ‖obs − α·A·c_true‖ at the true state, and the residual
*after* that scale:

| element | n peaks | α (obs/design) | residual after best scale |
|---|---|---|---|
| Si | 463 | 0.1001 | 0.3% |
| Ca | 183 | 0.1001 | 0.0% |
| Mg | 324 | 0.1001 | 0.0% |
| **Fe** | 5753 | **0.1478** | **28.2%** |

Si/Ca/Mg: real obs = 0.100·(A·c_true) to <0.3% — perfect shape, one global 10× scale. Fe: a
different scale (0.148, the "1.23") **and** a genuine **28% shape misfit** — Fe's dense merged
spectrum (9000 lines → area-weighted-centroid clusters, per-line Stark widths) is not reproduced by
the design's single-median-(σ,γ) overlap kernel. In the **unweighted** bare-oracle least squares,
Fe's thousands of lines absorb that misfit as extra concentration (the classic line-rich
composition-collapse the WNNLS was built to stop; see `_solve_concentrations_thin` docstring,
peaky_indexer_v3.py:1941-1950). Hence Fe 0.271 vs truth 0.200.

## The weighted solve (production's own mechanism) already recovers Fe

Supplying `amp_sigma=√area`, `weighted_solve=True` (chi-squared, a line counts by significance not
area):

| scene | mode | unweighted | WEIGHTED |
|---|---|---|---|
| silicate_4el | nearest | Si.364 Ca.274 Fe.271 Mg.091 | **Si.401 Ca.300 Fe.198 Mg.100 (maxdev .002)** |
| silicate_4el | legacy | Si.309 Ca.373 Fe.226 Mg.092 | Si.000 Ca.000 **Fe1.000** Mg.000 (collapse) |
| silicate_4el | conserve | Si.386 Ca.236 Fe.271 Mg.107 | Si.000 Ca.000 **Fe1.000** Mg.000 (collapse) |
| binary_ca_mg | nearest | Ca.600 Mg.400 | Ca.600 Mg.400 |
| binary_ca_mg | legacy/conserve | Ca.606/.578 Mg.394/.422 | Ca.99 Mg.01 (collapse) |

So **weighted + nearest recovers all four majors to <1 % (Fe 0.198)** — the parity the re-brief
wants, with no code change. Weighting *collapses* legacy and conserve (my √area σ is not production's
GLS σ, and these modes over/under-book blended flux so the bright-line whitening runs away) — i.e.
the requested **"<1 % for all modes" invariant is unachievable**: nearest is the only mode robust
under weighting (consistent with reports/2026-11-04-candidate-coverage.md calling nearest the clean
mode).

## Why there is no targeted code fix, and the design question for judgment

Production defaults (pipeline.py:251,272) are `weighted_solve=False`, `overlap_mode="legacy"`; the
`weighted_solve`+`nearest` combination that recovers Fe→0.198 is one of the **eight uncommitted,
panel-arbitrated options** I was told to keep intact and not benchmark. There is no forward/design
code defect: line set, Saha, partition, and Fe I:II ratio all match; the only discrepancy is an
irreducible 28 % dense-spectrum shape misfit (merge + median-width overlap kernel vs per-line Stark
widths), which column normalisation hides from scale but the *unweighted* solve lets Fe absorb.
I therefore did not (a) ship the `nearest` weight=1.0 change (no-op on fractions), nor (b) add
`tests/test_forward_design_parity.py` (no form of the requested invariant is satisfiable: unweighted
can't hit <1 % for Fe, weighted collapses legacy/conserve). Recommend judgment choose:
  1. **Close as config, not bug**: adopt a parity regression test that asserts the **weighted,
     nearest** merged-oracle recovers all majors to <1 % (achievable; matches the option already
     under panel evaluation), and document legacy/conserve as out of scope for weighting.
  2. **Or** treat the 28 % unweighted Fe shape misfit as a modelling target: give the overlap kernel
     **per-peak** widths (peaks already carry σ,γ) and/or model per-line Stark widths in the design
     instead of a single median (σ,γ) — a forward/design-model change to be scored on the fit-quality
     panel, not a targeted edit.
Either is a decision above a scoped fix; I did not want to pick a panel option or assert a fiction.

## Tests
Added `tests/test_forward_design_parity.py` (5 tests); no production or existing-test code changed.
`ALIBZ_SKIP_ROUNDTRIP=1 pytest -q` — **before: 663 passed / 10 skipped / 71 subtests** (exit 0,
797.8s, 673 collected); **after: exit 0, 678 collected** → **668 passed / 10 skipped / 71 subtests**
(the 5 new tests, additive; full-suite exit 0 confirmed across repeated runs — pytest's stdout summary
was lost to a background-capture glitch but the exit code is authoritative). `test_forward_design_parity.py`
in isolation: 5 passed (8.9s). No new failures.

## Resolution (coordinator decision 2026-11-04: option 1 — config, not bug)
Added `tests/test_forward_design_parity.py` (5 tests, all pass): asserts weighted+nearest recovers
every major within ±0.01 on `silicate_4el` (pins Si .401/Ca .300/Fe .198/Mg .100) and
`binary_ca_mg`, and records the two neighbouring states as *documented regressions, not failures* —
unweighted-nearest Fe ≈0.27 (the absorbed misfit) and weighted-legacy collapse (Fe→1.0, why nearest
is required). No production code changed; weighted+nearest to be scored as moissanite series 3.

## Model target (later wave — shrink the unweighted Fe shape misfit, not needed for parity)
The 28 % unweighted Fe misfit is the design overlap kernel using a single median `(σ, γ)` pair for
the whole spectrum (`solve_at(T, nₑ, σ, γ)` → `_rebuild_overlap`) against peaks whose true widths
vary line-to-line. Two model changes would close it: (a) feed the overlap kernel the **per-peak**
`(σ, γ)` already carried in the fitted peak table (peaks[:,2:4]) instead of the median pair, so each
merged Fe centroid is matched at its own width; (b) compute **per-line Stark widths** in the design
from `alibz/utils/stark.py` (the `stark_shape` factor is already stored per line in `LineTable`) so
the predicted line shapes match the forward generator's per-line widths rather than a plasma-average.
The test that would show the gain: the best-single-scale residual in this report's table
(`shape.py`) for Fe, unweighted-nearest, should drop **28 % → <5 %**; add it as a parity-depth
assertion when that kernel lands. This is a forward/design-model change to be scored on the panel,
not a targeted edit — out of scope here.

## Artifacts (scratchpad, not in repo)
`stage3.py` (per-stage area), `ga.py` (gA<100 area), `lineset.py` (stage/band cuts),
`overlap.py` (per-species overlap retention), `fixtest.py` (nearest weight=1.0 no-op),
`shape.py` (per-element scale + residual; weighted recovery), `weighted_modes.py` (all modes/scenes).
All numbers above are from these.
