# Wave-3 model option A — per-peak / per-line width overlap kernel

Date 2026-11-04. Agent: specialist (Opus 4.8). Provider: Claude subscription (no switch).
Working tree: `scratchpad/w3k/` (db symlinked to main). `.venv/bin/python`, db `db`.
Deliverables: patch `reports/patches/2026-11-04-w3k.patch`; panels copied into
`reports/fit-panel/w3k-*.json|md`; this report.

## TL;DR
Implemented the three overlap kernels behind `overlap_kernel ∈ {median,peak,peak+stark}`.
`median` is bit-identical to today (base-equality: all 36 panel items unchanged). The
kernel is correct and live on offset/fitted peaks, but **the oracle cannot arbitrate the
per-peak kernel** (oracle peaks sit on db lines at Δλ≈0, where every width gives overlap 1),
so acceptance (b) as written is not reproducible — the panel is the arbiter. On the fast
panel the strongest variant is **peak+stark + conserve**, which improves every panel's
primary headline (real r² 0.478→0.482, real stage-disagreement 0.500→0.445, synthetic
median rel-err 0.156→0.127, Fe fraction 0.546→0.563) but regresses two headline *means*
(synthetic |T| 0.069→0.073 and a negligible synthetic spurious-mass 0→8.5e-5), so it does
**not** strictly pass the win rule. Recommendation: **REVISE** — confirm peak+stark+conserve
at production settings and resolve the small |T| regression before adoption; reject both
`nearest` variants; treat plain `peak` as mixed.

## What was implemented (file:line in the working tree, alibz/peaky_indexer_v3.py)
- `_freeze_stark_assignments` (≈1000): guard widened so the frozen per-peak dominant-line
  Stark shapes are also computed when `overlap_kernel=="peak+stark"` (needed to profile
  γ_inst), independent of the Stark-width *cost*. Untouched behaviour when neither is on →
  median/peak stay bit-identical.
- `_kernel_gamma_inst(log_ne)` (≈1190): amplitude-weighted instrumental Lorentzian HWHM,
  profiled exactly as in `_width_cost` (Σ w·(γ_obs−stark)/Σ w, floored at 0; median-γ
  fallback when no peak carries a Stark shape).
- `_build_scaled_overlap_map(sigma, gamma, log_ne)` (≈1215): the per-peak kernels (see
  formulas below). Loops over peaks like `_build_observed_overlap_map`, applies
  `_apply_overlap_mode` (conserve/nearest) to the triplets exactly as the shared map does.
- `_rebuild_overlap(sigma, gamma, log_ne=None)` (≈2754): branches on the kernel — `median`
  → the historic `_build_overlap_map(..., apply_mode=True)`; `peak`/`peak+stark` →
  `_build_scaled_overlap_map`. Pseudo-observation rows keep the shared (σ,γ) in all kernels.
  Cache key extended with `self._overlap_kernel` and, for `peak+stark` only, `log_ne`
  bucketed to 0.05 dex (T-only moves still hit; median/peak key on `None`, so their cache
  behaviour is unchanged).
- Seven `_rebuild_overlap` call sites (solve_at, `_outer_objective`, final solve,
  `_select_basin_bic` scoring, posterior-spread, prune-refit) now pass the trial `log_ne`.
- Test double in `tests/test_peaky_indexer.py` (lines 112/144/581): the 2-arg stub lambdas
  for `_rebuild_overlap` updated to accept the optional `log_ne` (sanctioned signature
  change — this was the one suite failure, now fixed).

## Kernel formulas as implemented
With median peak widths `(σ_med, γ_med)=_median_peak_widths()` and the search's global
factors `scale_s=σ/σ_med`, `scale_g=γ/γ_med`, the per-peak scaled widths are
`σ_i'=σ_i·scale_s`, `γ_i'=γ_i·scale_g` (σ_i, γ_i = peak_array[:,2:4], floored at 1e-6).

- **median** (unchanged): `overlap(i,j) = V(Δλ; σ, γ) / V(0; σ, γ)`.
- **peak**: `overlap(i,j) = V(Δλ; σ_i', γ_i') / V(0; σ_i', γ_i')`.
- **peak+stark**: every line j gets its own profile `σ_j=σ`,
  `γ_j=γ_inst + stark_hwhm(shape_j, log_ne, c4, log_ne_ref)` with `shape_j=LineTable.stark_shape`,
  `c4=self._stark_c4` (run() default 1e-4), `log_ne_ref=self._stark_log_ne_ref` (17.0), and
  `overlap(i,j) = V(Δλ; sqrt(σ_i'²+σ_j²), γ_i'+γ_j) / V(0; sqrt2·σ_i', 2·γ_i')`.

`V` is `scipy.special.voigt_profile` (unit **area**), so the kernel is a height ratio. All
three give `overlap=1` for a line carrying the peak's own widths sitting on the peak
(Δλ=0), i.e. the same normalisation. **Synthetic parity**: the generator renders per-line
widths as `γ = inst_γ + stark_hwhm(stark_shape_factor(E_ion,E_k,z), log_ne, 1e-4, 17.0)`
(alibz/synthetic.py `_line_widths`, ≈743), and `LineTable.stark_shape` is populated with the
same `stark_shape_factor(e_ion, E_k, ion_stage)` (peaky_indexer_v3.py ≈310), so the
peak+stark per-line Stark widths use the generator's own convention.

## `_obs_amp` normalisation and why the kernel is consistent with it
`_obs_amp = peak_array[:,0]` is the per-peak **area** (the attenuated/observed emission
area): refinement.py's `sa_voigt(x, area, …) = area·_voigt(x−μ,σ,γ)` with a unit-area Voigt,
and the pipeline comments throughout call column 0 "per-peak area (amplitude)". The design
predicts `obs_amp[i]=Σ_s c_s·Σ_{j∈s} overlap(i,j)·ε_j` where `ε_j=line_emissivity` carries
the Saha stage population (so c_s is the element density, identical across stages — matches
the debug report's Ca I≈Ca II≈0.06 at truth). The overlap factor is a dimensionless
height-ratio in [0,1] with `overlap(on-centre, equal width)=1`; every kernel preserves that
unit, so A·c stays commensurate with the observed areas exactly as the `median` kernel is —
no change to the area/emissivity convention.

## Tests (tests/test_overlap_kernel.py; 6 pass, 2 xfail)
- `test_overlap_kernel_validated`: construction + bad-value rejection.
- `test_median_reproduces_legacy_map_exactly` **(a)**: `median` map is bit-identical to
  `_build_overlap_map(..., apply_mode=True)` (nnz and max|Δ| = 0).
- `test_peak_matches_median_solve_on_oracle`: on the merged oracle under nearest, `peak`
  and `median` recover the same fractions to <0.01 (the oracle can't discriminate the solve).
- `test_kernel_is_live_on_offset_peaks`: with peaks offset +0.03 nm (Δλ≠0, as in real data)
  `peak` and `peak+stark` map-sums differ from `median` — the kernel is live, not dead code.
- `test_fe_shape_residual_recorded` **(b, diagnostic)**: Fe best-single-scale residual on the
  merged silicate oracle at truth — **median 0.0404, peak 0.0408, peak+stark 0.1389**
  (Si/Ca/Mg < 2% for median/peak). This does NOT reproduce the parity report's 28% baseline
  (see "Oracle cannot arbitrate" below); recorded, not asserted as the target.
- `test_conserve_peak_no_worse_spurious_mass_than_median` **(c)**: conserve+peak spurious
  mass at truth (0.0522) ≤ conserve+median (0.0524)+0.01 — passes.
- **xfail** `test_unweighted_nearest_kernel_recovers_majors_within_002[peak|peak+stark]`: the
  brief's ±0.02 majors target. NOT achievable on the oracle (reason recorded in the xfail):
  `peak` is identical to `median` under nearest (Fe stuck at the 0.271 aggregation floor
  that test_oracle_roundtrip documents as unmovable by any peak-table change), and
  `peak+stark` is *worse* at truth (unmerged-oracle Fe→0.327, Ca→0.175).

Full suite: `ALIBZ_SKIP_ROUNDTRIP=1 pytest -q -x` → **674 passed, 10 skipped, 2 xfailed,
71 subtests** (878s). Baseline 668 passed + my 6 new passing = 674; the 2 xfailed are mine.
No pre-existing test regressed (the one failure was the stub-signature fix above).

## Oracle cannot arbitrate the per-peak kernel (the design question)
The parity report's `shape.py` 28% Fe misfit is an **unmerged dense-spectrum** quantity and
is not fully specified (α_Fe=0.148 vs 0.10, n_peaks off by a few from my rebuild). More
fundamentally: on BOTH the merged and unmerged oracle every peak sits at (a centroid of) its
database line, i.e. Δλ≈0, where the height-ratio kernel is 1 for **any** width. So under
`nearest` the `peak` kernel is byte-identical to `median` (verified: same nnz and map sum,
same recovered fractions), and the Fe residual I can define on the oracle is ~4% for both —
already below the 28% baseline the per-peak kernel was meant to shrink. The kernel only
diverges from `median` when peaks are **offset** from the line centres (verified on +0.03 nm
peaks and end-to-end on the panel), which is the real-fitted-data regime. **Therefore the
fit-quality panel, not the oracle, is the only valid arbiter for the per-peak kernel**, and
acceptance (b)'s two oracle targets (±0.02 majors; 28%→<5% Fe residual) are not reproducible
as stated. I implemented the kernel faithfully, recorded the measured oracle numbers, marked
the unreachable target xfail with the mechanism, and relied on the panel below. This is the
only point where I departed from the literal acceptance spec; flagging it for judgment.

## Base-equality (critical correctness gate) — PASS
`compare cpi-s3.json w3k-base.json` (overlap_kernel OFF, `{}`): **all 36 items unchanged**
(0 better / 0 worse; only runtime differs). The default path is untouched.

## Panel — fast settings (n_calls 8, draws 2, seed 11, fe_pixel_offset 18; moissanite)
Headline **means**, base → variant (▲ better, ▼ worse, = neutral; paired better/worse in
parentheses where a gate matters):

| metric (better dir) | base | peak-cons | peak-near | pstark-cons | pstark-near |
|---|---|---|---|---|---|
| syn median_rel_err ↓ | 0.1559 | 0.1409 ▲ | 0.1851 ▼ | **0.1273 ▲** | 0.3249 ▼ |
| syn abs_T_rel_err ↓ | 0.0686 | 0.0695 ▼ | 0.0803 ▼ | 0.0729 ▼ | 0.0491 ▲ |
| syn spurious_mass ↓ | 0.0 | 4.6e-5 ▼ | 0.0830 ▼ | 8.5e-5 ▼ | 0.0805 ▼ |
| syn dominant_correct ↑ | 1.00 | 1.00 = | 1.00 = | 1.00 = | 0.80 ▼ |
| real r_squared ↑ | 0.4782 | 0.4747 ▼ | 0.4685 ▼ | **0.4822 ▲** | 0.4719 ▼ |
| real stage_disagree ↓ | 0.4997 | 0.4743 ▲ | 0.4943 ▲ | **0.4446 ▲** | 0.5631 ▼ |
| fe fe_fraction ↑ | 0.5456 | 0.6109 ▲(14/10) | 0.6000 ▲(14/9) | 0.5629 ▲(12/11) | 0.5774 ▲(17/6) |
| fe n_non_fe ↓ | 1.154 | 1.038 ▲ | 1.154 = | 1.000 ▲(3/0) | 1.154 = |

Per-item runtime medians (confounded — base at workers 20 under 3-agent contention, variants
at workers 10): base 93.4 s; peak-cons 64.6 s; peak-near 65.6 s; pstark-cons 70.4 s;
pstark-near 69.6 s. Among the equal-footing variants peak+stark is ~8% above peak — the
nₑ-dependent overlap rebuilds, as expected (cache still hits on T-only moves). Full MD
tables in `reports/fit-panel/w3k-*.md`.

### Verdict per variant (rule: every headline mean ▲/= and no paired worse >2/5, >11/26)
- **peak+stark + conserve** — the strongest. Improves every panel's *primary* metric
  (syn median, real r², real stage-disagreement, Fe fraction & n_non_fe) and all paired
  gates pass (Fe fraction 11 worse = the 11/26 ceiling). FAILS the strict rule only on two
  *mean* regressions: syn |T| +0.004 (0.069→0.073) and syn spurious-mass +8.5e-5 (≈0). Not
  a clean win, but close.
- **peak + conserve** — mixed: Fe fraction (0.611) and real stage-disagreement gains, but
  real r² and syn |T| means regress and syn median paired fails (3 worse > 2).
- **peak + nearest** — REJECT: synthetic regresses broadly (median 0.156→0.185, 1/12),
  real regresses; only Fe fraction up.
- **peak+stark + nearest** — REJECT: synthetic (median 0.156→0.325, a major missed, one
  dominant flips) and real stage-disagreement regress; Fe fraction up (0.577) only.
  (Consistent with wave-2: the unweighted solve lands badly under `nearest`.)

No variant strictly wins, so per the brief I did **not** run production settings; the
production confirmation of peak+stark+conserve is the recommended next step, not done here.

## What is unverified / caveats
- Panel is 5/5/26 items, one seed, fast search; Fe axis is a shifted approximation.
- Runtime comparison is confounded by worker count (20 vs 10) and shared-box contention.
- The 28% parity baseline was not reproduced (unmerged-spectrum, under-specified); the oracle
  Fe-residual numbers above use my merged-oracle dominant-element reconstruction.
- peak+stark+conserve NOT confirmed at production settings (n_calls 40, draws 8).
- The two other agents' runs (w3t-*) shared moissanite throughout; my tree stayed ≤20 workers.

## Recommendation — REVISE
Keep the implementation (median default is provably inert; the kernels are correct and live).
`peak+stark + conserve` is the one promising candidate: on the fast panel it lifts every
panel's primary headline, with the only regressions a +0.004 synthetic |T| error and a
negligible synthetic spurious mass. Because it does not strictly pass the win rule, I
recommend the coordinator (judgment) authorise a **production-settings confirmation** of
peak+stark+conserve (vs cpi-prod2) and decide whether the small |T| regression is fast-search
noise; adopt only if production holds the real-r²/stage-disagreement/Fe gains without
widening |T|. **Reject** both `nearest` variants and treat plain `peak` as mixed. Separately,
the oracle acceptance (b) should be redefined on offset/fitted peaks (or dropped in favour of
the panel) since oracle peaks at Δλ≈0 cannot discriminate the per-peak kernel.

## Production-settings confirmation (coordinator, 2026-11-04): peak+stark+conserve vs cpi-prod2 (n_calls 40, draws 8)

Mixed → NOT adopted; kept as an option. Paired: syn median 0.170→0.174 (3/2), |T err| 0.077→0.068 (3/2), dominant_correct 0.8→1.0
(the binary flip recorded for cpi is repaired), real r² 0.489→0.509 (4/1), stage dis. 0.489→0.474 (2/3), Fe 0.563→0.545 (10/14),
n_non_fe 1.08→1.12 (1/3). Wall 554 s (not slower). Files: reports/fit-panel/w3k-peakstark-conserve-prod.{json,md}.

## Aggregate (items better / worse / unchanged per headline metric)

| panel | metric | better | worse | unchanged | n/a | base mean | var mean |
|---|---|---|---|---|---|---|---|
| synthetic | median_rel_err | 3 | 2 | 0 | 0 | 0.1697 | 0.1741 |
| synthetic | spurious_mass | 0 | 0 | 5 | 0 | 0 | 0 |
| synthetic | abs_T_rel_err | 3 | 2 | 0 | 0 | 0.07663 | 0.06848 |
| synthetic | n_majors_missed | 0 | 0 | 5 | 0 | 0 | 0 |
| synthetic | dominant_correct | 1 | 0 | 4 | 0 | 0.8 | 1 |
| real | r_squared | 4 | 1 | 0 | 0 | 0.4885 | 0.5092 |
| real | median_stage_disagreement | 2 | 3 | 0 | 0 | 0.4889 | 0.4735 |
| real | n_detected | 0 | 0 | 5 | 0 | 11.6 | 11.6 |
| real | ne_at_bound | 0 | 0 | 5 | 0 | 1 | 1 |
| fe | fe_fraction | 10 | 14 | 2 | 0 | 0.5625 | 0.5454 |
| fe | non_fe_mass | 10 | 14 | 2 | 0 | 0.4375 | 0.4546 |
| fe | n_non_fe | 1 | 3 | 22 | 0 | 1.077 | 1.115 |
| fe | abs_T_dev_10kK | 12 | 9 | 5 | 0 | 1.283 | 1.287 |

## Verdict

- **synthetic**: improved (7 better vs 4 worse metric-comparisons)
- **real**: improved (6 better vs 4 worse metric-comparisons)
- **fe**: regressed (33 better vs 40 worse metric-comparisons)
