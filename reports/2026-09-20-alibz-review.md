# alibz review — sampling fidelity, residual-raising fit steps, shared line shapes

Date: 2026-09-20. Provider: Claude subscription (no switch). Baseline test
suite before any change: 299 passed, 3 skipped (252 s).

Supporting evidence:

- `reports/2026-09-20-stage-residual-audit.md` — stage-by-stage residual
  audit on the four local sample spectra (Sonnet worker; headline finding
  re-verified independently, see §2.1).
- `reports/2026-09-20-native-grid-recovery.md` — native-grid recovery
  prototype (Sonnet worker; see §1.3 for what was and was not confirmed).
- `scripts/audit_stage_residuals.py` — reusable per-stage residual audit.

---

## 1. The wavelength grid is a vendor resampling, and it is recoverable

### 1.1 Finding (verified directly, this session)

Every input CSV (`data/remote_samples/*.csv`, and per `alibz/mw2_112.py`
`grid_ok` the whole MW2-112 corpus) is 23,431 rows on an exactly uniform
1/30 nm grid from 180 to 961 nm. That grid is **not** the detector's native
sampling. It is a piecewise-cubic interpolation of three coarser, unevenly
pitched detector segments:

| segment | zero 4th-difference runs | knot spacing (grid px) | noise rho(1) | PSD cutoff (cyc/px) | implied native pitch |
|---|---:|---:|---:|---:|---:|
| UV 190–365 nm | 0 % | (unresolvable from d4) | 0.76 | ~0.18 | ~0.09 nm |
| VIS 365–620 nm | 2.5 % | ~4 | 0.83 | ~0.14 | ~0.14 nm |
| NIR 620–910 nm | 29 % | 5–6 (median 5.5) | 0.91 | ~0.10 | ~0.18 nm |

Method: `np.diff(y, 4)` is exactly zero wherever five consecutive export
samples lie inside one cubic piece; in the NIR those runs are separated by
5–6 px, i.e. the interpolation knots (the native pixels). The fraction of
zero runs per segment matches `(r-4)/r` for oversampling ratio `r`
(NIR r≈5.6, VIS r≈4.1, UV r<4), consistent with the noise autocorrelation
and power-spectrum cutoffs. Roughly 1,800–2,000 native pixels per segment,
consistent with 2048-px CCDs.

**This corrects `docs/development_guide.md` §"Future work — instrument
line-spread function"**, which states the 1/30 nm grid "is the instrument's
own CCD→wavelength export, not a resampling" and that "upsampling does not
help". The first claim is wrong; the second is right for the wrong reason
(the data are already a spline, so re-splining is near-identity). The
"export ringing notches" described there are cubic-interpolation
overshoot on steep flanks. The previously noted empirical
"export-kernel correlation rho(1)=0.92" (`alibz/peaky_indexer_v3.py`
weighted-solve notes, project memory) is this same effect.

### 1.2 Why it matters for fidelity

Line FWHM on these spectra is ~0.19 nm, i.e. **about one native pixel in
the NIR**. Fitting a point-sampled Voigt to the spline-resampled export
therefore fits the interpolation kernel, not the line. Synthetic check
(FWHM 0.19 nm Voigt, sigma=gamma=0.05 nm, native pitch 0.183 nm, pixel
integrated, cubic-spline resampled to 1/30 nm, point-Voigt fit, 11
sub-pixel phases):

| quantity | bias on export grid |
|---|---|
| area | −4 to −6 % |
| centre | ±15 pm, phase-dependent |
| FWHM | +37 to +80 % |
| Lorentz fraction (true 0.50) | 0.00–0.21 |

The same line fitted on the native grid with a pixel-integrated model is
recovered exactly, and the Lorentz fraction is identifiable at single-line
SNR ≳ 150 (Δχ² ≈ 16 at ±0.25) but not below. This is the mechanism behind
the real-data degeneracy in §3.1.

Second-order consequences currently baked into the pipeline:

- Every pixel-count statistic (BIC `n` in `refinement._bic`, χ² degrees of
  freedom, matched-filter `sigma_area`, the noise-rescale in
  `classify_feature`) treats 23,431 correlated samples as independent;
  information is over-counted ~4–5×, so model-selection penalties are
  systematically too weak relative to χ².
- `PeakyFinder._noise_scale` already works around this (multi-lag max);
  the docstring attributes it to a different, 0.01 nm corpus path.
- The ±1 px centre bound in `fit_peaks`/`fit_all` is ±0.033 nm, i.e. a
  fraction of a native pixel; centres rail onto grid points.

### 1.3 Recovery — status

Recovery is feasible in principle because the export is a *linear* map
`S` (n_export × n_native) of the native samples; with the knots and kernel
known, `native = lstsq(S, export)` is massively over-determined and exact
if the kernel is right. A worker was briefed to identify the kernel
(interpolating cubic spline variants, Keys cubic convolution, PCHIP,
Akima), fit a smooth dispersion polynomial per segment, invert, and
validate (synthetic round trip; whitened native noise; cross-file
consistency), delivering `alibz/utils/native_grid.py` + tests. See
`reports/2026-09-20-native-grid-recovery.md` for the outcome; the section
below is updated from it.

**Verified this session (NIR, REE_01, 620–910 nm):**

- Kernel: interpolating **cubic spline**. With the knots below, the
  inversion residual is 4.5e-5 relative (0.012 counts RMS against ~3
  counts of noise) for not-a-knot and natural boundary conditions alike;
  PCHIP and Akima give 9e-3 (200× worse). Keys cubic convolution was
  rejected by the worker (2–10× worse).
- Knots: the tight bracket for the knot between consecutive zero-d4 runs
  `(a1,b1)`, `(a2,b2)` is `[b1+4, a2]` export px; 1590 of 1593 brackets are
  0–1 px wide. Sequential indexing and a **degree-3 polynomial** fit the
  bracket midpoints to 0.26 px RMS with no residual autocorrelation
  (degrees 4 and 6 identical), i.e. the dispersion is a smooth cubic.
  Native pitch runs from 0.198 nm at 620 nm to 0.162 nm at 910 nm
  (5.96 → 4.87 export px), ~1,600 native pixels in 620–910 nm.
- Whitening: on the recovered native samples the high-passed noise has
  lag-1 autocorrelation ≈ 0 in all three segments (export grid: 0.76 /
  0.80 / 0.90). This holds even with the worker's first-pass calibration.

**Per-file check (all four samples, NIR 620–910 nm, bracket procedure):**

| file | knots | polyfit RMS (px) | inversion rel. residual | knot shift vs REE_01 at 700/800/900 nm |
|---|---:|---:|---:|---|
| REE_01 | 1590 | 0.258 | 4.5e-5 | 0 / 0 / 0 |
| REE_44 | 1590 | 0.261 | 3.0e-4 | +0.053 / −0.114 / +0.075 nm |
| argon_noAr | 1587 | 0.264 | 2.3e-4 | −0.082 / −0.113 / +0.024 nm |
| scan9x9 | 1588 | 0.263 | 1.9e-4 | −0.075 / −0.053 / −0.029 nm |

So the recovery is essentially exact per file, but the native knot
wavelengths move by up to ~0.1 nm (about half a native pixel) between
acquisitions: the vendor applies a per-acquisition wavelength calibration
before resampling. **A fixed calibration table cannot work; the recovery
must be run per spectrum.**

**Prototype status (`alibz/utils/native_grid.py`, `tests/test_native_grid.py`,
worker report `reports/2026-09-20-native-grid-recovery.md`):** kernel
identified correctly (cubic spline; synthetic round trip 2.4e-15); its
first NIR polynomial converged to a wrong basin (~3 % too few knots), which
I flagged; after recalibration from the tight brackets it reaches 2.2e-4
on REE_01 but, because `SEGMENT_CALIBRATION` is a fixed table fit to
REE_01, it gives 5–7 % on the other three files (verified). VIS is at
0.6 % and UV at 3.4 % even on REE_01 (few or no brackets; needs per-file
residual minimisation seeded from the NIR shift). **Per-file redesign landed and verified (module run on all four files):**

| file | NIR relres (brackets) | VIS relres | UV relres | time |
|---|---:|---:|---:|---:|
| REE_01 | 4.5e-5 (1590) | 6.4e-3 | 4.0e-2 | 71 s |
| REE_44 | 3.0e-4 (1590) | 8.1e-3 | 3.6e-2 | 73 s |
| argon_noAr | 2.3e-4 (1587) | 1.5e-2 | 2.5e-2 | 80 s |
| scan9x9 | 1.9e-4 (1588) | 2.0e-2 | 6.2e-2 | 69 s |

`recover_native_grid` at that point derived the NIR knots per spectrum
from the brackets (`exact=True` below 1e-3 on all four files) and refined
VIS/UV per spectrum by an affine correction of a seed polynomial, which
left VIS/UV approximate (0.6–2 % / 2.5–6 %) with unphysical pixel-count
variation between files.

**Exact recovery in all three segments (2026-09-21, REE_01, window
pitch/phase search, `scripts/native_grid_phase_search.py`):** the
brackets are not needed. Per 6 nm window, a (pitch, phase) grid search
plus bounded polish of the cubic-spline inversion residual locates the
knots to ~0.05 px (verified against the bracket knots: NIR < 0.07 px, VIS
≤ 0.04 px); windows are chained through pitch continuity, the knots are
fit by a degree-3 polynomial in pixel index, and the four coefficients are
refined on the full-segment residual with 2 nm excluded at each dead edge
(the vendor spline is off-model in the last ~2 nm before a zero pad).

| segment | knots | polyfit RMS (px) | native pitch start → end | inversion rel. residual |
|---|---:|---:|---|---:|
| UV 193–362 nm | 1935 | 0.029 | 0.0959 → 0.0788 nm | 6.1e-10 |
| VIS 367–618 nm | 1960 | 0.017 | 0.1416 → 0.1130 nm | 1.4e-12 |
| NIR 623–946 nm | 1812 | 0.015 | 0.1982 → 0.1564 nm | 4.5e-12 |

Native noise is white in every segment (lag-1 autocorrelation −0.10 to
−0.07 versus 0.75–0.90 on the export grid). Cross-file: REE_44 UV
1.1e-12 (1936 knots), scan9x9 VIS 1.5e-12 (1960 knots), same pitch
curves as REE_01 to 4 digits. The residual floor is
machine precision: the export is exactly an interpolating cubic spline of
~5,700 native pixels whose wavelengths follow a smooth cubic dispersion
per detector, re-calibrated per acquisition (knot shifts up to ~0.1 nm
between files). Earlier chaining failures were a too-high VIS pitch floor
(the true pitch falls to 3.39 export px at 620 nm) and an unbounded
polish; both are fixed in the script. The reference script takes ~3–5 min
per segment.

**Module status (`alibz/utils/native_grid.py`, final, 2026-09-21):** the
native grid is treated as hardware. `INSTRUMENT_CALIBRATION` holds the
exact per-segment dispersion polynomials from the REE_01 search, and
`recover_native_grid` (default `mode="calibrated"`) measures only the
vendor's per-acquisition wavelength recalibration: a cubic correction in
pixel index per segment, from the shifts of 13 short windows (each a
coarse scan over one pitch plus a bounded polish, unwrapped modulo the
pitch), robustly fit and then polished jointly with Levenberg–Marquardt on
the window residuals, followed by one full-segment solve as the exactness
check. Deterministic, no random starts. Verified by me on all four files:

| file | UV | VIS | NIR | correction (UV / VIS / NIR shift, stretch) | time |
|---|---:|---:|---:|---|---:|
| REE_01 | 6.2e-11 | 6.2e-11 | 6.3e-11 | 0 (reference) | 6.4 s |
| REE_44 | 6.0e-11 | 7.3e-11 | 7.0e-11 | +0.25 / +0.49 / +1.89 px; +25 / +14 / +117 ppm | 6.2 s |
| argon_noAr | 5.7e-11 | 6.8e-11 | 5.6e-11 | +0.46 / +0.66 / −3.24 px; +146 / +61 / −301 ppm | 5.8 s |
| scan9x9 | 6.3e-11 | 6.5e-11 | 6.5e-11 | +0.87 / +0.83 / −1.71 px; +184 / +56 / +226 ppm | 5.9 s |

Every segment is at the solver floor; 14 min → 6 s per spectrum. The
per-spectrum correction is affine in the UV/VIS (window-shift profiles are
straight lines to 0.001 px) and needs the cubic term in the NIR (an
affine-only correction leaves 1e-3 on two files). The full search remains
available as `mode="search"` and as the fallback when a calibrated segment
misses the exact bar, and `calibrate_instrument` packages it for another
instrument. Tests: exact-and-fast on all four files, determinism, recovery
of an imposed shift and stretch; the slow search test is opt-in
(`ALIBZ_SLOW_TESTS=1`).

### 1.4 The better route: get the native export

Recovery by inversion is a workaround. The SciAps software holds the
per-pixel spectra before resampling; check the instrument/Profile Builder
export options for per-spectrometer pixel data (and the "Z300 exported"
campaign in the corpus for any non-CSV formats). I could not check the
corpus this session: on `moissanite` the drive is now mounted at
`/media/xc/Corpus_One` (owner uid 1005) and is permission-denied for
`mwhittaker`; `Samsung_T51/data` holds cryo-EM data only.

### 1.5 Recommendation

1. Fit on the native grid with a **pixel-integrated** line model (the
   synthetic generator already integrates over explicit cells:
   `alibz/synthetic.py` `ChannelGrid`; make the fitter consistent with it).
   Where the export grid must be kept, forward-model through the export
   operator `S` rather than fitting a point-sampled Voigt to spline output.
2. Replace pixel counts in every BIC/χ²/dof with native-sample counts
   (or the operator's rank), and estimate noise on the native samples.
3. Fix the development-guide section; keep the robust `soft_l1` loss only
   as a stopgap.

---

## 2. Fit steps that raise the least-squares error

Measured with `scripts/audit_stage_residuals.py` on the four local spectra
(REE_01, REE_44, argon_noAr, scan9x9). Pixel-space SSE against
`y − background` after every stage, with the transition-level increases
attributed to the responsible action.

### 2.1 Post-fit filter block in `fit_spectrum` (raises SSE in 4/4 files; no physics)

`alibz/peaky_finder.py` ~L1340–1360: widths below half a grid pixel are
snapped to 0; components wider than 100× median FWHM are dropped; implied
height below 2× local noise is dropped. Effect on total SSE: +0.25 %
(argon), +0.62 % (REE_01), **+2.3 % (REE_44)**, +0.26 % (scan9x9).

Re-verified independently on REE_44 with a true pre-filter snapshot:
+2.32 % total, of which **+1.86 % is the snap-to-zero alone**. 59 % of the
fitted components had a small but nonzero Lorentz width (0 < γ < 0.017 nm),
none were exactly at the zero bound; the snap converts them to pure
Gaussians and changes the profile. "Half a grid pixel" is one-tenth of a
native pixel, so the threshold has no physical meaning either. The 2σ
height gate also removes fitted flux that was supporting the residual,
with no cost accounting.

Verdict: the snap is a pure fidelity loss with no physical justification;
replace it with the shared-shape model of §3 (widths then cannot be
spiky) and report the SSE cost of any drop.

### 2.2 `refine_fit` blend splits and wing-soaker absorption (largest local increases)

On argon_noAr the largest per-window increases anywhere in the pipeline
are `split` actions (551.17 nm +2.9e5, 548.05 nm +1.8e5, 547.43 nm
+9.7e4) even though the stage total fell. Two mechanisms in
`alibz/refinement.py`:

- `classify_feature` compares S/A/B by a *robust* (`soft_l1`, f_scale 8σ)
  cost on a local window, then the winner is stored; the plain SSE of the
  chosen model can be higher than the incumbent's.
- Components classified as "wing-soakers" (`refine_fit` L~480–495: broader
  than 1.5× the weighted-median width and < 25 % of the feature area) are
  **dropped together with the feature** but their flux is only folded into
  the model as an `extra_area` seed, so any real wing flux they carried is
  lost from the stored model.

Verdict: the split verdict itself is data-driven (BIC + two db lines), but
the stored result is not the least-squares optimum of the window, and the
absorbed-component drop is unaccounted. Store the model that minimises the
plain χ² of the window after the verdict, and keep (or re-fit) the
absorbed flux.

### 2.3 Self-absorption tags (`sa-tag`) — documented, physically motivated, cost not surfaced

`refine_fit` (`asymmetric="only"` pass) replaces phantom pairs by a single
symmetric Voigt at the observed area. Local SSE rises (REE_44 248.96 nm
+3.8e4 at τ_a = 2.6; 221.00 nm +1.3e4 at τ_a = 3.8; REE_01 188 nm pair).
`docs/fit_pipeline.md` §5a admits and explains this (storage convenience:
downstream consumers want `[area, μ, σ, γ]` rows). The justification is
strong (resonance-capable lower level, BIC-gated), so this is the one
residual-raising step that meets the "very strong justification" bar —
but the model that *did* fit the data (model A, `params_asym`) is thrown
away from the profile, and the pipeline's outputs never report the cost.

Verdict: keep the physics; stop discarding the better fit. Carry the
attenuated profile in the stored model (a 6-parameter row type, or a
per-row profile callable) so the residual reflects what the fit achieved.

### 2.4 Whole-pattern r² collapse at indexer pass 2 — diagnosed and repaired

**Symptom.** scan9x9: r² 0.969 (pass 1) → 0.237 (pass 2); REE_44: 0.737 →
0.048, while pixel-space SSE kept falling.

**Diagnosis (controlled experiments, `scripts/diagnose_indexer_passes.py`,
2026-09-20).** In pass 2 the two strongest lines in each spectrum, Ca II K
and H at 393.4 / 396.8 nm, were predicted at ~0 and carried 81 % of the
residual; Ca II was absent from the pass-2 candidate table. Separating the
table change from the configuration change:

| experiment | scan9x9 r² | REE_44 r² |
|---|---:|---:|
| pass-1 table, cold config (= pass 1) | 0.969 | 0.736 |
| pass-2 table, cold config | 0.876 | 0.717 |
| pass-1 table, warm-started at pass-1 T | 0.240 | 0.095 |
| pass-2 table, warm-started at pass-1 T (= pass 2) | 0.237 | 0.047 |
| pass-2 table, fixed solve at 8000 K, table built at T_init = 4000 K | 0.186 | 0.047 |
| same, table built at T_init = 6000 / 8000 / 12000 K | 0.80 / 0.81 / 0.81 | 0.72 / 0.72 / 0.70 |

The warm start is the carrier, not the table. Pass 1 rails to the 4000 K
floor of the search (the wide-kernel degenerate basin: σ = γ = 0.3 nm at
the bounds, Bi or H ≈ 1.0), and the pipeline hands that temperature to
pass 2 as `temperature_init`. `build_candidate_matrix` evaluates its two
species prefilters (`_prefilter_species_by_initial_strength`,
`_prefilter_species_by_line_evidence`) at that single state; at 4000 K
every ion-stage species has a negligible Saha population, so Ca II (and
the other ions) are pruned from the design before the optimiser runs.
The candidate table shrinks from 21–29 species to 13–15 and pass 2 cannot
explain the H and K lines at any (T, n_e).

**Repair (shipped).**

1. `alibz/peaky_indexer_v3.py`: the initial-strength prefilter (the Saha
   population test that pruned Ca II) is now evaluated at the init state
   AND at the default 10,000 K state (`PREFILTER_TEMPERATURES_K`), keeping
   a species that passes at either. A cold start is unchanged. A wider
   ladder (5–20 kK, applied to both prefilters) was tried first and
   rejected: it re-admits line-rich species that pass the relative tests
   at some temperature, and the synthetic Ca/Mg round-trip test then
   collapsed onto Sn 0.98 (4 failures in `tests/test_roundtrip_synthetic.py`);
   restricting the ladder to the strength prefilter alone still failed the
   same way. The line-evidence prefilter stays at the init state.
2. `alibz/pipeline.py`: `_warm_start_temperature` replaces a pass-1 T that
   sits on a search bound by the 10,000 K default before it is used as
   `temperature_init` (pseudo-observation selection, Stark assignments and
   doublet anchoring evaluate at the init state too).

**Result on the same spectra (pass 2, same peak tables):** scan9x9 r²
0.237 → 0.869; REE_44 0.047 → 0.719 (K 0.60 / Al 0.13 / Na 0.09 / Si 0.08
at 11,700 K; Ca I and Ca II retained, 22 candidates). Round-trip synthetic
tests pass again (0 failures); unit tests added in
`tests/test_peaky_indexer.py` (hedge keeps the ion stage under a cold warm
start, no-hedge regression, evidence prefilter stays single-state, guard).

**Not repaired, and now the visible problem:** the pass-1 degenerate basin
itself. With Ca II back, the pass-2 composition on scan9x9 is still Bi 0.99 at
T = 4000 K (r² 0.87), i.e. the same degenerate corner pass 1 lives in. The
unweighted amplitude objective with a free, wide overlap kernel rewards
line-rich or single-huge-line species; physical width bounds alone move
the degeneracy to Zn (E7/E9 in the diagnosis log). That is the objective
problem already recorded in project memory (basin collapse guard,
pixel-likelihood estimator) and is out of scope here.

### 2.5 Physics-over-data gates that do not touch the residual (for completeness)

- Iterative deepening rejects a whole round if `composition_collapsed`
  fires (`alibz/pipeline.py` L235–252, L765–775) regardless of residual.
  Defensible as a guard, but it is a symptom of the amplitude objective
  (§2.4), not a fix.
- The width cap (`FWHM_CAP_FACTOR = 2×` median FWHM, commit 2942fdb) and
  the fit-window span cap in `fit_all` bound widths; the commit message
  documents an r² *improvement* in the affected windows, so these are not
  residual-raising and are physically reasonable (except that the median
  is measured on the resampled grid, §1.2).
- `recover_sa_areas` and `profiles` do not change the pixel model (bit-
  identical SSE, 4/4 files).

---

## 3. Shape correlations the physics supports, and how weakly they are used

### 3.1 State of the code

| stage | σ, γ treatment |
|---|---|
| `fit_peaks`, `fit_shoulders`, `fit_all` | free per component, bounds `[0, cap]` only; seed splits FWHM 50/50 |
| post-fit filters | snap < 0.5 px to 0 |
| `refine_fit` S/A/B | free per model within `[1e-4, 3×FWHM]` (model B: 8 free params, two independent shapes) |
| `deblend_shoulders` | new component clamped to 0.7–1.3× the main's σ, γ |
| `seed_minor_lines` / `recover_residual_lines` | template = segment medians ±10–15 % (good) |
| `profiles.segment_width_floor` | per-segment 20th-percentile FWHM (diagnostic only) |
| indexer | one global (σ, γ) for the overlap kernel; Stark–n_e width channel exists (`_width_cost`) but `stark_width_weight` defaults to 0 and is never set by the pipeline |

Measured consequence (blind fit, all four files; REE_44 shown, re-verified):
γ = 0 for 69 % (UV), 59 % (VIS), 42 % (NIR) of retained peaks; σ = 0 for
~10 %; median Gaussian fraction 1.00. The per-peak σ/γ split is not
identified on these data (§1.2 explains why), so every "shape" the
pipeline stores is mostly noise plus the export kernel.

### 3.2 What physics says should be shared

- **Gaussian width** is instrument LSF (per segment, slowly varying with
  wavelength) plus Doppler, which is negligible here (Fe at 10 kK, 500 nm:
  ~5 pm vs ~190 pm instrument). So σ should be **one smooth function of
  wavelength per segment shared by every line of every element and ion
  stage**, not a per-peak free parameter.
- **Lorentzian width** is Stark (electron-impact) plus small van der Waals
  and natural terms. Stark HWHM = `w_line(n_e)`; within a multiplet the
  parameter is essentially identical, across lines of one species it
  follows the `n_eff⁴/z²` scaling already implemented in
  `alibz/utils/stark.py`, and across ion stages of one element the same
  `n_e` applies. So γ should be **hierarchical**: one `n_e` for the plasma;
  per-line factors from atomic data (shared across stages through `z`);
  per-multiplet equality as a hard tie; a small per-line nuisance only for
  bright lines where it is identifiable.
- Ion stages of one element additionally share the plasma T (through
  Saha) — already modelled in the indexer — and, for resonance lines,
  self-absorption is *not* shared across stages (different lower-level
  populations), so τ must stay per line.

### 3.3 Recommendation

1. Replace per-peak free σ, γ in the blind fit with a **two-level model**:
   σ(λ) = per-segment low-order polynomial (or the `profiles` floor) shared
   by all peaks; γ_i = γ_inst,seg + c₄·shape_i·n_e/n_ref with `shape_i`
   from `stark_shape_factor` for the peak's dominant candidate line (the
   `_freeze_stark_assignments` machinery already does this assignment) and
   `n_e` a single global unknown; multiplet members tied exactly. Fit the
   global parameters jointly with per-peak areas and centres (VarPro: areas
   are linear). This removes ~2×N nuisance parameters, makes widths
   physical, and makes §2.1's snap unnecessary.
2. Turn the indexer's Stark channel on once widths are physical
   (`stark_width_weight > 0`), so n_e is measured from the shared
   Lorentzian widths rather than only from Hα.
3. In `refine_fit` model B, tie the two components' σ to the shared
   instrument width and their γ to the Stark model; the blend test then
   asks only about centres and areas, which is what the data can answer.
4. Do all of this on the native grid (§1) — on the export grid the shared
   σ will absorb the kernel and the Stark γ will be under-estimated.

---

## What remains unverified

- The native-grid inversion result (§1.3) is worker-reported; its report's
  residual numbers and tests are to be re-run before trusting the module.
- The worker's audit reproduces the stage sequence of
  `scripts/fit_inspector.py`, which differs slightly from
  `analyze_spectrum` (`kT_ev`, `segment_edges` in `seed_minor_lines`);
  mechanisms are the same, exact numbers may differ.
- §2.4 (r² collapse at pass 2) was observed, not diagnosed.
- Whether SciAps can export per-pixel data was not checked (corpus drive
  unreadable this session).
