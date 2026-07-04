# alibz Peak-Fitting Pipeline

This document walks the peak-fitting pipeline (Component 2 in the
[development guide](development_guide.md)) step by step, with `file:line`
anchors so each stage can be reviewed against the source. It also defines
precisely what an "area" is at each stage and answers a question that recurs
when reading the `fit_inspection` notebook: **how the self-absorption
refinement can raise the pointwise residual yet still be the right thing to
do** — and where it genuinely costs data fidelity.

The pipeline exercised by [`notebooks/fit_inspection.ipynb`](../notebooks/fit_inspection.ipynb)
is three passes:

```
raw spectrum
   │  find_background            (per-segment arPLS)
   ▼
y_bgsub
   │  ── PASS 1 : fit_spectrum ─────────────────────────────────────────
   │    fourier_peaks           blind detection (windowed maxima + noise gate)
   │    fit_peaks               per-peak windowed Voigt seeds
   │    fit_shoulders           add components for leftover residual structure
   │    fit_all                 windowed block-coordinate joint refit
   │    (width + significance filters)
   ▼
first-pass fit  (303 peaks on MW2-112 spectrum 0)
   │  ── PASS 2 : refine_fit ───────────────────────────────────────────
   │    reclassify ambiguous features under 3 local models (S / A / B),
   │    gated by the line database; merge phantom splits, quantify
   │    self-absorption, split genuine blends
   ▼
refined fit  (288 peaks)
   │  ── PASS 3 : seed_minor_lines ─────────────────────────────────────
   │    predict every other line of each established element from one
   │    Boltzmann scale; fit/deblend the predicted minor lines
   ▼
final fit  (299 peaks)   ──►  downstream: indexer / whole-pattern fit
```

A fourth, **separate** self-absorption mechanism (doublet-ratio anchoring)
lives in the *indexer*, is off by default, and is **not** part of the fit the
notebook produces — it is described last so it is not confused with Pass 2.

---

## 0. Convention: a "peak" is an area, not a height

Every fitted component is a length-4 vector `[area, center, sigma, gamma]`.
`scipy.special.voigt_profile` is **unit-area normalised**, and
`multi_voigt` multiplies each profile by its `area`
([utils/voigt.py:46](../alibz/utils/voigt.py#L46)). So component 0 is the
**integrated area**, and the implied peak height is
`area · V(0; sigma, gamma)`. Seeding a component from a measured *height*
(rather than area) would overestimate narrow-line areas ~4× — this is why
`_voigt_seed` converts height→area explicitly
([peaky_finder.py:104](../alibz/peaky_finder.py#L104)).

Keep this in mind throughout: **"area" always means integrated flux under the
line**, and every mention of "correct area" below is about *which* integrated
flux (observed vs emitted), never about a height.

---

## 1. Background subtraction — per-segment arPLS

**Entry:** [`PeakyFinder.find_background`](../alibz/peaky_finder.py#L354),
kernel [`_arpls`](../alibz/peaky_finder.py#L313).

The continuum is estimated by **asymmetrically reweighted penalised least
squares (arPLS)**: minimise `‖y − z‖²_w + λ‖D²z‖²` where `D²` is the
second-difference operator, then iteratively drive the weight of any pixel
sitting *above* the current baseline (an emission line) toward zero. Emission
lines of any width therefore cannot attract the baseline.

- **Per-segment.** The spectrum is split at the SciAps detector junctions
  `DEFAULT_SEGMENT_EDGES = (365, 620)` nm and each segment is solved
  independently ([:410-417](../alibz/peaky_finder.py#L410)), so the smoothness
  penalty never spans a hardware step (a spanned 620 nm junction left a
  +54-count ledge that spawned spurious peaks).
- **Grid-independent stiffness.** `lam` defaults to `(smooth_nm / pitch)⁴`
  with `smooth_nm = 1.9` nm ([:404-407](../alibz/peaky_finder.py#L404)), so the
  *physical* smoothing length is fixed regardless of the wavelength grid.

| Parameter | Default | Meaning |
|---|---|---|
| `smooth_nm` | 1.9 nm | physical baseline smoothing length (sets `lam`) |
| `segment_edges` | (365, 620) | detector junctions; `()` disables splitting |
| `max_iter` | 20 | arPLS reweighting iterations |

**Caveats to review.**
- There are **two** background implementations in the tree. The live fit path
  uses *only* `find_background` (arPLS). The `DetectorModel` / `background_pca`
  path ([detector.py:213](../alibz/detector.py#L213)) — PCA junction-artifact
  removal + median-anchor baseline — is exported but **never called** by the
  fitting pipeline; it is an offline utility, and it still contains the retired
  anchor baseline that arPLS replaced.
- A failed `find_background` is swallowed and falls back to a **zero baseline**
  with no warning ([:1198-1201](../alibz/peaky_finder.py#L1198)) — a failed
  baseline is then indistinguishable from `subtract_background=False`.
- Segments are solved independently with no cross-junction blending, so a line
  straddling 365/620 nm is split across two solves.

---

## 2. Blind peak detection — windowed maxima + noise gate

**Entry:** [`fourier_peaks`](../alibz/peaky_finder.py#L477) →
[`filter_peaks`](../alibz/peaky_finder.py#L428) →
[`find_peaks`](../alibz/peaky_finder.py#L275).

> **The name is historical and misleading.** Detection is *not* Fourier- or
> cepstral-based. `find_peaks` is a strict windowed local-maximum test (a
> sample must exceed 4 of its 5 window neighbours), and `filter_peaks` keeps a
> maximum only if it clears a noise-referenced significance threshold. The
> FFT/cepstrum/autocorrelation code in `fourier_peaks`
> ([:513-552](../alibz/peaky_finder.py#L513)) is **vestigial diagnostics**
> wrapped in a broad `try/except` and is not consumed by the pipeline. A
> `PowerTransformer` is fit and then **discarded**
> ([:456-458](../alibz/peaky_finder.py#L456)) — the significance test runs on
> the raw background-subtracted intensities.

- **Significance gate** ([:467-468](../alibz/peaky_finder.py#L467)):
  `threshold = 3.0 + max(n_sigma, 0)`, keep peak iff
  `y[peak] > threshold · noise[peak]`, where `noise` is the **per-pixel local**
  scale. So the notebook's `n_sigma=0` is already a hard 3σ gate; `n_sigma`
  *adds* to the floor and cannot loosen below 3σ.
- **Local noise** ([`_noise_scale_local`](../alibz/peaky_finder.py#L165)):
  a robust multi-lag MAD point-noise estimate (`1.4826·median|Δ|/√2`, max over
  lags 1/2/4/8), computed in blocks that **restart at each detector junction**
  so noise never bleeds across segments.

**Caveats.** Flat-topped / plateau peaks fail the strict `>` test and are
missed here (they are recovered later — a self-absorbed flat top is exactly
what Pass 2 handles). A genuinely flat segment can yield `noise = 0`, disabling
the gate locally.

---

## 3. First-pass windowed Voigt fitting — `fit_peaks`

**Entry:** [`fit_peaks`](../alibz/peaky_finder.py#L759), seed helper
[`_voigt_seed`](../alibz/peaky_finder.py#L104).

Each detected peak defines a half-maximum window
([`peak_parameter_guess`](../alibz/peaky_finder.py#L557)). Within the window,
co-located sub-peaks are re-detected and each new peak is fit with a Voigt via
bounded least-squares (or, on the notebook's `fast=True` path, seeded directly
from a height/FWHM estimate with a strongest-first pedestal subtraction and the
solve skipped unless a seed is degenerate).

- **Bounds** ([:894-900](../alibz/peaky_finder.py#L894)): `area ≥ 0`;
  `sigma, gamma ∈ [0, fwhm_limit]` with `fwhm_limit = 5·median_FWHM`; and the
  **center pinned to seed ± one grid step** — the fitter is not allowed to move
  a line more than one pixel from where it was detected.
- **De-blending** ([:905-912](../alibz/peaky_finder.py#L905)): the modelled
  contribution of every *other* already-fit peak is subtracted from the window
  target, so a strong line's wings don't act as an unmodelled pedestal.

**Caveats.** Peaks whose half-max window spans ≤5 samples are silently dropped
in this pass ([:806](../alibz/peaky_finder.py#L806)). The Gaussian/Lorentzian
split (`sigma` vs `gamma`) is only weakly constrained — they share one bound
and are not regularised. Solver failures are caught and printed, leaving that
window unfit.

---

## 4. Shoulders + global refit + sanity filters

**Entry:** [`fit_shoulders`](../alibz/peaky_finder.py#L926),
[`fit_all`](../alibz/peaky_finder.py#L1058), filters in
[`fit_spectrum`](../alibz/peaky_finder.py#L1252).

1. **`fit_shoulders`** finds residual structure exceeding `mean + 2·std`, and
   adds a new Voigt wherever a residual maximum is farther than one linewidth
   from any existing peak. Existing peaks are held **fixed** while the new
   shoulder floats.
2. **`fit_all`** is, despite the name, a **windowed block-coordinate refit**,
   *not* a single global joint solve. It processes peaks strongest-first; for
   each, it jointly refits only the peaks whose indices fall in a `±50 px`
   window ([:1096-1103](../alibz/peaky_finder.py#L1096)) and subtracts all
   out-of-window peaks as a fixed pedestal
   ([:1128-1132](../alibz/peaky_finder.py#L1128)). Widths are capped at the
   window span — an unbounded width once let a `sigma = 289 nm` component absorb
   89% of the fitted area ([:1110-1118](../alibz/peaky_finder.py#L1110)).
3. **Filters** ([:1280-1298](../alibz/peaky_finder.py#L1280)): drop components
   whose width exceeds `100·median_FWHM` (baseline remnants, using a robust
   `median_FWHM` restricted to widths `< span/50`), snap sub-pixel widths to 0,
   and apply a final **significance gate**: implied height must exceed
   `2·local noise` or the component is dropped.

The survivors, sorted by descending area, are `sorted_parameter_array` — the
first-pass fit the notebook shows as "303 peaks".

---

## 5. Second-iteration refinement — `refine_fit`

**Entry:** [`refine_fit`](../alibz/refinement.py#L419),
classifier [`classify_feature`](../alibz/refinement.py#L224),
model [`sa_voigt`](../alibz/refinement.py#L56).

The first pass fits everything with **symmetric** Voigts, which fails two ways
on real spectra: a **self-absorbed line** (flat-topped/shaded) gets split into
two *phantom* components, and a **genuine blend** narrower than one FWHM gets
fit as one peak. `refine_fit` re-examines every ambiguous feature (sub-FWHM
component pairs, and significant leftover residual structure) under three local
models:

| model | form | params |
|---|---|---|
| **S** | one symmetric Voigt | 4 |
| **A** | Voigt attenuated by a cold same-shape absorber: `A·V(x−μ)·exp(−τ_a·V(x−μ−δ)/V(0))` | 6 |
| **B** | two symmetric Voigts | 8 |

They are compared by **noise-rescaled BIC** (raw χ² is rescaled by the best
model's reduced χ², floored at 1, so margins measure *relative* improvement on
systematics-dominated real windows), and the verdict is **gated by physics**
([:309-398](../alibz/refinement.py#L309)):

- a **blend** verdict needs two *distinct* database lines matching the two
  fitted centers with a consistent separation (not just "some line within
  tolerance"), and a large BIC margin;
- an **asymmetric** (self-absorption) verdict is only *applied* when the
  primary line has a **resonance-capable lower level** (`Eᵢ ≤ 0.2 eV`) — the
  class of lines that self-absorb — and the fitted emission center hasn't
  walked off that line. Otherwise it is recorded under a qualified name
  (`asymmetric-nonresonant`, `-displaced`, `-saturated`) and **left alone**.

### 5a. Self-absorption and the two areas — read this before trusting a merged row

This is the subtle part. When model A wins on a resonance line, `refine_fit`
**merges** the phantom split into one line and records
([:534-553](../alibz/refinement.py#L534)):

| quantity | definition | where it lives |
|---|---|---|
| `emission_area` = `A` | the **unattenuated, thin-equivalent** area — the flux the line *would* emit with no self-absorption (∝ n·gA, the quantity CF-LIBS quantification needs) | decision record (`params_asym`) |
| `observed_area` | the integral of the **attenuated** profile actually seen, wing-corrected for the Lorentzian flux outside the fit window | the peak-table row |
| `tau_a`, `delta` | absorber optical depth and center shift | decision record |

So there are **two** areas, and "correct area" is ambiguous unless you say
which: the table row carries the flux that is faithful to the *data*
(`observed_area`); the physically meaningful emission area for quantification
(`emission_area`) is in the decision record. **Do not** additionally apply a
downstream self-absorption correction to a merged row — the attenuation is
already accounted for.

**Why the residual looks worse — and why that is *not* a loss of fit fidelity.**
The merged row is stored as a plain **symmetric** Voigt carrying
`observed_area` ([:553](../alibz/refinement.py#L553),
[:569](../alibz/refinement.py#L569)), because every downstream consumer
(peak table, uncertainties, indexer) assumes uniform `[area, μ, σ, γ]` rows.
That symmetric proxy conserves *flux* but not *shape*, so it leaves a
core-shaped residual on a flat-topped line — which is what raises the
full-spectrum RMS after refinement. **It is a storage choice, not an inability
to fit.** Model A itself follows the self-absorbed shape well.

Measured on MW2-112 spectrum 0 (residual RMS in each merged line's window):

| line | τ_a | observed→emission | plain-Voigt proxy (stored) | model A (the fit) | first-pass 2-phantom |
|---|---|---|---|---|---|
| Li I 670.7 | 0.59 | 1488 → 2109 (×1.42) | **227** | 59 | 100 |
| K I 766.5 | 0.46 | 1097 → 1381 (×1.26) | **101** | 19 | 89 |
| Fe 261.2 | 3.30 | 78 → 465 (×5.9) | **141** | 22 | 21 |
| 383.0 (7 comps) | 1.66 | 926 → 2028 (×2.19) | **640** | 549 | 76 |

Three conclusions:

1. **Fidelity is preserved at the fit.** Model A's residual is 2–8× *lower*
   than the stored symmetric proxy, and for clean lines (Li 670.7, K 766.5) it
   *beats* the 8-parameter two-phantom fit while using fewer parameters and
   yielding a physical `τ` and an emission area. The proxy's residual overstates
   the true modeling cost.
2. **The emission/observed ratio is real and useful.** It ranges 1.24–7.0
   across the 15 merged lines; ignoring it (using the raw two-phantom areas)
   would feed self-absorbed, under-counted flux into quantification.
3. **There is a genuine cost, in crowded windows.** Where a window holds many
   first-pass components (e.g. 383.0 nm collapses 7), even model A underfits
   (RMS 549 vs the two-phantom 76): the resonance-primary gate lets the merge
   fire, but the window may also contain other real lines. **These are the cases
   where the merge does trade real pointwise fidelity** — a reviewer should
   treat high-`n_phantom` merges (visible in the decision record) with
   suspicion.

Every number above comes from the `decisions` list returned by `refine_fit`:
each merged decision carries `observed_area`, `emission_area`, `tau_a`,
`delta_nm`, `params_asym`, and its `window`, so the comparison is reproducible
directly from the notebook's `refined, decisions = refine_fit(...)` call.

### 5b. Verdicts and actions

| verdict | action | effect |
|---|---|---|
| `blend` (db-supported) | split | replace with the two-Voigt refit / add the second component |
| `asymmetric` (resonance) | merge | replace phantoms with one row @ `observed_area`; record emission area + τ |
| `single` on a pair | merge | the symmetric single Voigt won even after the BIC penalty; the split was redundant |
| `asymmetric-nonresonant/-displaced/-saturated`, `ambiguous` | none | recorded, not applied |

A decision whose noise rescale exceeds `s2_action_max` (window explains nothing
at the stated noise) is recorded but not acted on.

---

## 6. Prior-driven minor-line seeding — `seed_minor_lines`

**Entry:** [`seed_minor_lines`](../alibz/minor_lines.py#L172),
scaling [`match_and_scale`](../alibz/minor_lines.py#L79).

Once an element is established from its strong lines, every *other* line of that
element in the same ion stage is predicted: within one stage the Boltzmann
factor `gA·exp(−Eₖ/kT)` fixes all line ratios up to a single per-(element,
stage) area **scale**, measured from that stage's clean strong lines. Those
predictions let the pass (1) fit low-intensity lines the blind pass rejected,
and (2) deblend an established line hidden under a stronger line of another
element (fit jointly, center pinned at the db position).

- **Scale calibration** ([match_and_scale](../alibz/minor_lines.py#L118)):
  per stage, keep a db line as a reference only if its same-element multiplet
  contamination is ≤10%, its nearest fitted peak is within `tol_nm = 0.06` and
  clears an SNR-derived amplitude, and no other peak blends it. The scale is the
  robust median of `peak_area / strength` over references; its MAD is the
  `spread`.
- **Trust + falsification gates** ([:245-319](../alibz/minor_lines.py#L245)):
  a stage predicts only with `≥ min_ref_lines = 3` consistent references and
  `spread ≤ max_scale_spread`. A stage is **falsified** if its brightest
  predicted line would far exceed anything actually matched (the spurious-Na II
  case). Every addition must also agree with its Boltzmann prediction to within
  `consistency_factor ≈ 5×`, or it is recorded `inconsistent`/`missing` rather
  than added.
- **Acceptance uses a prior-*free* area** ([:438-472](../alibz/minor_lines.py#L438)):
  the decision is made on an NNLS area against the frozen-geometry basis (not
  the prior-regularised fit), plus a BIC improvement — so the prior justifies
  *looking*, but the *data* decide.

Record actions: `added`, `missing` (confident prediction the data refused),
`inconsistent`, `rejected` (plus internal `coincident-skip` / `no-data` /
`fit-failed`).

**Caveats to review.**
- **Single hardcoded temperature** `DEFAULT_KT_EV = 0.76` eV (corpus-median for
  MW2-112, [minor_lines.py:40](../alibz/minor_lines.py#L40)); all ratios assume
  optically-thin emission at that one T.
- **Same stage only, no Saha** — stage I and II are calibrated independently;
  cross-stage coupling is the indexer's job.
- **Wavelength-frame dependence** — `shift_nm` defaults to 0 and `tol_nm` is
  60 pm, but db wavelengths are vacuum while spectra are air (0.11–0.24 nm
  offset). Correctness depends on the caller passing the right `shift_nm` (the
  notebook does, via `estimate_wavelength_shift`).

---

## 7. Downstream, and *separate*: doublet self-absorption anchoring (indexer)

**Not part of the notebook fit.** The indexer has its *own* self-absorption
mechanism ([`_freeze_doublet_taus`](../alibz/peaky_indexer_v3.py#L1141)) that
inverts the measured area ratio of two resolved ground-state resonance lines
into an optical depth to anchor a species' self-absorption. It is:

- **gated off by default** (`sa_doublets = False`);
- **distinct** from Pass 2's `sa_voigt` — it works on the indexer's line table,
  not the fitted profile, and its saturation cap is `τ_max = 50` (vs Pass 2's
  `TAU_MAX = 8`);
- hardened (commit 60e0939) to sum fitter-split components back into one feature
  and to reject coincidence pairs that invert to the saturated cap.

Mentioned here only so it is not mistaken for the refinement above.

---

## How to review / reproduce

- **Live walkthrough:** [`notebooks/fit_inspection.ipynb`](../notebooks/fit_inspection.ipynb)
  runs Pass 1 → 2 → 3 on MW2-112 spectrum 0 and visualises each with
  `plot_spectrum_overview` / `plot_peak_zoom`, the refinement decision records,
  and the seeded minor lines.
- **The two areas in code:** [refinement.py:534-553](../alibz/refinement.py#L534).
- **Every `file:line` anchor above** points at the exact routine; the honest
  caveats are collected inline per stage rather than hidden.
