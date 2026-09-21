# Stage-by-stage least-squares residual audit — alibz pipeline

**Task:** measurement only. No pipeline code was changed. This audit walks
`alibz.pipeline.analyze_spectrum` stage by stage (reusing the exact stage
sequence and function calls from `scripts/fit_inspector.py::capture`, per
the brief), additionally instrumenting the internal steps of
`PeakyFinder.fit_spectrum` (fit_peaks -> fit_shoulders -> fit_all ->
post-fit filters), and measures where the pixel-space least-squares
residual against the data increases, and attributes each increase to the
action responsible.

**New script:** `scripts/audit_stage_residuals.py` (reusable; run with
`--file FILE.csv --outdir DIR` or `--all`).

**Data:** `data/remote_samples/{REE_01,REE_44,argon_noAr,scan9x9}.csv`,
loaded via `alibz.pipeline.load_spectrum_csv`. Database:
`alibz.pipeline.resolve_dbpath()` (this checkout's `db/`).

**Commands run:**
```
python3 scripts/audit_stage_residuals.py --file data/remote_samples/argon_noAr.csv \
    --outdir <scratchpad>/audit --summary-json <scratchpad>/audit/summary_argon.json
python3 scripts/audit_stage_residuals.py --file data/remote_samples/REE_01.csv \
    --outdir <scratchpad>/audit --summary-json <scratchpad>/audit/summary_REE_01.json
python3 scripts/audit_stage_residuals.py --file data/remote_samples/REE_44.csv \
    --outdir <scratchpad>/audit --summary-json <scratchpad>/audit/summary_REE_44.json
python3 scripts/audit_stage_residuals.py --file data/remote_samples/scan9x9.csv \
    --outdir <scratchpad>/audit --summary-json <scratchpad>/audit/summary_scan9x9.json
```
(argon_noAr run twice: first attempt hit a numpy array-truthiness bug in
the script, fixed, then re-run; the other three ran once each, in
parallel background processes, after the fix was validated.)

**Timings (wall-clock, this machine, `.venv` interpreter):**

| sample | seconds | stages captured |
|---|---:|---:|
| argon_noAr | 91.5 | 13 |
| REE_01 | 33.2 | 10 (no deepening rounds added: 0 confident elements) |
| REE_44 | 191.4 | 13 |
| scan9x9 | 391.3 | 13 |

No failures — all 4 files completed. All raw CSVs (`stage_metrics_<sample>.csv`,
`window_attribution_<sample>.csv`) and JSON summaries
(`summary_<sample>.json`) are in the scratchpad directory used for this
run; the tables below are built directly from them (see
`<scratchpad>/build_report.py`).

---

## Methodology

- **Residual metric**: for every stage, `model = multi_voigt(x, peaks)`
  evaluated on the native `x` grid; `resid = y_bgsub - model` where
  `y_bgsub = y - fit['background']` (background fixed at the value
  computed once by the initial `fit_spectrum` call, per
  `fit_inspector.capture` and `analyze_spectrum`, which never
  recompute it). SSE, RMS reported per segment (UV `<365` nm, VIS
  `365-620` nm, NIR `>620` nm, from `PeakyFinder.DEFAULT_SEGMENT_EDGES`)
  and total.
- **Noise-normalized chi2**: `sum((resid/noise_local)**2)` where
  `noise_local = PeakyFinder._noise_scale_local(y_bgsub,
  segment_indices=...)` computed once (same field used throughout the
  real pipeline for significance gates). An "effective" `chi2/4` is also
  reported as an indicative correction for pixel-to-pixel correlation —
  **this is a stated caveat, not a rigorous effective-N**: the export
  grid is oversampled ~3-5x relative to native pixels with
  rho(1) ~ 0.8-0.9 (per prior project notes), so raw chi2 overstates
  significance; dividing by 4 is a rough compensating factor, not derived
  from an autocorrelation measurement on these 4 files.
- **Internal fit_spectrum steps**: captured via a thin tracing subclass
  (`TracedFinder` in the audit script) that overrides `fit_peaks`,
  `fit_shoulders`, `fit_all` to snapshot their return value (deep copy)
  before returning it *unchanged* — the real, unmodified, inherited
  `fit_spectrum` method (including the post-fit filter block) still runs
  exactly as shipped. The filter-block "reason" per component (sigma/gamma
  snap-to-zero, `>100x` median-FWHM drop, height-vs-local-noise gate) is
  re-derived from the exact same formulas read from
  `alibz/peaky_finder.py` (`fit_spectrum` lines ~1330-1362) applied to the
  before/after component arrays — this is a diagnostic re-derivation, not
  a re-fit, so it cannot be wrong about *what* changed (that's measured
  directly), only about *why* in the rare case two reasons could apply
  simultaneously (both are then listed).
- **Window delta-SSE attribution**: rather than a blind full-spectrum
  window scan, each action record (a `refine_fit` decision, a
  `seed_minor_lines`/`recover_residual_lines` "added" record, a
  `deblend_shoulders` "deblended" record, or a re-derived filter action)
  supplies its own center wavelength; the window is `center +/- 1.0 nm`.
  `delta_sse = sum((y_bgsub-model_after)**2) - sum((y_bgsub-model_before)**2)`
  restricted to that window, evaluated between the two stages' full-array
  models. This is anchored on the actions themselves (guaranteed to find
  the responsible action, since it *is* the window definition) rather
  than a symmetric measure-then-attribute search — see caveats below for
  what this trades away.
- **Degeneracy stats**: per retained component of the **blind fit**
  (`fit['peak_dictionary']`, the dict `fit_spectrum` mutates in place
  during its own filter block — i.e. exactly the post-filter, pre-`refine_fit`
  state), `sigma==0` / `gamma==0` are exact (the filter block literally
  sets them to `0.0` below half a pixel). "At upper bound" is an
  **approximation**: it recomputes `fwhm_cap = finder._fwhm_cap(x, y_bgsub,
  peak_indices)` (the same helper `fit_all` uses) and flags
  `sigma >= 0.995*sigma_max` or `gamma >= 0.995*gamma_max` where
  `sigma_max = cap/2.3548...`, `gamma_max = cap/2.0` (with the `H_CAP_RELAX`
  factor near Balmer lines) — but the **true** per-window bound in `fit_all`
  is `min(cap-based, window_span)`, and `window_span` (each peak's local
  fit-window width) is not recoverable after the fact without re-deriving
  the windowing itself. This proxy therefore slightly **undercounts**
  bound-hits that were actually window-span-limited (expected to be rare,
  since fit windows are usually wide relative to the cap, but not
  verified).

---

## Highlights (see per-sample sections below for full detail)

1. **The single most consistent SSE-increasing transition across all 4
   spectra is `1c fit_all -> 1d post-fit filters`** (the width-snap /
   `>100x`-median-FWHM / height-vs-noise-gate filter block). It increases
   *total* SSE in every one of the 4 files (argon +148k / +0.25%,
   REE_01 +2.4k / +0.62%, REE_44 +1.02M / +2.3%, scan9x9 +1.27M / +0.26%)
   and is overwhelmingly attributable to **gamma/sigma snap-to-zero** and
   the **height-vs-local-noise gate drop**, per the top-window tables —
   i.e. genuinely low-significance components being zeroed, which is the
   filter's documented intent, paid for in pointwise fidelity with **no
   quantified cost anywhere in the code or docs** (the filter just zeros
   `value[0]`; no residual accounting is emitted).
2. **`refine_fit`'s asymmetric self-absorption merges (`sa-tag`) do
   measurably raise the residual locally**, consistent with
   `docs/fit_pipeline.md` section 5a's explicit admission (quantified
   there for the *different* MW2-112 spectrum, not for these 4 files).
   Example windows found here: REE_44 248.96 nm (+3.8e4 SSE, tau_a=2.59),
   REE_44 220.999/221.65 nm, REE_01 188.08/188.76 nm (tau_a=2.02/4.43).
   This has a **stated physical justification** (resonance-capable
   self-absorption, BIC-gated) and the doc *does* quantify the cost — but
   only qualitatively via one archetype spectrum's table, not
   automatically per-run; nothing in the pipeline's own output surfaces a
   per-merge residual cost number.
3. **`refine_fit` blend `split` actions are the single largest per-window
   SSE increases observed** in the `1d -> 3a` transition on argon_noAr
   (551.17 nm: +2.9e5; 548.05 nm: +1.8e5; 547.43 nm: +9.7e4) even though
   the *stage total* SSE for that transition actually *decreased*
   overall (-6.6e6) — the split action's own window got locally worse
   while unrelated windows elsewhere improved far more. This is exactly
   the kind of transition the brief's segment-level increase criterion is
   built to catch (a decreasing *total* can still hide a real local
   regression); the split verdict's justification is purely statistical +
   database-line-count (BIC margin), with **no cost accounting**.
4. **Indexer whole-pattern r2 (amplitude-space) does not track pixel-space
   SSE monotonically, and sometimes craters at pass 2**: scan9x9 goes
   0.969 (pass 1) -> 0.237 (pass 2) -> 0.188 (final) even as pixel SSE
   keeps falling monotonically (7.3e8 -> ... -> 3.1e7); REE_44 shows the
   same pattern (0.736 -> 0.048 -> 0.705, i.e. it partially recovers
   during deepening, scan9x9 does not). This means **pixel-space fit
   quality and amplitude-space (composition) fit quality are not
   interchangeable diagnostics** — a pipeline stage can *improve* the
   pointwise residual while *degrading* the whole-pattern chemical fit
   (candidate matrix churn as new/removed lines change which species
   explain which peaks). This was not something the brief asked me to
   explain, only to report side-by-side (task 4) — flagged here as the
   most actionable-looking finding for follow-up, not diagnosed further.
5. **`recover_sa_areas` (the final documented stage) never changes the
   pixel-space model** — it only rescales `indexer._obs_amp` and re-solves
   the composition linearly; `final['sorted_parameter_array']` (the peak
   table `multi_voigt` is evaluated against) is untouched. So "profiles ->
   recover_sa_areas" contributes **zero** pixel-space SSE change in all 4
   files (verified: stage "10" SSE is bit-identical to the last deepening
   stage in every stage table below) — only the reported r2 changes. This
   is a structural fact of the code, not a per-file measurement.

---

## Per-spectrum stage tables, attribution, degeneracy, r2 progression

### argon_noAr -- per-stage residual table

| # | stage | n_peaks | UV SSE | VIS SSE | NIR SSE | total SSE | total RMS | total chi2 | total chi2/4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1a . fit_peaks (windowed Voigt seeds) | 274 | 1.224e+05 | 3.073e+07 | 1.768e+09 | 1.799e+09 | 277.1 | 2.187e+07 | 5.469e+06 |
| 1 | 1b . fit_shoulders (+residual components) | 274 | 1.224e+05 | 3.073e+07 | 1.768e+09 | 1.799e+09 | 277.1 | 2.187e+07 | 5.469e+06 |
| 2 | 1c . fit_all (block-coordinate joint refit) | 274 | 6.621e+04 | 4.868e+06 | 5.448e+07 | 5.941e+07 | 50.36 | 9.625e+05 | 2.406e+05 |
| 3 | 1d . post-fit filters (= 1 . blind fit) | 241 | 6.816e+04 | 4.881e+06 | 5.461e+07 | 5.956e+07 | 50.42 | 9.678e+05 | 2.419e+05 |
| 4 | 3a . data-only refine (asym deferred) | 228 | 6.604e+04 | 5.598e+06 | 4.73e+07 | 5.297e+07 | 47.54 | 9.562e+05 | 2.39e+05 |
| 5 | 3b . self-absorption tags | 228 | 6.604e+04 | 5.598e+06 | 4.647e+07 | 5.214e+07 | 47.17 | 9.376e+05 | 2.344e+05 |
| 6 | 5 . Boltzmann-seeded minor lines | 240 | 6.394e+04 | 5.382e+06 | 4.647e+07 | 5.192e+07 | 47.07 | 9.177e+05 | 2.294e+05 |
| 7 | 6 . residual recovery | 272 | 6.395e+04 | 3.054e+06 | 1.936e+07 | 2.248e+07 | 30.98 | 4.58e+05 | 1.145e+05 |
| 8 | 7 . shoulder deblends | 284 | 6.333e+04 | 2.969e+06 | 8.388e+06 | 1.142e+07 | 22.08 | 3.319e+05 | 8.296e+04 |
| 9 | 9.0 . deepening round bar=3.0 | 303 | 6.334e+04 | 1.778e+06 | 3.195e+06 | 5.036e+06 | 14.66 | 2.071e+05 | 5.179e+04 |
| 10 | 9.1 . deepening round bar=2.5 | 323 | 6.157e+04 | 1.737e+06 | 2.936e+06 | 4.735e+06 | 14.22 | 1.987e+05 | 4.967e+04 |
| 11 | 9.2 . deepening round bar=2.0 | 338 | 6.157e+04 | 1.654e+06 | 2.801e+06 | 4.516e+06 | 13.88 | 1.908e+05 | 4.77e+04 |
| 12 | 10 . final (post SA-area recovery; peaks unchanged) | 338 | 6.157e+04 | 1.654e+06 | 2.801e+06 | 4.516e+06 | 13.88 | 1.908e+05 | 4.77e+04 |

### argon_noAr -- stage transitions where SSE increased (total or per-segment)

| from | to | dUV | dVIS | dNIR | dTOTAL |
|---|---|---:|---:|---:|---:|
| 1c . fit_all (block-coordinate joint refit) | 1d . post-fit filters (= 1 . blind fit) | +1955 | +1.273e+04 | +1.333e+05 | +1.48e+05 |
| 1d . post-fit filters (= 1 . blind fit) | 3a . data-only refine (asym deferred) | -2122 | +7.171e+05 | -7.312e+06 | -6.597e+06 |
| 3a . data-only refine (asym deferred) | 3b . self-absorption tags | +0.362 | -172.4 | -8.271e+05 | -8.273e+05 |
| 3b . self-absorption tags | 5 . Boltzmann-seeded minor lines | -2103 | -2.156e+05 | +515 | -2.172e+05 |
| 5 . Boltzmann-seeded minor lines | 6 . residual recovery | +4.262 | -2.328e+06 | -2.711e+07 | -2.944e+07 |
| 7 . shoulder deblends | 9.0 . deepening round bar=3.0 | +8.946 | -1.191e+06 | -5.193e+06 | -6.384e+06 |
| 9.1 . deepening round bar=2.5 | 9.2 . deepening round bar=2.0 | +0.734 | -8.305e+04 | -1.352e+05 | -2.182e+05 |

**argon_noAr: 9.1 . deepening round bar=2.5 -> 9.2 . deepening round bar=2.0** (top windows by +dSSE)

| window center (nm) | n_px | +dSSE | action kind | action/verdict | n_comp | tau_a | bar | justification (truncated) |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 430.2833 | 60 | 2868 | recover_residual_lines(deepen) | added |  |  | 2.0 | statistical: matched-filter SNR + BIC gate on the predicted (minor_lines) or residual (rec... |

**argon_noAr: 1c . fit_all (block-coordinate joint refit) -> 1d . post-fit filters (= 1 . blind fit)** (top windows by +dSSE)

| window center (nm) | n_px | +dSSE | action kind | action/verdict | n_comp | tau_a | bar | justification (truncated) |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 620.1 | 61 | 7.75e+04 | filter | sigma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 620.0032 | 60 | 7.75e+04 | filter | sigma snap-to-zero (<0.5 px); gamma snap-to-zero (<0.5 px); height-vs-local-noise gate drop (min_height=19.16) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 766.4279 | 60 | 5.437e+04 | filter | gamma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 526.1667 | 61 | 9427 | filter | gamma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 518.8484 | 60 | 1377 | filter | gamma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 639.5031 | 60 | 1331 | filter | gamma snap-to-zero (<0.5 px); height-vs-local-noise gate drop (min_height=19.16) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 362.1423 | 60 | 1193 | filter | gamma snap-to-zero (<0.5 px); height-vs-local-noise gate drop (min_height=7.788) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 362.6667 | 59 | 1168 | filter | height-vs-local-noise gate drop (min_height=7.788) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 425.2667 | 59 | 758.5 | filter | height-vs-local-noise gate drop (min_height=7.716) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 398.3333 | 59 | 650.5 | filter | height-vs-local-noise gate drop (min_height=9.28) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |

**argon_noAr: 1d . post-fit filters (= 1 . blind fit) -> 3a . data-only refine (asym deferred)** (top windows by +dSSE)

| window center (nm) | n_px | +dSSE | action kind | action/verdict | n_comp | tau_a | bar | justification (truncated) |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 551.1667 | 61 | 2.931e+05 | refine_fit | split | 1 |  |  | statistical + database: >=2 distinct db lines match the two fitted centers with consistent... |
| 548.052 | 60 | 1.813e+05 | refine_fit | split | 2 |  |  | statistical + database: >=2 distinct db lines match the two fitted centers with consistent... |
| 620.3831 | 60 | 1.073e+05 | refine_fit | none | 2 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 547.4312 | 60 | 9.726e+04 | refine_fit | split | 2 |  |  | statistical + database: >=2 distinct db lines match the two fitted centers with consistent... |
| 558.8333 | 61 | 1.143e+04 | refine_fit | deferred | 1 | 0.9865745082761106 |  | asymmetric verdict recorded but deliberately NOT applied in this data-only pass (asymmetri... |
| 620.9667 | 61 | 7278 | refine_fit | none | 1 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 612.2606 | 60 | 5816 | refine_fit | split | 7 |  |  | statistical + database: >=2 distinct db lines match the two fitted centers with consistent... |
| 559.4104 | 60 | 5515 | refine_fit | deferred | 1 | 4.955681041299571 |  | asymmetric verdict recorded but deliberately NOT applied in this data-only pass (asymmetri... |
| 606.6794 | 60 | 1936 | refine_fit | split | 2 |  |  | statistical + database: >=2 distinct db lines match the two fitted centers with consistent... |
| 431.8333 | 59 | 990.7 | refine_fit | none | 1 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |

**argon_noAr: 3b . self-absorption tags -> 5 . Boltzmann-seeded minor lines** (top windows by +dSSE)
(no positive-dSSE windows recorded for this transition -- either no action records had a center, or all per-window deltas were <=0 even though the stage total/segment increased -- e.g. many small negative-delta windows outweighed by pixels outside any recorded action window)

**argon_noAr: 5 . Boltzmann-seeded minor lines -> 6 . residual recovery** (top windows by +dSSE)

| window center (nm) | n_px | +dSSE | action kind | action/verdict | n_comp | tau_a | bar | justification (truncated) |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 443.8478 | 60 | 743.3 | recover_residual_lines | added |  |  |  | statistical: matched-filter SNR + BIC gate on the predicted (minor_lines) or residual (rec... |

**argon_noAr: 3a . data-only refine (asym deferred) -> 3b . self-absorption tags** (top windows by +dSSE)

| window center (nm) | n_px | +dSSE | action kind | action/verdict | n_comp | tau_a | bar | justification (truncated) |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 622.3345 | 60 | 234.9 | refine_fit | none | 4 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 625.3812 | 60 | 80.93 | refine_fit | none | 6 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 627.4059 | 60 | 3.139 | refine_fit | none | 3 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 588.953 | 60 | 2.725 | refine_fit | none | 1 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 589.548 | 60 | 2.271 | refine_fit | none | 1 | 3.1374146541522796 |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 612.1664 | 60 | 2.004 | refine_fit | none | 1 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 598.4642 | 60 | 1.592 | refine_fit | none | 2 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 445.4499 | 60 | 1.068 | refine_fit | none | 2 | 0.6415439812182605 |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 526.9825 | 60 | 0.7746 | refine_fit | none | 1 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 526.3506 | 60 | 0.7037 | refine_fit | none | 2 | 2.2196015249799865 |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |

**argon_noAr: 7 . shoulder deblends -> 9.0 . deepening round bar=3.0** (top windows by +dSSE)

| window center (nm) | n_px | +dSSE | action kind | action/verdict | n_comp | tau_a | bar | justification (truncated) |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 671.3467 | 60 | 8602 | recover_residual_lines(deepen) | added |  |  | 3.0 | statistical: matched-filter SNR + BIC gate on the predicted (minor_lines) or residual (rec... |
| 393.45 | 60 | 4379 | recover_residual_lines(deepen) | added |  |  | 3.0 | statistical: matched-filter SNR + BIC gate on the predicted (minor_lines) or residual (rec... |
| 444.9167 | 60 | 669 | recover_residual_lines(deepen) | added |  |  | 3.0 | statistical: matched-filter SNR + BIC gate on the predicted (minor_lines) or residual (rec... |


### argon_noAr -- shape-parameter degeneracy (blind fit, n=274)

| segment | n | frac sigma==0 | frac gamma==0 | frac at upper bound (approx) |
|---|---:|---:|---:|---:|
| UV | 55 | 0.073 | 0.600 | 0.036 |
| VIS | 139 | 0.129 | 0.403 | 0.252 |
| NIR | 80 | 0.188 | 0.362 | 0.237 |

Gaussian fraction sigma/(sigma+gamma): median=0.620, IQR=0.607 (n=272)

### argon_noAr -- indexer whole-pattern r2 progression

- pass 1: 0.4156
- pass 2: 0.4408
- deepening rounds: [[3.0, 0.05620217820096718], [2.5, 0.07245157729077512], [2.0, 0.07491711740015217]]
- final (post-deepening): 0.0749
- final (post SA-area recovery, amplitude-space only -- pixel-space peaks unchanged): 0.0514
- timings (s): blind_fit=1.6, pass1=8.1, pass2=7.7, total_wallclock=91.5

---

### REE_01 -- per-stage residual table

| # | stage | n_peaks | UV SSE | VIS SSE | NIR SSE | total SSE | total RMS | total chi2 | total chi2/4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1a . fit_peaks (windowed Voigt seeds) | 227 | 2.459e+04 | 4.842e+05 | 6.883e+06 | 7.392e+06 | 17.76 | 1.637e+06 | 4.094e+05 |
| 1 | 1b . fit_shoulders (+residual components) | 227 | 2.459e+04 | 4.842e+05 | 6.883e+06 | 7.392e+06 | 17.76 | 1.637e+06 | 4.094e+05 |
| 2 | 1c . fit_all (block-coordinate joint refit) | 227 | 2.112e+04 | 2.85e+04 | 3.298e+05 | 3.794e+05 | 4.024 | 4.815e+04 | 1.204e+04 |
| 3 | 1d . post-fit filters (= 1 . blind fit) | 210 | 2.135e+04 | 2.92e+04 | 3.312e+05 | 3.818e+05 | 4.037 | 4.874e+04 | 1.219e+04 |
| 4 | 3a . data-only refine (asym deferred) | 199 | 2.135e+04 | 2.245e+04 | 2.694e+05 | 3.132e+05 | 3.656 | 3.285e+04 | 8214 |
| 5 | 3b . self-absorption tags | 198 | 2.137e+04 | 2.18e+04 | 2.694e+05 | 3.126e+05 | 3.652 | 3.272e+04 | 8181 |
| 6 | 5 . Boltzmann-seeded minor lines | 198 | 2.137e+04 | 2.18e+04 | 2.694e+05 | 3.126e+05 | 3.652 | 3.272e+04 | 8181 |
| 7 | 6 . residual recovery | 206 | 2.028e+04 | 2.068e+04 | 1.635e+05 | 2.044e+05 | 2.954 | 2.569e+04 | 6422 |
| 8 | 7 . shoulder deblends | 209 | 2.028e+04 | 2.043e+04 | 1.507e+05 | 1.914e+05 | 2.858 | 2.487e+04 | 6218 |
| 9 | 10 . final (post SA-area recovery; peaks unchanged) | 209 | 2.028e+04 | 2.043e+04 | 1.507e+05 | 1.914e+05 | 2.858 | 2.487e+04 | 6218 |

### REE_01 -- stage transitions where SSE increased (total or per-segment)

| from | to | dUV | dVIS | dNIR | dTOTAL |
|---|---|---:|---:|---:|---:|
| 1c . fit_all (block-coordinate joint refit) | 1d . post-fit filters (= 1 . blind fit) | +229.4 | +693.8 | +1452 | +2375 |
| 3a . data-only refine (asym deferred) | 3b . self-absorption tags | +23.16 | -650.5 | -0.01518 | -627.4 |
| 6 . residual recovery | 7 . shoulder deblends | +0.02058 | -245.4 | -1.274e+04 | -1.298e+04 |

**REE_01: 1c . fit_all (block-coordinate joint refit) -> 1d . post-fit filters (= 1 . blind fit)** (top windows by +dSSE)

| window center (nm) | n_px | +dSSE | action kind | action/verdict | n_comp | tau_a | bar | justification (truncated) |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 947.7176 | 60 | 625.2 | filter | sigma snap-to-zero (<0.5 px); gamma snap-to-zero (<0.5 px); height-vs-local-noise gate drop (min_height=4.231) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 589.5448 | 60 | 561 | filter | gamma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 629.7048 | 60 | 457.3 | filter | gamma snap-to-zero (<0.5 px); height-vs-local-noise gate drop (min_height=8.595) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 891.042 | 60 | 166.5 | filter | sigma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 891.4823 | 60 | 166.5 | filter | gamma snap-to-zero (<0.5 px); height-vs-local-noise gate drop (min_height=5.022) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 364.251 | 60 | 94.38 | filter | gamma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 364.1854 | 60 | 94.38 | filter | gamma snap-to-zero (<0.5 px); height-vs-local-noise gate drop (min_height=4.429) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 364.9333 | 59 | 94.34 | filter | gamma snap-to-zero (<0.5 px); height-vs-local-noise gate drop (min_height=4.429) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 935.3717 | 60 | 93.64 | filter | gamma snap-to-zero (<0.5 px); height-vs-local-noise gate drop (min_height=4.231) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 935.7 | 61 | 93.63 | filter | sigma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |

**REE_01: 6 . residual recovery -> 7 . shoulder deblends** (top windows by +dSSE)

| window center (nm) | n_px | +dSSE | action kind | action/verdict | n_comp | tau_a | bar | justification (truncated) |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 465.9667 | 59 | 47.9 | deblend_shoulders | deblended |  |  |  | shape-based: one-sided residual flank inconsistent with symmetric Voigt (profiles.py); no ... |

**REE_01: 3a . data-only refine (asym deferred) -> 3b . self-absorption tags** (top windows by +dSSE)

| window center (nm) | n_px | +dSSE | action kind | action/verdict | n_comp | tau_a | bar | justification (truncated) |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 188.7574 | 60 | 41.42 | refine_fit | none | 2 | 4.433728448764647 |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 188.0777 | 60 | 23.36 | refine_fit | sa-tag | 2 | 2.0160387739034578 |  | physical: resonance-capable lower level (Ei<=0.2 eV) + self-absorption growth-curve model ... |
| 193.0853 | 60 | 0.2621 | refine_fit | none | 1 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 383.2173 | 60 | 0.1221 | refine_fit | none | 2 | 3.0029495092450635 |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 383.9029 | 60 | 0.1141 | refine_fit | none | 2 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 468.1674 | 60 | 0.01805 | refine_fit | none | 2 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 471.2205 | 60 | 0.01139 | refine_fit | none | 2 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 466.538 | 60 | 0.008002 | refine_fit | none | 3 | 1.2955948830996307 |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 463.599 | 60 | 0.00556 | refine_fit | none | 2 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 386.519 | 60 | 0.003322 | refine_fit | none | 2 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |


### REE_01 -- shape-parameter degeneracy (blind fit, n=227)

| segment | n | frac sigma==0 | frac gamma==0 | frac at upper bound (approx) |
|---|---:|---:|---:|---:|
| UV | 31 | 0.097 | 0.839 | 0.000 |
| VIS | 123 | 0.089 | 0.569 | 0.146 |
| NIR | 73 | 0.178 | 0.370 | 0.260 |

Gaussian fraction sigma/(sigma+gamma): median=1.000, IQR=0.541 (n=224)

### REE_01 -- indexer whole-pattern r2 progression

- pass 1: -0.2364
- pass 2: -0.4400
- deepening rounds: []
- final (post-deepening): -0.4400
- final (post SA-area recovery, amplitude-space only -- pixel-space peaks unchanged): -0.4609
- timings (s): blind_fit=1.2, pass1=5.9, pass2=6.6, total_wallclock=33.2

---

### REE_44 -- per-stage residual table

| # | stage | n_peaks | UV SSE | VIS SSE | NIR SSE | total SSE | total RMS | total chi2 | total chi2/4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1a . fit_peaks (windowed Voigt seeds) | 480 | 3.278e+06 | 5.425e+07 | 1.062e+09 | 1.12e+09 | 218.6 | 6.825e+06 | 1.706e+06 |
| 1 | 1b . fit_shoulders (+residual components) | 480 | 3.278e+06 | 5.425e+07 | 1.062e+09 | 1.12e+09 | 218.6 | 6.825e+06 | 1.706e+06 |
| 2 | 1c . fit_all (block-coordinate joint refit) | 480 | 9.167e+05 | 8.047e+06 | 3.484e+07 | 4.381e+07 | 43.24 | 4.504e+05 | 1.126e+05 |
| 3 | 1d . post-fit filters (= 1 . blind fit) | 438 | 1.058e+06 | 8.559e+06 | 3.521e+07 | 4.482e+07 | 43.74 | 4.541e+05 | 1.135e+05 |
| 4 | 3a . data-only refine (asym deferred) | 429 | 1.03e+06 | 7.082e+06 | 2.772e+07 | 3.583e+07 | 39.1 | 3.902e+05 | 9.754e+04 |
| 5 | 3b . self-absorption tags | 426 | 1.07e+06 | 7.085e+06 | 2.401e+07 | 3.217e+07 | 37.05 | 3.553e+05 | 8.881e+04 |
| 6 | 5 . Boltzmann-seeded minor lines | 426 | 1.07e+06 | 7.085e+06 | 2.401e+07 | 3.217e+07 | 37.05 | 3.553e+05 | 8.881e+04 |
| 7 | 6 . residual recovery | 453 | 5.712e+05 | 4.988e+06 | 1.239e+07 | 1.795e+07 | 27.67 | 2.298e+05 | 5.744e+04 |
| 8 | 7 . shoulder deblends | 465 | 5.711e+05 | 5.02e+06 | 6.901e+06 | 1.249e+07 | 23.09 | 2.097e+05 | 5.242e+04 |
| 9 | 9.0 . deepening round bar=3.0 | 506 | 5.025e+05 | 3.992e+06 | 5.211e+06 | 9.706e+06 | 20.35 | 1.799e+05 | 4.498e+04 |
| 10 | 9.1 . deepening round bar=2.5 | 521 | 4.778e+05 | 4.664e+06 | 4.554e+06 | 9.696e+06 | 20.34 | 1.647e+05 | 4.119e+04 |
| 11 | 9.2 . deepening round bar=2.0 | 530 | 4.408e+05 | 4.481e+06 | 4.391e+06 | 9.312e+06 | 19.94 | 1.626e+05 | 4.064e+04 |
| 12 | 10 . final (post SA-area recovery; peaks unchanged) | 530 | 4.408e+05 | 4.481e+06 | 4.391e+06 | 9.312e+06 | 19.94 | 1.626e+05 | 4.064e+04 |

### REE_44 -- stage transitions where SSE increased (total or per-segment)

| from | to | dUV | dVIS | dNIR | dTOTAL |
|---|---|---:|---:|---:|---:|
| 1c . fit_all (block-coordinate joint refit) | 1d . post-fit filters (= 1 . blind fit) | +1.408e+05 | +5.116e+05 | +3.627e+05 | +1.015e+06 |
| 3a . data-only refine (asym deferred) | 3b . self-absorption tags | +4.046e+04 | +3863 | -3.704e+06 | -3.66e+06 |
| 6 . residual recovery | 7 . shoulder deblends | -43.86 | +3.202e+04 | -5.485e+06 | -5.453e+06 |
| 9.0 . deepening round bar=3.0 | 9.1 . deepening round bar=2.5 | -2.469e+04 | +6.719e+05 | -6.569e+05 | -9671 |

**REE_44: 1c . fit_all (block-coordinate joint refit) -> 1d . post-fit filters (= 1 . blind fit)** (top windows by +dSSE)

| window center (nm) | n_px | +dSSE | action kind | action/verdict | n_comp | tau_a | bar | justification (truncated) |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 620.0165 | 60 | 3.598e+05 | filter | sigma snap-to-zero (<0.5 px); gamma snap-to-zero (<0.5 px); height-vs-local-noise gate drop (min_height=26.92) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 429.5 | 60 | 2.819e+05 | filter | sigma snap-to-zero (<0.5 px); height-vs-local-noise gate drop (min_height=54.97) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 430.2208 | 60 | 2.816e+05 | filter | gamma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 288.1153 | 60 | 7.341e+04 | filter | gamma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 390.5278 | 60 | 6.647e+04 | filter | gamma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 391.6333 | 59 | 6.505e+04 | filter | height-vs-local-noise gate drop (min_height=87.36) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 392.0 | 59 | 6.488e+04 | filter | sigma snap-to-zero (<0.5 px); height-vs-local-noise gate drop (min_height=87.36) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 392.2667 | 59 | 5.983e+04 | filter | height-vs-local-noise gate drop (min_height=87.36) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 398.1767 | 60 | 3.973e+04 | filter | gamma snap-to-zero (<0.5 px); height-vs-local-noise gate drop (min_height=87.36) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 398.9667 | 59 | 3.971e+04 | filter | sigma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |

**REE_44: 9.0 . deepening round bar=3.0 -> 9.1 . deepening round bar=2.5** (top windows by +dSSE)

| window center (nm) | n_px | +dSSE | action kind | action/verdict | n_comp | tau_a | bar | justification (truncated) |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 396.4887 | 60 | 5.581e+05 | recover_residual_lines(deepen) | added |  |  | 2.5 | statistical: matched-filter SNR + BIC gate on the predicted (minor_lines) or residual (rec... |
| 443.5487 | 60 | 3233 | recover_residual_lines(deepen) | added |  |  | 2.5 | statistical: matched-filter SNR + BIC gate on the predicted (minor_lines) or residual (rec... |

**REE_44: 6 . residual recovery -> 7 . shoulder deblends** (top windows by +dSSE)

| window center (nm) | n_px | +dSSE | action kind | action/verdict | n_comp | tau_a | bar | justification (truncated) |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 627.0783 | 60 | 5.576e+04 | deblend_shoulders | deblended |  |  |  | shape-based: one-sided residual flank inconsistent with symmetric Voigt (profiles.py); no ... |
| 772.8152 | 60 | 3378 | deblend_shoulders | deblended |  |  |  | shape-based: one-sided residual flank inconsistent with symmetric Voigt (profiles.py); no ... |
| 732.4076 | 60 | 1262 | deblend_shoulders | deblended |  |  |  | shape-based: one-sided residual flank inconsistent with symmetric Voigt (profiles.py); no ... |

**REE_44: 3a . data-only refine (asym deferred) -> 3b . self-absorption tags** (top windows by +dSSE)

| window center (nm) | n_px | +dSSE | action kind | action/verdict | n_comp | tau_a | bar | justification (truncated) |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 248.9606 | 60 | 3.802e+04 | refine_fit | sa-tag | 2 | 2.5888142991535665 |  | physical: resonance-capable lower level (Ei<=0.2 eV) + self-absorption growth-curve model ... |
| 221.6543 | 60 | 1.349e+04 | refine_fit | none | 1 | 2.1612384730964336 |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 220.9992 | 60 | 1.322e+04 | refine_fit | sa-tag | 2 | 3.7758877638843615 |  | physical: resonance-capable lower level (Ei<=0.2 eV) + self-absorption growth-curve model ... |
| 503.9866 | 60 | 4085 | refine_fit | sa-tag | 2 | 2.7523013777648817 |  | physical: resonance-capable lower level (Ei<=0.2 eV) + self-absorption growth-curve model ... |
| 623.308 | 60 | 484 | refine_fit | none | 2 | 1.2938200527447377 |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 766.45 | 60 | 317.9 | refine_fit | none | 2 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 621.0865 | 60 | 277.7 | refine_fit | none | 2 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 277.8297 | 60 | 224.9 | refine_fit | none | 2 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 763.4802 | 60 | 202.1 | refine_fit | none | 1 |  |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |
| 656.1328 | 60 | 82.88 | refine_fit | none | 2 | 2.195924618453352 |  | verdict recorded but NOT applied (BIC margin insufficient / noise_rescale too high / not t... |


### REE_44 -- shape-parameter degeneracy (blind fit, n=480)

| segment | n | frac sigma==0 | frac gamma==0 | frac at upper bound (approx) |
|---|---:|---:|---:|---:|
| UV | 184 | 0.098 | 0.707 | 0.033 |
| VIS | 162 | 0.105 | 0.586 | 0.074 |
| NIR | 134 | 0.119 | 0.418 | 0.284 |

Gaussian fraction sigma/(sigma+gamma): median=1.000, IQR=0.541 (n=478)

### REE_44 -- indexer whole-pattern r2 progression

- pass 1: 0.7365
- pass 2: 0.0484
- deepening rounds: [[3.0, 0.6695469719185021], [2.5, 0.702841432503472], [2.0, 0.7045230067128126]]
- final (post-deepening): 0.7045
- final (post SA-area recovery, amplitude-space only -- pixel-space peaks unchanged): 0.6982
- timings (s): blind_fit=2.2, pass1=5.8, pass2=6.4, total_wallclock=191.4

---

### scan9x9 -- per-stage residual table

| # | stage | n_peaks | UV SSE | VIS SSE | NIR SSE | total SSE | total RMS | total chi2 | total chi2/4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1a . fit_peaks (windowed Voigt seeds) | 426 | 5.335e+07 | 3.416e+08 | 3.345e+08 | 7.295e+08 | 176.4 | 4.007e+06 | 1.002e+06 |
| 1 | 1b . fit_shoulders (+residual components) | 428 | 5.019e+07 | 3.416e+08 | 3.345e+08 | 7.263e+08 | 176.1 | 3.996e+06 | 9.989e+05 |
| 2 | 1c . fit_all (block-coordinate joint refit) | 428 | 8.883e+06 | 1.82e+08 | 3.005e+08 | 4.914e+08 | 144.8 | 3.074e+06 | 7.685e+05 |
| 3 | 1d . post-fit filters (= 1 . blind fit) | 385 | 1.007e+07 | 1.82e+08 | 3.006e+08 | 4.926e+08 | 145 | 3.079e+06 | 7.697e+05 |
| 4 | 3a . data-only refine (asym deferred) | 390 | 7.894e+06 | 1.82e+08 | 2.988e+08 | 4.887e+08 | 144.4 | 3.055e+06 | 7.637e+05 |
| 5 | 3b . self-absorption tags | 386 | 7.619e+06 | 2.965e+07 | 2.861e+08 | 3.234e+08 | 117.5 | 2.756e+06 | 6.89e+05 |
| 6 | 5 . Boltzmann-seeded minor lines | 386 | 7.619e+06 | 2.965e+07 | 2.861e+08 | 3.234e+08 | 117.5 | 2.756e+06 | 6.89e+05 |
| 7 | 6 . residual recovery | 411 | 3.567e+06 | 2.426e+07 | 6.754e+07 | 9.536e+07 | 63.79 | 8.277e+05 | 2.069e+05 |
| 8 | 7 . shoulder deblends | 423 | 3.489e+06 | 2.251e+07 | 6.408e+07 | 9.007e+07 | 62 | 7.154e+05 | 1.789e+05 |
| 9 | 9.0 . deepening round bar=3.0 | 489 | 2.674e+06 | 2.089e+07 | 1.393e+07 | 3.75e+07 | 40 | 2.168e+05 | 5.421e+04 |
| 10 | 9.1 . deepening round bar=2.5 | 517 | 2.446e+06 | 2.155e+07 | 8.996e+06 | 3.299e+07 | 37.52 | 2.335e+05 | 5.837e+04 |
| 11 | 9.2 . deepening round bar=2.0 | 533 | 2.075e+06 | 2.034e+07 | 8.841e+06 | 3.126e+07 | 36.53 | 1.343e+05 | 3.356e+04 |
| 12 | 10 . final (post SA-area recovery; peaks unchanged) | 533 | 2.075e+06 | 2.034e+07 | 8.841e+06 | 3.126e+07 | 36.53 | 1.343e+05 | 3.356e+04 |

### scan9x9 -- stage transitions where SSE increased (total or per-segment)

| from | to | dUV | dVIS | dNIR | dTOTAL |
|---|---|---:|---:|---:|---:|
| 1a . fit_peaks (windowed Voigt seeds) | 1b . fit_shoulders (+residual components) | -3.164e+06 | +1.013 | +0.2462 | -3.164e+06 |
| 1c . fit_all (block-coordinate joint refit) | 1d . post-fit filters (= 1 . blind fit) | +1.184e+06 | +3.053e+04 | +6.053e+04 | +1.275e+06 |
| 9.0 . deepening round bar=3.0 | 9.1 . deepening round bar=2.5 | -2.276e+05 | +6.582e+05 | -4.935e+06 | -4.505e+06 |

**scan9x9: 1c . fit_all (block-coordinate joint refit) -> 1d . post-fit filters (= 1 . blind fit)** (top windows by +dSSE)

| window center (nm) | n_px | +dSSE | action kind | action/verdict | n_comp | tau_a | bar | justification (truncated) |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 251.406 | 60 | 1.09e+06 | filter | gamma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 334.9167 | 60 | 8.068e+04 | filter | gamma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 236.8333 | 61 | 5.91e+04 | filter | gamma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 237.1533 | 60 | 5.871e+04 | filter | sigma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 237.3126 | 60 | 5.863e+04 | filter | gamma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 236.6817 | 60 | 5.846e+04 | filter | gamma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 854.1501 | 60 | 5.685e+04 | filter | gamma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 237.9 | 61 | 5.577e+04 | filter | sigma snap-to-zero (<0.5 px) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 385.3444 | 60 | 2.408e+04 | filter | height-vs-local-noise gate drop (min_height=67.57) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |
| 310.0067 | 60 | 1.415e+04 | filter | gamma snap-to-zero (<0.5 px); height-vs-local-noise gate drop (min_height=52.89) |  |  |  | engineering safeguard: sub-pixel width snap-to-zero, >100x median-FWHM baseline-remnant dr... |

**scan9x9: 1a . fit_peaks (windowed Voigt seeds) -> 1b . fit_shoulders (+residual components)** (top windows by +dSSE)
(no positive-dSSE windows recorded for this transition -- either no action records had a center, or all per-window deltas were <=0 even though the stage total/segment increased -- e.g. many small negative-delta windows outweighed by pixels outside any recorded action window)

**scan9x9: 9.0 . deepening round bar=3.0 -> 9.1 . deepening round bar=2.5** (top windows by +dSSE)

| window center (nm) | n_px | +dSSE | action kind | action/verdict | n_comp | tau_a | bar | justification (truncated) |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 588.9107 | 60 | 7.766e+05 | recover_residual_lines(deepen) | added |  |  | 2.5 | statistical: matched-filter SNR + BIC gate on the predicted (minor_lines) or residual (rec... |
| 588.6962 | 60 | 5.078e+05 | recover_residual_lines(deepen) | added |  |  | 2.5 | statistical: matched-filter SNR + BIC gate on the predicted (minor_lines) or residual (rec... |


### scan9x9 -- shape-parameter degeneracy (blind fit, n=428)

| segment | n | frac sigma==0 | frac gamma==0 | frac at upper bound (approx) |
|---|---:|---:|---:|---:|
| UV | 156 | 0.071 | 0.609 | 0.019 |
| VIS | 137 | 0.073 | 0.555 | 0.088 |
| NIR | 135 | 0.133 | 0.319 | 0.289 |

Gaussian fraction sigma/(sigma+gamma): median=0.841, IQR=0.556 (n=426)

### scan9x9 -- indexer whole-pattern r2 progression

- pass 1: 0.9692
- pass 2: 0.2370
- deepening rounds: [[3.0, 0.1882890595862433], [2.5, 0.1868214683784123], [2.0, 0.1877078406133269]]
- final (post-deepening): 0.1877
- final (post SA-area recovery, amplitude-space only -- pixel-space peaks unchanged): 0.1988
- timings (s): blind_fit=1.9, pass1=5.5, pass2=5.9, total_wallclock=391.3

---

## Unverified / caveats (explicit)

1. **`fit_inspector.capture()` fidelity to `analyze_spectrum` is close but
   not byte-identical**, and this audit inherited the same divergences
   (it follows `capture()` per the brief's instruction to reuse its stage
   sequence): `seed_minor_lines` is called with the *default* `kT_ev`
   (`DEFAULT_KT_EV`) and no `segment_edges` in stages 5/6/7, whereas
   `analyze_spectrum` passes `kT_ev=KB_EV_K*res1.temperature` and
   `segment_edges=(620.0,)`. This could shift which minor lines seed_minor_lines
   accepts and by how much, though not the *mechanism* being measured. Not
   independently re-verified against a full `analyze_spectrum` run on these
   4 files (that would require also reproducing `sa_merges` bookkeeping,
   `telemetry`, and the detection/support report, which was out of scope
   for a residual audit).
2. **"At upper bound" (degeneracy stat, task 3) is an approximation**, as
   detailed in Methodology — the `window_span` component of `fit_all`'s
   true bound is not reconstructed. Treat the reported fractions as a
   lower bound on true bound-hitting.
3. **Window delta-SSE attribution windows (+/-1 nm) are wider than the
   local optimization windows the underlying actions actually use**
   (`recover_residual_lines`/`seed_minor_lines` fit jointly within a
   window of roughly a few line-widths, well under 1 nm in these spectra).
   This means an "added" action's reported window can pick up SSE changes
   from a *different*, nearby action in the same transition, or from
   genuine (small) drift in neighboring components' pedestal subtraction —
   observed directly in the top-window tables where a `recover_residual_lines`
   "added" row shows a *positive* delta-SSE (e.g. scan9x9 588.91/588.70 nm,
   REE_44 396.49 nm) even though the action itself is BIC-gated to improve
   its own fit window. **This is a real limitation of the reporting
   granularity, not evidence the action made things worse** — it was not
   possible to disentangle within the ±1 nm window using only the
   information in the action records (they do not report their own fit
   window's bounds). A tighter, action-specific window (e.g. read the
   actual local window from `_feature_window`/the candidate's `window`
   field, which `refine_fit` decisions *do* carry but `minor_lines`
   records do not) would remove this specific ambiguity for `refine_fit`
   actions; that refinement was not implemented here given the time
   budget.
4. **`refine_fit` "none"/"deferred" verdict rows never directly cause their
   own window's delta-SSE** (by construction — the driver only mutates
   peaks for `split`/`merge`/`sa-tag` actions). Any nonzero delta shown for
   a "none"/"deferred" row is from a *different*, nearby action in the same
   transition (see the patched justification text in each such row).
   These rows are kept in the tables for completeness (they show what
   *wasn't* actioned nearby) but should not be read as "this decision cost
   SSE."
5. **chi2/4 "effective" figure is explicitly indicative**, not a measured
   autocorrelation-derived effective-N for these 4 files (see Methodology).
6. **r2 values (task 4) come from `FitResult.r_squared`**, the indexer's
   own amplitude-space (whole-pattern line-intensity design-matrix) fit
   quality — confirmed by reading `alibz/peaky_indexer_v3.py:2392`
   (`r_squared = 1.0 - ss_res / ss_tot`) — this is **not** the same
   quantity as the pixel-space SSE/RMS reported elsewhere in this report;
   the two are reported side-by-side per the brief (task 4) without being
   reconciled or explained further (see Highlight 4).
7. **REE_01 and argon_noAr never populate `established`/`confirmed`
   element lists strongly enough to run stage 5 (Boltzmann seeding) or the
   deepening rounds** in the same way REE_44/scan9x9 do (REE_01: 0
   deepening rounds ran at all; argon_noAr: established list non-empty but
   very few elements). Whether this reflects genuinely sparse/weak spectra
   or an upstream identification problem was not investigated — out of
   scope for a residual audit.
8. **No independent unit test was added**; this is a one-off measurement
   script per the brief, not a pipeline change. `python3 -m py_compile`
   was run on the script; the script's numeric output was spot-checked
   against the raw CSVs/JSON by hand (see Highlights section derivations)
   but not against a second, independently-written implementation.
9. Per-repo test suites (`tests/`) were **not** run — this task is a
   read-only measurement/analysis task against pipeline code that was
   explicitly not to be modified, and `scripts/audit_stage_residuals.py`
   is a new, standalone script with no existing test coverage to compare
   against. `python3 -m py_compile scripts/audit_stage_residuals.py`
   passed.

## Output manifest (scratchpad; not committed)

- `<scratchpad>/audit/stage_metrics_<sample>.csv` — per-stage SSE/RMS/chi2/chi2_eff/n_peaks, per segment + total (4 files)
- `<scratchpad>/audit/window_attribution_<sample>.csv` — per-transition, per-window delta-SSE + attribution (4 files)
- `<scratchpad>/audit/summary_<sample>.json` — r2 progression, degeneracy stats, timings, stage names (4 files)
- `<scratchpad>/audit/{argon,REE_01,REE_44,scan9x9}.log` — full stdout/stderr of each run
- `<scratchpad>/build_report.py` — script that generated the per-sample tables in this report from the CSVs/JSON above (reusable)

## Repo changes

- `scripts/audit_stage_residuals.py` — new file (the audit script named in
  the brief). No existing file was modified.
