# Fe calibration boundary for independent Ar I / O I wavelength estimates

## Finding

There is no Fe-specific hardware calibration or Fe-only wavelength correction
in the production pipeline. The only Fe-specific wavelength estimator is the
offline study in `scripts/fe_plasma_analysis.py`. Production uses a generic,
mixed-element residual estimator from `alibz/utils/wavelength.py`. The hardware
wavelength axis is a separate input layer: native Z300 bundles carry their own
per-shot cubic pixel-to-wavelength polynomials, while vendor CSV recovery uses
the instrument dispersion model in `alibz/utils/native_grid.py` plus a
per-spectrum correction.

Therefore Ar I and O I residual wavelength estimates can and should be computed
independently from the incoming wavelength axis and persisted as analysis
evidence. Per the selected policy, a validated Ar or O estimate may replace the
baseline analysis shift inside only its bracketed wavelength support. This is a
reversible analysis overlay: retain the immutable incoming/native/vendor axis,
the generic baseline estimate, both separate gas estimates, and exact applied
ranges. It must not rewrite hardware calibration, raw/native wavelengths, or
the frozen session calibration.

## Exact code boundaries

### Fe-only analysis (offline, not production calibration state)

- `scripts/fe_plasma_analysis.py:186-216` builds isolated Fe I/II anchors below
  620 nm. `AR_NIR_ANCHORS` at lines 219-222 supplies Ar I anchors in the NIR,
  where the study says Fe has essentially no emission.
- `estimate_shift_model` at lines 289-343 fits a per-segment linear residual
  model with the explicit sign convention `observed = db + shift`. It is Fe in
  UV/VIS and Ar in NIR, so even this historical "Fe calibration" is not purely
  Fe over the full wavelength range.
- The study estimates one model per wavelength-grid family from each family's
  grand mean (`scripts/fe_plasma_analysis.py:571-590`). This is analysis-time
  re-registration, not a write to the analyzer or decoder.
- It persists the results only in
  `reports/2026-09-22-fe-calibration-plasma-analysis.md` and figures under
  `reports/figures/fe-plasma-20260922/` (`scripts/fe_plasma_analysis.py:1480-1497`).
  There is no machine-readable Fe calibration JSON/CSV emitted by that script.
  The report records the surviving native-family result as global -154 pm,
  UV -119 pm, VIS -154 pm, NIR -154 pm, with nine total anchors
  (`reports/2026-09-22-fe-calibration-plasma-analysis.md:49-57`). This dated,
  grid-family grand-mean result is not an appropriate universal instrument
  calibration.

### Generic production residual estimator (mixed elements)

- `_anchor_catalog` iterates over every eligible `db.elements` entry and keeps
  locally dominant bright ion-I/II features (`alibz/utils/wavelength.py:72-127`).
  It has no Fe filter.
- `estimate_wavelength_shift` returns the median fitted-minus-database offset
  (`alibz/utils/wavelength.py:155-190`), and
  `estimate_wavelength_shift_segments` groups the same mixed-element matches by
  UV/VIS/NIR with significance-gated segment medians (lines 246 onward).
- `alibz/pipeline.py:463-483` estimates the pooled shift from the blind peak
  table, then the segment shifts from the refined peak table. That
  `SegmentShift` is the pipeline's internal residual registration.
- Current gas detection is *not* independent: the pipeline passes this generic
  shift into `detect_background_gases` (`alibz/pipeline.py:624-630`), and
  `alibz/gas_detection.py:105-108,378-396` evaluates Ar and O database groups at
  `db_wl + shift_nm`. Thus `background_gases.json` contains detection evidence
  in the generic-shift frame, not element-specific Ar/O wavelength calibration.

### Shared session calibration (generic, persisted, still not Fe-specific)

- `alibz/session_calibration.py:27-32,70-115` uses the same mixed-element
  `_anchor_catalog`/`_match_deltas` functions to collect per-segment residuals.
  `build_shared_calibration` pools nearby shots without crossing acquisition
  breaks (lines 135 onward).
- `alibz/mw2_112.py:204-223` writes these results to
  `session_calibration.csv`; four frozen examples exist under
  `runs/mw2_112_profile/*/session_calibration.csv`. They contain per-position
  `shift_prior_{uv,vis,nir}_nm`, uncertainties and counts. The canonical MW2-112
  workflow hashes and reuses the frozen table, as documented in
  `docs/mw2_112/provenance.md:70-85`.
- These files are appropriate only when reproducing their named MW2-112 run or
  when deliberately using the generic neighboring-shot priors. They should not
  be overwritten by Ar/O results: their schema and provenance mean
  mixed-element shared instrumental priors, and downstream relative-profile
  measurement directly adds them to all feature centers
  (`alibz/relative_profiles.py:274-309`).

### Hardware/vendor wavelength mapping (separate layer)

- Native Z300 ingestion evaluates each shot's stored four cubic wavelength
  polynomials and retains their provenance in the archive manifest
  (`scripts/z300_opal_ingest.py:437-469,675-696`). FlatBuffer decoding uses the
  empirically inferred pixel offset -18; legacy JSON uses raw zero-based pixel
  index. The notes explicitly say absolute accuracy is not certified. This path
  reads the calibration supplied with the acquisition; it does not learn it
  from Fe, Ar or O.
- For uniform vendor CSVs, `INSTRUMENT_CALIBRATION` is the per-segment hardware
  dispersion seed (`alibz/utils/native_grid.py:430-455`).
  `recover_native_grid` measures a fresh per-spectrum correction and returns
  its coefficients in `info` (`:675-717,749-765`). `calibrate_instrument`
  derives an instrument table (`:649-672`). Neither function writes analyzer
  state. The checked-in constant was derived from REE_01 spline inversion, not
  Fe atomic lines.
- An Ar/O analysis must treat these axes and manifest values as immutable
  inputs. Reporting `observed - database` alongside them is safe; rewriting the
  native wavelengths or hardware coefficients would erase the distinction
  between vendor calibration and line-based residual evidence.

## Wavelength medium and sign convention

Incoming spectral `x` is expected to be the instrument's wavelength axis in
the NIST ASD convention: standard air at and above 200 nm, vacuum below 200 nm.
The bundled database starts from Ritz vacuum wavelengths and converts them once
on load (`alibz/utils/database.py:111-121`) using `vacuum_to_air`; observed-line
records use the same convention (`:157-177`). The convention is stated directly
in `alibz/utils/wavelength.py:1-25`. All Ar I and O I diagnostic groups here are
above 200 nm, so comparisons are standard-air nanometres.

Residual sign is consistently intended as

`shift_nm = observed_wavelength_nm - database_air_wavelength_nm`, hence
`observed = database + shift`; subtract the shift to express an observed center
in the database frame (`alibz/utils/wavelength.py:163-176,193-203`).

`docs/fit_pipeline.md:345-348` still says the runtime database wavelengths are
vacuum. That statement is stale relative to the loader and should not guide a
new calibration implementation.

## Selected application and persistence policy

Keep one analysis record per spectrum with separate `Ar` and `O` entries. Each
entry should carry the element-only estimate, uncertainty, anchor/group count,
supported wavelength range/segments, abstention reasons, sign convention and
wavelength medium. A natural home is an extension or sibling of the existing
`background_gases.json`, which is already the CLI's JSON-safe per-file gas
evidence export (`alibz/cli.py:150-154`). A sibling such as
`gas_wavelength_calibration.json` is clearer because the current file describes
detection and currently consumes the generic shift.

The selected behavior is to apply each accepted absolute gas residual in its
supported wavelength region, while reporting Ar and O separately. The overlay
must replace the baseline residual there rather than add to it. Outside support,
on abstention, and in a conflicting overlap, retain the mixed-element baseline.
Persist the untouched baseline alongside proposed/applied regions and the two
source records. Do not replace `session_calibration.csv`, native CSV wavelength
columns, archive manifest calibration coefficients, or
`INSTRUMENT_CALIBRATION`.

The current integration implements this as a separate
`wavelength_calibration.json`; the historical summary shift columns retain the
baseline, while Ar and O status/offset/uncertainty are separate columns. That is
the appropriate persistence boundary.

## Review of the regional overlay implementation

Reviewed files: `alibz/gas_calibration.py`,
`alibz/wavelength_calibration.py`, the frame-aware changes in
`alibz/utils/wavelength.py`, and current pipeline consumers.

### Correct properties

- `calibrate_background_gases` accepts the raw incoming `x`, `y`, and an
  observed-frame peak table, but no Fe/mixed shift argument
  (`alibz/gas_calibration.py:511-527`). Its offsets are therefore absolute
  `observed - database` measurements. Ar and O records remain separate.
- Calibrated segment support is database-coordinate and bracketed by the first
  and last accepted inlier anchors; a segment needs at least two anchors
  (`alibz/gas_calibration.py:428-447`). The apply layer accepts only finite,
  calibrated, positive-uncertainty records with the element/global and segment
  count gates (`alibz/wavelength_calibration.py:55-81`).
- `RegionalShift` replaces the baseline with `offset_nm`; it does not add the
  gas value to the baseline (`alibz/wavelength_calibration.py:39-52`). Direct
  verification with baseline +0.03 nm and Ar +0.12 nm returned +0.12 nm inside
  support, +0.03 outside. `off` and `report` returned the original baseline.
- Database-position consumers now request `frame="database"` in gas detection,
  detection support, minor-line matching/seeding, and the pipeline support map.
  Observed-center consumers (`_db_frame`, refinement, observed coverage bounds)
  use the default observed frame. This is the correct direction: add an offset
  to a database coordinate; subtract one from an observed coordinate.
- Conflicting Ar/O database-range overlaps fall back only in the overlap and
  retain nonconflicting flanks.

### Boundary defect found and corrected during review

The first implementation abruptly changed offsets at support boundaries. I
reproduced a non-injective observed-frame map with two disjoint database ranges:
Ar `[700, 800]` at +0.2 nm and O `[800.1, 900]` at -0.2 nm produced overlapping
observed images and source-order-dependent inversion without a reported
conflict.

The revised policy and implementation correct this with a continuous linear
taper *inside* each supported database interval
(`alibz/wavelength_calibration.py:39-51,148-163`). The shift equals the baseline
at both anchor-hull endpoints, reaches the absolute gas offset only over the
reported `full_offset_range_nm`, and uses transition width
`max(0.5 nm, 4 * |gas - baseline|)`. Thus `|d shift/d wavelength| <= 0.25`, the
forward map `observed = database + shift(database)` has slope at least 0.75,
and fixed-point inversion is contractive. Ranges crossing a detector edge or
too narrow for two transitions abstain explicitly. Compatible adjacent pieces
using the same reference are merged before tapering.

`RegionalShift.at_in_frame(..., frame="observed")` now inverts the forward map
only inside the stored observed image and leaves the baseline unchanged outside
(`alibz/wavelength_calibration.py:53-72`). Independent dense-grid verification
of both the original adjacent-range reproduction and an Ar/O conflict with a
baseline overlap found strictly positive forward increments (minimum
`0.00153 nm` on a `~0.00204 nm` database step) and maximum database -> observed
-> database round-trip error `1.14e-13 nm`. Applied records export transition
width, full-offset range, observed image and boundary policy.

### Independent-source semantics

The gas estimator runs first on raw `x`, `y`, and the original observed-frame
blind peak table (`alibz/pipeline.py:467-476`). The generic scalar shift then
drives refinement without a gas overlay, and the saved per-segment
`baseline_shift` is estimated from that baseline path (`:478-532`). Only after
the baseline (including any authorized frozen session prior) is complete does
`apply_gas_calibrations` construct the effective regional overlay (`:532-534`).
This preserves the counterfactual baseline and prevents gas calibration from
becoming either a prior to itself or an increment added twice.

The verified stale statement in `docs/fit_pipeline.md:345-348` was corrected to
match the current loader: stored Ritz vacuum values are converted on load to
standard air at/above 200 nm, with vacuum retained below 200 nm.

No provider switch, network access, external application, or hardware action
occurred during this audit.
