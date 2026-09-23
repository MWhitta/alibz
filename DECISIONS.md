# Decisions

## 2026-09-22 — Optimize Acquisition scores any composition, not only Fe

The Pantheum delay/period study no longer requires element Fe; the panel is
"Optimize Acquisition" and takes a composition (symbol, formula such as Fe2O3,
or weighted list such as "Fe 70, Cr 20, Ni 10"). The FIRST element entered is
the primary element (formula cation-first), so Fe2O3 is scored as an Fe study.
The primary element must have a bundled line reference; other constituents are
scored when a reference exists and reported as unscored otherwise. Batch score =
fraction-weighted mean of per-element window scores; eligibility follows the
primary element. Lines within 0.25 nm of another constituent's reference line
are flagged potential_blend. References for 82 further elements are generated
by scripts/build_line_references.py with the exact fe-v1 recipe (verified by
--check-fe) plus an observable-range filter 180–961 nm (the export grid span),
so vacuum-UV lines never occupy candidate windows.

## 2026-09-22 — Complete Z300 acquisitions from native data converted on Opal

Use Pantheum's existing Moissanite worker and authorized SSH path to invoke a
transient producer on Opal. The producer reads the acknowledged test's exact
current Couchbase Lite revision and its spectrum bundle through Opal's authorized
USB ADB connection. Support both FlatBuffers and the API's ZIP/JSON format,
using shell for metadata and `adb pull` for binary bytes on the old firmware.
Do not depend on vendor CSV exports or turn MTP back on.
No new Opal service, network listener, credential, or physical action is needed.

Retain the raw test/revision/bundle on Opal with immutable native CSVs and
hashes. Preserve detector sample spacing rather than resampling to a uniform
export grid. Require identical axes for the native average. Exclude the known
fourth-channel placeholder by its exact calibration signature and retain it
in raw evidence; unknown calibration layouts remain pending/error. The existing
decoder's empirical wavelength offset and validation limits remain explicit
in provenance.

Acquisition completion requires the exact test identity, complete expected shot
set, and validated native artifacts in Pantheum. Commit dataset publication,
analysis jobs, and acquisition success together. Retrieval can retry after
connection loss or restart; it cannot replay instrument actions. Existing
deferred runs require an explicit ID allowlist for adoption, and uncertain
physical outcomes are never adopted.

Evidence, deployment hashes, test results, and limits belong in
`reports/2026-09-22-z300-automatic-ingestion.md` and the associated audit,
producer, and queue reports. No provider switch occurred.

## 2026-09-22 — Early Ar/O diagnostics; general physical triage remains opt-in

Run database-grounded neutral Ar/O group checks before the first composition
fit. Require independent locally significant groups, calibrated noise, coverage,
and interference checks; expected purge/background context never forces a
detection or identifies oxygen's source. Export evidence separately from fitted
material fractions. Gas sensitivity remains unvalidated on independent gas
standards.

Keep general physical candidate triage off by default, with explicit report and
provisional pass-1 prune modes. In 26 nominal Fe run means (252 shots, one
material), triage retained Fe but produced no final candidate reduction and
about fourfold higher candidate-build cost. Existing downstream line-evidence
filters removed Fe in both modes; this pre-existing failure was diagnosed and
left unchanged. Later pipeline passes can recover candidates independently of
new pass-1 exclusions. Do not use rigid resonance-doublet ratios or missing
lower stages as unconditional element vetoes: 926 of 928 measured K pairs fall
outside +/-20% of the thin ratio.

Evidence, reproducible archive, limits, and validation:
`reports/2026-09-22-physical-triage.md` and
`provenance/physical-triage-validation-20260922.json`.
No provider switch, deployment, or hardware action occurred.

## 2026-09-23 — Independent wavelength registration; warm-up-drift hypothesis

New `alibz/wavelength_registration.py` re-registers each spectrum's wavelength
axis against lines with exactly known air wavelengths: composition-independent
ambient Ar I / O I (`ambient_registration`, NIR-dominant) and the sample's
strong isolated I/II lines across all three segments (`element_registration`),
combined by a per-segment rule with Ar as the NIR anchor
(`combined_registration`). The ELEMENT estimator is DETERMINISTIC **golden
lines** (`golden_lines` + `element_registration`), after two rejected attempts
(a per-line matcher and a statistical vote/window estimator — both measured each
element's ALIAS pattern, not the instrument: a vote-score offset+slope gave Fe
VIS +1.86 vs V VIS −1.49 pm/nm on the identical axis). A golden line is a
composition I/II line in the segment's top ~150 by predicted strength with NO
composition+Ar/N/O/H competitor within ±0.5 nm above 3 % of its strength (Ar I =
NIR golden set); each is measured with the Gaussian instrumental-profile fit and
a robust (Huber) per-segment OFFSET is fit. The vote estimator is now a
diagnostic only. Golden counts: Fe UV3/VIS12/NIR83, V UV6/VIS24/NIR78. Line
centres: Gaussian fit (`subpixel="gaussian"`; paired residual rms 62 vs 71 pm).

DECISIVE finding — offset is instrument, slope is database error. Only the
per-segment OFFSET transfers between metals: Fe VIS +71 ≈ V VIS +55 pm (Δ17),
V VIS reproducible to 24 pm run-to-run. The within-segment SLOPE does NOT
transfer (Fe VIS −5.8 vs V VIS +1.5 pm/nm; the Fe-derived Δλ(λ) applied to V
leaves a 597 pm residual, not ~30–50), so it is per-element database wavelength
error, not an instrument scale change. Deployed correction is the per-segment
OFFSET only; the slope is diagnostic and NOT applied; UV is unregistered
(too few golden lines); Ar anchors NIR.

Verdict on the owner's "calibration cold, collection warm" hypothesis (53 runs):
NOT supported as a significant cause. The VIS offset drift over 4–414 min is
< 6 pm/h (2σ, unresolved, n=50); NIR −36 pm/h (small). The dominant robust effect
is the FIXED pixel→nm OFFSET mis-calibration (vendor recal +70/−94/−223 pm) — a
per-segment offset, not wavelength-dependent (the apparent slope is db error).
`thermal_drift_model` returns zero in VIS with the bound.

DEPLOYED DEFAULT `wavelength_registration="ambient"`: apply the
composition-independent ambient (Ar/O/N/H) registration to the NIR segment only;
keep the legacy anchor shift for UV/VIS; RECORD the golden-line and vote-mode
element registrations as diagnostics but do NOT apply them unless
`apply_element` is set explicitly. Rationale: **element registration is
diagnostic until validated against a line lamp or a certified reference — the
strongest lines of a pure-metal plasma are optically thick and displaced, hence
unsuitable as wavelength anchors** (Fe I 438.35 has no maximum at its position,
strongest feature at 438.035; the V I 437.92/438.47/439.00 triplet collapses to
a single 438.31 peak; strongest-60 match fraction to each metal's own top-300
list only 50–62 % even for the pure Fe run). The Fe→V transfer failure
(250–600 pm) therefore cannot be blamed on NIST-grade database errors; the safe
production behaviour is ambient-NIR-only until a proper standard is measured.
NOTE: reaching ≤30 pm everywhere is not possible from single-element metal
forests; a lamp / multi-element CRM is needed for a wavelength-dependent
instrument function.

Integrated behind `AnalysisConfig.wavelength_registration` (default `apply`,
per-segment quality-gated, legacy anchor shift kept elsewhere) and
`AnalysisConfig.subpixel` (default `gaussian`); `PeakyIndexerV3` accepts the
registration as a prior and reports a disagreement flag (matching unchanged when
absent). Provenance to feed the drift model is written by pantheum-I
(`wl_calibration_time`, `collected_at`, `warmup_minutes`,
`analyzer_temperature_c` = null-with-source, `wl_calibration_coefficients`).

OPEN: the analyzer local clock is mis-set (~+3 h vs UTC) and per-run
`calibrationTime` is only on Moissanite, so absolute warm-up zero is uncertain
(the drift SLOPE is offset-invariant, so the verdict stands); pantheum assumes
analyzer-clock=UTC, this study infers +3 h — neither verified against hardware.
Evidence and figures: `reports/2026-09-23-wavelength-registration.md`,
`reports/registration-study-20260923.json`,
`reports/figures/registration-20260923/`. No provider switch or hardware action.
