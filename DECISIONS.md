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

## 2026-09-23 — Two argon wavelength engines: both kept, one rule, neither applies to 10-shot Fe means

Two sessions implemented the "Ar I / O I as separate wavelength calibrations"
request in parallel: `gas_calibration` + `wavelength_calibration` (regional,
native-maxima peak table, ≥ 3 consistent Ar groups, default `apply`) and
`wavelength_registration` (line-shape-aware, Gaussian sub-pixel, per-segment,
default `ambient` = NIR when ≥ 3 lines). Both are committed. Binding rule in
`analyze_spectrum`: ambient registration may replace the NIR segment shift;
the gas engine's regional calibration overrides inside its bracketed support
when `calibrated`; `['gas_cross_check']` records their difference (20 pm
single-line floor) and QC flags `ar-registration-disagreement` beyond 3 σ.

Measured on the 26 pinned Fe run means: gas engine 0 calibrated (24 no
evidence: no candidate at SNR ≥ 6), ambient registration one line per run
(Ar I 696.5 nm, −0.188 nm, MAD 11 pm). Every other Ar I anchor (706, 738, 763,
794, 811, 826, 842 nm) matches a repeatable IRON feature 0.2–0.6 nm away (Fe II
763.19 / 794.51 / 810.98 / 842.06 nm); only 696.5, 772.4, 801.5 nm are argon
(−0.17 ± 0.03 nm). Hence: (a) `ambient_registration` gains a composition anchor
gate (removal only; the config path existed but was dead); a stricter isolation
fraction was tried and REJECTED (removes 696.5, keeps 794); (b) the earlier
"−0.18 nm trusted" statement is corrected to a single-line measurement; (c) both
defaults stay because both only act when their gates pass, which they do not on
this data; (d) an argon-rich registration acquisition (50–100 shots on Al/Cu/
glass) is the prerequisite for a multi-line NIR registration. Two of the closed
session's tests were stale against its own code and updated. Evidence:
`reports/2026-09-23-gas-registration-reconciliation.md`.

## 2026-11-03 — Accuracy target recorded; manuscript initialised

Recorded from the 2026-07-01 planning session so the manuscript can cite a repo
source: the working definition of done for quantitative CF-LIBS is mass/mole
fractions within about 20 % relative on MAJOR elements, validated first by
composition invariance on the paired argon/air series and then on certified
reference materials (none measured yet). The manuscript lives in `manuscript/`
(LaTeX; arXiv then *Applied Spectroscopy*); every number in it must cite the
repo report, test, notebook or commit that produced it, in a LaTeX comment on the
same line. Literature is found by search workers, Crossref-verified, and fetched
centrally with the literature-retrieval-agent (`manuscript/lit/fetch.sh`).
Evidence: `reports/2026-11-03-manuscript-init.md`. No provider switch occurred.

## 2026-11-03 — Literature-driven A/B round: nothing adopted yet; design matrix must conserve emission first

Seven literature ideas were implemented behind default-OFF options (adaptive NN-LASSO, Huber IRLS,
Barklem–Collet partition tables, Debye ionization-potential depression, BH/BY FDR on the detection
score, Matérn/early-stop GP settings, effective-n BIC) and scored on a fixed fit-quality panel
(`scripts/fit_quality_panel.py`: suite S, real sample spectra, Fe-standard run means) on moissanite.
Verdicts in `reports/2026-11-03-ab-wave1-verdicts.md`: Huber rejected; FDR rejected as implemented;
sparse neutral; IPD mixed; partition tables improve the Fe standard (0.40 → 0.56) and are physically
correct but score worse on synthetics because they move the thermometer onto the true state.
Root cause found there: `_build_observed_overlap_map` replicates each database line over every peak
within the shift tolerance, so dense-forest elements are under-recovered (oracle areas at exact truth
→ Ca 0.73 vs 0.60). Decision: build the emission-conserving design matrix (`overlap_mode`) as wave 2,
re-score every variant on top of it, and only then flip defaults. Evidence:
`reports/2026-11-03-debug-composition-bias-at-truth.md`, `reports/2026-11-03-debug-partition-roundtrip.md`.
No provider switch occurred.

## 2026-11-04 — Adopt emission-conserving overlap, Barklem–Collet partition functions and Debye IPD as pipeline defaults

Scored on the fixed fit-quality panel on moissanite against a matching-code baseline (deterministic:
identical code → identical metrics). Fast settings (reports/2026-11-04-ab-wave2-verdicts.md) and the
production-settings confirmation (n_calls 40, draws 8; reports/fit-panel/{baseline,conserve-partition-ipd}-
prod.json): synthetic median relative error 0.205 → 0.147 (better on 4/5 cases), real-spectrum r² 0.477 →
0.486 (4/5), Fe-standard Fe fraction 0.398 → 0.562 (14/26) with spurious elements 1.08 → 0.92 (6/4); costs:
real stage disagreement 0.457 → 0.487 (worse on 4/5), synthetic |T error| mixed (+3/−2). Defaults flipped
at the pipeline/CLI level only (`AnalysisConfig`, `analyze_spectrum`, `analyze_directory`, `alibz-analyze`):
`overlap_mode="conserve"`, `partition_source="barklem2016"`, `ipd="debye"`; library class defaults
(`PeakyIndexerV3`, `SahaBoltzmann`, the synthetic generator) stay legacy/line-sum so existing tests and
round trips keep their meaning.
Not adopted (evidence in the same reports): Huber IRLS (worse everywhere), adaptive NN-LASSO (neutral),
BH/BY FDR on detection (collapses the Fe standard), weighted+nearest (exact at the true state, bad
end-to-end), nearest alone (exact at truth, worse end-to-end on synthetic error), Matérn/early-stop GP and
effective-n BIC (mixed signs on 5-item panels; deferred). Known caveats: 5/5/26 items, one seed, Fe axis
is a shifted approximation; `conserve` hallucinates ~1.5 % Dy/Pb/U at the true state on the silicate oracle
where `nearest` is clean — the panel favours `conserve` end-to-end, so `nearest` waits for the per-peak
width kernel (reports/2026-11-04-forward-design-parity.md, model target). Off-path numeric drift between
option plumbings is being bisected (reports/2026-11-04-offpath-drift-bisection.md). No provider switch.

### 2026-11-04 addendum — two unflagged default changes found and gated; adoption re-confirmed

Bisection (reports/2026-11-04-offpath-drift-bisection.md) found two changes that had altered the default
path without a flag during the A/B round: soft intra-element stage-coupling rows (`INNER_STAGE_COUPLING_
WEIGHT = 0.15`) and `run(stark_width_weight=0.05)`. Both are now opt-in (`inner_stage_coupling_weight=0.0`,
`stark_width_weight=0.0`, plumbed through `AnalysisConfig`/`analyze_spectrum`); HEAD-equivalence verified
(binary 0.1701; series-3 `head` = series-1 baseline). Re-confirmation on the true baseline (fast settings,
reports/2026-11-04-ab-wave2-verdicts.md, series 3): the adopted defaults improve synthetic median relative
error 0.207→0.156, |T error| 0.082→0.069, real r² 0.471→0.478 (4/5), stage disagreement 0.515→0.500 (4/5),
Fe fraction 0.404→0.546 (13/26). Stark coupling and stage-coupling rows stay OFF (mixed). Rule going
forward: no default-path change without a flag and a panel score. Full fast suite on the gated tree: 668
passed; round-trip gates 5/5.

### 2026-11-04 — Manuscript figures from real data: showcase selection, stage capture, five figures; two findings
- **Showcase spectra** (manuscript/data/showcase.json, loader scripts/showcase_data.py, sha256-verified): MW2-112 shot 1000; JChristensen MDD017-24-15 2nd #1, Alaskire MDD27, MDD017-29-2c #1; user_spec1; Fe standard runs idx 17 (fresh, d5 p25) and idx 9 (reused-spot failure). Chosen because each is already named in the manuscript text. The 4 909-file Z300 corpus drive (Corpus_One) is not attached to moissanite; four undocumented Drive campaigns (20250529LIBSAuto 1 462, mapping_01 1 513, NETL, paper_revision 3) were not used — no provenance in the repo.
- **Stage capture in production** (`analyze_spectrum(capture_stages=True)` → `analysis["stages"]`, 13 keys): OFF by default, proven byte-identical when off (synthetic test + user_spec1 re-run), 691/691 suite. `scripts/fit_inspector.py::capture()` is a drifted copy of the chain and is not used for figures.
- **Figures F1–F5** (`scripts/manuscript_figures.py`, cache `manuscript/figures/cache/` gitignored, regenerable): overview (user_spec1), stage walkthrough and refinement (MW2-112 1000), confounder contest (MDD017-29-2c #1), two thermometers on the Fe standard. Captions quote the cache JSON (`% src:`); `make` clean, 54 pp.
- **Finding 1 (Fe standards):** the whole-pattern solve on native Fe data sits at 20–25 kK with nₑ at the H-alpha bound and assigns every Fe peak to Fe II (25/26 panel runs; the figure run 22.6 kK), while the line-selected Fe I VIS Boltzmann fit on the same spectrum gives 11.5 ± 0.4 kK (10 lines, 1.25 eV span, below the campaign's 2.4 eV resolution threshold) and the campaign median is 10.8 kK. F5 presents this as the two-thermometer discrepancy; it is a known failure mode, not a figure artefact.
- **Finding 2 (JChristensen, current defaults):** a 38-spectrum CLI survey (conserve/barklem2016/debye, n_calls 40, draws 32; products in reports/jchristensen-20261104/) shows the Mn→Mg contest firing on 6/38 with Mn as-fit 0.03–0.10 → 0 resolved and Mn upper-limit on 10; the 2026-07 hot-Mn cases (40–53 %) no longer occur. The 07-results "Confounder resolution" prose still quotes the 2026-07 numbers as history; F4's caption states the current outcome. Median r² 0.78.
- 05-datasets corrected: `data/remote_samples` is gitignored, not committable (identified by name and checksum).
- Nothing committed (owner).

### 2026-11-05 — Native-grid finder gate fixed; pipeline assumption survey; adoption decisions pending

- **Applied (uncommitted, owner to review):** `fit_peaks` native-grid seeding gate and the per-channel width filter
  (alibz/peaky_finder.py; `tests/test_native_gate.py`). Rationale: the `len(x_window) > 5` gate (from the 2025 export-era
  scripts, packaged 2026-03-20) discarded 85–96 % of native detections; the width filter then zeroed resolution-limited
  lines. Export behaviour is byte-identical by construction (`is_native` gate; verified on user_spec1/mw2_1000 and the
  unchanged synthetic + real panels). Evidence: reports/2026-11-05-native-gate-fix.md.
- **Not adopted as "better" on the Fe panel:** fe_fraction mean 0.563 → 0.287. Recorded as the exposure of a downstream
  defect, not a regression of the finder: the solve now lands in a hot Na basin (20.9 kK, Na 0.81 from 4 Na I peaks)
  while the stage-tie minimum and the VIS Fe I Boltzmann T agree at 10.6–11.5 kK (reports/2026-11-05-pipeline-
  assumption-survey.md §3.4). The Fe panel headline is known to reward the hot basin (§3.3 G1).
- **Survey verdict (owner question "patchwork of development attempts"):** one defect class — constants and width
  references fixed on the 0.0333 nm export grid, applied to native data without a regime record, scored by a panel whose
  synthetic and real items are all exports. Verified findings and a fix order are in the synthesis §3/§5. Rule proposed:
  no further default adoptions until the panel has a native tier and records the grid regime and git head.
- **Independent defects recorded (not fixed):** Saha ion fraction read by column position vs stage labels present in the
  db (Pr II 0.0013, Tb II 0.0008, Gd I 0.72 chained to a hydrogen-like ion, H never ionises; 11 elements with no stage-I
  lines); 620 nm response never measured (fallback 3.9 on 7/7 showcase runs); confounder-contest indexer built with
  library physics (line-sum partition, no IPD); `evidence_max_missing_lines=6` absolute cap on peak-table absence.
- No provider switch (Claude subscription throughout).
