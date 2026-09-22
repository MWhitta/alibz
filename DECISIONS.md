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
