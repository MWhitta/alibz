# Decisions

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
