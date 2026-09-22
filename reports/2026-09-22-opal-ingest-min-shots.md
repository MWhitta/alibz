# Opal producer: tolerate dropped frames down to --min-shots

Repo: alibz (uncommitted; not committed per instructions). No host contacted.

## Problem

`scripts/z300_opal_ingest.py` required the stored shot count to equal
`--expected-shots` exactly, so every test with an analyzer frame drop
(8-9 of 10 shots stored, active since 2026-09-22) sat through 40 refusals
and failed. The consumer (`pantheum-I/pantheum/alibz/retrieval.py`) already
tolerates `min(MIN_STORED_SHOTS=6, expected) <= manifest['shots'] <= expected`.
This change brings the producer in line.

## Change (scripts/z300_opal_ingest.py)

- `MIN_STORED_SHOTS = 6` constant (scripts/z300_opal_ingest.py:54).
- `Config.min_shots: int | None = None`, added as the last dataclass field so
  existing positional `Config(...)` calls (tests) are unaffected
  (scripts/z300_opal_ingest.py:95).
- `effective_min_shots(config)`: returns `config.min_shots` if set, else
  `min(MIN_STORED_SHOTS, expected_shots)` (scripts/z300_opal_ingest.py:98-102).
  This makes the CLI default effectively 6 for the common expected>=6 case,
  while never producing an invalid (min_shots > expected_shots) default for
  small `--expected-shots` runs — matching the consumer's own `min(6, expected)`
  pattern cited in the brief.
- `validate_config`: rejects explicit `min_shots` outside `1..expected_shots`
  (scripts/z300_opal_ingest.py:113-114).
- `decode_native(blob, expected_shots, min_shots)` (scripts/z300_opal_ingest.py:378):
  `count > expected_shots` -> IngestError ("exceeds expected"); `count < min_shots`
  -> PendingData naming both numbers ("below min_shots N of expected M"). The
  FlatBuffer shot vector is inherently contiguous (positional), so no separate
  contiguity check is needed there.
- `decode_legacy_native(blob, expected_shots, min_shots)` (scripts/z300_opal_ingest.py:488):
  members must be a subset of `{"-1", 0..expected_shots-1}` (else IngestError
  "unexpected" — this also rejects any member >= expected_shots, i.e. "more
  than requested"); missing `"-1"` -> PendingData ("average"); stored numeric
  members must be exactly contiguous `0..n-1` for the achieved `n`, otherwise
  IngestError ("not contiguous"); `n < min_shots` -> PendingData ("below
  min_shots N of expected M").
- `ingest()` (scripts/z300_opal_ingest.py:642-649): computes
  `min_shots = effective_min_shots(config)` and passes it to both decoders.
  `manifest["shots"]` was already `len(shots)` (dynamic, no change needed).
  Added `provenance["min_shots"]` and `provenance["dropped_frames"] =
  expected_shots - len(shots)` (scripts/z300_opal_ingest.py:675-676).
- `verify_archive()` (scripts/z300_opal_ingest.py:579-598): now checks
  identity via `schema/run_id/test_id/grid`, then requires
  `min_shots <= manifest["shots"] <= expected_shots` and
  `manifest["provenance"]["expected_shots"] == config.expected_shots` (so a
  later `--expected-shots` change on the same run_id is still caught), and
  requires `shots/shot-{i}.csv` for `i in range(manifest["shots"])` (the
  archive's own count) instead of `range(expected_shots)`. This makes an
  archive created with `n < expected` verify idempotently.
- CLI: added `--min-shots N` (default `None` -> auto; scripts/z300_opal_ingest.py:730-732).
  Success line for `--output` now reads the actual stored count from the
  written manifest: `native ZIP archived: {n} of {expected} expected shots`
  (scripts/z300_opal_ingest.py:760-763), replacing the old
  `native ZIP archived: {expected_shots} shots`.
- Docstring (scripts/z300_opal_ingest.py:14-17): replaced the "incomplete
  acquisition exits 3" line with one describing the min/expected bounds.

## Design choice flagged for the user

The brief said "`--min-shots N` (default 6 ...)". A literal argparse
`default=6` would break every existing test/CLI call using
`--expected-shots` < 6 (e.g. the fixture's `expected_shots=2`) by making
`validate_config` reject `min_shots=6 > expected_shots=2`. I implemented the
default as `min(6, expected_shots)` (computed lazily, sentinel `None` in
`Config`/CLI), which is literally 6 whenever `expected_shots >= 6` (the real
10-shot case this bug is about) and degrades sensibly otherwise, mirroring
the consumer's own `min(MIN_STORED_SHOTS, expected)` pattern quoted in the
brief. All existing tests pass unchanged under this interpretation.

## Tests (tests/test_z300_opal_ingest.py)

Updated existing direct `decode_native`/`decode_legacy_native` calls to pass
an explicit `min_shots` (kept at the old exact-match value so their intent
is unchanged): lines ~263, 269, 275, 460-462, 510-514, 517-524.

Restructured `test_legacy_missing_average_or_shot_is_pending` (old exact
semantics) into three explicit cases matching the new contiguity rule:
`test_legacy_missing_average_is_pending`, `test_legacy_trailing_shot_missing_is_pending_below_min_shots`
(valid still-filling-in prefix -> PendingData), `test_legacy_gap_in_shots_is_rejected`
(non-contiguous -> IngestError). Missing "0" while "1" is present is now an
IngestError, not PendingData, since shots are written in order and a gap is
a broken bundle, not one still filling in.

New tests added:
- `test_flatbuffer_dropped_frame_tolerated_down_to_min_shots`,
  `test_flatbuffer_below_min_shots_is_pending`,
  `test_flatbuffer_more_shots_than_expected_is_rejected`
- `test_legacy_dropped_frame_tolerated_down_to_min_shots`,
  `test_legacy_below_min_shots_is_pending`,
  `test_legacy_more_members_than_expected_is_rejected`
- `test_native_dropped_frame_archived_and_reverifies_without_instrument`
  (end-to-end `ingest()`: expected=2, stored=1, `min_shots=1`; checks
  manifest `shots`/`min_shots`/`dropped_frames`, then a second `ingest()`
  call with `adb.read` set to fail-on-call proves idempotent re-verification
  never touches the instrument)
- `test_min_shots_outside_expected_range_rejected` (parametrized 0, 3, -1
  against `expected_shots=2`)
- `test_min_shots_defaults_to_min_of_six_and_expected_shots`
- `test_cli_min_shots_outside_range_is_rejected` (exit code 2, stderr
  mentions `min_shots`)
- `test_cli_output_reports_dropped_frames` (new stderr message format)
- Added a stderr-content assertion to the existing
  `test_cli_output_is_zip_or_empty_on_failure` for the unchanged
  full-count case (`native ZIP archived: 2 of 2 expected shots`).

## Docs (docs/z300_device_notes.md)

Updated item 0 under "Anticipated problems" (grep for "exact" found only
this section) to state the Opal producer no longer requires an exact match,
citing `--min-shots`, the manifest provenance fields, and pantheum-I's
matching `MIN_STORED_SHOTS=6` decision, while noting the other pipeline
stages named there (retrieval validation, optimizer `shots != 10`, metrics)
are outside this repo and still need the owner's decision — not overclaiming
a fix scoped to this repo's producer script only.

## Commands and counts

- `python3 -m pytest tests/test_z300_opal_ingest.py -q` -> before: 71 passed
  (stated in brief); after: **84 passed** (13 new tests, 0 removed, 3
  restructured from 1 parametrized-3 into 3 explicit).
- `python3 -m pytest tests/test_deploy_z300_ingest.py tests/test_validate_z300_opal.py -q`
  -> **9 passed, 10 subtests passed** (unchanged from before; these files
  don't reference `expected_shots`/`min_shots` behavior).

## Not verified / out of scope

- Did not run the full `tests/` suite: unrelated pre-existing collection
  errors (`ModuleNotFoundError: No module named 'scipy'` in
  `alibz/peaky_finder.py`, hit by ~30 unrelated test modules such as
  `test_detector.py`) exist in this environment independent of this change;
  confirmed by running `tests/test_detector.py` alone and seeing the same
  import error. Not caused by this change; brief only asked for the three
  named test invocations, both run above.
- Did not touch `scripts/validate_z300_opal.py` (builds the PowerShell
  invocation) or `scripts/prepare_z300_ingest_bundle.py` to add a
  `--min-shots` passthrough — brief did not ask for it and neither hardcodes
  the old exact-match assumption; they pass through `--expected-shots` only,
  so the producer's new auto-default (`min(6, expected)`) applies
  transparently. If the user wants an explicit `--min-shots` surfaced through
  `validate_z300_opal.py`'s CLI, that's a follow-up.
- Did not edit `pantheum-I` (retrieval.py / DECISIONS.md) — out of scope for
  this alibz-repo brief; the report cites its existing behavior for context
  only, taken from the brief's own description, not independently re-read
  in this session.
- No commit made (per instructions).
