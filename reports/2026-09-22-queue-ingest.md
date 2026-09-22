# Pantheum Opal native acquisition ingestion

Status: implementation and final local verification complete, staged tree `/private/tmp/pantheum-z300-ingest-20260922` only.
Provider switches: none (subscription retained).
Scope: queue retrieval, exact native archive validation, durable retry/import state, optimization pending-data gating; no deployment or hardware access.
Baseline commands launched for both required pytest invocations. The staged Pantheum checkout has no `.venv/bin/python`; dependency-enabled invocation cannot establish pymatgen availability.

## Changes and evidence

All implementation paths below are relative to the staged tree, not deployed source.

- `pantheum/alibz/acquire.py:59`: admits `opal_database`; `:123` adds nullable per-run mode plus claim, attempt count/time, sanitized last error and retry deadline columns; `:141` validates trusted command configuration before acquisition; `:222` preserves no-data-API probing; `:379` hides claim token from public run metadata; `:458` persists mode with dispatch intent; `:574` explains pending Opal completion; `:630` selects existing deferred fire-only path for this new mode. Existing deferred detail string stays byte-identical and existing API retrieval implementation is unchanged.
- `pantheum/alibz/retrieval.py:37`: validates argv placeholders, finite bounded timeouts/retry delays and exact-ID allowlist; `:64` executes trusted argv with shell=False, DEVNULL input/stderr, finite timeout and bounded streamed ZIP output. Test IDs are restricted before interpolation to remain safe through SSH's remote shell.
- `pantheum/alibz/retrieval.py:101`: rejects duplicate/nonfinite JSON; `:113` verifies schema, exact run/test IDs, integer full shot count, native grid label, provenance object, exact member/hash inventory, ascending finite CSVs through existing parser plus original-order check, bounded sizes and traversal/symlink/duplicate rejection. Does not use ZIP extraction.
- `pantheum/alibz/retrieval.py:176`: durable atomic writes and directory fsync; `:197` implements per-root process-local exclusion, database claims, explicit allowlist adoption, worker-only restart claim reset, persistent retry state and compare-and-set guards. `_complete` commits the dataset, metadata, jobs and acquisition success in one transaction after all artifact fsyncs. Retry IDs use the existing SHA/source/example identity scheme. The original raw files, native average and shot CSVs are preserved byte-identically, with original Opal manifest and local manifest hashes.
- `pantheum/alibz/__main__.py:473`: constructs retrieval only for explicitly configured opal_database mode inside the existing elected worker lock, resets data claims, attempts retrieval before downstream jobs, and reconciles only eligible data completions. Other modes retain the previous worker lifecycle.
- `pantheum/alibz/optimization.py:520`: reconsider automatic pending-data batches, retaining old deferred terminal semantics; `:550` blocks metrics and proposals for new pending-data runs; `:663` guards pending batch reentry; `:725` prevents closing active automatic data batches while preserving legacy deferred close; `:760` adds read-only data reconciliation selected by known acquisition rows. This deliberately does not run startup hardware recovery during a healthy serve dispatch gap. Existing native Fe metrics are unchanged.
- `tests/test_alibz_retrieval.py:76`: exact native success, raw byte/hash/provenance preservation and idempotency; `:103` malformed identity/count/hash/native CSV rejection; `:121` archive traversal/symlink/extra-shot rejection; `:137` retry/restart without reservation changes; `:160` shared-claim exclusion and stale state protection; `:173` transactional rollback on artifact failure; `:187` real subprocess timeout/output bound and invalid config checks; `:200` explicit CLI retrieval/reconcile/process ordering without hardware recovery; `:219` disable-mode pause; `:228` worker data reconciliation cannot recover active dispatch; `:239` exact allowlist adoption rejects uncertain/unlisted rows; `:249` single fire and no data API even after reentry; `:267` timing/shot-plan blocking and unblocking without automatic next dispatch.
- `docs/acquisition-optimization.md:135`: operator configuration, payload contract, retry/security/size bounds, adoption and scientific limits.
- `DECISIONS.md:404`: persists the architectural decision and rationale.

## Tests and iteration history

Before edits, both mandated commands were run from staged Pantheum:

1. `PYTHONPATH=src python3 -m pytest tests/ -q` — 553 passed, 167 failed, 26 skipped, 37 errors; 214 subtests passed. `/private/tmp/queue-baseline.log`.
2. `PYTHONPATH=src:$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])") python3 -m pytest tests/ -q` — same counts; `.venv/bin/python` missing in staged Pantheum so this fell back to system Python. `/private/tmp/queue-baseline-pymatgen.log`.

The baseline errors include sandbox-denied socket bind `PermissionError: Operation not permitted`, not verified code regressions. Socket-based suites were rerun with approved sandbox escalation.

Focused after implementation: `PYTHONPATH=src python3 -m pytest tests/test_alibz_retrieval.py -q` — 12 passed, 13 subtests passed (1.55 s), `/private/tmp/queue-focused.log`.

Broader focused after implementation (before two final pause/worker-gap tests were added): `PYTHONPATH=src python3 -m pytest tests/test_alibz_acquire.py tests/test_alibz_optimization.py tests/test_alibz_optimization_metrics.py tests/test_alibz_retrieval.py -q` — 103 passed, 47 subtests passed (44.98 s), `/private/tmp/queue-regression.log`.

Full suite iteration and final passing unittest evidence are recorded below.

## Verification limits

This agent made no live hardware, Opal, analyzer or network-control calls and deployed nothing. Real producer/native-grid scientific validation and live end-to-end operation remain the parent's responsibility. Native spacing may fail the unchanged minimum sample support per Fe scoring window; no interpolation was introduced to force eligibility. Historical deferred shot-plan batches that were already marked completed retain that historical state; allowlist adoption backfills their acquisition without replaying the batch. Pymatgen availability is not established in this Pantheum staging tree (no .venv); no structures were used or modified.


## Full-suite iteration and final verification

Both full after-edit pytest commands completed with 759 passed, 11 failed, 26 skipped and 294 subtests passed. Logs: `/private/tmp/queue-after.log`, `/private/tmp/queue-after-pymatgen.log`. The dependency invocation again reported missing `.venv/bin/python`, so it did not establish pymatgen coverage. Failures were inspected: ten were omitted staged repository fixtures (`protocols/`, `ros2/`, `data/logos/`); one was a real lifecycle integration regression against an existing SimpleNamespace worker mock. The lifecycle regression was fixed by constructing/recovering/reconciling Opal retrieval only when `config.acquire.retrieval == opal_database`, preserving existing default-worker behavior. An explicit new CLI order test verifies the enabled path.

The parent authorized restoring omitted fixtures from original source. Copied 2 protocol, 4 ROS interface and 2 logo files from `/Users/mwhittaker/Projects/github/pantheum-I` into the stage and compared every SHA-256 against source. No production source or fixture content was changed. These copies are staging repairs, not implementation changes.

After the lifecycle fix: `PYTHONPATH=src python3 -m pytest tests/test_alibz_retrieval.py tests/test_alibz_checkouts.py -q` — 30 passed, 17 subtests passed, exit 0 (2.97 s); `/private/tmp/queue-lifecycle.log`. This run preceded addition of the enabled CLI order test, which is included in final unittest discovery.

The first full unittest run, before fixture restoration and lifecycle fix, reported 795 tests with 1 failure, 10 errors and 26 skips (`/private/tmp/queue-unittest.log`). It is superseded by the final run `/private/tmp/queue-unittest-final.log`.

At parent request, stopped two redundant full pytest reruns after they had started; only those own pytest child processes were terminated (PIDs 72698 and 72753, selected from exact output-log parent command lines). The required final unittest process was preserved. Do not interpret their partial logs `/private/tmp/queue-after-final.log` and `/private/tmp/queue-after-pymatgen-final.log` as completed test evidence.

Final code SHA-256:
- acquire.py: `6d9bb099d46b1018f5c00fac2027a8f49af21293737e27c2936ef6c6364c776f`
- retrieval.py: `fbd2cd965ca4858f63a553f58abe4564d216c835d652f4f681afdc2f14735b02`
- __main__.py: `ddf06310bee86f96b1a6a867d8338fdd94f29b8642dc966cd6debef51fbdc717`
- optimization.py: `2c96f4a6b9dd3a1dd6f35bb468aff3ae7af4be81b409ab593da22665b1b1e5cd`
- tests/test_alibz_retrieval.py: `00a5a7def4373aeef28202c2ee7e694d95995bdba2647b32f79e5e8898afb72c`


## Final result

`python3 -m unittest discover -s tests -v` from the final staged tree, with loopback access: **exit 0; 796 tests run in 156.774 seconds; 770 passed, 26 skipped; no failures or errors**. Complete log: `/private/tmp/queue-unittest-final.log`. This final run includes all 13 new retrieval tests, the fixed legacy worker lifecycle test, acquisition, optimization, reservation and remaining backend regressions. No test failure was skipped to reach this result; omitted staged fixtures were restored byte-identically from source before their suites ran.

All requested queue-side work is complete in the stage. Live deployment, actual Opal producer/transport validation and physical acquisition were intentionally outside this agent's scope. Provider switches: none.

## Concurrent source work preserved (follow-up merge)

The parent detected source changes after the first passing full suite. No deployment was performed by this agent. A read-only comparison to `/private/tmp/pantheum-z300-ingest-baseline.json` found six changed source files: acquisition implementation/tests, native-grid optimization metrics/tests, and camera-layout JavaScript/CSS. Byte copies of all six and current hashes were saved under `/private/tmp/pantheum-z300-source-concurrent-20260922/`; `changed-files.json` maps original baseline and current source hashes; `source-current-hashes.json` holds the source snapshot for the parent's compare-and-set install. Source files and UI work were never overwritten.

The staged acquisition now retains our retrieval changes plus only the exact concurrent native-grid changes from the saved source:

- `pantheum/alibz/acquire.py:61`: `GRID = 'native'`, and removal of the uniform resampling import.
- `pantheum/alibz/acquire.py:671`: the source's `_sorted_unique_csv` body transplanted byte-identically; preserves original detector samples, keeps first strictly increasing samples at shared segment boundaries, does not interpolate.
- `pantheum/alibz/acquire.py:691` and `:713`: native grid recorded in `test.json` and acquisition manifest.
- `tests/test_alibz_acquire.py:78`: copied the concurrent test file byte-identically, including a cubic-calibration native-grid regression. Existing tests were unchanged by our prior task, proven against the baseline hash before replacement.

The parent subsequently authorized including the related native metric changes in stage/deployment:

- `pantheum/alibz/optimization_metrics.py:57`: concurrent minimum three-pixel windows; `:61` line/background windows widen to original detector samples where narrow wavelength windows lack support; widened sidebands leave one sample gap; existing boundary/incomplete/saturation/quality rejection remains. `:142` records point count and minimum/median/maximum wavelength spacing; `:194` returns grid metadata with the metrics. Implementation copied byte-identically from source.
- `tests/test_alibz_optimization_metrics.py:71`: copied concurrent coarse, uneven-grid quality test byte-identically; original fine-grid tests retained.
- `docs/acquisition-optimization.md:48` and automatic-retrieval section: updated our documentation to explain the preserved concurrent minimum-pixel behavior, with no interpolation.
- `DECISIONS.md:414`: records the concurrent native-window design.

The camera-layout edits in `web/alibz/app.js` and `web/alibz/styles.css` were inspected and snapshotted only; they were not transplanted into stage or modified at source. Current source acquisition/test hashes still matched their captured snapshots after the native metric copy. The acquisition merge script is `/private/tmp/merge_concurrent_native_acquire.py`; merged artifact hashes are `/private/tmp/pantheum-z300-source-concurrent-20260922/staged-merged-hashes.json`.

Hashes needed for installation:

- Source acquire.py: `aa6a97501557b2589b1c3b9bb59f9c41bbf8683fb435f00423d2794525a9791a`.
- Staged merged acquire.py: `4a23a6915c1bc88454c13dacdce5e1de9cbea6a358fb06c9276e324c7d6f147b`.
- Original baseline optimization_metrics.py: `e01eb1cfd7fbef328d67082f37c84b1560e3c53c43d96b0454808c284cc40c60`.
- Source/staged optimization_metrics.py: `4bbcc186819f5c5725099559b731a8761ce80b8c1492c12caf0dc2a183a57fe9`.
- Source/staged tests/test_alibz_acquire.py: `e4ef0227ca0a91fbf4b35e8a422113f20eed4448f42933e303e009ac995165be`.
- Source/staged tests/test_alibz_optimization_metrics.py: `736ac8a2184380c585da99b2e2a9523eaaf29a7eeb1b46b07770723f4defbf7c`.

Pre-merge test evidence is the prior final full suite (796 tests, 770 passed, 26 skipped). Post-merge focused verification:

- `PYTHONPATH=src python3 -m pytest tests/test_alibz_acquire.py tests/test_alibz_retrieval.py -q`: **exit 0, 80 passed, 32 subtests passed**, 38.61 s; `/private/tmp/queue-native-merge-focused.log`.
- `PYTHONPATH=src python3 -m pytest tests/test_alibz_optimization_metrics.py tests/test_alibz_optimization.py -q`: **exit 0, 28 passed, 15 subtests passed**, 3.16 s; `/private/tmp/queue-native-metrics-focused.log`.
- Single final post-merge full suite: `python3 -m unittest discover -s tests -v`, `/private/tmp/queue-native-merge-unittest.log` (result pending).

Review limitation: concurrent generic metric code labels its grid native as an assumption; this does not establish that previously exported uniform historical spectra were converted. New Opal acquisitions independently require an exact native bundle and preserve its producer provenance. Widened windows operate on real samples and have different wavelength widths on coarse versus fine grids; older stored metric results are not automatically recomputed by this change.

The parent reported additional concurrent live acquisition/configuration changes and retained responsibility for fresh live-state checks and reservation locking before installation. This agent neither changed live configuration nor adopted or modified any live acquisition state.

### Latest native-metric revision pinned without losing source work

The first native merge full suite passed: `python3 -m unittest discover -s tests -v` — **exit 0, 798 tests, 772 passed and 26 skipped**, 153.634 s; `/private/tmp/queue-native-merge-unittest.log`.

During that run, source metric code/tests changed again. After the completed suite, captured and copied the latest source revision exactly once (parent explicitly authorized this). Prior source metric/test copies are retained under snapshot `revision-1/`; current snapshot metadata is updated atomically. No source file was overwritten.

Latest concurrent changes, preserved byte-identically:

- `pantheum/alibz/optimization_metrics.py:60`: central minimum becomes five original samples so a peak one pixel off the reference can remain interior; sidebands retain three samples and widened sidebands retain the one-pixel gap. Existing fine-grid windows exceeding the minima are retained. `:147` records separate central/sideband minima and a measured uniformity flag, replacing the earlier unconditional native flag.
- `tests/test_alibz_optimization_metrics.py:71`: assertions updated to separate central minimum and measured nonuniformity. The prior report's native-label caveat is therefore resolved in the pinned final code; producer provenance remains the native-source evidence.
- Documentation/decision text now states five line samples and three background samples, without interpolation.

Latest source/staged metrics SHA-256: `68d4d4859d349f7629fa6d6a451569632c89a85729247a0f298269735eee850a`.
Latest source/staged metric tests SHA-256: `ace889f1aeab481a66565c11756bd7cb0a8362b343c097b6c253f14d17983ffa`.
Acquisition merged SHA and all retrieval code hashes remain unchanged from the preceding native merge. `staged-merged-hashes.json` now pins the final metric and documentation revisions. Source-current hash files are install guards, not an instruction to overwrite future source edits; the parent will compare again immediately before installation.

Combined focused command: `PYTHONPATH=src python3 -m pytest tests/test_alibz_optimization_metrics.py tests/test_alibz_optimization.py tests/test_alibz_retrieval.py -q`; log `/private/tmp/queue-native-revision2-focused.log`.
Final full command: `python3 -m unittest discover -s tests -v`; log `/private/tmp/queue-native-revision2-unittest.log`. **Exit 0; 798 tests in 153.526 seconds; 772 passed, 26 skipped; no failures or errors.**


## Final merged outcome (supersedes earlier iteration results)

Concurrent acquisition and latest metric work are preserved, retrieval changes remain intact, and source/UI files are untouched. Final combined native metrics/optimizer/retrieval verification: **41 passed, 28 subtests passed; exit 0** (`/private/tmp/queue-native-revision2-focused.log`). Final full backend verification: **798 unittest cases, 772 passed, 26 skipped; exit 0** (`/private/tmp/queue-native-revision2-unittest.log`). This compares with 796 cases before the two concurrent native regression tests were preserved. No requested checks remain unfinished in this staged task.

The install bundle must include the merged acquisition implementation, retrieval/worker/optimizer changes and latest native metric implementation; current source metric/test files are already byte-identical to the staged versions and need no source overwrite. Parent retains responsibility for an immediate source/runtime hash comparison, live-state/reservation checks, actual transport and deployment. No provider switches occurred.
