# Z300 live fire timeout audit — 2026-09-21

> Correction from the later USB log and approved live-response audit: the
> epoch-millisecond claim and `List<LIBZTest>` claim below were incorrect.
> Sending `since=1790027760000` caused an Android `NumberFormatException`
> and stopped the remote API service. `since=0` returned an object containing
> 200 string IDs under `data` and `next=1613659041`. The pinned generic method
> signature is `List<String>`. See `2026-09-21-api-service-recovery.md` and
> `2026-09-21-tests-since-semantics.md` for the superseding evidence. Do not
> follow the epoch-millisecond preflight recommendation below.

## Scope and conclusion

Read-only diagnosis of optimization run `run-ca9c938cc88c42e98b07493803992d00`.
No instrument or production service was contacted, and no source or configuration was
changed. The pinned Profile Builder artifact was examined locally. No provider switch
occurred.

The immediate adapter defect is a response timeout that is much shorter than the vendor
client's: Pantheum gives the synchronous `/data/firetest` call 5 seconds total, while
Profile Builder uses a 5,000 ms connection timeout and a 90,000 ms socket/read timeout.
The run moved from dispatch at `2026-09-21T21:57:46.972896Z` to uncertain at
`21:57:51.985560Z`, 5.013 seconds later, closely matching Pantheum's configured
five-second request bound. The timing strongly supports a client-side timeout, but the
live error detail was discarded, so it does not prove the exception type or establish
whether the analyzer accepted, began, or completed physical work.

A second, independent blocker is now observed: `/instrument/id` and
`/instrument/defaultParams?mode=4` answer quickly, while
`GET /data/tests?since=1790027760000` closes with an empty reply (`HTTP 000`) in about
0.018 s (and an earlier bounded probe also failed). Therefore generic reachability is
not adequate fire readiness. A new fire must remain disabled until the data-list endpoint
works and its response is parsed according to the pinned protocol.

## Deployed path and why the result is ambiguous

- `pantheum/alibz/acquire.py:529-545` constructs `Z300Client(..., timeout=5.0)`, persists
  `dispatch_started_at`, then calls `client.fire_test(params)`. The configured acquisition
  timeout is used only after `fire_test` returns (`acquire.py:546-555`).
- `pantheum/alibz/z300.py:78-93` passes the same timeout to `urllib.request.urlopen` and
  then blocks in `response.read()`. `z300.py:105-110` maps the expiry to a timeout error.
- `z300.py:151-162` synchronously POSTs `/data/firetest` and obtains the test id only from
  the response body. The observed five-second failure is consistent with the response
  read not completing; regardless of the discarded underlying exception, `test_id` was
  never returned to or saved by Pantheum.
- `acquire.py:448-449` preserves the request parameters, but `acquire.py:467-475` replaces
  every live `Z300Error`, including its endpoint and error code, with the same generic
  uncertain detail. The local `since_ms` at `acquire.py:531` is also not durable.
- Polling `/data/tests` starts only after a test id is returned (`acquire.py:545-550`).
  Pantheum therefore performs no post-timeout reconciliation and retains neither a raw
  response nor the exact transport diagnostic. This explains `test_id: null` and the
  loss of evidence.

The user's observation that no shots fired is valuable bench evidence. The later
`triggerLocked=0`, START screen, and idle confirmation support clearing the hardware
hold under the project's reconciliation rules. They do not retroactively prove what the
POST did, and the original run and scientific batch must remain uncertain/blocked rather
than be replayed.

## Pinned vendor protocol proof

Artifact:
`/private/tmp/pantheum-acquisition-ProfileBuilder.jar`, Profile Builder 2.19.1a,
SHA-256 `c8164f159eb610b0c7b154df7876010e9699332da5d9cf0704138c59c5814dd7`.
The extracted bytecode evidence is saved at
`reports/2026-09-21-fire-timeout-bytecode.txt`.

`com.sciaps.common.webserver.LIBZHttpClient` proves:

1. Constructor bytecode offsets 38-48 call
   `HttpConnectionParams.setConnectionTimeout(params, 5000)` and
   `setSoTimeout(params, 90000)`. These are separate connect and response-read bounds.
2. `takeTest(RasterParams)` offsets 10-69 construct `%s/data/firetest` as `HttpPost`,
   serialize the complete `RasterParams` with Gson, wrap it in a UTF-8 `StringEntity`,
   set `application/json`, attach it, and execute synchronously.
3. `takeTest` maps HTTP status 520 to `LaserNotArmedException` and 521 to
   `TriggerLockedException` (offsets 88-143), then parses the response entity as a JSON
   string and returns it as the test id (offsets 144-201).
4. `getTestsSince(J)` formats exactly `%s/data/tests?since=%d`, passing the `long`
   unchanged. It also constructs `new Date(long)` only for logging, confirming Unix epoch
   milliseconds rather than a formatted date.
5. The vendor response shape is a JSON object: `getTestsSince` calls
   `JsonElement.getAsJsonObject()`, retrieves member `"data"`, then deserializes that
   member as `List<LIBZTest>`. Pantheum instead requires a top-level list at
   `z300.py:173-177`; its fake server also models a top-level list (`z300.py:342-344`).
   Once the endpoint recovers, this mismatch can fail reconciliation even after a valid
   response.

The 2026-09-12 protocol report correctly identified the endpoint and payload, but marked
the POST verb as inferred (`instrument-control/reports/2026-09-12-pb-programmatic-control.md:61-64`).
The bytecode above replaces that inference with direct artifact evidence.

## Calibration, arming, and request parameters

Profile Builder's `AcquisitionAndResultsPanel.onInstrumentStatusEvent` maps
`wlCalibrationNeededCode` as follows: `0` sets `mWLCalibrationNeeded=true` and a red
indicator; `1` sets false/green; `-1` sets false/yellow (unknown); `-100` is unsupported.
Its `doScan` path asks for confirmation when calibration is needed and returns if the
operator declines; acceptance proceeds. Thus code 0 before this batch meant wavelength
calibration was needed, while code 1 afterward means the vendor UI considers it okay.
This is an advisory GUI gate, not a demonstrated hard HTTP precondition. The 0→1 state
change is evidence that analyzer state changed during the episode; it does not prove a
sample shot or stored test.

Pantheum discards `wlCalibrationNeededCode` when building its probe result
(`acquire.py:168-184`) and consequently does not include it in status/start gating
(`acquire.py:217-239`, `acquire.py:404-422`). That allowed dispatch while the pre-run
code was 0. The adapter does gate `triggerLocked`, while vendor `takeTest` additionally
recognizes explicit not-armed/locked status codes. A START screen and
`triggerLocked=0` do not prove the separate data API is healthy.

The submitted request (delay 10, period 25, ten data shots, one location, no cleaning,
`pulsePeriod=10`, start=end `[0,0,0]`, step 0, `resetStage=false`, argon preflush 0)
does not violate the pinned Profile Builder UI validations. `RasterSettingPanel.doSave`
requires nonnegative start x/y/z and end x/y, end x/y at least start x/y, nonnegative
step/delay/period/preflush/cleaning count, positive shots/locations/averaging, and at most
600 data shots. It has no upper coordinate bound or calibration-specific coordinate
condition. Therefore there is no evidence that zero coordinates caused the timeout.
Current mode-4 defaults (`[134,76,70]` to `[206,124,70]`, step 24, reset true,
`pulsePeriod=100`, preflush 300) must not be substituted automatically; physical
coordinates and reset behavior require an operator-verified same-spot recipe.

## Minimal corrective changes, in order

1. Before enabling or starting live fire, perform a bounded read-only
   `/data/tests?since=<recent epoch-ms>` preflight in addition to `/instrument/id`.
   Require a complete valid response and expose a specific `data API unhealthy` reason.
   Do not claim that this proves the laser can fire; it proves that the evidence/recovery
   path needed after dispatch is available.
2. Correct `tests_since` to unwrap the pinned `{ "data": [...] }` response. If a raw
   list is retained for compatibility, cover both explicitly and record which shape was
   observed; update the fake server to use the vendor envelope by default.
3. Split connect timeout from response-read timeout. Retain a short connect bound, but
   allow at least the pinned 90-second vendor read window for `/data/firetest`; keep the
   longer run/list polling deadline separate. This avoids declaring uncertainty at five
   seconds while the analyzer may still be processing.
4. Persist, before dispatch, the list-reconciliation watermark and the exact request.
   Persist endpoint, error code/type, elapsed time, and safe bounded response metadata on
   failure. After a fire-response timeout, reconcile by the stored time/request evidence
   without replay; never infer completion merely from reachability.
5. Preserve and expose `wlCalibrationNeededCode`, including ready, needed, unknown, and
   unsupported states. The pinned vendor GUI treats code 0 as an operator-overridable
   warning, so a hard software block would be a future scientific-policy decision rather
   than a protocol requirement and is not part of the minimal transport fix.
6. Do not auto-copy mode defaults or change stage coordinates. Keep the original run
   immutable and require a new run id only after data-API health, calibration readiness,
   and operator-controlled coordinates are established.

## Useful tests for the implementation

- A fake `/data/firetest` that responds after 6 seconds but before the configured read
  bound: exactly one POST, succeeds, and never retries.
- Connect timeout and response-read timeout are independent; a response timeout records
  endpoint/code/elapsed and leaves the operation uncertain.
- Vendor `{ "data": [...] }` test-list envelope parses; malformed, empty, prematurely
  closed, and (if supported) legacy raw-list responses have explicit outcomes.
- Live start is refused when `/instrument/id` works but `/data/tests` closes empty or is
  invalid, and the public reason identifies data API health.
- Calibration codes 0, 1, -1, and -100 map to needed/ready/unknown/unsupported in status;
  a separate policy test is warranted only if a future requirement makes code 0 blocking.
- Timeout after dispatch persists the reconciliation watermark and request; recovery
  may associate one matching later test but never sends a second fire POST.
- The existing request with start=end zero continues to pass schema validation, while no
  test invents a device coordinate maximum absent vendor evidence.

## Verification and limits

Required local `alibz` suite, before and after this read-only audit:

- `PYTHONPATH=src python3 -m pytest tests/ -q`: collection failed with 30 errors because
  the host Python lacks SciPy; 0 tests ran. This is unrelated to the audited Pantheum
  code.
- `PYTHONPATH=src:$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])") python3 -m pytest tests/ -q`:
  326 passed, 4 skipped, 55 subtests passed, 1 warning in 277.14 s.

After the two report artifacts were added, the dependency-free invocation reproduced the
same 30 SciPy collection errors (0 tests ran). A repeated pymatgen-enabled invocation was
stopped at the main session's request because this audit changed no scientific source; at
interruption it reported 147 passed, 1 skipped, 19 subtests passed, plus one timing
subfailure (`scan9x9.csv` took 189 s against a 120 s threshold) while the suites were run
concurrently. The completed pre-write pymatgen-enabled suite above is the usable result.

No source changed, so pre/post source behavior is byte-identical. Source hashes remained
`acquire.py` `d5e793e7a051cecdd1a79b3f9b181aecea70ed2b4a65618402a3900a84005658`
and `z300.py` `3243843ce91a979009e3d470d20824cfc31e232f7131a17c6fdd9f3d6eebb808`.

Unverified: the analyzer-side reason for the current `/data/tests` empty close; whether
the timed-out POST caused a calibration-only action, partial internal work, or no action;
whether firmware v2.19.7 ever emits a raw-list response contrary to the pinned client;
and physical coordinate limits. No hardware access was used to settle these questions.
