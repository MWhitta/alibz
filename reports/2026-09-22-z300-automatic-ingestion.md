# Automatic Z300 database retrieval and native conversion — deployed

Enabled at 2026-09-22 10:05 PDT. Pantheum now uses
`acquire.retrieval=opal_database`. Both `pantheum-alibz.service` and
`pantheum-alibz-worker.service` are active, installed source hashes match the
reviewed bundle, and private configuration outside the retrieval settings is
unchanged. No laser, gantry, cancellation, settings-push, or new acquisition
command was issued by this task. No provider switch occurred.

## Behavior

After a run is acknowledged, its exact test ID enters `awaiting_data`. The
existing single worker invokes a transient process on Opal over the already
authorized Moissanite → bridge `192.168.50.112:52222` → Opal SSH connection.
Opal reads the exact current nondeleted Couchbase Lite revision over USB ADB,
pulls the referenced spectrum bundle twice, and verifies the revision and bytes
remained stable. The old Android firmware supports shell and `adb pull`, but
not `exec-out`; the implemented transport reflects live verification.

Both database formats are supported: handheld `shotTable.all_fb` FlatBuffers
and API `shotTable.all` ZIPs of gzip JSON records. Conversion happens on Opal,
using the format's stored per-shot wavelength calibration. The known dummy
fourth channel is retained in raw evidence and excluded from scientific CSVs.
No uniform export grid is introduced. FlatBuffers use the established empirical
pixel offset -18; JSON uses the existing zero-based calibration contract. Their
different calibration provenance and limits are recorded in every manifest.
For JSON, the instrument's `-1` average is preserved; FlatBuffers use a mean
only when all native axes are identical.

Opal stores an immutable, checksummed per-run archive under
`C:\LabData\LIBS\pantheum-acquisitions`. The producer and a hash-verified copy of
the existing ADB binaries live under
`C:\InstrumentControl\alibz\z300-ingest-20260922`. No new Opal service,
scheduled task, network listener, key, or credential was added.

Pantheum validates identity, count, file inventory, SHA256 hashes, finite values,
and strict wavelength ordering. It durably writes the native artifacts, then
commits the dataset, analysis job, and acquisition success together. Study
batches wait for that completion. Missing/incomplete data remains pending and
retries after 15 seconds; the transfer timeout is 240 seconds. No retrieval
retry can replay a physical operation. Worker restart recovers interrupted
retrieval claims. Disabling `opal_database` pauses these attempts.

## Live verification

Existing acquisition `run-eae9d54f22834928ae2f6af8d4a23779`, analyzer test
`63f26855-ae3e-4275-a34f-5fee887e5248`, was retrieved and converted on Opal:

- **10 individual shots**, each **5,848 native wavelength samples**.
- Native sample spacing: **0.0783188620–0.1983658586 nm**.
- Exact ZIP contents `-1, 0..9` were verified locally on Opal; the database
  reports individual spectra and an averaging count of 1.
- Result ZIP SHA256:
  `264be5fc7870ca782196d9d2cf45e70e417aef2817365f7246cadc92fb7ae4bd`.
- The actual Moissanite worker command fetched that Opal archive over SSH. In
  an isolated temporary queue, acquisition reached `succeeded`, its native
  preview job reached `completed`, and retry produced no duplicate dataset or
  job. Production records were read-only during this verification.

The three previously waiting acquisitions were recovered by a concurrent
session before this deployment. Their completed production records were
preserved; the final adoption allowlist is empty. After installation the
production queue had zero waiting-data runs and zero unresolved hardware holds.
The new route is enabled for subsequent runs. A newly fired physical run has
not been requested or performed by this task.

## Tests and reviewed changes

- Final Pantheum backend suite: `python3 -m unittest discover -s tests -v`:
  **798 cases, 772 passed, 26 optional skips, exit 0**. Log:
  `reports/2026-09-22-ingest-backend-tests.log`.
- Final local producer/deployment/validation-helper checks:
  **80 passed, 10 subtests passed**. Log:
  `reports/2026-09-22-ingest-local-tests.log`.
- A genuine previously archived 16-shot FlatBuffer also passed the producer →
  Pantheum validator contract, yielding 5,848 native points per shot. The final
  legacy extension retained FlatBuffer regression coverage.
- Deployment dry run checked current hashes, idle acquisitions/analysis, and
  unresolved reservations before installation. Backup:
  `/home/mwhittaker/pantheum-ingest-backup-20260922T170511149189Z`.

The implementation lives in alibz `scripts/z300_opal_ingest.py` and sibling
Pantheum `pantheum/alibz/retrieval.py`, with acquisition mode/state, worker, and
study integration. Local Pantheum source was installed with hash guards and
source backups. Concurrent native acquisition/metrics changes were merged and
tested, including five-sample line windows and three-sample sidebands; unrelated
UI changes were preserved. A concurrent deployment was caught by the runtime
hash guard; its files matched the reviewed native-grid snapshot and were
preserved before installation proceeded.

## Evidence and operational limits

Installed hashes, verified service states, configuration checks, source backup,
and live receipt are in:

- `provenance/z300-ingest-runtime-20260922.json`
- `provenance/z300-ingest-deployment-manifest-20260922.json`
- `provenance/z300-ingest-source-20260922.json`
- `reports/2026-09-22-opal-ingest-audit.md`
- `reports/2026-09-22-opal-producer.md`
- `reports/2026-09-22-queue-ingest.md`

The analyzer must remain USB-connected and authorized on Opal, with Opal
reachable from Moissanite. Otherwise the queue waits and retries. Unsupported
averaging, unknown calibration layouts, changing revisions, incomplete shot
sets, and integrity failures cannot be published as complete acquisitions.
Native sampling alone does not establish absolute wavelength accuracy.

The initial audit request to return live database metadata to the laptop was
rejected by automatic approval review. The implemented validation kept those
contents on Opal and delivered native results directly to Moissanite; only
format flags, counts, and verification hashes were returned to the laptop.
There is no remaining approval blocker. Android's old SQLite CLI has no
`-readonly` flag; the producer issues only SELECT transactions and checks the
database exists first, but the SQLite engine may perform its normal journal
recovery. No instrument-side temporary database or backup is created.
