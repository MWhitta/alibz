# Opal exact-test Z300 native producer — 2026-09-22

## Legacy database bundle extension implemented

Parent's live read now reaches the exact API-created test revision, which lacks
`shotTable.all_fb`. Historical storage documentation identifies the other exact
document reference `shotTable.all`: a SHA1-addressed ZIP in
`libzdata/spectrum/<first-two-hex>/<sha1>`, containing shot entries `0..N-1` and
average `-1`, each gzip JSON with knots, cubic calibrations and pixel counts.
The producer supports that documented format, with bounded ZIP/gzip decoding
and a separate zero-based-pixel calibration provenance. The instrument-provided
`-1` average is calibrated directly for `average.csv`; it is not recomputed. Existing
FlatBuffer math and offset -18 remain unchanged. No live probes by this worker.

## Live transport correction implemented

Parent's Opal validation found this firmware returns exit `-1` and empty
stdout for ADB `exec-out`, while ordinary `adb shell` works. The producer now
uses normalized text shell output for SQLite metadata and
`adb pull` into a bounded temporary Opal file for binary FlatBuffers. Binary
data never passes through the CRLF-transforming legacy shell transport.
The regression runs a real fake-ADB subprocess to verify CRLF normalization,
exact binary CR/LF/NUL/0xFF preservation, oversize termination, timeout and temp
cleanup. The transport regression passes with the full 71-case focused suite; this worker performs no live probes.

Implemented the bounded producer assigned by `/root`. No deployment, remote
modification, acquisition, app control, reboot, root command, or host `lrc`
access was performed by this worker. No provider switch occurred.

## Delivered files and behavior

- `scripts/z300_opal_ingest.py:45`: ZIP schema `alibz.opal-native.v1`; limits
  of 8 MiB/document, 128 MiB/raw blob, 8 MiB/CSV, 256 MiB/ZIP and aggregate
  uncompressed payload, 1000 expected shots. Explicit calibration limitations
  and the known fourth-channel placeholder signature are embedded in provenance.
- `scripts/z300_opal_ingest.py:88`: validated CLI configuration and safe run,
  exact test, serial and filename handling. `:128` bounds ADB output and each
  subprocess lifetime. Shell is restricted to text reads; binary data uses ADB
  pull with local temporary-file size monitoring every 20 ms plus a final size
  check, and process timeout. No device control or raw diagnostic disclosure.
- `scripts/z300_opal_ingest.py:226`: existing-file guard prevents SQLite from
  creating a DB when the SD mount is missing. Transactional exact
  `docs.docid`/`revs.doc_id` selection requires `current=1` and `deleted=0`;
  conflicting revisions fail. Revision sequence, revid, docid and raw JSON are
  retained. `:265` accepts only the selected document's plain `shotTable.all_fb`
  filename. No latest/name/time guessing or full-database transfer is performed.
- `scripts/z300_opal_ingest.py:290`: reject explicitly stored averages through
  top-level `onlyAvgSaved` or `config.numShotsToAvg != 1`; retain observed
  configuration fields without inventing defaults for missing document fields.
- `scripts/z300_opal_ingest.py:308`: strict FlatBuffer bounds and field checks.
  `:365` requires the exact expected stored shot count, four cubic coefficients,
  2066 finite samples per segment, finite strictly monotonic calibration axes,
  and strictly increasing seam-trimmed native output. All shot wavelength axes
  must match exactly before averaging; no interpolation/recovery approximation.
- `scripts/z300_opal_ingest.py:365`: excludes only fourth segment 3 with exact
  cubic `[961,-0.0004,1e-12,1e-12]` and seam `[960,961]`; other fourth-channel
  signatures fail closed for scientific review. Raw all-four-channel bundle is
  preserved. This intentional change removes the documented inactive-channel
  placeholder from scientific CSVs. Existing decoder implementation is unchanged.
- `scripts/z300_opal_ingest.py:513`: nonblocking OS advisory lock (Windows
  `msvcrt.locking`, POSIX `flock`) prevents simultaneous same-run publication and
  releases on process exit/crash. The lock file remains, preventing inode races.
- `scripts/z300_opal_ingest.py:546`: native CSV uses 17 significant digits for
  binary64 round trips. `:556` checks existing archives and ZIP receipt hashes;
  repeated requests return the same bytes without contacting the instrument.
- `scripts/z300_opal_ingest.py:588`: exact document read, two full blob reads
  with matching length/SHA-256, repeated identical revision query, strict decode,
  per-shot CSVs and arithmetic native-grid mean, raw document/blob/revision,
  manifest with hashes and provenance, fsync, then atomic directory rename.
  Staging files use write-capable handles for Windows fsync compatibility.
- `scripts/z300_opal_ingest.py:686`: CLI options requested in the brief, plus
  `--source-revision`, `--timeout`, and optional atomic `--output PATH`.
  Default success emits only a binary ZIP on stdout. Errors return 2;
  pending/incomplete/concurrent acquisitions return 3 with empty stdout.
  `--dry-run` emits a plan on stderr and performs no ADB access or archive writes.
- `tests/test_z300_opal_ingest.py:22`: generated FlatBuffer fixture builder;
  `:88` mocked ADB executes the producer's actual SQL against synthetic CBL1
  SQLite tables. Tests `:140` onward cover exact selection, complete manifest,
  hash integrity, native axes/means, idempotence, corruption and wrong-test
  rejection, changing blobs/revisions, missing/deleted/obsolete records, wrong
  shot count, conflicts, averaged storage, traversal and malformed bounds,
  invalid calibration/data, preservation of decoder math, dry run, stdout ZIP,
  atomic output, concurrent/process-exit locking and payload limits.
- `tests/test_z300_opal_ingest.py:344`: real fake-ADB subprocess verifies old
  Android shell text/newline handling, exact binary pull, bounded size/time and
  temporary cleanup; `:382` rejects binary pulls outside safe spectra paths.

## Verification

Both mandated project invocations were run before implementation:

1. `PYTHONPATH=src python3 -m pytest tests/ -q` — **30 collection errors**, no
   tests executed: this interpreter lacks `scipy`. Pre-existing environment
   limitation; the repository has `alibz/`, not a `src/` package tree.
2. `PYTHONPATH=src:$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])") python3 -m pytest tests/ -q`
   — **326 passed, 4 skipped, 55 subtests passed, 1 warning**, 277.12 s.

Focused final producer suite:

` .venv/bin/python -m pytest tests/test_z300_opal_ingest.py -q `

— **71 passed**, 1.10 s. Windows fsync-handle change, missing-DB guard, legacy
Android transport and legacy ZIP/gzip JSON fallback are included. The SQLite mock asserts the
actual existing-file guard before its transactional SELECT, and atomic-output/
stdout tests verify emitted ZIP bytes. No broad rerun was needed after the
transport and format corrections; their focused regressions verify them.

Post-change mandated invocations:

1. Base interpreter: **30 collection errors**, same missing `scipy`, 11.51 s.
   Log: `reports/2026-09-22-opal-producer-base-after.log`.
2. Pymatgen-enabled invocation: **375 passed, 4 skipped, 55 subtests passed,
   1 warning**, 280.26 s (includes five concurrently added tests outside this
   worker's 44-case producer suite). Log:
   `reports/2026-09-22-opal-producer-enabled-after.log`.

Already-local historical raw bundles (not newly transferred) were checked:

| Bundle prefix | Stored shots | Scientific native points/shot | Retained channels identical to old decoder |
|---|---:|---:|---|
| `89899c99` | 15 | 5848 | yes, exact array equality |
| `c544de3a` | 16 | 5848 | yes, exact array equality |
| `dc6ab9b1` | 14 | 5848 | yes, exact array equality |
| `f5022c9d` | 16 | 5848 | yes, exact array equality |

These 61 shots have identical axes within each bundle. Original decoder output
has 7895 points including 2047 fourth-channel placeholder samples; the producer
retains exactly the same first three channel wavelengths and intensities and
excludes that placeholder. Native range is about 187.756–950.674 nm in the three
recent bundles; the older bundle is about 187.770–950.731 nm. No uniform grid was
introduced. Historical placeholder basis: sibling repository
`pantheum-I/reports/2026-09-12-acquire-calibration.md:239`, plus exact coefficient
and seam checks in all 61 local shots. Existing decoder math and existing CLI
remain unchanged, demonstrated by the synthetic equivalence test and these
historical comparisons.

## Limits and handoff

- Parent explicitly revised the initial whole-DB snapshot request to exact
  transactional document selection + repeated blob/revision verification. The
  archive records `database_archive: null`; no whole DB snapshot is claimed.
- SQLite schema is exercised against a synthetic CBL1 fixture. Live DB schema,
  Android shell/ADB behavior, Windows process execution and deployment are the
  parent's Opal-only validation work; this worker did not claim live success.
- Android SQLite 3.7.11 lacks `-readonly` per parent inspection. Commands request
  no changes, but SQLite itself can recover an existing hot journal when opening
  the DB. The file guard prevents accidental empty DB creation if absent.
- Matching repeated reads prove a bounded stable observation, not that the
  instrument can never alter that source later. Expected shot count and current
  revision checks prevent publishing an observed incomplete bundle.
- The empirically inferred pixel offset is **not** a certified absolute
  wavelength calibration. Provenance explicitly retains its historical validation
  limits. Unknown fourth-channel interpretation and differing shot axes fail.
- Lock crash-release is tested on POSIX here; Windows branch needs the parent's
  actual Opal check. No new physical acquisition was performed.

Final source hashes supplied to parent for staging:

```
edaa9748520281f4a2b8452fdfb4c1b994fb27bdbf80bb288674f5b8cdf4d301  scripts/z300_opal_ingest.py
3130e640202bfa40861d58babdc2dcde9066f8bb2f5a49c4c108165733225b23  scripts/z300_fb_decode.py
959e0d2dbc384bfec93acb4d6a5e39fd2b3e8770c28b44c24e29b3eee8ecc7f8  tests/test_z300_opal_ingest.py
```

## Legacy-format extension details

- `scripts/z300_opal_ingest.py:275`: source selection keeps `all_fb` precedence;
  only its absence permits exact `shotTable.all`, requiring a 40-character
  lowercase SHA1 and its matching two-character shard. The read transport
  `:128` allows only these safe paths; no guessed/latest source or API calls.
- `scripts/z300_opal_ingest.py:420`: production legacy spectrum validation
  requires four finite cubic coefficient vectors, four finite 2066-value
  detector arrays, five increasing knots, monotonic calibrated axes, and the
  exact fourth-placeholder signature. Raw zero-based pixel math and closed
  clipping intervals match `pantheum/alibz/z300_calibration.py` independently
  checked by the scalar reference regression; offset -18 remains FB-only.
- `scripts/z300_opal_ingest.py:472`: ZIP must contain exactly the unique names
  `-1`, `0` through expected-minus-one. Missing members are pending; extras,
  duplicate names, encryption, corrupt or oversized members fail. Each ZIP
  member and gzip JSON expansion is capped at 8 MiB, with 256 MiB aggregate
  limits. Plain JSON ZIP members are supported as well. Per-shot and vendor
  average wavelength axes must match exactly.
- `scripts/z300_opal_ingest.py:556` and `:588`: immutable archive verification
  accepts raw/all.zip for the explicit legacy source format, raw/all.fb for FB.
  Instrument average -1 becomes legacy average.csv directly. Provenance records
  source format, average origin, pixel offset zero, native calibration limits
  and the pinned reference-module SHA256. FB uses its existing arithmetic mean
  and unchanged decoder arrays. No extra Opal runtime dependency is introduced.
- `tests/fixtures/z300_real_spectrum_trimmed.json:1` is copied byte-for-byte
  from the sibling pantheum-I fixture: the real four calibrations and first
  100 pixels per channel. SHA256:
  `6b1af2213ffc1186078ab5f6f1f19d2ffa7eb576f3070a1dd6ba49031809a974`.
  The production-shaped synthetic legacy fixture (`tests/test_z300_opal_ingest.py:398`)
  uses those real coefficients with explicitly synthetic 2066-value arrays;
  it is not represented as a complete real acquisition. Its scientific grid
  has 5849 points after excluding the known inactive fourth segment.
- Tests `tests/test_z300_opal_ingest.py:417` onward cover direct vendor average
  preservation (deliberately different from per-shot mean), exact source/hash
  constraints, FB precedence, native math, missing/extra members, malformed
  shapes/calibration/nonfinite values, mismatched axes, decompression bounds,
  safe sharded pull and idempotent legacy archives. All prior 47 cases pass.
- Reference algorithm SHA256:
  `8a6135d058f09c8de62542695c7a1ac923d577906fdc88b8db2e8ded89bd156d`
  (`pantheum-I/pantheum/alibz/z300_calibration.py`). No unrelated code modified.

Parent reported Opal-only structural verification of the existing requested
test: legacy source present, FlatBuffer source absent, onlyAvgSaved false,
numShotsToAvg 1.0, 154825-byte ZIP with exactly eleven required entries, and
four 2066-value channels. This worker did not access that raw test metadata or
blob and does not claim final live producer success until parent verifies it.
