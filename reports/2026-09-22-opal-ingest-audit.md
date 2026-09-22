# Opal Z300 per-run ingestion audit — 2026-09-22

Audit only: no deployment or source edits. No provider switch. Existing subscription used. No laser, gantry, cancellation, analyzer configuration, or persistent service changes.

## Transport and installed prerequisites

The practical path is a bounded transient command from Pantheum's Moissanite worker to Opal, using existing SSH authorization, then ADB reads on Opal. Conversion and durable raw/native files belong on Opal; return only the exact run's native artifact bundle to Pantheum. No new listener, firewall rule, scheduled task, key forwarding, or credentials are needed.

The parent independently verified the actual worker-side SSH route:

```sh
ssh moissanite 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=8 -p 52222 whittaker@192.168.50.112 whoami'
```

Result reported by parent: `mwhittaker\whittaker`. My shorter `ssh moissanite 'ssh ... opal hostname'` failed because Moissanite has no `opal` hostname/alias. Therefore configure the concrete forwarded endpoint, not a nonexistent worker alias. Authoritative laptop SSH configuration is `../lab-networking/ssh/lab.ssh_config:147` (Opal uses bridge `.112:52222`, ProxyJump Moissanite).

Read-only SSH/PowerShell probes directly verified:

| Item | Observation |
|---|---|
| Opal computer name | `MWHITTAKER` |
| General Python | `C:\Users\Whittaker\AppData\Local\Programs\Python\Python312\python.exe` |
| alibz Python | `C:\InstrumentControl\alibz\.venv\Scripts\python.exe` |
| alibz repository | `C:\InstrumentControl\alibz` |
| NumPy | `2.4.4`, imported successfully from alibz Python |
| Pipeline config | `C:\InstrumentControl\opal\opal-pipeline.json` |
| Pipeline alibz enabled | `true` |
| Pipeline MTP enabled | `false` |
| Data root | `C:\LabData\LIBS` |
| State root | `C:\LabData\.labdesk\libs` |
| Tasks | `OpalCamera`, `OpalLIBSIngest`, State `3` (raw PowerShell enum value) |
| ADB executable | `C:\Users\Whittaker\AppData\Local\Temp\alibz-api-recovery-20260921\platform-tools\adb.exe` |
| ADB server port | `5038` |
| ADB serial | `0123456789ABCDEF`, authorized `device` |
| ADB shell identity | `uid=0(root) gid=0(root)` |
| Analyzer database | `/sdcard/libzdata/libzdb.cblite` |
| Analyzer database size | `20,643,840` bytes, timestamp `2026-09-22 09:26` |
| Rollback journal | `218,024` bytes, same timestamp |
| Analyzer SQLite CLI | `sqlite3` and `/system/xbin/sqlite3`, version `3.7.11` |

ADB automatically started its existing host daemon on port 5038 during a probe; no persistent service was registered. The executable currently lives in a Windows temporary directory, so deployment should either pin and verify that existing path or stage a durable executable without downloading unreviewed binaries. Prior source pin: `reports/2026-09-21-api-service-recovery.md:8` (Google platform-tools 37.0.1 ZIP SHA-256 recorded there).

The actual probes used PowerShell `-EncodedCommand` over SSH, `Get-Content ...opal-pipeline.json` with only the table's selected keys returned, `Get-ScheduledTask -TaskName '*Opal*'`, alibz Python `-B -c 'import sys,numpy;...'`, and these ADB read commands:

```text
adb -P 5038 devices -l
adb -P 5038 -s 0123456789ABCDEF shell 'id; ... ls -l ...'
adb -P 5038 -s 0123456789ABCDEF shell 'for p in /sdcard /mnt/extsd /storage/sdcard1; do if [ -f $p/libzdata/libzdb.cblite ]; then echo DB_PATH=$p/libzdata/libzdb.cblite; fi; done; sqlite3 -version; /system/xbin/sqlite3 -version; ...'
```

## Exact test correlation, consistency, and completeness

Known successful run from existing local state: `run-e825d9f567204c5cbc6fc33c9c16bbf7` acknowledged test `d45a1971-68aa-4fa6-b511-e75f3e62fc0c` at 08:58 PDT, left `awaiting_data` (`reports/STATUS.md`, earlier “FIRST SUCCESSFUL API FIRE” section). This is historical evidence; audit did not requery the queue. A later 09:30 test was refused for invalid laser parameters and is not a successful validation sample.

The database is Couchbase Lite 1.x/SQLite; historical documents contain `config`, `onlyAvgSaved`, `shotTable`, `unixTime`, and user metadata (`../instrument-control/reports/2026-09-12-pb-programmatic-control.md:95`). A modern flatbuffer test uses `shotTable.all_fb` naming a UUID file directly under `/sdcard/libzdata/spectra/`; older `shotTable.all` points into the distinct sharded `libzdata/spectrum/<xx>/<sha1>` format and should not be silently interpreted as a flatbuffer (`scripts/z300_fb_decode.py:4`; `reports/2026-09-21-fe-purity-checkout.md:151`).

Expected CBL schema is `docs(doc_id,docid)` joined to `revs(sequence,doc_id,revid,current,deleted,json)`. **This schema is inferred from CBL conventions, not verified against a returned live schema by this audit.** Producer must validate locally on Opal. Select only the exact acknowledged test ID with current, nondeleted revision; never infer a match from newest test, timestamp, sample name, or count. A candidate query is:

```sql
SELECT revs.sequence,revs.revid,revs.json
FROM revs JOIN docs ON revs.doc_id=docs.doc_id
WHERE docs.docid=? AND revs.current=1 AND revs.deleted=0;
```

Run the exact read transaction through ADB into an Opal-local file. Do not return raw database metadata to the laptop. Validate UUIDs before constructing CLI SQL/shell arguments. Query before and after bounded blob transfer and require unchanged revision plus stable blob size/hash; refuse missing, conflicting, malformed, partial, or changed data. A plain `adb pull` of the live `.cblite` plus journal is not a consistent SQLite snapshot. SQLite `.backup` could produce one but writes an analyzer temporary file and therefore is not recommended for the requested instrument-read-only design. SQLite 3.7.11 is old: do not assume modern `-readonly`, URI, or `PRAGMA query_only` support without capability checks.

Only mark queue completion after the entire exact run's raw evidence and native outputs are durable on Opal and imported/validated by Pantheum. The archive manifest should contain run/test identity, selected DB revision, config/averaging semantics, blob and output hashes, ordered shot entries, decoder provenance, and explicit native grid metadata. Expose retryable ingestion failure rather than replaying acquisition.

The historical flatbuffer root field 3 is an ordered vector of stored spectra (`scripts/z300_fb_decode.py:124`); fields are described in its module docstring. **Stored spectra are not inherently laser pulse count or proven raster coordinates.** Existing reports show a configured 16-location test yielding 15 stored shots; missing location could not be recovered, and a known prior export skipped mid-raster (`reports/2026-09-21-fe-purity-checkout.md:163`). Validate the planned expected stored count under `numShotsPerLocation`, `numShotsToAvg`, `rasterNumLocations`, and `onlyAvgSaved`; reject unsupported averaging or short sets rather than assigning invented coordinates or claiming completeness.

## Native grid and scientific provenance

`scripts/z300_fb_decode.py:135` decodes every segment's own wavelength polynomial and intensity array. It evaluates the coefficient polynomial at stored index **minus 18**, clips each segment to its supplied seam interval, concatenates, and sorts ascending (`:153`, `:162`). This produces the detector's nonuniform native spacing without interpolation. `decode()` already returns native arrays (`:193`). `write_csv(..., export_grid=False)` preserves them; the CLI requires `--native-grid` because its default currently resamples to a uniform 0.1 nm vendor export grid (`:203`, `:270`).

Offset −18 is empirically fitted against vendor exports, not vendor-documented calibration. Historical matched exports produced r=0.94/0.98, RMS ~1.2% full-scale, line-center agreement ±51 pm (`reports/2026-09-21-fe-purity-checkout.md:182`). Those values do not establish exact numerical equivalence or instrument-wide calibration. Keep the decoded raw bundle, per-segment calibration/edges, pixel-offset provenance and decoder hash; native spacing is not permission to silently change calibration assumptions. For averages, require identical native wavelength axes (or refuse); interpolation would cease to be native per-pixel averaging.

Already-local genuine bundles and two vendor CSVs are available for integration verification at:

```text
/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/0ba52321-8cb9-4d7d-a516-ac08c42fef42/scratchpad/fb/
/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/0ba52321-8cb9-4d7d-a516-ac08c42fef42/scratchpad/gt/
```

No `libzdb.cblite` already-local archive was found under `/private/tmp`, including hidden/ignored paths.

## Existing paths that do not satisfy this request

- `../pantheum-I/deploy/alibz/pull-z300-usb.sh:2` is bounded MTP copying of **CSV exports**; no exact test association, DB extraction, native conversion, or Pantheum completion. It depends on manually produced exports and should not become an after-run default.
- MTP background offload historically wedged the analyzer; its live disabled state is intentional (`../instrument-control/reports/2026-09-14-z300-display-and-upload.md:116`). Do not reenable it for automation.
- `../instrument-control/libs_pipeline/z300_pull.py:156` still multiplies `since` by 1000, a known firmware-crashing query corrected in Pantheum on Sep 21 (`reports/2026-09-21-api-service-recovery.md:16`). It also fetches only one test page, terminates at the first shot 404, records incomplete sets without planned count validation, and forces a 0.1 nm grid (`:99`, `:169`, `:247`). Do not reuse it unchanged.
- `OpalLIBSIngest` is an existing logon-triggered passive archive/reduction/upload runner (`../instrument-control/deploy/opal/opal-install.ps1:33`; `opal-pipeline.ps1:100`). Its presence does not demonstrate integration with Pantheum's per-run acquisition completion.

## Approval review limit

Automatic approval review rejected a live query that would have returned cached database schema plus a bounded sample of acquisition metadata/configuration to the agent host. Its stated reason: the user authorized transfer to Opal, not disclosure of this private database payload to that destination. The action was not executed; no workaround or indirect payload extraction was attempted. Unaffected implementation can keep database documents on Opal and return only verification booleans/counts/hashes here, while delivering requested native result artifacts directly to Pantheum's Moissanite queue. Parent was informed of this boundary immediately.
