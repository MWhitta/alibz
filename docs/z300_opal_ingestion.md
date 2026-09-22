# Z300 acquisition data on Opal

Pantheum's `opal_database` retrieval mode connects a completed instrument run
to its data. The acquisition stays `awaiting_data` until its exact acknowledged
test has been retrieved, converted on Opal, and validated by Pantheum. Analysis
jobs are created in the same database transaction that marks acquisition
`succeeded`.

The existing Moissanite worker invokes a transient process over its authorized
SSH connection to Opal. No new Opal listener or scheduled task is required.
Opal uses USB ADB shell reads for the exact current Couchbase Lite test revision
and binary `adb pull` for its spectrum bundle. Both `shotTable.all_fb`
FlatBuffers and `shotTable.all` ZIPs of compressed JSON spectra are supported.
The producer checks two full bundle reads and
rechecks the document revision. Missing or incomplete data stays pending;
retrying retrieval never repeats a laser or motion command.

`scripts/z300_opal_ingest.py` keeps the raw test document, database revision,
source bundle, native per-shot CSVs, an average, and a hash manifest in an immutable
per-run directory on Opal. Pantheum receives that run's ZIP over SSH and checks
identity, shot count, hashes, file inventory, and wavelength ordering before
import. Existing deferred runs are adopted only through the private
`acquire.opal_retrieval.adopt_run_ids` allowlist. Uncertain physical outcomes
cannot be adopted.

## Native wavelength samples

The converter evaluates each shot's stored wavelength polynomials at its
detector samples and trims the supplied detector seams. It does not create a
uniform export grid. Averaging requires exactly matching native wavelength
axes. Raw evidence is retained, including the known dummy fourth channel;
that channel is excluded from scientific CSVs only when its known placeholder
calibration is recognized. An unknown fourth calibration is rejected.

The FlatBuffer decoder's historical `-18` pixel offset was inferred from matched
vendor exports; JSON spectra use their existing zero-based calibration contract.
Format-specific calibration provenance travels in the manifest; native sampling
does not establish a new absolute-wavelength calibration. An incomplete shot
set or unsupported averaging cannot be labeled a complete acquisition.

## Installation and checks

The private deployment configuration supplies the SSH command as an argument
list with `{run_id}`, `{test_id}`, and `{expected_shots}` placeholders. It also
sets the timeout, retry interval, and optional explicit adoption list. No
credential is placed in this repository or on the converter's command line.

`scripts/stage_z300_opal.py` previews staging; `--apply` copies and verifies the
producer, decoder, and the already-installed ADB binaries at a durable path.
`scripts/deploy_z300_ingest.py --bundle PATH` validates a pinned Pantheum bundle;
`--apply` installs it under the reservation lock, with source/configuration/DB
backups and rollback of source/configuration if an installation write fails.
Deployment refuses active acquisition, queued analysis, or unresolved hardware.

The converter itself accepts `--dry-run`, which validates arguments and prints
a plan without reading the device or creating an archive. `--output PATH`
keeps its result ZIP on Opal for local validation. Normal worker invocation
writes the ZIP to stdout. The device must remain connected to Opal with the
authorized ADB connection available; otherwise the queue waits and retries.

Run focused checks with:

```sh
.venv/bin/python -m pytest tests/test_z300_opal_ingest.py tests/test_z300_sync_tests.py tests/test_deploy_z300_ingest.py
```

Pantheum owns queue/state-machine tests in `tests/test_alibz_retrieval.py` and
requires its backend suite before deployment. The dated deployment report in
`reports/` records installed hashes and live validation separately from mocked
tests.
