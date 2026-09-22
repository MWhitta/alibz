#!/usr/bin/env python3
"""Validate one existing run on Opal, returning only a receipt; dry run default."""
import argparse
import re
import subprocess

try:
    from .stage_z300_opal import powershell
except ImportError:
    from stage_z300_opal import powershell


def command(run_id, test_id, expected_shots, apply=False):
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', s) for s in (run_id, test_id)):
        raise ValueError('Use exact plain run/test IDs')
    if type(expected_shots) is not int or not 1 <= expected_shots <= 1000:
        raise ValueError('Expected shot count must be 1..1000')
    root = r'C:\LabData\LIBS\pantheum-acquisitions'
    target = root + '\\' + run_id
    script = rf"""$ErrorActionPreference='Continue'
& 'C:\InstrumentControl\alibz\.venv\Scripts\python.exe' -B 'C:\InstrumentControl\alibz\z300-ingest-20260922\z300_opal_ingest.py' --run-id {run_id} --test-id {test_id} --expected-shots {expected_shots} --adb 'C:\InstrumentControl\alibz\z300-ingest-20260922\platform-tools\adb.exe' --archive-root '{root}' --output '{target}\native.zip' {'--dry-run' if not apply else ''} 2>&1 | ForEach-Object {{ $_.ToString() }}
if ($LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }}
"""
    if apply:
        script += rf"""
$m=Get-Content -Raw '{target}\manifest.json' | ConvertFrom-Json
@{{verified=$true; run_id=$m.run_id; test_id=$m.test_id; schema=$m.schema; grid=$m.grid; shots=$m.shots; native_points=$m.provenance.native_points_per_shot; zip_sha256=(Get-FileHash '{target}\native.zip').Hash.ToLower(); archive='{target}'}} | ConvertTo-Json -Compress
"""
    return script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--test-id', required=True)
    parser.add_argument('--expected-shots', required=True, type=int)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        print(powershell(command(args.run_id, args.test_id, args.expected_shots, args.apply)).strip())
    except subprocess.CalledProcessError as exc:
        # The remote command returns only the producer's fixed diagnostics or
        # receipts; the spectra and database document remain on Opal.
        print((exc.stdout or '')[:3000].strip())
        raise SystemExit(exc.returncode)


if __name__ == '__main__':
    main()
