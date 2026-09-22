#!/usr/bin/env python3
"""Stage the transient Opal converter and existing ADB binaries; dry run first."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

REMOTE = 'C:/InstrumentControl/alibz/z300-ingest-20260922'
ADB_SOURCE = 'C:/Users/Whittaker/AppData/Local/Temp/alibz-api-recovery-20260921/platform-tools'
PYTHON = 'C:/InstrumentControl/alibz/.venv/Scripts/python.exe'


def powershell(script):
    encoded = base64.b64encode(script.encode('utf-16le')).decode()
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                           'opal', 'powershell.exe', '-NoProfile', '-NonInteractive',
                           '-EncodedCommand', encoded], check=True, capture_output=True, text=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    files = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
             for name in ('z300_opal_ingest.py', 'z300_fb_decode.py')}
    print(json.dumps({'apply': args.apply, 'target': REMOTE, 'files': files,
                      'adb_source': ADB_SOURCE, 'new_services': 0}), flush=True)
    if not args.apply:
        return
    powershell(f"$ErrorActionPreference='Stop'; New-Item -ItemType Directory -Force '{REMOTE}' | Out-Null")
    for name in files:
        subprocess.run(['scp', '-q', str(source / name), f'opal:{REMOTE}/{name}'], check=True)
    checks = '\n'.join(
        f"if ((Get-FileHash -Algorithm SHA256 '{REMOTE}/{name}').Hash.ToLower() -ne '{digest}') {{ throw 'producer hash mismatch' }}"
        for name, digest in files.items())
    script = f"""$ErrorActionPreference='Stop'
{checks}
New-Item -ItemType Directory -Force '{REMOTE}/platform-tools' | Out-Null
foreach ($name in @('adb.exe','AdbWinApi.dll','AdbWinUsbApi.dll')) {{
  $source = Join-Path '{ADB_SOURCE}' $name
  $target = Join-Path '{REMOTE}/platform-tools' $name
  if (Test-Path $target) {{
    if ((Get-FileHash $source).Hash -ne (Get-FileHash $target).Hash) {{ throw 'existing ADB differs' }}
  }} else {{ Copy-Item -LiteralPath $source -Destination $target }}
  if ((Get-FileHash $source).Hash -ne (Get-FileHash $target).Hash) {{ throw 'ADB copy mismatch' }}
}}
& '{PYTHON}' -B '{REMOTE}/z300_opal_ingest.py' --help | Out-Null
if ($LASTEXITCODE -ne 0) {{ throw 'producer import check failed' }}
@{{verified=$true; target='{REMOTE}'; adb_sha256=(Get-FileHash '{REMOTE}/platform-tools/adb.exe').Hash.ToLower()}} | ConvertTo-Json -Compress
"""
    print(powershell(script).strip())


if __name__ == '__main__':
    main()
