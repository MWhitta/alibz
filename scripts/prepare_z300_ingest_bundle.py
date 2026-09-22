#!/usr/bin/env python3
"""Build the five-file Pantheum update with explicit runtime/config hash pins."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

FILES = ['pantheum/alibz/' + name for name in (
    'acquire.py', 'retrieval.py', 'optimization.py', '__main__.py', 'optimization_metrics.py')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True, help='JSON mapping of installed paths to SHA256')
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adopt-run-id', action='append', default=[])
    parser.add_argument('--apply', action='store_true', help='write bundle; default only prints planned hashes')
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    after = {name: hashlib.sha256((args.source / name).read_bytes()).hexdigest() for name in FILES}
    command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
               '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
               '-p', '52222', 'whittaker@192.168.50.112',
               r'C:\InstrumentControl\alibz\.venv\Scripts\python.exe', '-B',
               r'C:\InstrumentControl\alibz\z300-ingest-20260922\z300_opal_ingest.py',
               '--run-id', '{run_id}', '--test-id', '{test_id}', '--expected-shots', '{expected_shots}',
               '--adb', r'C:\InstrumentControl\alibz\z300-ingest-20260922\platform-tools\adb.exe',
               '--serial', '0123456789ABCDEF', '--adb-port', '5038',
               '--archive-root', r'C:\LabData\LIBS\pantheum-acquisitions']
    manifest = {'files': after, 'before': {name: baseline.get(name) for name in FILES},
                'config_sha256': args.config_sha256,
                'acquire_patch': {'retrieval': 'opal_database', 'opal_retrieval': {
                    'command': command, 'timeout_seconds': 240, 'retry_seconds': 15,
                    'adopt_run_ids': args.adopt_run_id}}}
    print(json.dumps({'apply': args.apply, 'output': str(args.output), 'files': after}))
    if not args.apply:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source / name, target)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copy2(Path(__file__).with_name('deploy_z300_ingest.py'), args.output / 'deploy.py')


if __name__ == '__main__':
    main()
