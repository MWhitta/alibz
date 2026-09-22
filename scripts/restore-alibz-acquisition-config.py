#!/usr/bin/env python3
"""Restore the pre-recovery acquisition config after verifying the tested runtime.

Run on Moissanite. Default is a read-only dry run; --apply restores only the
previously backed-up acquire.enabled flag. No physical commands are issued.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess


ORIGINAL_SHA = '37e26ff18ff5b9b82ea949ea2f8c6678dbf9dd84de807fb487471676f9587728'
PAUSED_SHA = '2d10dea71fdb10b960968df48279031d92b54c975be3ee2ee83da2e93b4c3ae4'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    base = Path.home()
    config = base / '.config/pantheum/alibz.json'
    backup = config.with_name(config.name + '.api-recovery-20260921')
    state = base / '.local/state/pantheum/alibz'
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    with (state / 'reservation.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if digest(config) != PAUSED_SHA or digest(backup) != ORIGINAL_SHA:
            raise RuntimeError('Configuration changed; preserve it for review')
        paused = json.loads(config.read_text())
        original = json.loads(backup.read_text())
        if paused['acquire']['enabled'] is not False or original['acquire']['enabled'] is not True:
            raise RuntimeError('Unexpected enablement state')
        paused['acquire']['enabled'] = True
        if paused != original:
            raise RuntimeError('Configuration differs beyond acquire.enabled')
        for name, expected in manifest['files'].items():
            rel = Path(name)
            if rel.is_absolute() or '..' in rel.parts:
                raise RuntimeError('Unsafe runtime path')
            if digest(base / 'pantheum-I' / rel) != expected:
                raise RuntimeError('Tested runtime is not installed: ' + name)
        with sqlite3.connect(str(state / 'alibz.sqlite')) as db:
            queries = [
                "SELECT COUNT(*) FROM acquisitions WHERE state IN ('queued','running','cancelling')",
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')",
                "SELECT COUNT(*) FROM hardware_operations WHERE state IN ('pending','uncertain')",
            ]
            if any(db.execute(query).fetchone()[0] for query in queries):
                raise RuntimeError('Active work or unresolved hardware')
        print(json.dumps({'apply': args.apply, 'verified_runtime': True,
                          'only_change': 'acquire.enabled: false -> true'}), flush=True)
        if not args.apply:
            return
        services = ['pantheum-alibz-worker.service', 'pantheum-alibz.service']
        subprocess.run(['systemctl', '--user', 'stop', *services], check=True)
        try:
            temp = config.with_name(config.name + '.recovery-tmp')
            with temp.open('wb') as stream:
                os.chmod(temp, config.stat().st_mode & 0o777)
                stream.write(backup.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, config)
            if digest(config) != ORIGINAL_SHA:
                raise RuntimeError('Restored configuration hash mismatch')
        finally:
            subprocess.run(['systemctl', '--user', 'start', *reversed(services)], check=True)
        print(json.dumps({'restored_sha256': digest(config)}))


if __name__ == '__main__':
    main()
