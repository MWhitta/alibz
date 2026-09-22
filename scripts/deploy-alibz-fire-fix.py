#!/usr/bin/env python3
"""Deploy only manifest-pinned files while holding alibz's reservation lock.

Default is a dry run. No physical commands or private configuration edits.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    home = Path.home()
    root = home / 'pantheum-I'
    state = home / '.local/state/pantheum/alibz'
    config = home / '.config/pantheum/alibz.json'
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    services = ['pantheum-alibz-worker.service', 'pantheum-alibz.service']
    with (state / 'reservation.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, after in manifest['files'].items():
            rel = Path(name)
            if rel.is_absolute() or '..' in rel.parts:
                raise RuntimeError('Unsafe manifest path')
            if digest(args.bundle / rel) != after:
                raise RuntimeError('Bundle changed: ' + name)
            before = manifest['before'][name]
            live = root / rel
            if before is None:
                if live.exists():
                    raise RuntimeError('Live file already exists: ' + name)
            elif digest(live) != before:
                raise RuntimeError('Live source changed: ' + name)
        if digest(config) != manifest['config_sha256']:
            raise RuntimeError('Private configuration changed')
        database = sqlite3.connect(str(state / 'alibz.sqlite'))
        checks = [
            "SELECT COUNT(*) FROM acquisitions WHERE state IN ('queued','running','cancelling')",
            "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')",
            "SELECT COUNT(*) FROM hardware_operations WHERE state IN ('pending','uncertain')",
        ]
        if any(database.execute(sql).fetchone()[0] for sql in checks):
            raise RuntimeError('Active work or unresolved hardware; refusing deployment')
        print(json.dumps({'apply': args.apply, 'files': list(manifest['files']),
                          'active_work': 0, 'configuration': 'unchanged'}), flush=True)
        if not args.apply:
            database.close()
            return
        backup = home / ('pantheum-fire-fix-backup-' + time.strftime('%Y%m%dT%H%M%S'))
        backup.mkdir()
        shutil.copy2(config, backup / 'alibz.json')
        with sqlite3.connect(str(backup / 'alibz.sqlite')) as target:
            database.backup(target)
        database.close()
        for name in manifest['files']:
            live = root / name
            if not live.exists():
                continue
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(live, target)
        subprocess.run(['systemctl', '--user', 'stop', *services], check=True)
        try:
            for name, expected in manifest['files'].items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.fire-fix-tmp')
                shutil.copy2(args.bundle / name, temporary)
                os.replace(temporary, target)
                if digest(target) != expected:
                    raise RuntimeError('Installed hash mismatch')
            if digest(config) != manifest['config_sha256']:
                raise RuntimeError('Private configuration changed during deployment')
        except BaseException:
            for name, before in manifest['before'].items():
                target = root / name
                if before is None:
                    if target.exists():
                        target.unlink()
                elif (backup / name).exists():
                    shutil.copy2(backup / name, target)
            raise
        finally:
            subprocess.run(['systemctl', '--user', 'start', *reversed(services)], check=True)
        print(json.dumps({'backup': str(backup), 'installed_hashes': manifest['files']}))


if __name__ == '__main__':
    main()
