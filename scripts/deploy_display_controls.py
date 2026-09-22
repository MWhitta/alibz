#!/usr/bin/env python3
"""Install the reviewed Pantheum display controls; dry run by default.

The manifest pins existing source and private configuration. Only its listed
files and top-level display configuration change. No instrument command runs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def inspect_bundle(bundle, root, config):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for name, after in manifest['files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or not rel.parts:
            raise ValueError('Unsafe manifest path')
        if digest(bundle / rel) != after:
            raise ValueError('Bundle changed: ' + name)
        if digest(root / rel) != manifest['before'][name]:
            raise ValueError('Installed source changed: ' + name)
    if digest(config) != manifest['config_sha256']:
        raise ValueError('Private configuration changed')
    patch = manifest['display_patch']
    if set(patch) != {'enabled', 'command', 'timeout_seconds', 'status_ttl_seconds'} or patch['enabled'] is not True:
        raise ValueError('Unexpected configuration patch')
    updated = json.loads(config.read_text())
    updated['display'] = patch
    return manifest, updated


def check_idle(database):
    checks = {
        'acquisitions': "SELECT COUNT(*) FROM acquisitions WHERE state IN ('queued','running','cancelling') OR (state='uncertain' AND hardware_resolved=0)",
        'hardware holds': "SELECT COUNT(*) FROM hardware_operations WHERE state IN ('pending','uncertain')",
    }
    for name, query in checks.items():
        if database.execute(query).fetchone()[0]:
            raise RuntimeError('Active ' + name + '; retry deployment after work finishes')


def atomic_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.display-tmp')
    shutil.copy2(source, tmp)
    os.replace(tmp, target)


def install(bundle, root, config, state, apply=False, run=subprocess.run):
    import fcntl  # deployment host is Linux; unit tests also support macOS
    services = ['pantheum-alibz.service']
    with (state / 'reservation.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, updated = inspect_bundle(bundle, root, config)
        with sqlite3.connect(state / 'alibz.sqlite') as database:
            check_idle(database)
        result = {'apply': apply, 'files': list(manifest['files']),
                  'display_controls': True, 'active_work': 0}
        if not apply:
            return result
        backup = root.parent / ('pantheum-display-backup-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        backup.mkdir(mode=0o700)
        # Stop under the physical-operation lock, then recheck the queue: no
        # acquisition can start while it is held; the unrelated analysis worker stays running.
        run(['systemctl', '--user', 'stop', *services], check=True)
        installed = False
        try:
            manifest, updated = inspect_bundle(bundle, root, config)
            with sqlite3.connect(state / 'alibz.sqlite') as database:
                check_idle(database)
                with sqlite3.connect(backup / 'alibz.sqlite') as target:
                    database.backup(target)
            shutil.copy2(config, backup / 'alibz.json')
            for name in manifest['files']:
                target = root / name
                if target.exists():
                    atomic_copy(target, backup / name)
            # Backups complete before the first source/config mutation.
            installed = True
            for name, expected in manifest['files'].items():
                atomic_copy(bundle / name, root / name)
                if digest(root / name) != expected:
                    raise RuntimeError('Installed hash mismatch: ' + name)
            tmp = config.with_name(config.name + '.display-tmp')
            with tmp.open('w', encoding='utf-8') as handle:
                os.chmod(tmp, 0o600)
                json.dump(updated, handle, indent=2)
                handle.write('\n')
            os.replace(tmp, config)
        except BaseException:
            if installed:
                for name, before in manifest['before'].items():
                    if before is None:
                        (root / name).unlink(missing_ok=True)
                    else:
                        atomic_copy(backup / name, root / name)
                atomic_copy(backup / 'alibz.json', config)
            raise
        finally:
            run(['systemctl', '--user', 'start', *reversed(services)], check=True)
        result.update(backup=str(backup), installed_hashes=manifest['files'],
                      config_sha256=digest(config))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path.home() / 'pantheum-I')
    parser.add_argument('--config', type=Path, default=Path.home() / '.config/pantheum/alibz.json')
    parser.add_argument('--state', type=Path, default=Path.home() / '.local/state/pantheum/alibz')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(install(args.bundle, args.root, args.config, args.state, args.apply)))


if __name__ == '__main__':
    main()
