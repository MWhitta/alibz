#!/usr/bin/env python3
"""Deploy the three manifest-pinned alibz static assets; default is dry run.

Run on Moissanite. This never restarts services, changes configuration, or
calls hardware APIs. It refuses to overwrite concurrent frontend edits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    base = Path.home()
    root = base / 'pantheum-I'
    config = base / '.config/pantheum/alibz.json'
    config_before = digest(config)
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    names = ['web/alibz/styles.css', 'web/alibz/app.js', 'web/alibz/index.html']
    if set(manifest['after']) != set(names) or set(manifest['before']) != set(names):
        raise RuntimeError('Unexpected deployment paths')
    for name in names:
        if digest(args.bundle / name) != manifest['after'][name]:
            raise RuntimeError('Bundle changed: ' + name)
        if digest(root / name) != manifest['before'][name]:
            raise RuntimeError('Concurrent live edit: ' + name)
    print(json.dumps({'apply': args.apply, 'files': names, 'configuration': 'unchanged'}), flush=True)
    if not args.apply:
        return
    backup = base / ('pantheum-panel-errors-backup-' + time.strftime('%Y%m%dT%H%M%S'))
    backup.mkdir()
    for name in names:
        target = backup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    installed = []
    try:
        for name in names:
            target = root / name
            if digest(target) != manifest['before'][name]:
                raise RuntimeError('Live file changed during deployment: ' + name)
            temporary = target.with_name(target.name + '.panel-errors-tmp')
            shutil.copy2(args.bundle / name, temporary)
            os.replace(temporary, target)
            installed.append(name)
            if digest(target) != manifest['after'][name]:
                raise RuntimeError('Installed hash mismatch: ' + name)
        if digest(config) != config_before:
            raise RuntimeError('Configuration changed during deployment')
    except BaseException:
        for name in installed:
            if digest(root / name) == manifest['after'][name]:
                shutil.copy2(backup / name, root / name)
        raise
    print(json.dumps({'backup': str(backup), 'installed_hashes': manifest['after'],
                      'configuration_sha256': config_before, 'services_restarted': False}))


if __name__ == '__main__':
    main()
