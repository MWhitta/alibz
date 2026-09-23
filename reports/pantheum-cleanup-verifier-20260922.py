#!/usr/bin/env python3
"""Verify preserved Pantheum backups; remove the six originals only with --apply."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

ROOT = Path('/Users/mwhittaker/Projects/github')
REPO = ROOT / 'pantheum-I'
ARCHIVE = Path('archive/source-backups/2026-09-21')
COMMIT = '3284cbc8791041fe60975d50e98b09322c20f22d'
NAMES = {
    'pantheum-fire-fix-source-backup-20260921T153917',
    'pantheum-I-acquisition-backup-20260921T130028',
    'pantheum-I-acquisition-backup-20260921T141426',
    'pantheum-I-acquisition-backup-20260921T142747',
    'pantheum-panel-errors-source-backup-20260921T165416',
    'pantheum-query-fix-source-backup-20260921T162743',
}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO)


def verify_tree(source, snapshot):
    if not source.is_dir() or source.is_symlink():
        raise RuntimeError('Missing or redirected source: ' + str(source))
    expected = {f['path']: f for f in snapshot['files']}
    files, directories = set(), set()
    for directory, dirs, entries in os.walk(source, followlinks=False):
        for name in dirs + entries:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise RuntimeError('Unsupported entry: ' + str(path))
            if stat.S_ISDIR(mode):
                directories.add(str(path.relative_to(source)))
        for name in entries:
            path = Path(directory) / name
            rel = str(path.relative_to(source))
            if rel not in expected:
                raise RuntimeError('Unexpected file: ' + str(path))
            raw = path.read_bytes()
            item = expected[rel]
            if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise RuntimeError('Changed file: ' + str(path))
            files.add(rel)
    if files != set(expected) or directories != set(snapshot['directories']):
        raise RuntimeError('Directory inventory changed: ' + str(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    manifest_bytes = git('show', COMMIT + ':' + str(ARCHIVE / 'manifest.json'))
    manifest = json.loads(manifest_bytes)
    assert (REPO / ARCHIVE / 'manifest.json').read_bytes() == manifest_bytes
    assert {s['directory'] for s in manifest['snapshots']} == NAMES
    assert manifest['source_parent'] == str(ROOT)
    git('merge-base', '--is-ancestor', COMMIT, 'HEAD')
    git('merge-base', '--is-ancestor', COMMIT, 'origin/main')
    for snapshot in manifest['snapshots']:
        verify_tree(ROOT / snapshot['directory'], snapshot)
        verify_tree(REPO / ARCHIVE / snapshot['directory'], snapshot)
        for item in snapshot['files']:
            path = ARCHIVE / snapshot['directory'] / item['path']
            raw = git('show', COMMIT + ':' + str(path))
            assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], str(path)
    result = {'apply': args.apply, 'archive_commit': COMMIT,
              'verified_folders': len(NAMES),
              'verified_files': sum(len(s['files']) for s in manifest['snapshots'])}
    if args.apply:
        for snapshot in manifest['snapshots']:
            source = ROOT / snapshot['directory']
            verify_tree(source, snapshot)
            shutil.rmtree(source)
        assert not any((ROOT / name).exists() for name in NAMES)
        result['removed_folders'] = sorted(NAMES)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
