# Pantheum directory audit — 2026-09-22

Scope: metadata-only inventory of the six explicitly named directories under `/Users/mwhittaker/Projects/github`. No file contents were read, no symlinks were followed, and no source or Git state was changed.

Method: Python 3 `os.scandir` traversal and `DirEntry.stat(follow_symlinks=False)`. File counts and total byte sizes include regular files only; symlinks are counted separately. Byte sizes are logical file lengths (`st_size`), not allocated disk space. `.git` matches include entries of any type at any depth.

Command run: `python3 -` with a metadata inventory script specifying the six directory names, walking each directory without following symlinks, counting regular files and their `st_size`, collecting `.git` paths and top-level names, and writing this report.

## pantheum-fire-fix-source-backup-20260921T153917

- Regular files: 6
- Total regular-file bytes: 170280
- Symlinks (not followed): 0
- `.git` exists anywhere: no
- Top-level entry names: `DECISIONS.md`, `docs`, `pantheum`, `tests`

## pantheum-I-acquisition-backup-20260921T130028

- Regular files: 15
- Total regular-file bytes: 460506
- Symlinks (not followed): 0
- `.git` exists anywhere: no
- Top-level entry names: `DECISIONS.md`, `README.md`, `config`, `deploy`, `docs`, `pantheum`, `reports`, `tests`, `web`

## pantheum-I-acquisition-backup-20260921T141426

- Regular files: 16
- Total regular-file bytes: 549900
- Symlinks (not followed): 0
- `.git` exists anywhere: no
- Top-level entry names: `DECISIONS.md`, `README.md`, `deploy`, `docs`, `pantheum`, `tests`, `web`

## pantheum-I-acquisition-backup-20260921T142747

- Regular files: 6
- Total regular-file bytes: 324639
- Symlinks (not followed): 0
- `.git` exists anywhere: no
- Top-level entry names: `DECISIONS.md`, `docs`, `pantheum`, `tests`, `web`

## pantheum-panel-errors-source-backup-20260921T165416

- Regular files: 4
- Total regular-file bytes: 298384
- Symlinks (not followed): 0
- `.git` exists anywhere: no
- Top-level entry names: `tests`, `web`

## pantheum-query-fix-source-backup-20260921T162743

- Regular files: 5
- Total regular-file bytes: 163910
- Symlinks (not followed): 0
- `.git` exists anywhere: no
- Top-level entry names: `DECISIONS.md`, `docs`, `pantheum`, `tests`

## Completion

- Directories inventoried: 6 of 6
- Failures: none
- Hashes: not calculated because this audit forbids reading file contents.
