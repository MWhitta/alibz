"""Run provenance: what code, database, config, and inputs produced a result.

Promoted from the MW2-112 workflow (:mod:`alibz.mw2_112` re-exports these
for compatibility) into core so EVERY ``analyze_directory`` run records a
``run_manifest.json`` — the MW2-112 production run happened on a dirty
worktree whose untracked modules were only hash-recorded, not captured,
and the result bundle is consequently not byte-reconstructable (see
docs/mw2_112/provenance.md).  The policy here makes that non-recurring:

- **record by default**: a dirty worktree is captured as
  ``provenance/dirty.patch`` (``git diff --binary HEAD``) plus copies of
  untracked *source* files, with a loud warning;
- **strict mode** (``--strict-provenance``): refuse to run dirty.

The manifest's ``config_hash`` identifies (code, database, configuration)
— not the input set — so two runs over different directories with the
same hash are the same analysis.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

#: manifest schema; bump when the manifest structure changes.
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_NAME = "run_manifest.json"

_PACKAGE_NAMES = (
    "alibz", "numpy", "scipy", "scikit-learn", "scikit-optimize",
    "matplotlib", "pulp", "periodictable",
)

#: repo-relative directories whose UNTRACKED files are copied (not just
#: hash-listed) into the provenance capture — source code is small and
#: is exactly what the MW2-112 lesson showed must be preserved; data
#: directories are listed by name+hash only.
_CAPTURE_UNTRACKED_DIRS = ("alibz", "scripts", "tests", "bench")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path, chunk_size: int = 2 ** 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(paths: Iterable[Path]) -> str:
    """Historical path-dependent hash retained for manifest compatibility."""
    digest = hashlib.sha256()
    for path in sorted((Path(item).resolve() for item in paths), key=str):
        if not path.is_file():
            continue
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(hash_file(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as fh:
        json.dump(jsonable(value), fh, indent=2, sort_keys=True,
                  allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _git_value(repo: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True)
        return result.stdout.decode(errors="replace").strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def software_state(repo: Path) -> dict:
    repo = Path(repo)
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=repo, check=False,
        capture_output=True).stdout
    untracked = _git_value(repo, "ls-files", "--others", "--exclude-standard")
    source_paths = list((repo / "alibz").rglob("*.py"))
    source_paths += list((repo / "scripts").rglob("*.py"))
    versions = {}
    for package in _PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "git_commit": _git_value(repo, "rev-parse", "HEAD"),
        "git_branch": _git_value(repo, "branch", "--show-current"),
        "git_dirty": bool(diff or untracked),
        "git_diff_sha256": sha256_bytes(diff),
        "untracked_files": untracked.splitlines() if untracked else [],
        "source_tree_sha256": _tree_hash(source_paths),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": versions,
    }


def database_state(dbpath: Path) -> dict:
    dbpath = Path(dbpath)
    files = [path for path in dbpath.rglob("*") if path.is_file()]
    return {
        "path": str(dbpath.resolve()),
        "n_files": len(files),
        "tree_sha256": _tree_hash(files),
    }


def repo_root() -> Path:
    """The source checkout containing this alibz package."""
    return Path(__file__).resolve().parents[1]


class DirtyWorktreeError(RuntimeError):
    """Raised in strict mode when the worktree has uncaptured changes."""


def capture_dirty_worktree(repo: Path, out_dir: Path,
                           software: dict) -> dict:
    """Capture uncommitted state so the run is byte-reconstructable.

    Writes ``dirty.patch`` (``git diff --binary HEAD``) and copies
    untracked files under the source directories into
    ``out_dir/untracked/``; other untracked files are listed by
    name+hash only (data directories can be enormous).  Returns a
    capture record for the manifest.
    """
    repo = Path(repo)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=repo, check=False,
        capture_output=True).stdout
    patch_path = out_dir / "dirty.patch"
    patch_path.write_bytes(diff)
    captured, listed = [], []
    for rel in software.get("untracked_files", []):
        src = repo / rel
        if not src.is_file():
            continue
        entry = {"path": rel, "sha256": hash_file(src),
                 "size_bytes": src.stat().st_size}
        if rel.split("/", 1)[0] in _CAPTURE_UNTRACKED_DIRS:
            dst = out_dir / "untracked" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            captured.append(entry)
        else:
            listed.append(entry)
    return {
        "patch": patch_path.name,
        "patch_sha256": sha256_bytes(diff),
        "untracked_captured": captured,
        "untracked_listed_only": listed,
    }


def config_hash(manifest: dict) -> str:
    """Identity of (code, database, configuration) — input-set independent."""
    software = manifest.get("software") or {}
    database = manifest.get("database") or {}
    payload = json.dumps(jsonable({
        "schema_version": manifest.get("schema_version"),
        "config": manifest.get("config"),
        "git_commit": software.get("git_commit"),
        "git_diff_sha256": software.get("git_diff_sha256"),
        "source_tree_sha256": software.get("source_tree_sha256"),
        "database_tree_sha256": database.get("tree_sha256"),
    }), sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def run_manifest(
    output_dir: Path,
    config: dict,
    files: Sequence[str],
    *,
    dbpath: Optional[str] = None,
    repo: Optional[Path] = None,
    hash_inputs: bool = True,
    strict: bool = False,
    progress=print,
) -> dict:
    """Write ``run_manifest.json`` into ``output_dir`` and return it.

    ``config`` is the exact configuration snapshot (e.g.
    ``dataclasses.asdict(AnalysisConfig)``), so what ran and what is
    recorded cannot drift apart.  In strict mode a dirty worktree raises
    :class:`DirtyWorktreeError` BEFORE any analysis runs; otherwise the
    dirty state is captured under ``output_dir/provenance/``.
    """
    output_dir = Path(output_dir)
    repo = Path(repo) if repo is not None else repo_root()
    software = software_state(repo)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": utc_now(),
        "status": "running",
        "config": dict(config),
        "software": software,
        "database": database_state(Path(dbpath)) if dbpath else None,
        "inputs": [
            {"path": str(f), "size_bytes": os.path.getsize(f),
             **({"sha256": hash_file(Path(f))} if hash_inputs else {})}
            for f in files
        ],
    }
    if software["git_dirty"]:
        if strict:
            raise DirtyWorktreeError(
                "worktree has uncommitted or untracked changes; commit "
                "them or run without --strict-provenance "
                f"(untracked: {software['untracked_files'][:5]}...)")
        capture = capture_dirty_worktree(
            repo, output_dir / "provenance", software)
        manifest["dirty_capture"] = capture
        progress("WARNING: dirty worktree — uncommitted state captured "
                 f"under {output_dir / 'provenance'} (commit for clean "
                 "provenance, or use strict mode to refuse)")
    manifest["config_hash"] = config_hash(manifest)
    atomic_json(output_dir / MANIFEST_NAME, manifest)
    return manifest


def finalize_manifest(output_dir: Path, manifest: dict, rows) -> dict:
    """Mark the manifest complete with row/failure tallies and rewrite it."""
    by_reason: dict = {}
    n_ok = 0
    for row in rows:
        if row.get("status") == "ok":
            n_ok += 1
        else:
            reason = row.get("failure_reason") or "error"
            by_reason[reason] = by_reason.get(reason, 0) + 1
    manifest = dict(manifest)
    manifest["completed_utc"] = utc_now()
    manifest["status"] = "complete"
    manifest["n_rows"] = len(rows)
    manifest["n_ok"] = n_ok
    manifest["failures_by_reason"] = by_reason
    atomic_json(Path(output_dir) / MANIFEST_NAME, manifest)
    return manifest
