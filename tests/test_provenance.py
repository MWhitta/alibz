"""Provenance core: manifest identity, dirty capture, strict refusal."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from alibz.provenance import (
    DirtyWorktreeError,
    MANIFEST_NAME,
    config_hash,
    finalize_manifest,
    run_manifest,
)


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True,
                   capture_output=True)


def _make_repo(root: Path) -> Path:
    repo = root / "repo"
    (repo / "alibz").mkdir(parents=True)
    (repo / "alibz" / "mod.py").write_text("x = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-q", "-m", "init")
    return repo


class TestRunManifest(unittest.TestCase):
    def test_clean_repo_manifest_and_finalize(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            repo = _make_repo(root)
            out = root / "out"
            out.mkdir()
            spec = out / "s.csv"
            spec.write_text("wavelength,intensity\n500,1\n")
            m = run_manifest(out, {"n_calls": 4}, [str(spec)],
                             repo=repo, progress=lambda *_: None)
            self.assertEqual(m["status"], "running")
            self.assertFalse(m["software"]["git_dirty"])
            self.assertNotIn("dirty_capture", m)
            self.assertEqual(len(m["inputs"]), 1)
            self.assertIn("sha256", m["inputs"][0])
            on_disk = json.loads((out / MANIFEST_NAME).read_text())
            self.assertEqual(on_disk["config_hash"], m["config_hash"])

            rows = [{"status": "ok"},
                    {"status": "error: x", "failure_reason": "timeout"}]
            done = finalize_manifest(out, m, rows)
            self.assertEqual(done["status"], "complete")
            self.assertEqual(done["n_ok"], 1)
            self.assertEqual(done["failures_by_reason"], {"timeout": 1})

    def test_config_hash_input_set_independent_but_config_sensitive(self):
        base = {"schema_version": 1, "config": {"n_calls": 40},
                "software": {"git_commit": "abc", "git_diff_sha256": "d",
                             "source_tree_sha256": "s"},
                "database": {"tree_sha256": "t"},
                "inputs": [{"path": "a.csv"}]}
        other_inputs = dict(base, inputs=[{"path": "b.csv"}])
        other_config = dict(base, config={"n_calls": 80})
        self.assertEqual(config_hash(base), config_hash(other_inputs))
        self.assertNotEqual(config_hash(base), config_hash(other_config))

    def test_dirty_repo_captured_and_strict_refuses(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            repo = _make_repo(root)
            # modify a tracked file AND add an untracked source file
            (repo / "alibz" / "mod.py").write_text("x = 2\n")
            (repo / "alibz" / "new.py").write_text("y = 3\n")
            out = root / "out"
            out.mkdir()
            spec = out / "s.csv"
            spec.write_text("wavelength,intensity\n500,1\n")

            with self.assertRaises(DirtyWorktreeError):
                run_manifest(out, {}, [str(spec)], repo=repo, strict=True,
                             progress=lambda *_: None)

            m = run_manifest(out, {}, [str(spec)], repo=repo,
                             progress=lambda *_: None)
            cap = m["dirty_capture"]
            patch = (out / "provenance" / cap["patch"]).read_bytes()
            self.assertIn(b"x = 2", patch)          # tracked change captured
            captured = {e["path"] for e in cap["untracked_captured"]}
            self.assertIn("alibz/new.py", captured)  # untracked source copied
            self.assertTrue(
                (out / "provenance" / "untracked" / "alibz" / "new.py")
                .exists())


if __name__ == "__main__":
    unittest.main()
