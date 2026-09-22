import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock

from scripts import deploy_display_controls as deploy


class DeployDisplayTests(unittest.TestCase):
    def test_dry_run_apply_preserves_other_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, bundle, state = [base / part for part in ("root", "bundle", "state")]
            for path in (root, bundle, state):
                path.mkdir()
            config = base / "config.json"
            original = {"acquire": {"retrieval": "opal_database"}, "camera": {"enabled": True}}
            config.write_text(json.dumps(original))
            name = "pantheum/alibz/display.py"
            (bundle / name).parent.mkdir(parents=True)
            (bundle / name).write_text("# tested module\n")
            patch = {"enabled": True, "command": ["test", "{action}"],
                     "timeout_seconds": 45, "status_ttl_seconds": 10}
            manifest = {"files": {name: deploy.digest(bundle / name)}, "before": {name: None},
                        "config_sha256": deploy.digest(config), "display_patch": patch}
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            with sqlite3.connect(state / "alibz.sqlite") as db:
                db.executescript("CREATE TABLE acquisitions(state TEXT,hardware_resolved INTEGER);"
                                 "CREATE TABLE hardware_operations(state TEXT);")
            run = Mock()
            deploy.install(bundle, root, config, state, run=run)
            run.assert_not_called()
            self.assertEqual(json.loads(config.read_text()), original)
            self.assertFalse((root / name).exists())
            deploy.install(bundle, root, config, state, apply=True, run=run)
            self.assertEqual(json.loads(config.read_text()), {**original, "display": patch})
            self.assertEqual((root / name).read_bytes(), (bundle / name).read_bytes())
            self.assertEqual(run.call_args_list[0].args[0],
                             ["systemctl", "--user", "stop", "pantheum-alibz.service"])
            self.assertNotIn("pantheum-alibz-worker.service", repr(run.call_args_list))

    def test_active_or_unresolved_hardware_blocks_restart_but_retrieval_continues(self):
        with sqlite3.connect(":memory:") as db:
            db.executescript("CREATE TABLE acquisitions(state TEXT,hardware_resolved INTEGER);"
                             "CREATE TABLE hardware_operations(state TEXT);")
            db.execute("INSERT INTO acquisitions VALUES ('awaiting_data',1)")
            deploy.check_idle(db)  # The retrieval worker is not restarted.
            for state, resolved in (("running", 0), ("uncertain", 0)):
                with self.subTest(state=state):
                    db.execute("DELETE FROM acquisitions")
                    db.execute("INSERT INTO acquisitions VALUES (?,?)", (state, resolved))
                    with self.assertRaises(RuntimeError):
                        deploy.check_idle(db)
            db.execute("DELETE FROM acquisitions")
            db.execute("INSERT INTO hardware_operations VALUES ('uncertain')")
            with self.assertRaises(RuntimeError):
                deploy.check_idle(db)


if __name__ == "__main__":
    unittest.main()
