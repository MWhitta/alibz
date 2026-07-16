import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_mw2_112 import (
    SCHEMA_VERSION,
    _checkpoint_valid,
    choose_pilot_ids,
    main,
    parse_ids,
)


class TestMW2112Runner(unittest.TestCase):
    def test_parse_ids_supports_ranges(self):
        self.assertEqual(parse_ids("989,1000-1002,1001"),
                         [989, 1000, 1001, 1002])

    def test_pilot_keeps_zero_and_terminal_positions(self):
        entries = [{
            "test_id": test_id,
            "all_zero": test_id == 1632,
            "negative_fraction": test_id / 1e6,
            "intensity_p99": float(test_id),
        } for test_id in range(989, 1918)]
        selected = choose_pilot_ids(entries)
        self.assertIn(1632, selected)
        self.assertIn(1916, selected)
        self.assertIn(1917, selected)
        self.assertEqual(len(selected), 20)

    def test_checkpoint_requires_input_and_config_identity(self):
        entry = {"path": "/raw/a.csv", "size_bytes": 12,
                 "sha256": "abc"}
        checkpoint = {
            "schema_version": SCHEMA_VERSION, "config_hash": "cfg",
            "input": dict(entry), "row": {"status": "ok"},
        }
        self.assertTrue(_checkpoint_valid(checkpoint, entry, "cfg"))
        changed = dict(entry, sha256="different")
        self.assertFalse(_checkpoint_valid(checkpoint, changed, "cfg"))

    def test_inventory_only_never_writes_raw_directory(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            raw = root / "raw"
            out = root / "run"
            raw.mkdir()
            for test_id in (989, 990):
                (raw / f"{test_id}-Test.csv").write_text(
                    "wavelength,intensity\n180,0\n190,1\n910,2\n961,0\n")
            before = sorted(path.name for path in raw.iterdir())
            code = main([str(raw), str(out), "--inventory-only"])
            after = sorted(path.name for path in raw.iterdir())
            self.assertEqual(code, 0)
            self.assertEqual(before, after)
            manifest = json.loads((out / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "inventory_only")
            self.assertEqual(manifest["selected_ids"], [989, 990])
            self.assertTrue((out / "input_inventory.csv").exists())


if __name__ == "__main__":
    unittest.main()
