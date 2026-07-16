import csv
import tempfile
import unittest
from pathlib import Path

from scripts.mw2_112_provenance import (
    aggregate_manifest_hash,
    file_manifest,
    hash_file,
    verify_input_inventory,
)
from alibz.mw2_112 import load_frozen_calibration


class TestPortableProvenance(unittest.TestCase):
    def test_tree_identity_is_independent_of_absolute_root(self):
        with tempfile.TemporaryDirectory() as first, \
                tempfile.TemporaryDirectory() as second:
            for root in (Path(first), Path(second)):
                (root / "nested").mkdir()
                (root / "nested/value.txt").write_text("same bytes\n")
            first_rows = file_manifest(Path(first))
            second_rows = file_manifest(Path(second))
            self.assertEqual(first_rows, second_rows)
            self.assertEqual(aggregate_manifest_hash(first_rows),
                             aggregate_manifest_hash(second_rows))

    def test_raw_verification_uses_basename_size_and_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            spectrum = raw / "0989-test.csv"
            spectrum.write_text("wavelength,intensity\n500,1\n")
            inventory = root / "inventory.csv"
            with inventory.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=(
                    "file", "size_bytes", "sha256"))
                writer.writeheader()
                writer.writerow({
                    "file": spectrum.name,
                    "size_bytes": spectrum.stat().st_size,
                    "sha256": hash_file(spectrum),
                })
            self.assertEqual(verify_input_inventory(inventory, raw), [])
            spectrum.write_text("changed\n")
            self.assertTrue(verify_input_inventory(inventory, raw))

    def test_frozen_calibration_restores_vector_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calibration.csv"
            header = ["test_id", "response_measured", "response_prior"]
            for label in ("uv", "vis", "nir"):
                header.extend((f"shift_prior_{label}_nm",
                               f"profile_fwhm_{label}_nm"))
            with path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=header)
                writer.writeheader()
                writer.writerow({
                    "test_id": 989,
                    "response_measured": "",
                    "response_prior": "2.8",
                    "shift_prior_uv_nm": "-0.01",
                    "shift_prior_vis_nm": "0.02",
                    "shift_prior_nir_nm": "",
                    "profile_fwhm_uv_nm": "0.2",
                    "profile_fwhm_vis_nm": "0.23",
                    "profile_fwhm_nir_nm": "0.3",
                })
            got = load_frozen_calibration(path)[989]
            self.assertEqual(got["response_prior"], 2.8)
            self.assertEqual(got["shift_prior_nm"], [-0.01, 0.02, None])
            self.assertEqual(got["profile_fwhm_nm"], [0.2, 0.23, 0.3])


if __name__ == "__main__":
    unittest.main()
