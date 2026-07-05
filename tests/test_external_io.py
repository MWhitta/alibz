import os
import tempfile
import unittest

import numpy as np

from alibz.utils.dataloader import Data
from alibz.utils.database import Database


class TestExternalIO(unittest.TestCase):

    def test_load_data_returns_loaded_array_and_selected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_a = os.path.join(tmpdir, "2-Test.csv")
            file_b = os.path.join(tmpdir, "1-Test.csv")

            np.savetxt(
                file_a,
                np.array([[500.0, 5.0, 50.0], [501.0, 6.0, 60.0]]),
                delimiter=",",
                header="w,intensity,alt",
                comments="",
            )
            np.savetxt(
                file_b,
                np.array([[400.0, 1.0, 10.0], [401.0, 2.0, 20.0]]),
                delimiter=",",
                header="w,intensity,alt",
                comments="",
            )

            loader = Data(tmpdir)
            data = loader.load_data(w_col=0, i_col=2)

            self.assertIs(data, loader.data)
            self.assertEqual(data.shape, (2, 2, 2))
            np.testing.assert_allclose(data[0, 0], [400.0, 401.0])
            np.testing.assert_allclose(data[0, 1], [10.0, 20.0])
            np.testing.assert_allclose(data[1, 0], [500.0, 501.0])
            np.testing.assert_allclose(data[1, 1], [50.0, 60.0])

    def test_get_data_lazy_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_a = os.path.join(tmpdir, "1-Test.csv")
            np.savetxt(
                file_a,
                np.array([[400.0, 1.0], [401.0, 2.0]]),
                delimiter=",",
                header="w,intensity",
                comments="",
            )

            loader = Data(tmpdir)
            data = loader.get_data()

            self.assertIsNotNone(loader.data)
            self.assertEqual(data.shape, (1, 2, 2))

    def test_database_resolves_default_dbpath_outside_repo_cwd(self) -> None:
        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                db = Database("db")
        finally:
            os.chdir(original_cwd)

        self.assertIn("H", db.elements)
        self.assertTrue(len(db.no_lines) > 0)

    def test_database_default_honors_alibz_db_env(self) -> None:
        original = os.environ.get("ALIBZ_DB")
        dbpath = os.path.abspath("db")
        try:
            os.environ["ALIBZ_DB"] = dbpath
            db = Database("db")
        finally:
            if original is None:
                os.environ.pop("ALIBZ_DB", None)
            else:
                os.environ["ALIBZ_DB"] = original

        self.assertEqual(str(db.dbpath), os.path.abspath(dbpath))

    def test_database_resolves_installed_target_share_db(self) -> None:
        import alibz.utils.database as database_mod

        original_file = database_mod.__file__
        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                package_file = os.path.join(
                    tmpdir, "alibz", "utils", "database.py"
                )
                share_db = os.path.join(tmpdir, "share", "alibz", "db")
                os.makedirs(os.path.dirname(package_file))
                os.makedirs(share_db)
                database_mod.__file__ = package_file
                os.chdir(tmpdir)

                resolved = Database._resolve_dbpath("db")
        finally:
            database_mod.__file__ = original_file
            os.chdir(original_cwd)

        self.assertEqual(str(resolved), os.path.realpath(share_db))

    def test_database_ionization_energy_filters_by_ion_stage(self) -> None:
        db = Database("db")

        stage_1 = db.ionization_energy("Fe", ion=1)
        stage_2 = db.ionization_energy("Fe", ion=2)

        self.assertEqual(stage_1.shape, (1, 3))
        self.assertEqual(stage_2.shape, (1, 3))
        self.assertEqual(int(float(stage_1[0, 1])), 0)
        self.assertEqual(int(float(stage_2[0, 1])), 1)
        self.assertLess(float(stage_1[0, 2]), float(stage_2[0, 2]))


if __name__ == "__main__":
    unittest.main()
