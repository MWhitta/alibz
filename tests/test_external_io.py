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


if __name__ == "__main__":
    unittest.main()
