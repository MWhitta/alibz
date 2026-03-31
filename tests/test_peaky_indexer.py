import unittest
import warnings

import alibz
import numpy as np

from alibz import PeakyIndexer, PeakyIndexerV3
from alibz.peaky_indexer_v2 import PeakyIndexerV2


class _DummyFinder:
    data = object()

    def fit_spectrum_data(self, *_args, **_kwargs):
        return {}


class TestPeakyIndexerPublicApi(unittest.TestCase):

    def test_peaky_fitter_removed_from_public_api(self):
        self.assertFalse(hasattr(alibz, "PeakyFitter"))

    def test_public_alias_points_to_v3(self):
        self.assertIs(PeakyIndexer, PeakyIndexerV3)

        idx = PeakyIndexer(np.array([[1.0, 500.0, 0.05, 0.03]]))
        self.assertIsInstance(idx, PeakyIndexerV3)
        self.assertEqual(idx.n_peaks, 1)
        self.assertAlmostEqual(idx.T_init, 10_000.0)

    def test_v2_shim_warns_and_routes_to_v3(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            idx = PeakyIndexerV2(
                np.array([[2.0, 510.0, 0.04, 0.02]]),
                temperature=8_500.0,
            )

        self.assertIsInstance(idx, PeakyIndexerV3)
        self.assertAlmostEqual(idx.T_init, 8_500.0)
        self.assertTrue(any("deprecated" in str(w.message).lower() for w in caught))

    def test_legacy_module_rejects_finder_signature_with_clear_error(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from alibz.peaky_indexer import PeakyIndexer as LegacyIndexer

        with self.assertRaisesRegex(TypeError, "Legacy PeakyIndexer\\(finder, \\.\\.\\.\\) is retired"):
            LegacyIndexer(_DummyFinder())

    def test_legacy_module_warns_and_uses_v3_for_peak_arrays(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            from alibz.peaky_indexer import PeakyIndexer as LegacyIndexer

            idx = LegacyIndexer(np.array([[3.0, 520.0, 0.03, 0.01]]))

        self.assertIsInstance(idx, PeakyIndexerV3)
        self.assertTrue(any("deprecated" in str(w.message).lower() for w in caught))


if __name__ == "__main__":
    unittest.main()
