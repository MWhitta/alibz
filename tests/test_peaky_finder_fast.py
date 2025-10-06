import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from peaky_finder import PeakyFinder


class TestPeakyFinderFast(unittest.TestCase):

    def test_fast_fit_skips_optimizer(self) -> None:
        finder = PeakyFinder.__new__(PeakyFinder)

        x = np.linspace(-2.0, 2.0, 401)
        true_params = np.array([5.0, 0.25, 0.3, 0.2])
        y = finder.multi_voigt(x, true_params)
        y -= y.min()
        y[0] = 0.0
        y[-1] = 0.0

        peak_indices = np.array([int(np.argmax(y))])

        with patch("peaky_finder.least_squares") as mock_least_squares:
            result = finder.fit_peaks(x, y, peak_indices, fast=True)

        self.assertFalse(mock_least_squares.called)
        self.assertGreater(len(result), 0)
        best_key = min(result, key=lambda key: abs(result[key][1] - true_params[1]))
        estimated = result[best_key]
        self.assertEqual(estimated.shape, (4,))
        self.assertGreater(estimated[0], 0.0)
        self.assertAlmostEqual(estimated[1], true_params[1], delta=0.2)


if __name__ == "__main__":
    unittest.main()
