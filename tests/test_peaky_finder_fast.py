import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import numpy as np

from alibz.peaky_finder import PeakyFinder


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

        with patch("alibz.peaky_finder.least_squares") as mock_least_squares:
            result = finder.fit_peaks(x, y, peak_indices, fast=True)

        self.assertFalse(mock_least_squares.called)
        self.assertGreater(len(result), 0)
        best_key = min(result, key=lambda key: abs(result[key][1] - true_params[1]))
        estimated = result[best_key]
        self.assertEqual(estimated.shape, (4,))
        self.assertGreater(estimated[0], 0.0)
        self.assertAlmostEqual(estimated[1], true_params[1], delta=0.2)

    def test_fit_spectrum_data_lazy_loads(self) -> None:
        finder = PeakyFinder.__new__(PeakyFinder)
        finder.data = MagicMock()
        finder.data.get_data.return_value = np.array(
            [[[400.0, 401.0, 402.0], [1.0, 2.0, 3.0]]],
            dtype=float,
        )

        expected = {'sorted_parameter_array': np.empty((0, 4))}
        with patch.object(PeakyFinder, 'fit_spectrum', return_value=expected) as mock_fit:
            result = finder.fit_spectrum_data(0, plot=False)

        finder.data.get_data.assert_called_once_with()
        mock_fit.assert_called_once()
        self.assertIs(result, expected)

    def test_fit_peaks_handles_upper_edge_window_bounds(self) -> None:
        finder = PeakyFinder.__new__(PeakyFinder)

        x = np.linspace(0.0, 1.0, 8)
        y = np.array([0.0, 0.2, 0.4, 0.8, 1.6, 2.5, 3.0, 2.7], dtype=float)
        peak_indices = np.array([6], dtype=int)

        def fake_peak_parameter_guess(data, idx):
            idx_arr = np.asarray(idx)
            if idx_arr.ndim == 0:
                return 3.0, 2.0, 1.0, len(data), 1

            shape = idx_arr.shape
            return (
                np.full(shape, 3.0),
                np.full(shape, 2.0),
                np.full(shape, 1.0),
                np.full(shape, len(data)),
                np.full(shape, 1),
            )

        finder.peak_parameter_guess = fake_peak_parameter_guess
        finder.find_peaks = lambda arr: (np.array([5], dtype=int), np.array([], dtype=int))

        with patch(
            "alibz.peaky_finder.least_squares",
            return_value=SimpleNamespace(x=np.array([1.0, x[6], 0.05, 0.02])),
        ) as mock_least_squares:
            result = finder.fit_peaks(x, y, peak_indices, fast=False)

        self.assertTrue(mock_least_squares.called)
        self.assertIn(6, result)

    def test_fit_spectrum_handles_empty_peak_dictionary(self) -> None:
        finder = PeakyFinder.__new__(PeakyFinder)

        class _Transformer:
            lambdas_ = np.array([1.0])

            def fit_transform(self, arr):
                return np.asarray(arr, dtype=float)

        x = np.linspace(400.0, 401.0, 32)
        y = np.sin(np.linspace(0.0, 2.0 * np.pi, x.size)) ** 2

        finder.fourier_peaks = lambda y, n_sigma=0: (
            np.array([8, 24], dtype=int),
            _Transformer(),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        finder.fit_peaks = lambda x, y_bgsub, peak_indices, plot=False, fast=True: {}

        result = finder.fit_spectrum(
            x,
            y,
            subtract_background=False,
            plot=False,
            skip_profile=False,
        )

        self.assertEqual(result['sorted_parameter_array'].shape, (0, 4))
        self.assertEqual(result['spectrum_dictionary'], {})
        np.testing.assert_allclose(result['profile'], np.zeros_like(y))

    def test_fit_spectrum_handles_flat_spectrum(self) -> None:
        finder = PeakyFinder.__new__(PeakyFinder)

        x = np.linspace(400.0, 401.0, 32)
        y = np.ones_like(x) * 7.0

        result = finder.fit_spectrum(
            x,
            y,
            subtract_background=True,
            plot=False,
            skip_profile=False,
        )

        self.assertEqual(result['sorted_parameter_array'].shape, (0, 4))
        self.assertEqual(result['spectrum_dictionary'], {})
        np.testing.assert_allclose(result['background'], y)
        np.testing.assert_allclose(result['profile'], np.zeros_like(y))

    def test_fit_spectrum_forwards_background_kwargs(self) -> None:
        finder = PeakyFinder.__new__(PeakyFinder)

        class _Transformer:
            lambdas_ = np.array([1.0])

            def fit_transform(self, arr):
                return np.asarray(arr, dtype=float)

        x = np.linspace(400.0, 401.0, 32)
        y = np.sin(np.linspace(0.0, 2.0 * np.pi, x.size)) ** 2 + 1.0
        bg = np.linspace(0.0, 0.1, x.size)

        with patch.object(PeakyFinder, 'find_background', return_value=bg) as mock_bg:
            with patch.object(
                PeakyFinder,
                'fourier_peaks',
                return_value=(
                    np.array([], dtype=int),
                    _Transformer(),
                    np.array([], dtype=int),
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            ):
                with patch.object(PeakyFinder, 'fit_peaks', return_value={}):
                    result = finder.fit_spectrum(
                        x,
                        y,
                        subtract_background=True,
                        plot=False,
                        range=9,
                    )

        mock_bg.assert_called_once_with(x, y, range=9)
        np.testing.assert_allclose(result['background'], bg)


if __name__ == "__main__":
    unittest.main()
