"""Fixed H--U support schema and Se/Th/U forward-model coverage."""
import unittest

import numpy as np

from alibz.peaky_maker import PeakyMaker


class TestPeakyMakerElementSupport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.maker = PeakyMaker("db")

    def test_batch_retains_positions_and_never_labels_unsupported_mass(self):
        focus = ["Se", "Pm", "Th", "Po", "At", "Rn", "Pa", "U"]
        np.random.seed(1234)
        elements, fractions, _, _, _ = self.maker.batch_maker(
            focus_el=focus,
            n_elem=1,
            n_delta=0,
            abundance="random",
            batch=12,
            w_lo=260.0,
            w_hi=261.0,
            inc=0.2,
        )
        self.assertEqual(elements, focus)
        unsupported = [elements.index(el) for el in ("Pm", "Po", "At", "Rn", "Pa")]
        np.testing.assert_array_equal(fractions[:, unsupported], 0.0)
        np.testing.assert_allclose(fractions.sum(axis=1), 1.0)

    def test_unsupported_only_composition_is_rejected(self):
        fractions = np.zeros(92)
        fractions[self.maker.db.elements.index("Pm")] = 1.0
        with self.assertRaisesRegex(ValueError, "no supported"):
            self.maker.peak_maker(
                fractions, w_lo=300.0, w_hi=301.0, inc=0.2
            )

    def test_se_th_u_each_generate_nonzero_spectrum(self):
        centers = {"Se": 203.9851, "Th": 261.1627, "U": 285.5964}
        for element, center in centers.items():
            fractions = np.zeros(92)
            fractions[self.maker.db.elements.index(element)] = 1.0
            _, spectrum, per_element = self.maker.peak_maker(
                fractions,
                w_lo=center - 0.2,
                w_hi=center + 0.2,
                inc=0.02,
                temperature=10_000.0,
                ne=17.0,
                voigt_sig=0.03,
                voigt_gam=0.02,
            )
            self.assertTrue(np.all(np.isfinite(spectrum)), element)
            self.assertGreater(float(np.max(spectrum)), 0.0, element)
            self.assertGreater(
                float(np.max(per_element[self.maker.db.elements.index(element)])),
                0.0,
                element,
            )


if __name__ == "__main__":
    unittest.main()
