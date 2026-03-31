import unittest

import numpy as np

from alibz.peaky_indexer_v3 import LineTable
from alibz.utils.sahaboltzmann import SahaBoltzmann


def _reference_distribution(sb: SahaBoltzmann, element: str, temperature, log_ne):
    temperature = sb._temperature_array(temperature)
    ne_m3 = 10 ** float(log_ne) * 1e6

    kT = sb.boltzmann_constant * temperature[:, None]
    lamb = sb.plank_constant / np.sqrt(
        2 * np.pi * sb.me * kT / sb.speed_c**2
    )

    lines = np.asarray(sb.db.lines(element))
    ionization = lines[:, 0].astype(float)
    Eion = np.asarray(sb.db.ionization_energy(element), dtype=float)

    ions = np.sort(np.unique(ionization))
    Zi, _ = sb.partition(element, temperature)
    ci = np.ones((len(temperature), len(ions)), dtype=float)

    if len(ions) == 1:
        return ci

    Ei = np.full((len(temperature), len(ions) - 1), np.inf, dtype=float)
    for ii, ion_stage in enumerate(ions[:-1]):
        Eind = Eion[:, 1] == ion_stage - 1
        if np.any(Eind):
            Ei[:, ii] = Eion[Eind, -1][0]

    saha_prefactor = 2.0 * lamb[:, 0] ** -3 / ne_m3
    min_partition = 1e-10

    for iii in range(len(ions) - 1):
        Zi_i = np.clip(Zi[:, iii], min_partition, None)
        Zi_ip1 = np.clip(Zi[:, iii + 1], min_partition, None)
        a = saha_prefactor * (Zi_ip1 / Zi_i) * np.exp(-Ei[:, iii] / kT[:, 0])
        ci[:, iii + 1] = a * ci[:, iii]

    ci = ci / np.clip(np.sum(ci, axis=1, keepdims=True), min_partition, None)
    return np.round(ci, 10)


class TestSahaBoltzmann(unittest.TestCase):

    def test_ionization_distribution_converts_density_to_m3(self) -> None:
        sb = SahaBoltzmann("db")
        temperature = np.array([10_000.0])
        log_ne = 17.0

        expected = _reference_distribution(sb, "Ca", temperature, log_ne)
        actual, _ = sb.ionization_distribution("Ca", temperature, log_ne)

        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0.0)

    def test_line_table_preserves_saha_fraction_for_single_retained_stage(self) -> None:
        sb = SahaBoltzmann("db")
        lt = LineTable(sb.db, sb, (200.0, 910.0), max_ion_stage=2, min_gA=100.0)

        fractions = lt.compute_saha_fractions(10_000.0, 17.0)
        cs_indices = [i for i, sp in enumerate(lt.species) if sp.element == "Cs"]

        self.assertEqual(len(cs_indices), 1)

        stage_fraction = fractions[cs_indices[0]]
        ci, _ = sb.ionization_distribution("Cs", [10_000.0], 17.0)

        self.assertAlmostEqual(stage_fraction, ci[0, 0], places=10)
        self.assertLess(stage_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
