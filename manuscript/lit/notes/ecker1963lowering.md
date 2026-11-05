---
key: ecker1963lowering
title: "Lowering of the ionization energy for a plasma in thermodynamic equilibrium"
year: 1963
topic: theory
relevance: 2
---

**Method/claim.** Ecker-Kroll ionization-energy lowering with length set by mean particle separation r_EK^3=3/(4 pi (n_e+n_i)) at high density; Debye-Huckel limit at low density.

**Adoptable specifics.** Low-density limit Delta chi = z e^2/(4 pi eps0 lambda_D). At n_e=1e17 cm^-3, T=8 kK: lambda_D=19.5 nm, Delta chi=0.074 eV per unit charge (0.15 eV for charge 2), Saha ratio factor exp(Delta chi/kT)=1.11 (neutral->1+), 1.24 (1+->2+). This is ~5x smaller than the 'several tenths eV' in 02-theory.

**Where it would enter alibz.** `sahaboltzmann.ionization_distribution`: subtract Delta chi_z before Saha (queued IPD item).

**Fit-quality test.** Synthetic suite S: T error and log n_e error; real: stage_disagreement. Expected small (~10-25% stage ratios).

**Caveats.** N_D=(4pi/3)n lambda_D^3~3 at these conditions: Debye-Huckel marginal; effect comparable to partition-function error.
