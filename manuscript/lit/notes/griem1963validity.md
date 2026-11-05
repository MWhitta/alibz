---
key: griem1963validity
title: "Validity of local thermal equilibrium in plasma spectroscopy"
year: 1963
topic: theory
relevance: 2
---

**Method/claim.** Derives n_e lower bounds for LTE by requiring collisional rates to dominate radiative decay for the resonance transition of the lowest gap; excited-state populations obey Saha-Boltzmann above some level; multiply ionized species cannot reach complete LTE at moderate n_e.

**Adoptable specifics.** Criterion n_e >~ C (E21/E_H)^3 (kT/E_H)^1/2 with C of order 1e18 cm^-3 (hydrogenic, Sec. IV; coefficient was not recoverable from the extraction, verify in PDF); stated accuracy about a factor of 2 (p. near line 398). McWhirter (1965, not retrieved) is the practical form 1.6e12 T^1/2 dE^3.

**Where it would enter alibz.** Same as fujimoto1990validity: gap-dependent per-line n_e bound reported with the McWhirter flag (already queued).

**Fit-quality test.** Worked check: at T=8 kK, Ar I NIR (dE ~11.5 eV) needs n_e >~ 2e17 > assumed 1e17, so Ar I anchors fail McWhirter while Fe I (dE 3-4.5 eV, ~1e16) passes. Test on Fe means and Ar-bearing real spectra: stage_disagreement and doublet check should be no worse after dropping flagged lines from the thermometer.

**Caveats.** Factor-2 accuracy; necessary only; hydrogenic. Assumes steady state, which a 1 us-scale plasma violates (see cristoforetti2010lte).
