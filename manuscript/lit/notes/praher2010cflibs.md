---
key: praher2010cflibs
title: "Calibration free laser-induced breakdown spectroscopy of oxide materials"
year: 2010
topic: cflibs
relevance: 3
---

**Method/claim.** CF-LIBS for oxide materials: modified algorithm with fast deconvolution of the instrumental function, self-absorption correction, and oxide concentrations obtained through stoichiometric relations (closure on oxides, with O not measured). Pressed Fe2O3-MgO-CaO powders (Fe2O3 9.8-33.3, MgO 7.6-33.3, CaO 33.3-81.2 wt%), Nd:YAG 1064 nm, air, echelle + calibrated ICCD.

**Adoptable specifics.** Relative error |C_CF-C_N|/C_N: <10% Fe2O3, <20% MgO, <5% CaO (abstract). Closure by stoichiometric oxide formulas (sum of oxides = 100%) rather than sum of elements.

**Where it would enter alibz.** `peaky_indexer_v3.py` normalisation and reporting: add optional oxide closure (element -> default oxide, oxygen mass added to the closure sum) so unquantified O no longer inflates every fraction (assumption A10, 03-assumptions.tex:122). Not queued.

**Fit-quality test.** Synthetic suite S: add oxide-matrix cases (generator must emit O or oxide ratios); metric median relative error on majors and spurious mass (expect lower error on majors, since a missing O no longer rescales everything). Real: sum of oxides vs 100% as a QC statistic. Paper bar: 5-20% on three majors.

**Caveats.** Fe valence (FeO vs Fe2O3) and volatile phases (carbonate, OH, F) make the oxide choice a prior; wrong oxidation state biases all fractions. Argon purge removes air O, so measured O I is unusable as a closure term.
