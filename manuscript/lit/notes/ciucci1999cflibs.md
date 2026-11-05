---
key: ciucci1999cflibs
title: "New procedure for quantitative elemental analysis by laser-induced plasma spectroscopy"
year: 1999
topic: theory
relevance: 3
---

**Method/claim.** Original CF-LIBS: Boltzmann plots per species, Saha for stage mapping, concentrations closed by normalising the sum to 1; iterative loop on n_e when only one stage per element is seen.

**Adoptable specifics.** Closure: sum of concentrations = 1; N_e updated as sum of ions until convergence (text near p. 962). Accuracy 'few percent' on alloys.

**Where it would enter alibz.** Closure already in alibz (unnormalised issue fixed per audit). The n_e plasma-neutrality iteration could replace the n_e prior when Halpha is absent (`peaky_indexer_v3.py`).

**Fit-quality test.** log n_e error on synthetic S; n_e-at-bound flag frequency on real spectra; expected to reduce railing.

**Caveats.** Neutrality needs all elements and stages incl. H, O, N; unknown matrix breaks closure.
