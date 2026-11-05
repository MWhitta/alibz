---
key: aguilera2004spatial
title: "Characterization of a laser-induced plasma by spatially resolved spectroscopy of neutral atom and ion emissions."
year: 2004
topic: cflibs
relevance: 3
---

**Method/claim.** Spatially resolved (with deconvolution) emission of Fe I (21 lines) and Fe II (15 lines) from a Fe-Ni alloy plasma in air, 3.0-3.5 us window. Local Boltzmann / Saha-Boltzmann plots give the same T for neutrals and ions; spatially integrated plots give two different T (9100+/-150 K neutrals, 13700+/-300 K ions). The difference is explained by population-averaged emissivity distributions of the two stages.

**Adoptable specifics.** Numbers above (abstract). Integrated apparent T_neutral < T_ion because ions emit from the hotter core. Ionization degree is high and slowly varying in the inner plasma.

**Where it would enter alibz.** `peaky_indexer_v3.py` stage-consistency thermometer / `stage_disagreement`: model the ion-minus-neutral apparent-T offset as expected physics (stage-weighted effective T) rather than pure phase-heterogeneity or error; allow a prior offset sign (T_II >= T_I). Not queued.

**Fit-quality test.** Fe standard (26 means): fit Fe I and Fe II Boltzmann T separately, check T_II - T_I > 0 and stable; real spectra: stage_disagreement should shrink when a bounded offset is allowed. Expected direction: lower stage_disagreement, same or better r2.

**Caveats.** Laser in air at 1 atm, 3 us delay; Z300 is argon-purged, different expansion. Offset magnitude (4600 K) will differ; a free offset could mask real Saha errors (IPD, partition functions).
