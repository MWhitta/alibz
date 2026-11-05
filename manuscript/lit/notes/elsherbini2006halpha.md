---
key: elsherbini2006halpha
title: "Measurement of electron density utilizing the Hα-line from laser produced plasma in air"
year: 2006
topic: cflibs
relevance: 1
---

**Method/claim.** H-alpha (656.27 nm) Stark width as n_e standard in laser plasma in humid air, 0-10 us delay, 1 us gate; n_e from 1e18 down to 6e16 cm^-3, compared with Al II 281.62 nm; Griem standard theory vs Gigosos tables. Line is isolated, strong against background, long lived, broad, and not self-absorbed.

**Adoptable specifics.** n_e range 1e18 -> 6e16 cm^-3 across 0-10 us (abstract); agreement with Al II confirms Halpha as reference. Not self-absorbed.

**Where it would enter alibz.** ALREADY IN alibz: `utils/stark.py:80-85` (`halpha_log_ne`, 0.549 nm x (n_e/1e23 m^-3)^0.67965) and `pipeline.py:512` (`halpha_ne_bounds`). Provenance only.

**Fit-quality test.** No new test. Existing: n_e-at-bound flag on real spectra when H-alpha present.

**Caveats.** Requires H in the plasma (moisture, OH minerals); at 0.1-0.2 nm pixels the 1 nm-scale line is resolved, but the instrument LSF and Fe/Si blends near 656 nm need care.
