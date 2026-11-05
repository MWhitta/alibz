---
key: harmon2023muscovite
title: "Rapid Analysis of Muscovites on a Lithium Pegmatite Prospect by Handheld LIBS"
year: 2023
topic: cflibs
relevance: 3
---

**Method/claim.** SciAps Z-300 handheld LIBS (the same instrument family as alibz's target) used for >130 muscovites from Li pegmatites (Carolina Tin-Spodumene Belt). Li 0.04-0.74 wt%, K/Rb 63 to 8; spodumene-bearing pegmatites 0.21 +/- 0.12 wt% Li vs 0.14 +/- 0.08 for quartz-feldspar (abstract).

**Adoptable specifics.** Hardware (Sec. 3.1): 6 mJ, 50 um, 1 ns pulse; gate ~650 ns delay, 1 ms integration; three spectrometers 190-365 / 365-620 / 620-950 nm with FWHM 0.18 / 0.24 / 0.35 nm; 23,432 channels. Protocol: 4x3 raster, 2 cleaning + 4 data shots per point, 5 grids, 48 spectra averaged. Calibration via intensity ratios of Li 670.7, K 766.49, Rb 780.03/794.76 nm over 670-672, 765-768, 779-781, 793-796 nm windows; >=20 spectra per reference, kept only when total emission varied <15%. Valid range 0-3 wt% Li, 0-12 K, 0-1 Rb.

**Where it would enter alibz.** Test data and a regime check for alkali doublet handling (K/Na ratios, MW2-112 note); `pipeline.py` segment-3 (NIR) alkali windows; shot-averaging and a <15% total-intensity QC filter.

**Fit-quality test.** Real spectra: alibz Li/K/Rb ratios vs calibrated ratios on muscovite; n_detected for Li 670.7 and Rb; K/Rb ratio is the direction-of-effect metric (no CF accuracy reported).

**Caveats.** Empirical ratio calibration against microprobe references, not CF; no RMSE values quoted in text (software-internal); strong matrix (muscovite) specificity.
