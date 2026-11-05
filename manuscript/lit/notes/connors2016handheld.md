---
key: connors2016handheld
title: "Application of Handheld Laser-Induced Breakdown Spectroscopy (LIBS) to Geochemical Analysis"
year: 2016
topic: cflibs
relevance: 2
---

**Method/claim.** Handheld LIBS for geochemistry (SciAps-class: ~0.1 nm resolution, vacuum-UV to NIR, argon purge from 21 g cartridges, rastering). Calibration on OREAS/JOGMEC rock CRMs and NIST soils with analyte/matrix-line intensity ratios (Al, Si, Ca, Fe as internal standards) after Savitzky-Golay smoothing; mapping by integrated peak area.

**Adoptable specifics.** Argon flows ~1 s before the pulse train; purge and a short gate delay (extracted as '1 ms'; the text loses the micro sign everywhere, e.g. '125 mm' = 125 um, so read 1 us). 12 locations x 3 shots, 120 um spacing, spot 50 um. Sensitivity: percent level for Al, B, C, Ca, Cu, Fe, Mg, Mn, Na, Ni, S, Ti, Zn; ~1000 ppm Cr, Ba, Pb; <=100 ppm Be, Li. Soils and rocks gave one consistent curve. Matrix matching would improve it.

**Where it would enter alibz.** Operational reference for Z300 acquisition presets: argon purge on, short delay, multi-site averaging, heterogeneity requires several locations. No code change.

**Fit-quality test.** Z300 Fe standard: compare site-averaged vs single-site fit stability (Fe II/Fe I ratio scatter). Expect averaging to reduce scatter.

**Caveats.** Calibration curves, not first-principles; gate unit ambiguity above.
