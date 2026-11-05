---
key: korbel2024lithium
title: "Quantification of lithium using handheld instruments: application of LIBS and XRF spectroscopy to assay the lithium content of mineral proce"
year: 2024
topic: cflibs
relevance: 3
---

**Method/claim.** SciAps Z300 handheld LIBS (argon flux, 190-950 nm, 10 Hz, 1064 nm) and ED-XRF on pressed lepidolite-bearing mineral-processing pellets; Li quantified from Li I lines, from Rb (XRF proxy), or both with multiple linear regression. 6x6 raster, 5 shots per site (1 cleaning + 4 data), two passes, 72 spectra averaged.

**Adoptable specifics.** Table 5: indirect via Rb R2 0.99, RMSE 0.12, MAPE 11.33%; Li lines R2 0.99, RMSE 0.15, MAPE 11.75%; Li+Rb R2 0.99, RMSE 0.18, MAPE 14.53% (wt% Li2O). Li I ~610 and ~671 nm main lines (812.6 also used). Blank quartz still gave a Li signal; LOD estimated ~10 ppm. LIBS preferred below 0.8 wt% Li2O.

**Where it would enter alibz.** Z300 benchmark and line choice for Li; `detections.py` confounder contest should handle a Li blank/contamination signal (quartz). Compare with the project's own handheld Li behaviour. Not queued.

**Fit-quality test.** If a Li-bearing Z300 dataset is available: relative error vs assay (expect ~12-15% MAPE univariate), detection at <1000 ppm. Not in current datasets.

**Caveats.** Empirical regression on pellets, not a physics inversion; the file copy for li2014spectralresponse and palleschi2022selfabsorption is this same paper (wrong content).
