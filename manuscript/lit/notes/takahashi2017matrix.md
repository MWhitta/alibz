---
key: takahashi2017matrix
title: "Quantitative methods for compensation of matrix effects and self-absorption in Laser Induced Breakdown Spectroscopy signals of solids"
year: 2017
topic: cflibs
relevance: 3
---

**Method/claim.** Review of matrix-effect and self-absorption compensation for solids: calibration curves, internal standards, CF-LIBS (with its four assumptions), OPC, PCR/PLS, ANN. Recommends common figures of merit such as NRMSE.

**Adoptable specifics.** NRMSE = RMSE/(y_max - y_min) (Eq. 8); RMSE/RMSEC/RMSEP/RMSECV naming. CF-LIBS assumptions: LTE, optically thin, homogeneous, stoichiometric ablation; McWhirter alone insufficient (relaxation-time and diffusion-length criteria). Self-absorption fixes: COG, Csigma (better T), internal reference line (Sun & Yu 2009), Amamou, El Sherbini; Csigma on Fe, Ca, Mn glass: 9% mean RE. CF-LIBS accuracy: 5-20% metals, 1-190% geological (Herrera, Salle). Bottleneck: all elements must be detected; K and S often missed in soils -> only relative concentrations.

**Where it would enter alibz.** Reporting: add NRMSE to `bench/` summaries for comparability (cheap). Closure issue (undetected K/S) supports oxide/anchor closure. Reference list for the manuscript metrics section.

**Fit-quality test.** Suite S: add NRMSE column next to median relative error; no effect size, comparability only. Real: report relative-to-reference-element ratios.

**Caveats.** Review; numbers are secondary. Geological CF-LIBS errors quoted come from Mars-pressure stand-off, not handheld.
