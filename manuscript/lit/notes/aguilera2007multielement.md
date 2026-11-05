---
key: aguilera2007multielement
title: "Multi-element Saha-Boltzmann and Boltzmann plots in laser-induced plasmas"
year: 2007
topic: theory
relevance: 3
---

**Method/claim.** A single multi-element plot: y = ln(I lambda /(g A)) (plus a Saha shift for ions) against E_k, with element- and stage-specific intercepts, fitted jointly. Assumes LTE and stoichiometric conservation; 58 lines of Fe, Ni, Mn I/II gave high correlation. Spatially integrated data gave two different apparent temperatures for neutrals and ions.

**Adoptable specifics.** Joint fit: common slope -1/kT, intercept per (element, stage); ionic points shifted by ln[(2(2 pi m kT)^1.5/(h^3 n_e)) ] and E_ion (Sec. 2, Eq. for Saha-Boltzmann plot). Lesson: integrated plasma yields T(ion) > T(neutral), i.e. a built-in stage disagreement.

**Where it would enter alibz.** `peaky_indexer_v3.py` Boltzmann-plot seed (already queued) and the stage-consistency thermometer: allow separate neutral and ion T as an explicit 2-T mode and report the difference.

**Fit-quality test.** Z300 Fe means: Fe I vs Fe II T from separate fits; if difference is systematic, a 2-T model should reduce stage_disagreement and stabilise Fe II/Fe I ratio across the 26 runs.

**Caveats.** Their deconvolved emissivities are spatially resolved; spatially integrated data (as in the Z300) biases the plot. Uses 58 lines at high resolution.
