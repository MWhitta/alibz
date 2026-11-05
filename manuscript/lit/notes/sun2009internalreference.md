---
key: sun2009internalreference
title: "Correction of self-absorption in CF-LIBS by an internal reference method (IRSAC)"
year: 2009
topic: theory
relevance: 2
---

**Method/claim.** Per species pick a high-E_k reference line (assumed thin); ratio other lines of the species to it to estimate each line's self-absorption, then regress for the correction; Boltzmann points become more regular and T and composition more accurate.

**Adoptable specifics.** Reference line choice: highest upper-level energy among strong lines (abstract). Correction regressed iteratively.

**Where it would enter alibz.** `peaky_indexer_v3.py` Boltzmann-plot seed (queued) as pre-step; line weight downgrade for low E_k lines.

**Fit-quality test.** Z300 Fe means: Fe I Boltzmann scatter and T consistency across 26 runs; expect lower scatter.

**Caveats.** Needs wide E_k spread; Fe I VIS lines give limited spread.
