---
key: anderson2017submodels
title: "Improved accuracy in quantitative laser-induced breakdown spectroscopy using sub-models"
year: 2017
topic: cflibs
relevance: 1
---

**Method/claim.** Sub-model method: train several PLS (or other) regressions on restricted composition ranges and blend them into one result; improves RMSEP for almost all cases because emission response depends on the concentration of other elements (matrix effect). Used in the ChemCam calibration. [Abstract only.]

**Adoptable specifics.** Idea: compose-range-specific models, blended by a first-pass estimate. No numbers in provided text.

**Where it would enter alibz.** Conceptual analogue only: second-pass re-solve with matrix-class priors (iterative deepening already exists in `peaky_indexer_v3.py`).

**Fit-quality test.** Not worth separate test; iterative deepening already covers.

**Caveats.** Needs large labelled corpus; PLS-specific.
