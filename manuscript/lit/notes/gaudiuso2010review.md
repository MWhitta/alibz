---
key: gaudiuso2010review
title: "Laser Induced Breakdown Spectroscopy for Elemental Analysis in Environmental, Cultural Heritage and Space Applications: A Review of Methods and Results"
year: 2010
topic: cflibs
relevance: 1
---

**Method/claim.** Review (Sensors 10, 7434) of LIBS fundamentals, calibration-line vs calibration-free (CF) analysis, and applications (cultural heritage alloys, soils, space). States that CF needs strictly fulfilled stoichiometric ablation, LTE, homogeneity and optical thinness, and otherwise gives only semi-quantitative results (Sec. 2-3, text before Table 1).

**Adoptable specifics.** Line-selection rules (Sec. 2): exclude all lines involving the ground state (self-absorption), especially for majors; McWhirter criterion is necessary, not sufficient, for LTE and delay/gate must be chosen so Saha equilibrium holds. Table 1: fs- vs ns-CF-LIBS on three bronze standards agree; Cu was not quantified because of strong Cu self-absorption.

**Where it would enter alibz.** No new module. Supports the existing line-selection prior in `peaky_indexer_v3.py` (down-weight resonance/ground-state lines for high-concentration elements) and the validity gates in the report text.

**Fit-quality test.** Real sample spectra: compare r^2 / stage_disagreement with and without ground-state lines for the dominant element. No effect size is given; it is qualitative guidance.

**Caveats.** Narrative review, no algorithm or thresholds beyond the above; superseded by tognoni2010cflibs and hahn-style treatments for specifics.
