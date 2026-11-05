---
key: wilks1938lrt
title: "The large-sample distribution of the likelihood ratio for testing composite hypotheses"
year: 1938
topic: ml
relevance: 2
---

**Method/claim**: -2 ln Lambda is asymptotically chi-square with df = difference in parameters under regularity conditions.

**Adoptable specifics**: Theorem (l.135): df = h - m. Boundary caveat (not in Wilks): when null sits on a boundary (c_j=0 with c_j>=0), the null distribution is a 50:50 mixture of chi2_0 and chi2_1 (Chernoff 1954), so one-sided z = sqrt(2 ΔlnL).

**Where it would enter alibz**: Detection z in the evidence row is the signed square root of the LRT; justify threshold: z>1.645 (not 1.96) corresponds to 5% for a nonnegative parameter. Use in detections.py threshold documentation and BH inputs.

**Fit-quality test**: data/remote_samples/*.csv: spurious-element count at z threshold; no code change, calibration check on blanks.

**Caveats**: Regularity fails for NNLS with coincident lines, and sigma_eff misspecification inflates z.
