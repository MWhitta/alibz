---
key: fujimoto1990validity
title: "Validity criteria for local thermodynamic equilibrium in plasma spectroscopy"
year: 1990
topic: theory
relevance: 3
---

**Method/claim.** Collisional-radiative solution for H-like ions (Z=1,2,26) gives the lowest principal quantum number p that is within 10% of its Saha-Boltzmann population, for ionizing, recombining and ionization-balance plasmas, then extends to the ground state (complete LTE). Opacity relaxes the criteria; non-hydrogenic systems are treated only by analogy.

**Adoptable specifics.** Criteria are n_e thresholds scaling steeply with level (n_e >~ p^-17/2 form for ionization balance) and T^1/2 (Sec. III-IV, Eqs. 10, 17, 21, 25-27; exponents garbled in the text extraction, re-read the PDF before coding). Ground state needs far higher n_e than excited levels: levels with large energy gap are the ones to distrust. Relaxation-time condition for transient plasmas (Eq. 25-27).

**Where it would enter alibz.** Per-line LTE flag next to the queued McWhirter diagnostic (`detections.py`, `peaky_indexer_v3.py` reporting): mark lines whose lower/upper gap exceeds the n_e-limited gap. Diagnostic only, no change to the solve.

**Fit-quality test.** Real spectra: correlation of the flag with line residual and with Fe I Boltzmann-plot outliers on the Z300 Fe means (Fe I resolved-fit count, T consistency). Expected: flagged high-gap lines (Ar I NIR, resonance lines) show larger Boltzmann residuals; removing them lowers T scatter.

**Caveats.** Hydrogenic calculation, 10% criterion; the paper warns non-hydrogenic extension is approximate. Transient LIBS plasma needs the time-dependent condition, which a gated-less handheld cannot verify. Not a fit improvement by itself.
