---
key: cowan1948selfabsorption
title: "Self-absorption of spectrum lines"
year: 1948
topic: theory
relevance: 2
---

**Method/claim.** Classic review of curve-of-growth and self-reversal theory (Ladenburg-Reiche); line intensity vs absorber column for Doppler and Lorentz profiles.

**Adoptable specifics.** Escape factor (1-e^-tau)/tau for a uniform slab is the thin-to-thick interpolation alibz uses; Lorentz wings give sqrt(tau) growth, Doppler cores ln(tau)^1/2. Full text is an OCR scan, no tables extracted.

**Where it would enter alibz.** `absorption.py`: profile-dependent escape factor (Voigt tau0 integral) instead of the single (1-e^-tau)/tau.

**Fit-quality test.** Synthetic self-absorption round trips: majors error for K/Na resonance lines; expected lower bias for strongly saturated lines.

**Caveats.** Homogeneous slab; real plasma has a cold shell (self-reversal) not covered.
