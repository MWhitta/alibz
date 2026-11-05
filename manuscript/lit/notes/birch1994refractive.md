---
key: birch1994refractive
title: "Correction to the updated Edlen equation for the refractive index of air"
year: 1994
topic: theory
relevance: 3
---

**Method/claim.** Corrected Edlen dispersion: (n-1)_s x1e8 = 8342.54 + 2406147/(130 - s^2) + 15998/(38.9 - s^2), s=1/lambda_vac in um^-1; pressure/temperature scaling factor; uncertainty 3e-8 for 350-650 nm.

**Adoptable specifics.** Eq. (1)-(2) above. At 500 nm n-1=2.79e-4 so lambda_air=lambda_vac/n is 0.139 nm below vacuum; at 190 nm 0.06 nm; 950 nm 0.26 nm, matching the 0.11-0.24 nm offset noted. Scale (n-1)_tp=(n-1)_s p/96095.43 [1+1e-8(0.601-0.00972 t)p]/(1+0.003661 t): at 20 C and 85 kPa the shift changes by about -18%, ~25 pm at 500 nm, comparable to the +-30 pm Ar anchors.

**Where it would enter alibz.** Line-DB loader / air-vacuum conversion; verify existing conversion against this formula (S).

**Fit-quality test.** Z300 Fe: line-matching fraction and Ar anchor residual; expected: no change if conversion already standard, a 25 pm shift if p,T not standard.

**Caveats.** Valid 350-650 nm dry air; UV below 200 nm extrapolated; plasma in Ar atmosphere has slightly different n.
