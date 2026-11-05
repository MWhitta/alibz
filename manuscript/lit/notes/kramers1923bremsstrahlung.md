---
key: kramers1923bremsstrahlung
title: "On the theory of X-ray absorption and of the continuous X-ray spectrum"
year: 1923
topic: theory
relevance: 1
---

**Method/claim.** Classical origin of the free-free (bremsstrahlung) continuum: emissivity proportional to Z^2 n_e n_i T^-1/2 exp(-hv/kT).

**Adoptable specifics.** Kramers form with Gaunt factor g_ff=1 is the baseline for the free-free continuum in `synthetic.py`. No new numbers beyond the scaling; text is a scan (77 kB).

**Where it would enter alibz.** `synthetic.py` continuum: use Gaunt factor from vanhoof2014gaunt rather than 1.

**Fit-quality test.** Synthetic only: continuum shape vs T.

**Caveats.** Classical; hv/kT is ~0.2-0.5 at 190-950 nm for 8 kK, so the exponential matters.
