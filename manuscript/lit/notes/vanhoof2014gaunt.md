---
key: vanhoof2014gaunt
title: "Accurate determination of the free-free Gaunt factor. I. Non-relativistic"
year: 2014
topic: theory
relevance: 1
---

**Method/claim.** Tables of thermally averaged free-free Gaunt factor over log10 gamma^2 = -6..10 and log10 u = -16..13 (11 digits), plus a polynomial series fit for the frequency-integrated factor.

**Adoptable specifics.** Gaunt factor depends on u = hv/kT and gamma^2 = Z^2 Ry/kT. At LIBS (kT~0.7 eV, u~0.3-1, gamma^2 ~ 20 for Z=1) g_ff is of order 1.0-1.5 and varies slowly with wavelength.

**Where it would enter alibz.** `synthetic.py` free-free continuum: tabulated g_ff(u,gamma^2) instead of constant; inverse model background is empirical so no change.

**Fit-quality test.** Synthetic suite S only: round-trip continuum shape; effect small (~10-30% in the continuum amplitude slope), probably not visible in majors/T metrics.

**Caveats.** Free-bound continuum, which dominates early LIBS, is not covered; low value for the metrics used.
