---
key: gigosos1996hbeta
title: "New plasma diagnosis tables of hydrogen Stark broadening including ion dynamics"
year: 1996
topic: theory
relevance: 2
---

**Method/claim.** Computer-simulation tables of H Stark profiles including ion dynamics for n_e 1e20-1e24 m^-3 (1e14-1e18 cm^-3), with agreement to ~10% over runs.

**Adoptable specifics.** Tables give Hbeta/Halpha FWHM vs n_e,T. Equations extracted poorly (quaternion method text).

**Where it would enter alibz.** `stark.py` Halpha table lookup instead of power-law fit (Eq. halpha).

**Fit-quality test.** Real spectra with H alpha: compare log n_e from table vs current fit; expected difference <0.1 dex but removes a fitted constant.

**Caveats.** Halpha at 656.3 nm is a Z300 VIS-NIR segment line (0.1-0.18 nm pixel); Stark FWHM ~0.5 nm at 1e17 is resolved, fine.
