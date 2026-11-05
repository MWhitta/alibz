---
key: jochum2005georem
title: "GeoReM: A New Geochemical Database for Reference Materials and Isotopic Standards"
year: 2005
topic: cflibs
relevance: 1
---

**Method/claim.** GeoReM database of >750 geological reference materials (USGS, GSJ, MPI-DING, NIST glasses) with published values, uncertainty and method metadata.

**Adoptable specifics.** Source for reference compositions of natural-rock CRMs; each value carries uncertainty type and method.

**Where it would enter alibz.** `bench/` or `cal/` dataset construction: recommended-value lookup for CRM truth tables. No module change.

**Fit-quality test.** If CRM spectra are acquired: use GeoReM recommended values as truth for relative error on majors.

**Caveats.** Compilation values are not all certified.
