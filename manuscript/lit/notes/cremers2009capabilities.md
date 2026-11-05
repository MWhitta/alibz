---
key: cremers2009capabilities
title: "Laser-Induced Breakdown Spectroscopy—Capabilities and Limitations"
year: 2009
topic: cflibs
relevance: 2
---

**Method/claim.** General LIBS review with a CF-LIBS section: requires accurate T and n_e, lines of ALL elements and accurate gf; the normalisation factor F is the main source of inaccuracy; 'controlled calibration LIBS' uses a reference element to avoid F.

**Adoptable specifics.** CF-LIBS on siliceous soils/rocks validated with NIST SRMs: accuracy from a couple % up to 30% for most elements (ref. 41). Controlled calibration vs CF on soil (vs ICP): 3.2% vs 6.0% mean difference, 2.6x more accurate (Table 3, ref. 42). Bronze coins: CF and curves only partially agree. Matrix effects can reverse sign with pressure (Mn curves).

**Where it would enter alibz.** Anchor normalisation in `peaky_indexer_v3.py`: when one element's mass fraction is known or can be assumed (e.g. a stoichiometric host or a pure standard such as the Fe standard), use it to replace the closure sum (controlled-calibration style). Not queued.

**Fit-quality test.** Fe standard (99.98% Fe): composition invariance with anchor; suite S 'spurious mass' and error on majors with anchor vs closure (expect lower). Reference bar: 2.6x improvement (ICP-compared, one soil).

**Caveats.** Requires an independently known element; for unknown minerals this is not available. One-sample soil comparison; weak statistics.
