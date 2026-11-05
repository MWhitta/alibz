---
key: elsherbini2005selfabsorption
title: "Evaluation of self-absorption coefficients of aluminum emission lines in LIBS"
year: 2005
topic: theory
relevance: 2
---

**Method/claim.** Self-absorption coefficient SA (measured/thin intensity) from the Lorentzian Stark width: broadened width of a self-absorbed line vs the thin width at known n_e.

**Adoptable specifics.** From the abstract: needs n_e and the line's Stark parameter; relation of the form width ratio ~ SA^0.54 (exponent from memory, unverified in the retrieved 4 kB).

**Where it would enter alibz.** `absorption.py`: width-based tau estimate for lines without doublet partners.

**Fit-quality test.** K I/Na D doublet-ratio check and SA round trips; expected help only where Stark width resolvable.

**Caveats.** Stark widths ~0.002 nm vs 0.03-0.05 nm scatter: unresolvable on Z300.
