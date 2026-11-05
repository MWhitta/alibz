---
key: rezaei2020selfabsorption
title: "A review of the current analytical approaches for evaluating, compensating and exploiting self-absorption in Laser Induced Breakdown Spectro"
year: 2020
topic: cflibs
relevance: 2
---

**Method/claim.** Review of self-absorption (SA): causes (analyte density, lower-level population, plasma size; worse for resonance lines and double-pulse), and strategies to estimate, compensate and exploit it. [Provided fulltext is abstract + start of introduction only.] Standard methods (background knowledge, not in provided text): curve of growth, internal reference line, width-based (El Sherbini) SA coefficient, doublet ratio, Csigma, columnar density.

**Adoptable specifics.** Only qualitative content available in the provided text. Cross-reference urbinamedina2021branching (eqs. 8, 11) and takahashi2017matrix s4.3 for formulas.

**Where it would enter alibz.** None directly; context for `utils/absorption.py` design choices. 'Exploit' direction: use SA lines to retrieve columnar density (cristoforetti2013columnar) - not queued.

**Fit-quality test.** None until full text is obtained.

**Caveats.** Fulltext truncated; do not cite specific thresholds from this paper until retrieved in full.
