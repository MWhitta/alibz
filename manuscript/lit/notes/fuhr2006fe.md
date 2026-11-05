---
key: fuhr2006fe
title: "Critical compilation of atomic transition probabilities for neutral and singly ionized iron"
year: 2006
topic: theory
relevance: 3
---

**Method/claim.** NIST critical compilation of A values for Fe I and Fe II with accuracy grades.

**Adoptable specifics.** Grades (p. near Sec. 1): AA/A within +-3%, B +-10%, C +-25%, D +-50%, E worse. Use grade as a weight on gA in the Boltzmann fit.

**Where it would enter alibz.** Z300 Fe standard route: weight Fe I lines in `peaky_indexer_v3.py` WNNLS by 1/(grade uncertainty)^2 and exclude D/E. Database `gA` provenance in the line DB.

**Fit-quality test.** Z300 Fe means: Fe I Boltzmann T consistency and resolved-fit count; excluding D/E lines expected to reduce Boltzmann scatter (typical rms from 0.2-0.5 in ln I to <0.15).

**Caveats.** Grades refer to individual lines; multiplet blending at 0.1-0.2 nm pixels dominates over A uncertainty for many lines.
