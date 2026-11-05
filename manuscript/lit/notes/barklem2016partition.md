---
key: barklem2016partition
title: "Partition functions and equilibrium constants for diatomic molecules and atoms of astrophysical interest"
year: 2016
topic: theory
relevance: 3
---

**Method/claim.** NIST-level based partition functions U(T) for 284 atomic species (H-U), tabulated to 10 000 K, with a study of truncation of Rydberg levels (Hummer-Mihalas occupation probability).

**Adoptable specifics.** Sec. 5 text: when I/kT > ~10 the ground state dominates and U is insensitive to the level cut-off; sensitivity grows for low-IP neutrals (alkalis, alkaline earths) and with T. Data at CDS; range up to 10 000 K only (Irwin 1987 polynomials diverge outside their fit range).

**Where it would enter alibz.** Already queued (tabulated partition functions) in `sahaboltzmann.py`: replace sum over line-list levels by tabulated U(T) interpolated in log U vs log T; extrapolate carefully above 10 kK.

**Fit-quality test.** Synthetic suite S: T error and spurious mass for elements with sparse line lists; real: stage_disagreement for Na/K/Ca/Mg. Expected: lower U bias on elements whose line list misses levels (U underestimated) so smaller stage_disagreement.

**Caveats.** Tables stop at 10 kK, while the Z300 plasma reaches 12 kK; low-IP species sensitive to truncation, tables use no density-dependent lowering (consistent with a lowering-free alibz).
