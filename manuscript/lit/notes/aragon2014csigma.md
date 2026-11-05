---
key: aragon2014csigma
title: "CSigma graphs: A new approach for plasma characterization in laser-induced breakdown spectroscopy"
year: 2014
topic: cflibs
relevance: 3
---

**Method/claim.** Generalises curves of growth to Csigma (C-sigma) graphs: per line a cross section sigma_l is computed from atomic data, T and n_e; lines of several elements in ONE ionization state fall on a single curve. Homogeneous one-zone plasma, treated separately for neutrals (z=0) and ions (z=1). Fitting experimental graphs yields four parameters per stage (beta*A, N_l, T, N_e). A validity limit is stated. Fused glass (Fe, Mn, Mg, Si, Cr). [Abstract + intro only; no equations.]

**Adoptable specifics.** Abstract: four parameters per stage, T and N_e may differ between z=0 and z=1 (p.1). Takahashi 2017 s4.3: Csigma improved Boltzmann-T reliability when strong resonance lines are included.

**Where it would enter alibz.** `peaky_indexer_v3.py` stage-consistency thermometer: allow stage-specific effective (T_z, n_e_z) and report their offset as a diagnostic instead of a penalty; `utils/absorption.py`: adopt the model-validity limit as a line-exclusion rule (drop lines whose predicted tau exceeds the one-zone limit). Not queued.

**Fit-quality test.** Real spectra: stage_disagreement and residual r2 (expect lower disagreement, equal-or-higher r2); Fe standard: Fe II/Fe I ratio stability across 26 run means (expect tighter). Synthetic suite S: T error must not worsen (generator is single-zone, so neutral at best). No effect size in provided text.

**Caveats.** Needs a validated validity limit; Z300 resolution (0.1-0.2 nm) merges multiplets so per-line sigma is blended. One-zone assumption is exactly what fails for a 7-12 kK argon plasma with gradients.
