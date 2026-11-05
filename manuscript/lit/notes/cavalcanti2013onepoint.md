---
key: cavalcanti2013onepoint
title: "One-point calibration for calibration-free laser-induced breakdown spectroscopy quantitative analysis"
year: 2013
topic: cflibs
relevance: 3
---

**Method/claim.** One-point calibration (OPC): a CF-LIBS variant where ONE standard of known composition and similar matrix empirically fixes the poorly known experimental/spectroscopic parameters (line-wise factors absorbing wrong A_ki, response, geometry); analysis of unknowns then proceeds with the Boltzmann/Saha machinery. Demonstrated on Cu-based (bronze) samples; reported as improving trueness over plain CF-LIBS. [Fulltext = abstract + intro; equations and numbers not in provided text.]

**Adoptable specifics.** Abstract: only one similar-matrix standard needed; improves trueness vs CF-LIBS. Takahashi 2017 s4.2 notes OPC improves CF-LIBS accuracy. Borduchi 2022 uses OPC-corrected Saha-Boltzmann plots in practice.

**Where it would enter alibz.** New small module (e.g. `utils/onepoint.py`) fitting per-line (or per-segment) multiplicative correction factors from a reference spectrum with known composition (Fe standard `fe_aesar_9998_means_20260922.npz`, or a NIST glass); applied inside `peaky_indexer_v3.py` before the WNNLS solve. Complements the planned branching-ratio corpus response (assumptions A9). Not queued.

**Fit-quality test.** Fit factors on half of the 26 Fe run means, test on the rest: Fe I Boltzmann T consistency (expect lower scatter), line-matching fraction, resolved-fit count; real spectra r2 and stage_disagreement. Expected direction: improved; no effect size in provided text.

**Caveats.** Z300 spectra are multi-phase geology, standard must be matrix-similar; per-line factors can overfit and silently absorb the 620 nm gain step and wavelength-registration errors. Hold-out discipline is mandatory.
