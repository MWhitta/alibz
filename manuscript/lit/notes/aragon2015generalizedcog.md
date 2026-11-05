---
key: aragon2015generalizedcog
title: "Quantitative analysis by laser-induced breakdown spectroscopy based on generalized curves of growth"
year: 2015
topic: cflibs
relevance: 3
---

**Method/claim.** Cs-LIBS: quantitative analysis from Csigma graphs with a homogeneous-double (HD) model (one region, separate neutral/ion treatment). A model-limit criterion eliminates data (high intensity or concentration) not described by the HD model. Characterisation step gives (beta A)_z, (eta N_l)_z, T_z, N_e_z (z=0,1); then relative or absolute concentrations. Fused glass from certified slags and pure compounds.

**Adoptable specifics.** Accuracy (abstract): 3.2% mean for concentrations relative to Fe (Ca, Mn, Mg, V, Ti, Si, Al in three slag glasses); 9.2% mean for absolute Fe, Ca, Mn. Takahashi 2017 s4.3 quotes the 9% with Ca worst (matrix effect). Key idea: data-exclusion criterion from the model limit.

**Where it would enter alibz.** `utils/absorption.py` / `refinement.py`: add a pre-fit exclusion gate on lines whose estimated tau or intensity exceeds a validity limit, before the doublet-anchored tau fit. `peaky_indexer_v3.py`: per-stage T_z, n_e_z as in aragon2014csigma. Not queued.

**Fit-quality test.** Synthetic suite S self-absorption round trips: median relative error on majors with vs without the gate (expect lower when strong resonance lines are present). Real: K I 766/770 doublet check closer to thin ratio after tau correction. Paper effect size: 3.2% / 9.2% (reference accuracy to beat or approach).

**Caveats.** Calibrated on a reference glass (needs one standard), not purely standardless. Single-pass optical-depth model is for homogeneous sources; multi-phase minerals violate it. Gate threshold is untested at Z300 resolution.
