# Physical triage research for fast LIBS identification

Date: 2026-09-22  
Scope: scientific review; the separate Ar/O detector implementation is recorded
in `reports/2026-09-22-argon-oxygen-detection.md`.

## Outcome

Fast physical triage is defensible, but the hard-rejection surface is narrow.
The strongest implementable rules are:

1. Defer a proposed species for the supplied peak list when it has no wavelength
   support after propagating database, calibration, and fitted-center
   uncertainties. This is not evidence of chemical absence.
2. Reject a proposed species when one isolated, securely attributed reference
   line fixes a positive minimum species amplitude and at least two independent,
   non-resonant companion features would each exceed the measured local area
   noise by at least 5 sigma for every allowed temperature (recommended initial
   envelope: 4000--25000 K), yet both are absent. This requires calibrated
   response, valid detector coverage, unsaturated data, no credible blend, and
   transition probabilities whose lower uncertainty bounds still imply the
   contradiction.
3. Treat an absent same-upper-level branch as an especially strong companion
   contradiction because the upper population cancels. In the repository's
   energy-flux convention, the response-corrected thin ratio is
   `(gA_1/lambda_1)/(gA_2/lambda_2)`. It is hard evidence only after excluding
   self-absorption, saturation, blends, and response error.

Everything else requested here is best used to rank or annotate candidates:
ordinary doublet ratios, Saha ion-stage balance, a high-stage plausibility
envelope, excitation-temperature coherence, profile self-reversal, and
chance-coincidence penalties. None is a generally safe early rejection by
itself in transient, inhomogeneous LIBS plasma.

## Findings that affect the current code

- `alibz/utils/absorption.py` calls K I 766.5/769.9 a shared-upper pair. It is a
  shared-*lower* ground-state resonance doublet with two fine-structure upper
  levels. Na D has the same structure. Their near-2 thin ratios are weakly
  temperature dependent under LTE; they are not exact branching ratios.
  `invert_doublet_tau()` already accepts distinct `thin_ratio` and
  `strength_ratio`, so the numerical API can represent the distinction even
  though several comments do not.
- `alibz/utils/sahaboltzmann.py` derives partition functions only from the
  lower and upper levels present in the local transition table. These are not
  complete thermodynamic partition functions. Its ionization chain and
  `LineTable.compute_saha_fractions()` also assume stages are contiguous and
  addressable as I, II, ... . Therefore repo Saha fractions are unsuitable for
  hard rejection, especially for repaired or sparse heavy-element data.
- `PeakyIndexerV3._species_evidence_keep_mask()` says a species must have two
  supported strong lines but computes
  `min(evidence_min_supported_lines, strong_line_count)`. A one-line species
  can therefore satisfy that particular evidence gate. The separate initial
  strength gate partly mitigates this, but a candidate score must count
  independent observed resolution elements explicitly.
- Current missing-line evidence uses relative predicted strength at a single
  initial plasma state. A hard contradiction instead needs a local
  upper-limit calculation across the full plausible temperature envelope,
  actual per-feature area uncertainty, response/coverage masks, blends,
  saturation, and atomic-data uncertainty.
- Split lobes of a flat-topped or self-reversed resonance line must count as
  one physical feature. Counting both lobes as independent wavelength matches
  rewards the optical-depth pathology that triage should flag.
- NIST ASD observed intensities are source dependent, commonly uncorrected for
  spectral sensitivity, and sometimes qualitative. They must not be used as
  cross-line quantitative priors. NIST transition-probability accuracy codes
  range through `E` (>50% uncertainty); doubtful or intensity-derived values
  should never drive a hard missing-line rejection.

## Real-data check in this repository

The frozen 929-shot MW2-112 K result is a useful falsification test for rigid
doublet rules. In the run manifest and a reproducible re-analysis of
`k_wide_line_profiles.csv`:

- the expected thin K I 766.49/769.90 peak-height ratio is 2.0060;
- the median observed ratio is 1.1849 and all detected ratios are below 1.8;
- all 928 non-empty shots have positive, SNR >= 5 integrated areas for both
  resonance features; their area-ratio 5th/50th/95th percentiles are
  1.35127/1.44061/1.52738, and 926/928 ratios fall outside a +/-20% band around
  the thin value 2.006045;
- both resonance features are detected in 928 non-empty shots and have
  Spearman rho = 0.9938 across the scan;
- the weak non-resonant K I 693.876 nm anchor is detected in 789 shots and
  correlates with the resonance lines at rho = 0.757--0.767; and
- the homogeneous-slab inversion gives median finite weak-line optical depth
  1.684.

This supports the pair as strong K and optical-depth evidence, and shows that a
hard optically-thin ratio gate would reject a strongly corroborated signal. It
does not establish self-absorption as the unique cause. The corpus has no
independent certified composition, blanks, or replicate shots, so it does not
validate absolute concentration or a universal rejection threshold.

## Primary-source anchors

- NIST gives optically thin line power as proportional to upper-level
  population times transition probability times photon energy, and documents
  the Saha/Boltzmann assumptions used by its synthetic spectra:
  https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html
- NIST warns that ASD relative intensities are source dependent, often not
  sensitivity corrected, and qualitative; the same page provides transition
  probability accuracy codes and line flags for blends, reversal, uncertainty,
  and unresolved structure.
- Urbina Medina et al. (2021) specifically show that a branching-ratio test can
  appear consistent with optical thinness despite substantial self-absorption:
  https://doi.org/10.1177/00037028211006764
- El Sherbini et al. (2005) experimentally quantify self-absorption as measured
  intensity relative to the optically thin expectation:
  https://doi.org/10.1016/j.sab.2005.10.011
- Gornushkin et al. (1999) observed the curve-of-growth transition for the Cr I
  425.4 nm resonance line in steel plasmas:
  https://doi.org/10.1016/S0584-8547(99)00004-X
- Cristoforetti et al. (2010) show that the McWhirter criterion is necessary,
  not sufficient, for LTE in transient, inhomogeneous LIBS plasmas:
  https://doi.org/10.1016/j.sab.2009.11.005
- Amato et al. (2010) rank LIBS identifications using line strength and inverse
  local database-line density, an empirical precedent for a chance-coincidence
  penalty rather than raw match counts:
  https://doi.org/10.1016/j.sab.2010.04.019

## Recommended implementation boundary

Emit a transparent `physical_evidence_score` (or named component scores), not
a probability or posterior. Preserve the untriaged candidate list and attach
reason codes, feature-level evidence, upper limits, parameter-envelope extrema,
and atomic-data quality. Hard rejection should require all prerequisites to be
affirmatively true; unknown coverage, response, blending, saturation, or
optical depth must downgrade the rule to ranking only.

The detailed equations, decision table, feature definitions, citations, and
validation limits are in `docs/physical_triage_priors.md`. Source facts and
access dates are recorded in
`provenance/physical-triage-sources-20260922.json`.
