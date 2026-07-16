# MW2-112 peak-window PCA quantification

## Scope

This method estimates **within-element relative spatial abundance trends**
from all 929 single-pulse MW2-112 spectra. It does not estimate absolute
concentrations, and a score for one element is not numerically comparable with
a score for another. The profiles are chemical anchors for joint X-ray and
neutron diffraction interpretation, not standalone phase fractions.

The promoted local result is
`runs/mw2_112_profile/peak_window_pca_shift_validated_20260716/`. The earlier
`peak_window_pca_primary_20260716` and
`peak_window_pca_deconfounded_20260716` directories are developmental audit
runs and must not replace it.

## Fixed corpus peak-shape prior

The compact basis in
`corrections/corpus_peak_shape_pca_05x_10pc.npz` was recovered from the legacy
`peaky_data.ipynb` artifact. Its provenance manifest records 4,909 training
spectra and 13,502,524 baseline-subtracted, range-normalized peak windows.

The saved `05x` basis was trained with a 0.1557064239 nm half-window. The
0.0841 nm value calculated later in the notebook is inconsistent with the
saved basis and is not used. MW2-112 projection uses only PC1--PC5, which retain
99.883% of corpus peak-shape variance. Ten components would nearly saturate an
interpolated window containing only about ten native 0.0333 nm detector
samples.

The basis is projected onto signed physical templates. These are correlated
diagnostics, not uniquely identified mechanisms:

- PC1 is predominantly shift;
- PC2 is predominantly Gaussian broadening and broad-wing/splitting response;
- PC3 is predominantly flattening;
- PC4 is predominantly Lorentzian wings; and
- PC5 contributes a smaller shift coordinate.

Each shot/window retains PC1--PC5, reconstructed area, residual centroid,
broadening, Lorentzian-wing, flattening, and splitting coordinates. Positive
flattening can be consistent with self-absorption; splitting can also be an
unresolved blend. Neither label is treated as proof by itself.

## Spectrum and window processing

1. Remove the arPLS background from each raw spectrum.
2. Apply the frozen detector-segment response prior and preserve total positive
   line power as a shot-normalization nuisance variable.
3. Build an aligned mean spectrum from the 928 nonzero observations. Test ID
   1632 remains explicitly missing, never zero abundance.
4. Generate strong ion-stage features at 9,000 K from the atomic database and
   supplement them with pre-registered canonical H, Li, O, Na, Mg, Al, Si, K,
   Ca, Ti, Fe, Rb, Sr, and Ba lines.
5. Retain a discovery window at summed-spectrum SNR >= 1. This permissive gate
   is needed for broad K, O, Al, and Rb lines and is not sufficient evidence of
   an element.
6. Reproduce the corpus training transform in every window: endpoint baseline,
   range normalization, interpolation to the corpus grid, fixed-basis
   projection, and five-PC reconstruction. Relative line area is reconstructed
   shape area times the observed peak range, divided by shot line power.

The resulting amplitude is scaled by its within-line 90th percentile before
multiple lines are combined. A weighted median, rather than a sum, limits the
effect of one saturated or shape-anomalous resonance line.

## Shift correction as a tested physical prior

The frozen atomic-anchor table proposes a per-shot wavelength shift in each
detector segment. The method does not apply it automatically. Clean strong
windows are used to compare the robust dispersion of common peak centroids
before and after the proposed correction. A variable correction is retained
only if corrected dispersion is no more than 0.90 times raw dispersion using
at least five windows.

For the promoted run:

| Segment | Windows | Raw spread | Proposed corrected spread | Applied mode |
|---|---:|---:|---:|---|
| UV | 17 | 8.85 pm | 9.06 pm | constant segment median |
| Visible | 15 | 6.09 pm | 10.32 pm | constant segment median |
| NIR | 7 | 4.98 pm | 1.52 pm | per-shot frozen prior |

Thus only the NIR data support variable drift correction. Detector temperature
is a possible cause of common drift, but no temperature telemetry exists and
the analysis does not claim it as the demonstrated cause.

## Spectral independence and element support

Every selected feature is checked against the full catalog of material
transitions from all expected elements, not just other top-ranked candidates.
All database overlaps within 0.16 nm are recorded. An overlap becomes a
disqualifying competitor when its 9,000 K `gA exp(-E/kT)` times crustal
abundance prior is at least 0.25 times the candidate prior. This removes the
false preliminary Ag, Cu, Ga, Ge, and V assignments caused by Fe-rich blends.
The ratio is a stated physical prior, not a concentration correction; database
incompleteness and non-LTE plasma response remain limitations.

Same-element/stage windows within 0.16 nm form one independent spectral
cluster. Only one representative per cluster can contribute. A supported ion
stage requires at least two independent, uncontested representatives with:

- summed-spectrum SNR >= 1;
- per-shot SNR >= 3 in at least 25% of positions;
- median five-PC reconstruction R² >= 0.35; and
- same-stage Spearman profile coherence >= 0.25 with at least 100 common
  detections.

H, N, and O emission cannot become sample-abundance profiles without blanks or
atmosphere controls, even if their spectral gates pass.

## Promoted results

The fixed gate supports 52 independent windows and ten relative element
profiles:

| Element | Windows | Stages | Interpretation status |
|---|---:|---|---|
| Al | 8 | I | internally supported; vendor-discordant |
| Ba | 2 | II | supported, but many positions are single-line |
| Ca | 7 | I | supported |
| Fe | 9 | I, II | strong X-ray/transition-metal anchor |
| Li | 2 | I | neutron-sensitive/absorbing anchor; relative only |
| Mg | 5 | I, II | supported clay-chemistry anchor |
| Na | 5 | I | supported alkali profile |
| Si | 3 | I | strong silicate/X-ray anchor |
| Sr | 2 | I | supported, but many positions are single-line |
| Ti | 9 | I, II | strong transition-metal/X-ray anchor |

Matched-filter comparison Spearman correlations are Al 0.853, Ba 0.629, Ca
0.445, Fe 0.954, Li 0.690, Mg 0.683, Na 0.806, Si 0.968, Sr 0.791, and Ti
0.941. Vendor comparison is strongest for Si (0.904), Ti (0.899), and Fe
(0.686). Al is internally coherent but has vendor rho -0.045, so it must not be
used as an unqualified anchor.

K fails the *narrow corpus-PCA* gate because the method is invalid for its
optically thick resonance lines, not because K is absent. K I 766.490 and
769.896 nm have median FWHM 0.467 and 0.400 nm, wider than the entire 0.311 nm
PCA window; the old PCA noise sidebands therefore lie inside their wings. The
observed median 766/769 peak ratio is 1.185 versus the optically thin database
ratio 2.006, giving a homogeneous-slab diagnostic of median tau(769)=1.68 and
tau(766)=3.38. A supplementary +/-1 nm broad-window analysis detects both
resonance lines in all 928 nonzero shots and the weak K I 693.876 nm line in
789. Their spatial Spearman coherences are 0.76--0.99. The robust three-line
relative K trend is retained as **optical-depth compressed**, never absolute
concentration, in
`runs/mw2_112_profile/k_resonance_primary_20260716/`.

Rb has one clean strong line, while its second canonical line is Ti-confounded.
Ag, Cu, Ga, Ge, and V have no two-line uncontested support.

Median shape coordinates for accepted lines are generally within one corpus
standard deviation. Li, Na, Ca, Mg, and Sr resonance windows show moderate
positive flattening/wing coordinates, but fewer than 1% of accepted
shot/windows exceed the two-sigma flattening threshold. The data therefore
suggest broad/flat resonance profiles without establishing pervasive severe
self-absorption or resolved splitting. The multi-line median remains the main
nonlinearity protection.

## Products and regeneration

Key files in the promoted directory are:

- `element_profiles.csv`: primary per-position relative profiles and status;
- `element_summary.csv`: supported lines, stages, and shape summaries;
- `window_manifest.csv`: every screened window and its pass/fail evidence;
- `candidate_prior_manifest.csv`: screened and rejected line priors, including
  complete overlap annotations;
- `window_profiles.csv`: all per-shot PC and physical coordinates;
- `peak_physics_summary.csv`: accepted-line physical diagnostics;
- `shift_correction_policy.csv` and `detector_shift_profiles.csv`;
- `validation.csv`, figures, and `analysis_report.md`;
- `run_manifest.json` and `reproduction_recipe.json`; and
- `product_checksums.json`, which currently verifies all 21 other artifacts.

The original Drive raw and vendor directories pass the frozen upstream
provenance verifier. To regenerate, follow the exact argument vector in
`reproduction_recipe.json`, substituting only `RAW_DIR` and an empty
`NEW_OUTPUT`. Verify raw inputs first:

```bash
mw2-112 provenance verify \
  runs/mw2_112_profile/relative_profiles_primary_20260715 \
  --input-dir RAW_DIR --vendor-dir VENDOR_DIR

mw2-112 peak-pca RAW_DIR NEW_OUTPUT \
  --db db \
  --basis corrections/corpus_peak_shape_pca_05x_10pc.npz \
  --calibration \
    runs/mw2_112_profile/relative_profiles_primary_20260715/session_calibration.csv \
  --vendor-dir VENDOR_DIR \
  --previous-run \
    runs/mw2_112_profile/relative_profiles_primary_20260715
```

The full manifest records every non-default threshold, database hash, basis
hash, calibration hash, source state, package versions, input lineage, and
claim boundary. Scientific CSVs should be compared semantically across
platforms; generated timestamps and image metadata need not be byte-identical.
