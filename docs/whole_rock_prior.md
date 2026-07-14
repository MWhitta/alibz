# Whole-rock composition prior

## Source and scope

The natural-composition scene prior is fitted to Gard, Hasterok, and Halpin
(2019), *Global whole-rock geochemical database compilation*,
doi:10.5194/essd-11-1553-2019, and its pinned Zenodo 1.0.0 tables,
doi:10.5281/zenodo.2592823. The paper reports 1,022,092 samples; the pinned
download currently contains 1,023,490 `sample.csv` rows. Both counts are
recorded rather than silently reconciled.

The release hashes, URLs, fitting parameters, selection counts, and scientific
limitations are in `db/whole_rock_prior_v1_manifest.json`. Raw source tables
are not committed. The compact anhydrous artifact is
`db/whole_rock_prior_v1.npz`; its element/stratum counts and ranges are
reviewable in `db/whole_rock_element_ranges.csv`. Carbonate/volatile-rich
chemistry is kept in the separate
`db/whole_rock_carbonate_volatile_prior_v1.npz` artifact and
`db/whole_rock_carbonate_volatile_ranges.csv` table.

The paper establishes several limits on interpretation:

- the compilation is a heterogeneous literature/database aggregate, not a
  probability sample of Earth's rocks;
- igneous samples dominate, with volcanic rocks especially common;
- North America, Canada, Australia, and New Zealand are overrepresented,
  while Africa is underrepresented;
- methods, limits of detection, and upstream duplicate-merging rules vary;
- major-derived computations use LOI-free analyses with totals between 85 and
  120 wt%.

The pinned release yields 1,004,337 classified rows, of which 73.69% are in
the fitted igneous strata. This is close to, but not identical to, the paper's
72.37% of known rock groups because the release row count also differs. The
major-oxide quality rule retains 626,902 rows.

## Anhydrous statistical model

The model has ten lithologic strata: volcanic, plutonic, and other igneous;
igneous-protolith, sedimentary-protolith, and unspecified metamorphic;
clastic, chemical/biogenic, and unspecified sedimentary; and unclassified.
Training selects the nine classified strata uniformly by default so the
database's sampling imbalance is not learned as Earth's true rock-frequency
distribution. `stratum_policy="corpus"` reproduces the observed row weights
when that is specifically wanted.

Major oxides are converted stoichiometrically to elemental mass ppm after
LOI-free normalization. Total iron prefers FeO-total, then Fe2O3-total
converted to FeO-equivalent, then the separately reported FeO and Fe2O3.
Positive trace reports are merged only where a valid major-derived value is
not already available.

Database states remain distinct:

- positive values are quantitative detections;
- negative values are left-censor limits and are sampled below their absolute
  limit;
- blank fields are unreported/unknown and are never treated as zero;
- exact zero remains an explicit synthetic coverage condition, not an
  inference from database missingness.

Each rock stratum uses 101 empirical log10(ppm) quantile knots. A
pairwise-complete Gaussian copula captures major-major, major-trace, and
trace-trace dependence. Correlations require at least 200 co-reports, shrink
toward zero with a 500-sample pseudocount, and are projected to a positive-
semidefinite correlation matrix. A stratum with fewer than 100 reports for an
element inherits the global marginal and records that fallback in the range
table.

This gives empirical priors for 76 of the 83 training-enabled elements. H and
N have only 25 and 43 positive reports, respectively, and He, Ne, Ar, Kr, and
Xe have no quantitative whole-rock coverage, so none are fabricated. Ar/air
is already a separate ambient-gas component. The periodic-table scheduler
continues to supply explicit balanced coverage for positions lacking a
whole-rock prior.

## Carbonate and volatile-rich model

The separate model prevents hydrous/carbonate chemistry from distorting the
dominant LOI-free silicate distribution. It has two strata:

- `carbonate_rich`: at least 20 wt% CO2-equivalent, matching the paper's
  carbonatite boundary;
- `volatile_rich`: non-carbonate analyses with at least 5 wt% directly
  speciated H2O + CO2 + SO3.

H2O-total is preferred where present; otherwise H2O+ and H2O- are summed.
Explicit CaCO3 and MgCO3 replace their corresponding CaO or MgO component and
separately exported CO2, preventing mass from being counted twice. LOI is
never assigned to an element because it does not identify the relative H2O,
CO2, sulfur, organic, or other contributions.

The 85--120 wt% directly speciated closure gate retains 23,303 distinct major
analyses, linked to 29,492 sample rows: 12,865 carbonate-rich and 16,627
volatile-rich. A further 38,235 rows meet the 5 wt% threshold only through
LOI; they are recorded but excluded. The separate prior models 70
training-enabled elements, including H, C, O, S, Se, Th, and U.

Selected global 2.5th--97.5th percentile ranges for this model are:

| Element | Detected mass ppm | Sampled nuclei fraction |
|---|---:|---:|
| H | 184--16,139 | 0.00435--0.259 |
| C | 291--124,459 | 0.000409--0.184 |
| O | 438,881--520,515 | 0.443--0.597 |
| S | 0.082--58,749 | 1.71e-8--0.0346 |
| Se | 0.032--27.1 | 1.47e-9--1.28e-6 |
| Th | 0.217--106 | 1.27e-8--6.32e-6 |
| U | 0.094--75.1 | 5.22e-9--3.83e-6 |

The 5 wt% non-carbonate threshold is a versioned development boundary, not a
claim that rocks immediately below it are physically anhydrous.

## Equal-stratum training mixture

`WholeRockCompositionMixture.load_default()` combines the nine classified
anhydrous strata with the two separate carbonate/volatile strata. Training
draws each of these 11 strata with equal probability. The unclassified stratum
is retained for auditing but is not in the training mixture. Corpus-row
weighting remains available only as an explicit evaluation policy.
Released training sets use `balanced_stratum_schedule(repeats_per_stratum)`
to enforce exact, replayable quotas rather than relying on finite-sample
equality from random draws.

## Global abundance ranges

The full CSV contains every element and stratum. Selected global values are
shown below. Ppm bounds are the 2.5th--97.5th percentiles of positive reports;
nuclei bounds are equal quantiles from joint prior draws after mass-to-mole
conversion and censor-tail sampling.

| Element | Detected mass ppm, 2.5--97.5% | Sampled nuclei fraction, 2.5--97.5% |
|---|---:|---:|
| O | 422,879--500,866 | 0.510--0.665 |
| Si | 200,781--373,106 | 0.144--0.272 |
| Al | 5,688--108,771 | 0.00557--0.0836 |
| Fe | 2,808--124,982 | 0.000883--0.0494 |
| Mg | 491--145,621 | 0.000354--0.111 |
| Ca | 755--133,730 | 0.000292--0.0737 |
| Se | 0.055--131.8 | 7.28e-11--1.55e-5 |
| Th | 0.074--55.3 | 4.86e-9--4.54e-6 |
| U | 0.027--27.3 | 2.17e-9--2.46e-6 |

These are broad corpus priors, not hard physical bounds or model acceptance
limits. Ore-rich and unusual rocks legitimately widen the tails; explicit
abundance-decade scenes still cover exact zero and `1e-8` through 1 without
being clipped to these intervals.

## Synthetic use

```python
from alibz import (
    PeriodicCoverageScheduler,
    SyntheticSpectrumGenerator,
    WholeRockCompositionMixture,
    WholeRockCompositionModel,
    WholeRockSceneSampler,
)

prior = WholeRockCompositionModel.load("db/whole_rock_prior_v1.npz")
training_prior = WholeRockCompositionMixture.load_default("db")

# Natural-composition scene, with independent direct stage I--III draws.
scene_sampler = WholeRockSceneSampler(training_prior, stratum_policy="balanced")
scene = scene_sampler.scene(seed=4102, stratum="igneous_volcanic")

# Comprehensive abundance-decade focus scenes with realistic joint
# whole-rock backgrounds.
generator = SyntheticSpectrumGenerator("db")
coverage = PeriodicCoverageScheduler(
    generator,
    whole_rock_model=training_prior,
    whole_rock_policy="balanced",
)
```

The sampled whole-rock nuclei fractions currently act only as the shape prior
for emitting-plasma nuclei. The scene manifest marks the mapping
`identity-shape-proxy-not-calibrated`. Element-dependent ablation, transport,
and plasma-yield mapping must be fitted later from standards; it is not hidden
inside this prior. Stage fractions and per-stage temperatures remain
independent, and no Saha ionization constraint is introduced.

## Rebuild

Download the four files whose exact URLs and hashes are pinned in the
manifest, then run:

```bash
build-whole-rock-prior \
    --major /path/to/major.csv \
    --trace /path/to/trace.csv \
    --sample /path/to/sample.csv \
    --rockgroup /path/to/rockgroup.csv

build-whole-rock-prior \
    --composition-mode carbonate_volatile \
    --major /path/to/major.csv \
    --trace /path/to/trace.csv \
    --sample /path/to/sample.csv \
    --rockgroup /path/to/rockgroup.csv
```

The command refuses source checksum changes. A deliberate data-release update
requires a new schema/version and scientific review of changed distributions.
