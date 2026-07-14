# Explicit-stage synthetic spectrum generator

`SyntheticSpectrumGenerator` is the synthetic-training forward model. It is
separate from legacy `PeakyMaker`: ion-stage abundances are direct inputs and
the generator never calls a Saha ionization distribution.

## Minimal use

```python
import numpy as np

from alibz import (
    AtomicStrengthUncertainty,
    ChannelGrid,
    HierarchicalPlasmaSampler,
    InstrumentResponse,
    SyntheticScene,
    SyntheticSpectrumGenerator,
    WholeRockCompositionModel,
    WholeRockSceneSampler,
)
from alibz.elements import ATOMIC_NUMBER
from alibz.synthetic_calibration import IndividualShotCalibration

# Stages use columns I, II, III and must sum to one over target nuclei.
x = np.zeros((92, 3))
x[ATOMIC_NUMBER["Ca"] - 1, 1] = 0.60  # Ca II
x[ATOMIC_NUMBER["Al"] - 1, 0] = 0.40  # Al I

# Provisional hierarchical cross-element constraints. The returned Te values
# are independent for every populated element-stage; ne and N_eff are per
# element.
plasma = HierarchicalPlasmaSampler().component(
    x,
    seed=100,
    emission_scale=0.1,
    continuum_scale=200.0,
)

# A saved corpus summary can be loaded when available.
calibration = IndividualShotCalibration.load(
    "jobs/synthetic_instrument_mw2_individual.json"
)
generator = SyntheticSpectrumGenerator(
    "db",
    calibration.response,
    atomic_uncertainty=AtomicStrengthUncertainty(enabled=True),
)

scene = SyntheticScene(plasma, seed=101)
result = generator.render(scene, ChannelGrid.native(), add_noise=True)

wavelength_nm = result.wavelength_nm
exported_counts = result.intensity_counts
physical_counts = result.physical_channel_counts  # always nonnegative
manifest = result.manifest
```

For arbitrary or disjoint regions, construct `ChannelGrid` from increasing
wavelength centers. Supply explicit left/right edges when available. Gaps are
kept as gaps rather than converted into wide channels.

## Whole-rock composition scenes

`WholeRockCompositionModel` provides a rock-stratified joint composition
prior fitted to the Gard et al. (2019) global whole-rock compilation. It
distinguishes detections, left-censored reports, and unreported fields; models
76 training-enabled elements; and preserves major/trace correlations with a
regularized empirical Gaussian copula. Se, Th, and U are included. See
`whole_rock_prior.md` for source hashes, element ranges, biases, and rebuild
instructions.

`WholeRockSceneSampler` creates natural-composition scenes. Alternatively,
pass the prior to `PeriodicCoverageScheduler` to use a realistic joint
whole-rock background while its focus element-stage remains fixed at every
required abundance decade. Balanced lithologic strata are the training
default; corpus-row weighting is an explicit option.

The carbonate/volatile-rich prior is a separate artifact with directly
speciated H2O/CO2/SO3 and carbonate chemistry. The default training mixture
contains nine classified anhydrous strata plus carbonate-rich and
volatile-rich strata; all 11 receive equal training weight. LOI-only analyses
are not assigned fabricated element abundances.

The prior does not claim a calibrated bulk-to-plasma map. Whole-rock nuclei
fractions are currently an identity shape proxy for emitting-plasma nuclei,
marked as such in scene metadata. Ion stages are still independent direct
draws and never use Saha equilibrium.

## Scientific state

`PlasmaComponent` stores:

- `stage_abundance[92, 3]`, with columns I--III;
- `temperature_k[92, 3]`, independent for every element-stage;
- `log_ne_cm3[92]`, one effective electron density per element;
- `effective_column[92]`, one provisional self-absorption column scale per
  element;
- explicit emission and positive continuum scales.

Target stage abundances must sum to one. Pm, Po, At, Rn, and Pa retain their
schema positions but nonzero abundance in those positions is rejected. A
stage with no quantitative transition in 190--910 nm remains valid target
state but is marked structurally unobservable by `stage_observable`.

Tc, Fr, Ra, and Ac are a separate policy class: their atomic-data positions
remain technically supported, but they are excluded from the current training
round. `SyntheticSpectrumGenerator` can still render an explicitly requested
diagnostic scene for them; the default `PeriodicCoverageScheduler` cannot use
them as a focus or background element.

The renderer applies Boltzmann excitation and a partition function within each
supplied stage, Doppler and per-element-density Stark broadening, per-element
escape-factor self-absorption, and cell integration. It does not infer or
modify stage fractions.

## Ambient gas

`dry_air_component` creates a separate nuisance component between pure Ar and
fixed dry N2/O2/Ar. Its stage fractions are explicit; no Saha equilibrium is
used. Ambient emission is returned separately and never enters target
abundance normalization. Air-derived and target-derived N/O remain
source-confounded at inference time.

## Positivity and signed exports

Latent photon emission, continuum, and expected physical detector counts are
nonnegative. The observation layer then adds individual-shot photon/read
noise, dark-estimation error, offsets, and an export kernel. Consequently the
final exported counts may be negative, matching the behavior observed in
individual MW2-112 files. Negative outputs are not clipped.

The same scene seed and grid reproduce the spectrum exactly. The manifest
records hashes of the scene, grid, and atomic source manifest plus the complete
instrument configuration.

For Se, Th, and U, optional source-class strength perturbations distinguish
NIST `gA`, intensity-derived `MC`, Corliss--Bozman `CB`, multiplet-estimated,
and guessed values. Perturbations contain correlated source bias plus per-line
variation and replay exactly from the scene seed. Their provisional widths are
stored in the manifest and must be sensitivity-tested rather than interpreted
as measured database errors.

## Periodic-table coverage

`PeriodicCoverageScheduler` explicitly enumerates 92 elements, stages I--III,
exact zero, and every abundance decade from `1e-8` through 1. The five
unsupported elements remain in the coverage manifest but cannot produce
scenes. Tc, Fr, Ra, and Ac also retain coverage cells and output positions,
but their cells carry `training_enabled=false` and cannot produce training
scenes. This leaves 87 technically supported and 83 training-enabled element
positions. Supported, training-enabled stages with no quantitative line in
190--910 nm do produce target scenes marked structurally unobservable.

The manifest records independent 92-position `support_mask` and
`training_mask` arrays plus the explicit `training_excluded_elements` list.
Excluded elements are also removed from generated background mixtures, so
they cannot enter training incidentally.

Each coverage cell generates deterministic background mixtures, independent
stage temperatures, hierarchical per-element densities/columns, individual-
shot signal scales, and optional Ar/dry-air nuisance emission. Replicate quotas
and adversarial line-confusion pair schedules still need to be frozen before a
large dataset is released.

## Instrument calibration

Run the corpus summary on individual shots only:

```bash
calibrate-synthetic-instrument /path/to/individual/shots \
    --pattern '*.csv' \
    --out jobs/synthetic_instrument.json
```

The current MW2-112 artifact summarizes 929 individual shots. It estimates
native wavelength layout, per-segment baseline, local noise, negative-count
morphology, and dynamic range. Line widths are provisionally anchored by the
smallest UV/VIS/NIR modes summarized in `peaky_data`.

A reproducible full-band demonstration is available through:

```bash
demo-explicit-synthetic
```

The checked demonstration produces 21,600 channels with median 89.7 counts,
99th percentile 329 counts, maximum 8,870 counts, and 0.213% negative exported
channels. These values establish the intended individual-shot scale but do not
close the conditional-profile or line-density validation gates below.

## Current release gates

This is a working scientific core, not yet a final training generator. The
manifest remains `calibrated=false` until all of the following are complete:

- quality-gated PCA refit with recoverable conditional peak scores;
- wavelength-dependent empirical profile sampling rather than one provisional
  Voigt distribution per detector segment;
- dark/blank-frame separation of electronic offsets from plasma continuum;
- empirical vendor remapping/ringing kernel;
- reviewed numerical widths for source-aware oscillator-strength
  perturbations (the mechanism is implemented with provisional values);
- frozen replicate quotas and adversarial blend-pair extensions to the
  implemented element-stage/abundance-decade scheduler;
- standards-based element-dependent bulk-to-plasma ablation/transport mapping;
- held-out MW2-112 and other individual-shot real-versus-synthetic validation.

The generator emits an explicit warning while these gates remain open.
