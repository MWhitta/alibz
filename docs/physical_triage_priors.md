# Physical priors for fast LIBS candidate triage

## Purpose and claim boundary

This note defines physical features that may reduce obviously incompatible
element/ion candidates before the expensive whole-pattern solve. It separates
hard contradictions from ranking evidence. A triage score is a named heuristic
score, not a calibrated probability, likelihood, or posterior.

Hard rejection is intentionally rare. A transient LIBS plasma is spatially and
temporally inhomogeneous; optical depth, non-LTE populations, detector response,
line blending, wavelength drift, saturation, and incomplete atomic data can all
make a real line weak or absent. Cristoforetti et al. show that even satisfying
the McWhirter density criterion is necessary but not sufficient to establish LTE
in a LIBS plasma ([doi:10.1016/j.sab.2009.11.005](https://doi.org/10.1016/j.sab.2009.11.005)).

The recommended initial hard-rejection rules are only:

1. no observed wavelength support for any eligible feature of the species in
   the supplied peak list, after calibration and uncertainty propagation; or
2. one isolated, securely attributed reference feature plus at least two
   independent, non-resonant companions that must exceed the measured local
   5-sigma area limit for every allowed plasma temperature, but are absent.

All prerequisites must be affirmatively known. Unknown response, coverage,
blending, saturation, atomic strength, or optical depth changes a rejection to
a rank penalty.

## Line-intensity relationships

For an optically thin homogeneous source, NIST gives the emitted energy in a
line from upper level `u` to lower level `l` as proportional to

```text
I_ul = N_u A_ul h nu_ul = N_u A_ul hc/lambda_ul .             (1)
```

([NIST Atomic Spectroscopy, spectral lines](https://physics.nist.gov/Pubs/AtSpec/node17.html);
[NIST ASD lines help](https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html)).
For LTE excitation within one ion stage,

```text
N_u = N_stage (g_u/U_stage(T)) exp[-E_u/(k_B T)] .            (2)
```

The repository stores `gA = g_u A_ul` and uses energy flux, so its thin line
weight is

```text
w_j(T,n_e) proportional to
  (gA_j/lambda_j) exp[-E_u,j/(k_B T)] f_stage(T,n_e)/U_stage(T). (3)
```

The detector response `R(lambda)` multiplies the measured quantity. Whether
`R` is calibrated in photon counts or energy units must match the intensity
convention used to form the theoretical ratio.

### Same-upper-level branching pairs

For two transitions with the exact same upper level, `N_u` cancels:

```text
(Y_1/R_1)/(Y_2/R_2) = (A_1/lambda_1)/(A_2/lambda_2)
                     = (gA_1/lambda_1)/(gA_2/lambda_2).       (4)
```

This is the strongest ratio prior because it does not require a temperature,
partition function, ion fraction, or elemental abundance. The level identity
must be matched using ion stage, upper energy, configuration, term, and `J`;
similar energy alone is not a safe level key.

An absent strong branch can be a hard contradiction only when all of these are
true:

- the reference feature has an isolated, positive species-attributable lower
  amplitude bound;
- both branches lie in valid response-calibrated detector coverage;
- both are resolved from known competitors and from each other;
- neither is saturated, clipped, flat-topped, self-reversed, or otherwise
  optically thick;
- the stronger branch's lower predicted intensity bound, including transition
  probability and response uncertainty, exceeds the local 5-sigma area limit;
  and
- the measured upper limit is below that lower bound.

Even this is not an unconditional optical-depth test. Urbina Medina et al.
demonstrate cases in which a branching ratio remains close to the optically
thin value despite substantial self-absorption
([doi:10.1177/00037028211006764](https://doi.org/10.1177/00037028211006764)).

### Fine-structure and resonance “doublets”

The word *doublet* does not imply a fixed intensity ratio. K I
766.49/769.90 nm and Na I D are shared-lower-level ground-state resonance
pairs with different fine-structure upper levels. Their thin ratio is

```text
Q_12(T) = (R_1/R_2)
          [(gA_1/lambda_1)/(gA_2/lambda_2)]
          exp[-(E_u,1-E_u,2)/(k_B T)].                        (5)
```

The fine-structure energy difference is small, so the ratio is near 2 over
typical LIBS temperatures, but it is not an exact temperature-independent
branching ratio. Conversely, their absorption optical depths are related by
their shared lower-level populations and absorption oscillator strengths. The
emission ratio and absorption-strength ratio must remain separate inputs.

Use resonance-pair ratios to flag or estimate optical depth, never as a general
hard identity veto. A ratio driven toward unity can indicate saturation, while
an anomalously high ratio can indicate a blend, bad response correction, or a
wrong pairing.

## Optical depth and profile topology

The net LTE absorption coefficient contains both lower-level absorption and
stimulated emission. In relative form,

```text
kappa_j proportional to lambda_j^2 gA_j
  exp[-E_l,j/(k_B T)] f_stage/U_stage
  [1-exp(-hc/(lambda_j k_B T))].                              (6)
```

For a homogeneous slab, the repository uses the escape factor

```text
beta(tau) = [1-exp(-tau)]/tau,                                (7)
I_observed = I_thin beta(tau).
```

This is a useful approximation, not a unique radiative-transfer solution.
El Sherbini et al. experimentally define self-absorption through departure from
the thin intensity and show the resulting nonlinear line response
([doi:10.1016/j.sab.2005.10.011](https://doi.org/10.1016/j.sab.2005.10.011)).
Gornushkin et al. measured the curve-of-growth transition of the Cr I 425.4 nm
resonance line in steel plasmas
([doi:10.1016/S0584-8547(99)00004-X](https://doi.org/10.1016/S0584-8547(99)00004-X)).

Ground-state and low-lower-energy transitions deserve an optical-depth risk
flag. A cool absorbing periphery can flatten a line or remove its center,
yielding a self-reversed two-lobe profile. Such a profile has three triage
consequences:

1. merge the lobes into one physical feature for multiplicity counting;
2. mark its amplitude as a lower bound or model-dependent value; and
3. do not use its absence or amplitude as hard missing-line evidence.

A central dip is not unique proof of self-absorption; an unresolved blend,
detector artifact, or poor baseline can mimic it. Profile topology is therefore
a ranking/QC feature unless corroborated by a known low-level transition and
other lines of the same species.

## Missing stronger lines

The physically defensible missing-companion test begins with an isolated
reference line `r` whose species contribution has a positive lower bound
`Y_r^-`. For each plasma state `theta` in the allowed envelope, compute the
response-aware thin ratio `q_jr(theta)` and its lower uncertainty bound. Then

```text
Y_j^-(theta) = Y_r^- q_jr^-(theta).                           (8)
```

Companion `j` is contradictory only if

```text
min over allowed theta of Y_j^-(theta) >= 5 sigma_area,j      (9)
```

and its measured upper limit is below the same bound. Use at least two
independent companions in separate resolution elements. The initial temperature
envelope should cover the full optimizer range, 4000--25000 K, rather than the
warm start. Electron density is irrelevant for ratios within one stage but is
needed for cross-stage predictions.

Required eligibility masks are:

- in-band, sampled, and free of detector gaps or junction ambiguity;
- local area uncertainty measured from that window, with a nonzero count floor;
- unsaturated and not flat-topped/self-reversed;
- no supported competitor or unresolved blend;
- non-resonant (`E_lower` safely above the populated ground/low term);
- quantitative transition probability with an uncertainty bound;
- no NIST flags indicating uncertain identification, masking, perturbation,
  reversal, or unresolved structure; and
- no source-quality exclusion such as the repository's uncertain Se/Th/U
  oscillator-strength class.

NIST warns that ASD *relative intensities* are source dependent, often not
spectrally sensitivity corrected, may derive from photographic blackening, and
are qualitative. It also gives transition-probability uncertainty grades from
sub-percent through `E` (>50%) and flags blends, reversal, uncertainty, and
unresolved lines
([NIST ASD lines help](https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html)).
Use quantitative `A`, `gA`, or `gf` with its provenance; never use the ASD
relative-intensity column for a hard companion test.

## Ion-stage evidence and high-stage plausibility

Independent multi-line support in two stages of the same element is valuable:
it makes a chance identification less likely and can constrain the plasma state.
It is not required for detection. One stage can dominate, its useful lines can
fall outside the detector range, and different spatial/temporal plasma zones can
emit different stages.

Under LTE, adjacent stages satisfy the Saha relation

```text
n_(q+1) n_e / n_q =
  2 (2 pi m_e k_B T/h^2)^(3/2)
  [U_(q+1)(T)/U_q(T)] exp[-chi_q/(k_B T)].                    (10)
```

Use this only as a plausibility envelope. A suggested feature is the maximum
predicted fraction of the proposed stage over `T=4000--25000 K` and
`log10(n_e/cm^-3)=14--19`, plus the cumulative ionization energy required to
reach it. Very small fractions rank a high-stage candidate down. They do not
hard-reject it without an independently established LTE plasma state.

The current repository Saha implementation has additional hard-limit reasons:

- `SahaBoltzmann.partition()` sums only levels appearing as lower or upper
  states in the local transition table, not a complete evaluated level set;
- the ion chain assumes contiguous stages; and
- `LineTable.compute_saha_fractions()` indexes fractions as `ion_stage-1`.

Therefore a Saha-based early rejection is invalid for sparse/noncontiguous
stage coverage. `docs/atomic_data.md` already records the contiguous-low-stage
problem that motivated the Se/Th/U repair.

## Excitation coherence

For optically thin lines in one stage, define

```text
z_j = ln[Y_j lambda_j/(R(lambda_j) gA_j)].                    (11)
```

LTE predicts `z_j = constant - E_u,j/(k_B T)`. A robust line fit over at least
three independent upper levels yields an excitation temperature and a residual
scatter. Reward a plausible negative slope and small uncertainty-normalized
scatter; penalize positive slope, large scatter, or a temperature outside the
allowed envelope. This remains ranking evidence because non-LTE excitation,
optical depth, blends, and response error all bend a Boltzmann plot. Multiple
branches from one upper level validate branching behavior but supply only one
independent excitation level.

## Multiplicity and chance coincidences

Raw match counts strongly favor line-rich elements. Estimate the local observed
peak density `rho(lambda)` and match half-width `delta_j`; a simple local null
probability is

```text
p_chance,j = 1-exp[-2 rho(lambda_j) delta_j].                 (12)
```

Sum evidence over independent resolution elements, not database rows. Merge an
unresolved multiplet, a self-reversed lobe pair, and multiple transitions
matched to the same fitted peak. Downweight crowded wavelength regions and
candidate catalogs with many opportunities to match. Amato et al.'s automatic
LIBS ranking similarly weights a peak by intensity and inverse local database
line density rather than treating every coincidence equally
([doi:10.1016/j.sab.2010.04.019](https://doi.org/10.1016/j.sab.2010.04.019)).
Classical wavelength-coincidence work also emphasizes expected spurious results
and small-number failures
([Hartoog, Cowley & Cowley 1973](https://adsabs.harvard.edu/pdf/1973ApJ...182..847H)).

Report this as `coincidence_score`, `expected_random_matches`, or equivalent.
Do not call it a posterior probability without an explicitly calibrated null,
alternative model, prevalence prior, and held-out validation.

## Dedicated early Ar I and O I evidence

`alibz/gas_detection.py` measures common background-gas features before the
first whole-pattern indexer. It selects exact stage-I database rows with
positive quantitative `gA`, groups multiplet components into one independent
feature, applies the wavelength-shift model to raw native-grid windows, and
uses local sidebands and native-pitch-aware area noise.

NIST lists many persistent Ar I lines from 696.54 through 922.45 nm, including
763.51, 811.53, and 912.30 nm
([NIST persistent Ar I lines](https://physics.nist.gov/PhysRefData/Handbook/Tables/argontable3.htm)).
The detector uses 19 independent Ar I groups; the two 772.38/772.42 nm database
lines count as one group. Three clean groups are required for `detected`.

NIST lists O I components at 777.194, 777.417, and 777.539 nm and a separate
triplet at 844.625, 844.636, and 844.676 nm, with their level classifications
and `A` values
([NIST persistent O I lines](https://physics.nist.gov/PhysRefData/Handbook/Tables/oxygentable3.htm)).
The 777 nm components count as one independent group even if partially resolved.
O I requires two clean groups, such as 777 plus 844 nm, for `detected`; 777 alone
is `tentative`. Additional grouped anchors are near 615.6, 645.5, and 926.1--926.6
nm.

The detector deliberately imposes no multiplet intensity-ratio gate. O and Ar
excitation can be non-LTE, response varies across the detector, and optical
depth can change ratios. If no peak table is supplied, interference cannot be
fully checked and otherwise sufficient evidence remains `tentative`. A supported
competitor requires at least three independent strong-feature matches outside
the gas window in the same ion and local 100 nm band; a locally strong database
line coinciding with the gas window then marks that group blended. This prevents
strong UV lines from hiding a plausible NIR interferer and prevents the target
peak from corroborating its own alternative assignment.

Expected purge gas is contextual prior information, not observed evidence.
Absence of Ar I does not prove absence of argon: timing, excitation, ionization,
and coverage can suppress neutral lines. Atomic O I does not distinguish oxygen
from the sample, ambient air, purge impurity, or plasma chemistry, and it does
not establish O2 or concentration. The ordinary indexer remains responsible for
other ion stages and material attribution. No gas signal is subtracted from the
reported material fractions by this early detector.

## Real-data checks and limits

### MW2-112 K resonance pair

The frozen 929-shot MW2-112 result contains 928 non-empty shots with positive,
SNR >= 5 integrated areas for both K I 766.49 and 769.90 nm. The measured area
ratio 5th/50th/95th percentiles are 1.35127/1.44061/1.52738 versus the thin
value 2.006045; 926/928 ratios lie outside a +/-20% thin band. The two features
remain spatially coherent (`rho=0.9938`), and the independent weak K I 693.876
nm feature correlates with them at `rho=0.757--0.767`.

This supports resonance pairs as useful identity and optical-depth diagnostics:
a rigid thin-ratio veto would reject a strongly corroborated K signal. It does
not establish self-absorption as the unique cause. The check reuses one measured
corpus and has no independent certified composition, blanks, or replicate
shots, so it does not validate concentration or a universal threshold.

### Native-grid Fe calibration

The dedicated detector was checked on native 10-shot mean
`run-e825d9f567204c5cbc6fc33c9c16bbf7` from the 26-run Fe Aesar 99.98%
session (`~0.18 nm` NIR pitch). A truth-assisted calibration sanity check using
the recorded `-0.154 nm` shift and a raw prominence-10 peak table reports Ar I
`tentative` (one supported 696 nm group, also blend-flagged) and O I
`no_evidence`. A separate blind production-peak replay of the first three run
means reports `no_evidence` for both gases in all three. These outcomes do not
contradict the prior qualitative observation of visible argon features: the
detector deliberately requires quantitative local 5-sigma area support and
independent groups. This is reused Fe-session data, not independent certified
gas validation. Threshold calibration still requires
held-out argon/no-argon and oxygen/no-oxygen spectra across native grid families,
gate timings, purge conditions, and matrices.

## Decision table

| Feature | Early reject? | Recommended use |
|---|---|---|
| No eligible wavelength match in supplied peak list | Conditionally | Defer that candidate for this calibrated peak list; do not infer chemical absence |
| >=2 absent non-resonant companions, each forced above local 5 sigma over full parameter envelope | Yes, when every eligibility mask passes | Reject with per-line upper-limit audit |
| Absent same-upper branch | Sometimes; same masks plus optical-thin evidence | Strong contradiction or large penalty |
| Ordinary doublet ratio | No | Rank; estimate/flag optical depth |
| Flat, split, or self-reversed resonance profile | No | Merge lobes, flag saturation, exclude amplitude from hard evidence |
| Same-element support in two ion stages | No | Strong rank bonus; plasma-state diagnostic |
| Saha-improbable high stage | No | Rank penalty and warning |
| Boltzmann excitation coherence | No | Rank bonus/penalty |
| One isolated line | No | Tentative support only |
| Multiple matches in independent, uncrowded resolution elements | No by itself | Rank bonus with chance correction |
| Expected purge/air species | No | Context annotation; still require measured evidence |

## Audit payload

Every triage result should retain:

- candidate element and ion stage;
- original and post-triage status;
- rule/version and reason codes;
- matched feature groups and unique observed peak identifiers;
- expected and measured wavelengths, calibration uncertainty, and match windows;
- measured area, local area uncertainty, coverage/gap masks, response source,
  saturation/profile flags, and competitor elements;
- atomic-data source, transition-probability accuracy/flags, and level keys;
- extrema of every predicted companion over the parameter envelope; and
- separate named score components rather than one undocumented scalar.

Preserve the untriaged candidate list so false rejections can be measured on
standards and replayed when atomic data or thresholds change.
