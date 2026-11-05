# Fe standard hot basin: root cause, and the VIS Fe I thermometer proposal

Date: 2026-11-04 (evening). Trigger: owner comment on Fig. F5 ("the Fe VIS
thermometer looks very good; the VIS response is probably more linear than
the UV; perhaps the VIS thermometer should be used when enough lines exist").
Scope: diagnosis only. Nothing in production was changed; one scratch copy of
`alibz/peaky_finder.py` was patched for a mechanism test (see 4).

Inputs: `manuscript/figures/cache/fe_fresh_d5p25.pkl` (production run with
`capture_stages=True`, adopted defaults), `fe_reused_fail.pkl`, export caches
`user_spec1.pkl`, `mw2_1000.pkl`. Scratch scripts (session scratchpad
`/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/7c311a86-0d91-4217-8224-5b49f94b239b/scratchpad/`):
`fixed_t_resolve.py`, `cold_species.py`, `cold_species2.py`, `prefilter_test.py`,
`fe1_weights.py`, `fe1_restored.py`, `peaky_finder_patched.py`.

## 1. What the final solve actually contains (fe_fresh_d5p25)

Final peak table (deepen_final): 135 peaks = UV 119, VIS 5, NIR 11.
Assignments: Fe II 107 (UV) + 5 (NIR); Fe I 0; H I 2; unassigned 21.
The 21 unassigned peaks include the strongest Fe I lines in the spectrum:
344.07 (567 counts), 356.54 (452), 357.01 (1179), 358.12 (1170), 360.98 (1334),
361.86 (561), 364.78 (248), 365.09 (845) nm. The five VIS peaks are 365.09,
368.20, 417.71, 418.86, 588.99 nm.

Raw spectrum, VIS Fe I thermometer lines (y − bg peak; finder noise 21 counts):
404.58 nm 2238 (109 σ), 426.05 nm 917 (45 σ), 430.79 nm 1043 (51 σ),
438.35 nm 1825 (89 σ), 440.48 nm 1206 (59 σ). None is in the peak table.

Fe II Boltzmann points of the pipeline solve (figure-script convention),
fitted within subsets:
| subset | n | E_k span (eV) | slope (1/eV) | implied T |
|---|---|---|---|---|
| all 112 | 111 | 4.8–13.6 | −0.201 | 58 kK |
| UV only | 106 | 4.8–13.6 | −0.162 | 72 kK |
| E_k ≥ 7.4 eV | 92 | 7.4–13.6 | −0.047 | flat (≈ no Boltzmann population) |
| E_k < 7 eV | 19 | 4.8–5.9 | +1.93 | positive (strong resonance-connected lines sit low) |
Reading: 92 of the 112 "Fe II" peaks are explained by high-lying Fe II forest
lines whose amplitudes do not follow any Boltzmann distribution; they are the
Fe I UV lines (248–305, 340–365 nm) and others claimed by the forest at the hot
state. The 22.6 kK is the compromise that makes the forest available.

## 2. Fixed-state re-solves on the current table

Harness: rebuild `PeakyIndexerV3` exactly as `analyze_spectrum` does for the
deepening re-solve (same frame, amp_sigma, noise floor, conserve/barklem2016/
debye, ne prior from H-alpha) and call `solve_at`. Reproduction at the pipeline
state: r² 0.346, Fe 0.932 / H 0.068 (cache: 0.3457, 0.932 / 0.068).

| state | r² | Fe fraction | Fe I assigned | note |
|---|---|---|---|---|
| 22 601 K, 17.69 (pipeline) | 0.346 | 0.93 | 0 | reproduced |
| 11 467 K, 17.09 / 17.39 / 17.69 | 0.094 / 0.094 / 0.090 | 0.04 / 0.04 / 0.25 | 0 | H/Cu/Ag/Ir/Hg junk |
| 9 619 K, any | 0.000 | 0.11–0.14 | 0 | Cu/Ag/Ir junk |
| 8 036 K, any | −0.088 | 0.93–0.98 | 0 | |
With the line-evidence prefilter disabled (`evidence_top_k=0`) so Fe I stays
in the design: 11 467 K r² 0.264 (Zn/P/H junk, Fe I 33 peaks), 9 619 K 0.176,
8 036 K 0.088. Conclusion: on THIS peak table the VIS temperature cannot be
imposed; the amplitude pattern (forest + missing VIS) genuinely prefers the hot
state. The defect is upstream of the thermometer.

## 3. Why Fe I is not in the design at a cold state

`LineTable` holds Fe I (2367 lines, 1281 in 365–620 nm; Saha fraction 0.80 at
8 036 K, 0.10 at 11 467 K, 1.5e-4 at 24 370 K). Building the candidate matrix
at init 8–11.5 kK removes Fe I; `_prefilter_species_by_initial_strength` is not
the culprit (drops 31 species, none Fe); `_prefilter_species_by_line_evidence`
is (Fe I survives only with `evidence_top_k=0`). At a cold state Fe I's
strongest predicted lines are the near-UV/VIS ones (372–440 nm) that are absent
from the peak table, so Fe I fails the multi-line support test. At the
pipeline's hot pass-2 init (24 370 K) Fe I survives the prefilter but receives
no assignments. The stage-tie thermometer therefore has a single Fe stage with
support and no vote (`stage_tie_cost` 0.0 in the final result).

## 4. Where the VIS peaks are lost: the fit_peaks window gate

Blind-fit chain on fe_fresh (counts UV/VIS/NIR):
| step | total | UV | VIS | NIR |
|---|---|---|---|---|
| find_peaks (local maxima) | 1024 | 346 | 331 | 347 |
| filter_peaks (> 3 σ local) | 284 | 130 | 108 | 46 |
| fit_peaks (fast seeding) | 52 | 32 | 6 | 14 |
| full fit_spectrum table | 88 | 72 | 5 | 11 |
`fit_peaks` seeds a detected peak only inside
`if len(x_window) > 5:` (alibz/peaky_finder.py:924), where the window is the
span between the neighbouring minima from `peak_parameter_guess`; there is no
else-branch, so a narrower window is silently discarded. On the native Z300
grid the windows are 3–4 samples: ≤ 5 samples for UV 85 %, VIS 96 %, NIR 76 %
of detections (fe_reused_fail: 84 / 94 / 92 %). On vendor exports (0.033 nm
pitch) the gate rarely bites: user_spec1 16 / 2 / 0 %, mw2_1000 9 / 0 / 0 %,
and the blind-fit tables keep 85–100 % of detections. The UV later recovers
peaks through fit_shoulders/fit_all and the pipeline's recover stage (which
seeds lines of ESTABLISHED ions, i.e. Fe II only), the VIS does not: the loss
is self-reinforcing through the Fe I pruning in 3.

Mechanism test (scratch copy, gate `> 2`): fit_peaks 258 (118 / 96 / 44); full
blind-fit table 157 (92 / 40 / 25); strong Fe I lines present 3/11 → 5/11
(430.79 and 526.95 nm added; 404.58, 426.05, 438.35, 440.48 still dropped by
later width/height filtering). The gate is the first and dominant loss, not the
only one; a real fix needs the small-window seeding path designed (seed from
`peak_parameter_guess` directly, as the fast path already does for the window
sub-peaks) and the downstream filters checked on native data.

## 5. The VIS Fe I thermometer itself

Same spectrum, cal-script method (`build_thermo_set` / `robust_boltzmann`,
side-band areas at database positions, independent of the peak table):
| set | clip | n | span (eV) | T_B (K) | r² |
|---|---|---|---|---|---|
| Fe I 420–620 | 1.6 σ (cal default; F5) | 10 | 1.25 | 11 467 ± 367 | 0.99 |
| Fe I 420–620 | 2.5 σ | 25 | 3.35 | 7 663 ± 529 | 0.90 |
| Fe I 420–620 | 3.5 σ | 26 | 3.35 | 8 036 ± 664 | 0.86 |
| Fe I 420–620 | none | 27 | 3.35 | 11 289 ± 2 711 | 0.41 |
| Fe I 365–620 | 1.6 σ | 17 | 1.25 | 9 619 ± 504 | 0.96 |
| Fe I 365–900 | 1.6 σ | 16 | 2.98 | 10 636 ± 397 | 0.98 |
| Fe II 186–365 (cal set) | 1.6 σ | 8 | 1.30 | 7 995 ± 9 500 | 0.11 |
The F5 value (11.5 ± 0.4 kK, r² 0.99) is the product of shedding 17 of 27
lines; across clipping choices the Fe I thermometer spans 7.7–11.5 kK. The
365–900 nm set is the one that meets the campaign's own resolution criterion
(n ≥ 8, span ≥ 2.4 eV, r² ≥ 0.5): 10.6 ± 0.4 kK statistical, with a systematic
spread of about ± 2 kK from self-absorbed strong lines and the uncalibrated
within-segment response tilt. Every variant is far below 22.6 kK.

## 6. Detector response: what exists and what the data show

The pipeline corrects only the 620 nm gain step (`correct_segment_response`,
edges=(620,)); on this run the step was not measurable (continuum too weak) and
the corpus fallback 3.9 (q25–q75 3.0–5.2) was used. There is no within-segment
response curve anywhere in the chain. The 2026-09-24 calibration report already
states: within-segment Boltzmann slopes are largely immune to the response, the
cross-segment Fe II/Fe I ratio (hence Saha nₑ) is not, and a lamp calibration
is the remedy. The within-UV trend of ln(obs/pred) measured here (+2.6 per
100 nm below 260 nm) is dominated by the forest mis-assignment and is NOT
usable as evidence for a UV roll-off. Direction check: an uncorrected
short-wavelength roll-off would make high-E_k (short-λ) lines look weak, which
biases a Boltzmann T COLD; it cannot be what drives the hot basin.

## 7. Assessment of the proposal

- A within-segment, line-selected thermometer is sound and is what the cal
  campaign already recommends; it is immune to the cross-segment response and
  to the Fe II forest. It should enter the outer search as a T prior/box
  (analogous to the H-alpha nₑ box), measured at database positions with
  side-band areas (not from the peak table), gated on n ≥ 8, E_k span ≥ 2.4 eV,
  r² ≥ 0.5; available for line-rich neutrals (Fe, Ti, Cr, Ni, V, Mn), i.e.
  most Fe-bearing rocks, not for Ca/Mg/Si/Al/K/Na-only spectra.
- It would not fix F5 on its own: on the current table the solve at 11.5 kK is
  worse (r² 0.09). The first fix is the native-grid window gate (section 4),
  which should restore the VIS peaks, keep Fe I in the design, and give the
  existing stage-tie thermometer its Fe I vote; the pipeline may then land
  near the Fe I value without new machinery. Then the VIS thermometer becomes
  an independent check/prior, and its agreement with the stage-tie T is the
  test of the response assumption.
- The F5 caption currently frames the hot solve as a "thermometer discrepancy";
  after the gate fix it should be re-derived and the text updated.

## Unverified
- The gate's effect on the other 24 Fe panel runs (same grid, same expected
  behaviour; not re-run). The downstream filters that still drop 404.58/426.05/
  438.35/440.48 nm after the gate patch were not traced. No full-suite or panel
  run was made; production code is unchanged.
