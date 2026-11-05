# Native-grid fit_peaks window gate / width filter fix

Date: 2026-11-05. Tier: specialist. Branch: fix/z300-pixel-offset (working tree
has many uncommitted changes that are NOT mine; I touched only
`alibz/peaky_finder.py` and added `tests/test_native_gate.py` +
`tests/fixtures/z300_fe_native_trimmed.json`). Nothing committed.

Starts from reports/2026-11-04-fe-hot-basin-root-cause.md (sections 1,3,4).

## TL;DR

The two grid-regime defects are fixed and the lost lines are recovered:
`fit_peaks` 52 -> 288, blind-fit table 88 (72/5/11) -> 243 (125/85/33), and the
five strong Fe I VIS lines (404.58/426.05/430.79/438.35/440.48 nm) now reach the
table. Export data is **byte-identical** (proven). **But the gate fix does not
cure the hot basin**: the full pipeline on fe_fresh_d5p25 stays hot (22.6 kK ->
20.9 kK) and now mis-assigns the recovered VIS peaks to Na — Fe fraction falls
0.93 -> 0.12, Na rises to 0.81. The Fe-panel headline means regress. The fix is
mechanically correct and necessary (it restores physics the finder was deleting)
but is **not sufficient**; the next problem is in the indexer/solve, not the
finder. See "Decision for judgment".

## The two defects and the fixes

### Defect A — the window gate (the discard)
`fit_peaks` seeded a detected peak only inside `if len(x_window) > 5:` (old
line 924) with no else-branch, so any narrower window was silently dropped. On
the native grid (per-segment pitch 0.089/0.129/0.179 nm, LSF FWHM ~1.4/1.4/1.65
px) a resolution-limited line spans 1-2 samples and the minima-to-minima window
is 3-4 samples. Traced on fe_fresh (scratch gate/trace.py, gate/trace2.py): all
six target lines are detected by filter_peaks (284 survivors) but have windows
of len 3-4 -> all discarded; fit_peaks keeps 52, exactly reproducing the report.

### Defect B — the per-channel width filter (the second loss)
`fit_spectrum` filtered widths with `value[2] < key_inc/2 -> value[2]=0`,
`value[3] < key_inc/2 -> value[3]=0`, then `both==0 -> drop` (old lines
1495-1504). `_voigt_seed` splits the measured FWHM into EQUAL Gaussian and
Lorentzian components, so a line at the instrument FWHM (1.4 px) seeds
sigma≈0.36 px, gamma≈0.43 px — BOTH below key_inc/2 (0.5 px) -> both zeroed ->
dropped. Trace (gate/trace2.py) confirms: even with Defect A fixed, 404.58,
426.05, 438.35, 440.48 die here (sigma/gamma = 0.36/0.43 pitch); 430.79 and
526.95 survive (wider fwhm estimate). Exactly the four lines the brief's gate>2
scratch test still lost.

### The physical rule that replaced the sample count
A real emission line cannot be narrower than the instrument LSF. The LSF FWHM
is a per-segment physical constant (`alibz.utils.peakfit.instrument_sigma_px`,
UV/VIS 1.4 px, NIR 1.65 px). The fix expresses retention against that FWHM, not
against a raw sample count:

- `fit_peaks` (peaky_finder.py): the `> 5` branch is unchanged — it is a
  *numerical* requirement of the cumulative-integral sub-peak split, which needs
  several samples to run, not a detection threshold — so every window that ran
  the search before still runs the identical code. A new `elif _pf_native:`
  branch seeds the single detected peak directly (as the fast path seeds
  in-window sub-peaks), flooring its FWHM at the instrument LSF
  (`_instrument_fwhm_px`). The silent discard — the actual defect — is gone.
- `fit_spectrum` (peaky_finder.py): on the native grid the per-channel
  half-pixel gate is replaced by a test on the COMBINED Voigt width vs the
  instrument FWHM: drop only if `voigt_width(sigma,gamma) <
  NATIVE_MIN_FWHM_FRACTION (0.5) * instr_fwhm_nm` (a sub-resolution spike) or a
  channel exceeds the unchanged ultra-broad `100*median_fwhm` junk cap. Fitted
  widths are kept (both channels carry a narrow line).

Both new paths are gated on `segment_pitch_array(x).is_native`, which is **False
on every uniform / 1/30 nm vendor-export grid**, so the export code path is the
old code, byte-for-byte.

### Downstream-filter trace (constraint 3)
| line (nm) | killed by gate A | killed by filter B | after fix |
|---|---|---|---|
| 404.58 | yes (len 3) | yes (sig/gam 0.36/0.43 px) | PRESENT |
| 426.05 | yes (len 3) | yes | PRESENT |
| 430.79 | yes (len 4) | no | PRESENT |
| 438.35 | yes (len 3) | yes | PRESENT |
| 440.48 | yes (len 3) | yes | PRESENT |
| 526.95 | yes (len 4) | no | PRESENT |
The `min_height` (2σ) significance gate (peaky_finder.py ~1561) is **left
unchanged** on both grids: it is physical (implied height vs local point noise),
grid-independent, floors widths at key_inc/2 for the height calc, and all six
lines pass it (45-109 σ). The `100*median_fwhm` ultra-broad cap is kept on both
grids. No other width/height filter dropped the four lines. The `_arpls`
`if n<5` (line ~339) the brief mentioned is in the whole-spectrum baseline, not
the per-peak seeding path — left alone.

## Changes (file:line, post-edit)
- `alibz/peaky_finder.py:19` — import `instrument_sigma_px as _instrument_sigma_px`.
- `alibz/peaky_finder.py:46-62` — `_FWHM_PER_SIGMA`, `NATIVE_MIN_FWHM_FRACTION=0.5` constants + rationale.
- `alibz/peaky_finder.py:78-89` — `_instrument_fwhm_px(wavelength)` classmethod (LSF FWHM in native px).
- `alibz/peaky_finder.py:~1082-1110` — `elif _pf_native:` narrow-window direct-seed branch in `fit_peaks`.
- `alibz/peaky_finder.py:~1551-1580` — native-vs-export split of the width filter in `fit_spectrum`.
- `tests/test_native_gate.py` — 4 regression tests (native rescue + narrow-window regime + uniform-grid no-op).
- `tests/fixtures/z300_fe_native_trimmed.json` — 601-sample native fixture (UV 345-366 + VIS 398-445 nm slices of fe_fresh; is_native=True; 11 KB).
Patch of the source + test: reports/patches/2026-11-05-native-gate.patch (fixture is a data file, not inlined).

## Export byte-identity (constraint 2)
`PeakyFinder.fit_spectrum` re-run on the two export caches' x,y vs the cached
`blind_fit` stage `sorted_parameter_array`:
- user_spec1 (n=23431, is_native=False): shape (170,4) == (170,4), `np.array_equal`=True, max|Δ|=0.0
- mw2_1000 (n=23431, is_native=False): shape (303,4) == (303,4), `np.array_equal`=True, max|Δ|=0.0
Byte-identical, as guaranteed by the `is_native` gate.

## Blind-fit chain, fe_fresh_d5p25 (native)
| step | before | after |
|---|---|---|
| find_peaks | 1024 (346/331/347) | 1024 |
| filter_peaks | 284 (130/108/46) | 284 |
| fit_peaks | 52 (32/6/14) | 288 |
| blind_fit table | 88 (72/5/11) | 243 (125/85/33) |
All six Fe I VIS lines present in the final table.

## Tests
Targeted files (brief list), venv pytest, ALIBZ_SKIP_ROUNDTRIP unset:
`test_peaky_finder_fast test_peaky_finder_accuracy test_native_grid
test_peak_location test_capture_stages test_pipeline` -> **98 passed, 1 skipped,
7 subtests passed** (92 s).
New regression file: `tests/test_native_gate.py` -> **4 passed**.
Full suite (`pytest tests/`, ALIBZ_SKIP_ROUNDTRIP=1, 1241 s): **695 passed, 10
skipped, 2 xfailed, 71 subtests passed**. Baseline was 691 passed / 10 skipped /
2 xfailed; the +4 are the new `test_native_gate.py` tests. No regressions.
  NB: a stray duplicate copy of the suite under
  `data/jchristensen_20261104/provenance/untracked/tests/` (untracked, not mine)
  pollutes rootless `pytest` collection with import-file-mismatch; confining to
  `tests/` (default import mode) collects cleanly.

## Fit-quality panel (moissanite, 32 workers, --n-calls 40 --draws 8)
Trees: ~/alibz-gate-base (peaky_finder.py = git HEAD) vs ~/alibz-gate (fix). db
symlinked to ~/alibz/db; data/triage-real + data/remote_samples rsynced.

### Fe panel (fast look, 26 runs) — REGRESSED
Headline means (base -> fix):
| metric | better/worse/same | base mean | fix mean |
|---|---|---|---|
| fe_fraction (+) | 8/18/0 | 0.5625 | 0.2865 |
| non_fe_mass (-) | 8/18/0 | 0.4375 | 0.7135 |
| n_non_fe (-) | 1/21/4 | 1.077 | 2.346 |
| abs_T_dev_10kK (-) | 18/7/1 | 1.283 | 1.158 |
Verdict (repo rule): **regressed** (35 better vs 64 worse comparisons). Only
abs_T_dev_10kK improved. CAVEAT: this panel's npz
(data/triage-real/fe_aesar_9998_means_20260922.npz) uses an APPROXIMATE
+18-sample axis shift (FE_PIXEL_OFFSET=18); newly-recovered lines land at
approximate wavelengths, which plausibly inflates the non-Fe mis-assignment.
The properly-registered manuscript run (below) is the cleaner test and shows the
SAME direction, so the regression is real, not only an axis artifact.

### Full production panel (synthetic+real+fe, 36 items) — fe REGRESSED, rest UNCHANGED
Local copies: reports/fit-panel/gate-base-prod.json, gate-fix-prod.json,
2026-11-05-gate-compare-prod.md.
| panel | metric | better/worse/same | base mean | fix mean |
|---|---|---|---|---|
| synthetic | median_rel_err / spurious_mass / abs_T_rel_err / n_majors_missed / dominant_correct | 0/0/5 (all) | 0.170 / 0 / 0.077 / 0 / 0.8 | identical |
| real | r_squared / median_stage_disagreement / n_detected / ne_at_bound | 0/0/5 (all) | 0.489 / 0.489 / 11.6 / 1 | identical |
| fe | fe_fraction (+) | 8/18/0 | 0.5625 | 0.2865 |
| fe | non_fe_mass (-) | 8/18/0 | 0.4375 | 0.7135 |
| fe | n_non_fe (-) | 1/21/4 | 1.077 | 2.346 |
| fe | abs_T_dev_10kK (-) | 18/7/1 | 1.283 | 1.158 |
Verdict (repo rule): **synthetic — no net change; real — no net change; fe —
regressed (35 better vs 64 worse).** Overall FAILS the "every headline mean
improves, no paired count worse than ~2/5 or 11/26" rule, on the fe panel.
The synthetic and real panels being EXACTLY unchanged (every item, every metric)
at production settings independently confirms export/uniform-grid byte-identity:
those spectra are is_native=False, so the fix is a no-op there. The whole net
effect of the fix is confined to native Fe data.

Per-run Fe panel (fast fe json, 26 runs): mean T_rec 22 830 K -> 21 448 K
(both hot); mean fe_fraction 0.563 -> 0.287; mean n_detected 2.3 -> 4.0. The fix
roughly doubles the number of detected ELEMENTS, and on the approximate-axis npz
the surplus detections scatter into non-Fe species. The panel JSON does not carry
per-run Fe I/Fe II or VIS peak counts; I did not re-run 26 approximate-axis
pipelines to synthesize them (cost/scope) — the clean Fe I/Fe II/VIS breakdown is
the manuscript fe_fresh run below (Fe I 140, Fe II 108, VIS 87).

## Manuscript re-analysis, fe_fresh_d5p25 (local, properly registered)
| quantity | OLD (brief) | NEW (after fix) |
|---|---|---|
| T (K) | 22 601 | 20 929 |
| log nₑ | 17.69 | 17.655 |
| r² | 0.346 | 0.4206 |
| table size | 135 = 119/5/11 | 271 = 151/87/33 |
| Fe I / Fe II / unassigned | 0 / 112 / 21 | 140 / 108 / 15 |
| Fe fraction | 0.93 | 0.119 |
| element fractions | Fe 0.93 / H 0.07 | Na 0.809 / Fe 0.119 / Ga 0.051 / H 0.019 |
| stage_tie (Fe disagreement) | 0.0 | 0.970 |
| Fe I VIS Boltzmann T_B | 11 467 K (r² 0.99) | 11 467 K (r² 0.992, n=10, span 1.25 eV, resolved=False) |
fe_reused_fail (context): T 20 860 K, r² 0.625, Fe 0.581 / Ga 0.338 / H 0.081.
F5 re-rendered (fig F5 -> F5-standard.pdf/.png). I did NOT edit manuscript/sections/.

Reading: the fix did exactly what section 4 predicted at the finder level — Fe I
is back in the table (0 -> 140 assigned) and the VIS fills (5 -> 87 peaks) — and
the stage-tie thermometer now has a live Fe disagreement (0.97). But section 7's
hope ("the pipeline MAY then land near the Fe I value without new machinery")
did NOT materialise: the solve stays hot (20.9 kK) and the extra VIS peaks are
claimed by Na (0.81), so Fe fraction collapses. The Fe I VIS Boltzmann
thermometer is unchanged at 11.5 kK (it is computed independently of the table).

## Decision for judgment (I did not widen scope)
The gate/filter fix is correct, verified, and export-safe, and it removes a real
information-destroying bug in the finder. It is a PREREQUISITE for any cold-Fe
solution (Fe I cannot be solved for if its lines never reach the table). But on
its own it REGRESSES the headline Fe composition metrics, because the indexer's
solve still prefers a hot, now Na-dominated basin for the richer table. The open
question — ship the finder fix now (correct physics, defer the basin/assignment
problem to the indexer work) vs hold it until the solve can use the recovered Fe
I lines — is a solve/indexer design decision above this task's scope. Evidence is
all above; the next lever is the indexer (stage-tie / line-evidence prefilter /
Na-vs-Fe assignment at the hot basin), not the finder.

## Commands
- trace: scratch gate/trace.py, gate/trace2.py (venv python, OMP=1)
- tests: `.venv/bin/python -m pytest <files> -q`
- export identity: scratch inline (array_equal vs cached blind_fit)
- panels: `OMP/OPENBLAS/MKL=1 ~/alibz/.venv/bin/python -m scripts.fit_quality_panel run --variant <tag> --workers 32 --n-calls 40 --draws 8 --out ...`; compare via same CLI
- manuscript: `python -m scripts.manuscript_figures analyze --keys fe_fresh_d5p25 fe_reused_fail` then `fig F5`

## Provider
Claude subscription throughout (no usage-limit switch).

## Unverified / caveats
- Full production panel verdict: PENDING (see above).
- Full local suite count: PENDING.
- Fe-panel npz axis is approximate (+18 px); real-pipeline run corroborates the direction.
- The native width filter now passes un-snapped sub-pixel sigma/gamma downstream on native data; no test regression seen, but the indexer's sensitivity to those widths was not separately audited.
