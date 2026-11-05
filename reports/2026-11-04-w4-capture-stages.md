# W4 — opt-in per-stage capture in `analyze_spectrum`

2026-11-04. Branch `fix/z300-pixel-offset`. Adds `capture_stages` to
`alibz.pipeline.analyze_spectrum` so manuscript figures come from the REAL
production chain, not the drifted `scripts/fit_inspector.py::capture()`.

## Scope / result

Additive, gated change. When `capture_stages=False` (the default) the returned
dict is byte-for-byte identical (no `"stages"` key). When `True` the dict gains
one key `"stages"`: a list of 14 snapshots in documented pipeline order, each
decoupled from later mutation. Verified on a synthetic case (pytest) and on a
real spectrum (no-op proof). All related tests pass. No behaviour change.

## Files changed (line ranges are in the CURRENT working tree)

### `alibz/pipeline.py` — all additions below carry the sentinels
`_snap / stages / capture_stages / _STAGE_ORDER / _copy_fit / _snap_copy /
_stage_fractions / _solve_i / _seg_shift_returned / _reg_payload`, so they are
grep-identifiable within the file's larger (pre-existing) working-tree diff.

- L86 `import copy`
- L131 `from alibz.peaky_indexer_v3 import FitResult as _FitResult`
- L311 `AnalysisConfig.capture_stages: bool = False` (+ docstring L306–310)
- L569–575 `_STAGE_ORDER` map (documented stage order for the return-time sort)
- L577–592 `_copy_fit` (shallow dict copy; every ndarray value `np.array(copy=True)`)
- L594–611 `_snap_copy` (arrays copied, FitResult by reference, lists/dicts deep-copied)
- L613–636 `_snap(stages, *, stage, label, fit=None, n_peaks=None, **payload)` — no-op when `stages is None`
- L638–657 `_stage_fractions` (per-species element/stage/concentration rows)
- L697 `capture_stages: bool = False` added to `analyze_spectrum` signature
- L747–749 `stages = [] if capture_stages else None`; `_solve_i = 0`
- L757 blind_fit snapshot
- L783–784 refine_data snapshot
- L826 `_seg_shift_returned = shift` (raw per-segment shift before reassignment)
- L931–940 registration snapshot (after gas calibration, so the registration/
  calibration products ARE available and are included)
- L1093–1101 pass1 snapshot (+`_solve_i += 1`)
- L1154–1155 refine_physics snapshot
- L1165–1176 seed snapshot (skipped-branch included)
- L1181–1182 recover snapshot
- L1196–1197 deblend snapshot
- L1253–1264 pass2 snapshot (+`_solve_i += 1`)
- L1360–1388 two deepen per-bar snapshots (collapse branch + accepted branch)
- L1401–1409 deepen_final snapshot
- L1421–1425 sa_recovery snapshot
- L1460–1464 detections snapshot
- L1466 `return dict(...)` → `out = dict(...)`; L1495–1498 stable-sort + `out["stages"] = stages`; L1499 `return out`
- L1774 `capture_stages=cfg.capture_stages` forwarded at the config call site (was L1773)

### `alibz/cli.py` — NOT MODIFIED by this task
`cli.py` has no `analyze_spectrum(` call (it goes through `analyze_directory` →
`AnalysisConfig`), so there was nothing to forward there. Its working-tree diff
(argparse flags) is PRE-EXISTING, unrelated quant-fix-program work, and is
excluded from the patch.

### `tests/test_capture_stages.py` — new (7 tests)

## Snapshot schema as implemented (one line per stage, in list order)

1. `blind_fit`  "1 · blind fit" — `fit`, `n_peaks`
2. `registration` "2 · wavelength registration" — `shift` (raw segment shift
   exactly as returned by `estimate_wavelength_shift_segments`), `shift0`,
   `n_anchor`, `n_shift_anchor`, `wavelength_registration`,
   `wavelength_calibration` (last two included because the snapshot is taken
   after they are computed), `n_peaks`
3. `refine_data` "3a · data-only refinement" — `fit`, `decisions` (dec_data), `n_peaks`
4. `pass1` "4 · pass-1 solve" — `fit`, `result` (res1), `T`, `log_ne`, `sigma`,
   `gamma`, `element_fractions`, `stage_fractions`, `r_squared`, `established`,
   `solve_index`, `n_peaks`
5. `refine_physics` "3b · self-absorption tags" — `fit`, `decisions` (dec_phys), `sa_zones`, `n_peaks`
6. `seed` "5 · Boltzmann-seeded minor lines" — `fit`, `records`, `n_peaks`
   (+`skipped`/`reason` when gated out)
7. `recover` "6 · residual recovery" — `fit`, `recovered`, `n_peaks`
8. `deblend` "7 · shoulder deblends" — `fit`, `deblends`, `profiles_pre`, `n_peaks`
9. `pass2` "8 · pass-2 solve" — `fit`, `result` (res2), `T`, `log_ne`, `sigma`,
   `gamma`, `element_fractions`, `stage_fractions`, `r_squared`, `basin_info`,
   `solve_index`, `n_peaks`
10. `deepen` "9 · iterative deepening" — one PER re-solved bar: `fit`, `result`
    (resN), `bar`, `n_added_seed`, `n_added_recover`, `T`, `log_ne`,
    `element_fractions`, `accepted`, `collapsed`, `solve_index`, `n_peaks`
11. `deepen_final` "9 · iterative deepening" — `fit` (carried `final`), `result`,
    `T`, `log_ne`, `element_fractions`, `accepted`, `reason`, `solve_index`, `n_peaks`
12. `sa_recovery` "10 · self-absorption area recovery" — `result`,
    `element_fractions`, `sa_records`, `sa_used`, `profiles`, `n_peaks`
13. `detections` "11 · detections" — `detections`, `resolved_fractions`,
    `contested`, `support`, `element_uncertainty`, `n_peaks`

Decoupling: `fit` dicts are shallow-copied with every ndarray value
`np.array(copy=True)`; decision/record lists and plasma-state dicts are
`copy.deepcopy`-ed; `FitResult`s are stored BY REFERENCE (confirmed never
mutated after creation — `recover_sa_areas` returns a fresh dataclass, the
deepen re-solves build new `resN`, `analyze_detections` does not mutate the
result). The documented scalars (`T`, `log_ne`, etc.) are copied out at capture
time, so they stay correct even if a later stage replaced `result`.

### Notes on brief ambiguities (resolved, not widened)
- FitResult exposes no normalised per-stage fraction or z/sigma; `stage_fractions`
  therefore stores what EXISTS — one row per `result.species` with `element`,
  `stage` (ion), and the aligned `concentration` from `result.concentrations`.
- `solve_index` is attached only to the solve snapshots the brief names
  (pass1=0, pass2=1, deepen bars=2.., deepen_final=last), NOT to `sa_recovery`
  (the brief's sa_recovery payload omits T/log_ne), keeping test-2's invariant
  "every solve snapshot has T/log_ne/element_fractions" clean.
- Stage append happens at each stage's finish (execution order: blind,
  refine_data, registration, ...), then the list is stable-sorted to the
  documented order (registration before refine_data per its "2"/"3a" labels).

## Tests (verbatim summary lines)

New file (`ALIBZ_SKIP_ROUNDTRIP=1 .venv/bin/python -m pytest tests/test_capture_stages.py -q`):

    .......                                                                  [100%]
    7 passed in 44.16s

Related existing + new (`... -m pytest tests/test_pipeline.py tests/test_peaky_indexer.py tests/test_capture_stages.py -q`):

    ........................................................................ [ 86%]
    ...........                                                              [100%]
    83 passed in 54.84s

(Test files that exist: test_pipeline.py, test_peaky_indexer.py — plus
test_oracle_roundtrip.py, test_peaky_finder_accuracy.py, test_peaky_finder_fast.py,
test_peaky_pca.py, test_roundtrip_synthetic.py, which the coordinator runs in
the full suite.) I did not have a pre-change baseline run (no stash/checkout
allowed), but the change is additive and gated on `capture_stages=False`, and
the existing tests exercise only that path; they pass.

Test timing lever: a full `analyze_spectrum` run is ~100 s regardless of
`n_calls` (dominated by the pass1/pass2/deepen/SA solves), so the two in-test
runs use the suite-S `binary_ca_mg` scene CROPPED to 370–530 nm (both Ca and Mg
present) — `search="grid"` (deterministic, as the brief asked), `n_calls=2`,
`draws=1`, `gp_seed=11`. Each cropped run is ~21 s; the file runs in 44 s.

## Real-spectrum no-op proof (report-only; not a test)

Script: `/private/tmp/claude-501/-Users-mwhittaker-Projects-github-alibz/7c311a86-0d91-4217-8224-5b49f94b239b/scratchpad/capture_noop_real.py`
(`OMP/OPENBLAS/MKL_NUM_THREADS=1`; `data/remote_samples/user_spec1.csv`,
23431 pts 180–961 nm; `n_calls=8, draws=2, gp_seed=11`).

    runtime  False=49.0s  True=47.2s
    top-level key sets equal: True
    True-minus-stages == False: True  (first diff: root)
    n_snapshots=14
    VERDICT: PASS

The 14 snapshots and their keys matched the schema above exactly (2 deepen bars
on this spectrum). So on a real production spectrum the True result minus
`"stages"` is byte-equal to the False result.

## Could not verify
- Full 30-min suite (coordinator runs it) — only the pipeline/indexer subset here.
- `stage_fractions` z/sigma + normalised per-stage fraction: not present on
  FitResult, so not stored (documented above).

## diff --stat (working tree, for the task's files)
`git diff --stat -- alibz/pipeline.py`:

    alibz/pipeline.py | 367 ++++++++++... (includes PRE-EXISTING quant-fix changes
                      |     interleaved with this task's ~160 added lines)

New test file: `tests/test_capture_stages.py` (~230 lines).
`alibz/cli.py` unchanged by this task (its 17-line working-tree diff is pre-existing).

## Patch
`reports/patches/2026-11-04-w4-capture.patch` = `git diff -- alibz/pipeline.py`
(this task's additions + pre-existing pipeline.py changes) concatenated with
`git diff --no-index /dev/null tests/test_capture_stages.py`. cli.py excluded
(untouched by this task).
