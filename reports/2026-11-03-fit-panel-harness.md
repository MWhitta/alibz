# Fit-quality A/B panel harness (2026-11-03)

## Built
- `scripts/fit_quality_panel.py` (new): `run` and `compare` subcommands; console entry `fit-quality-panel = "scripts.fit_quality_panel:main"` added at `pyproject.toml:45` (not installed in the venv; run with `.venv/bin/python -m scripts.fit_quality_panel ...`).
- `tests/test_fit_quality_panel.py` (new, 6 tests, fakes for analyze_spectrum/render_case): scoring, error rows + JSON row schema, aggregate, Fe dead-channel masking, compare sign conventions, missing-item listing.
- Not committed.

## Run
```
.venv/bin/python -m scripts.fit_quality_panel run --variant NAME [--config-json '{"weighted_solve": true}'] --workers 4 [--panel synthetic,real,fe] [--items binary_ca_mg,REE_01] [--out PATH] [--timeout-s 900] [--fe-pixel-offset 18]
.venv/bin/python -m scripts.fit_quality_panel compare BASE.json VAR.json --md out.md
```
Defaults: n_calls 8, draws 2, seed 11, db `db`, out `reports/fit-panel/<variant>-<stamp>.json`. Items ids: synthetic case names, real CSV stems, Fe run ids. `--config-json` is merged verbatim into `analyze_spectrum` kwargs.

## Design notes
- Workers: one spawned process per item (up to `--workers`), BLAS thread env vars set to 1 at module top before numpy import (children inherit). Per-item SIGALRM repeating itimer (the pipeline's pattern) plus a hard kill at timeout+60 s and a worker-died check; all produce error rows, never abort. `--workers 1` runs inline (used by tests).
- Real-panel flags/qc_status/qc_reasons come from `alibz.pipeline._summary_row` (private function; wrapped in try/except). `ne_at_bound` is my derived metric: |log_ne - bound| < 0.05 against `analysis["ne_bounds"]`. basin keys from `analysis["basin_info"]`.
- JSON: `variant, created, git_head, config, settings, items[{panel,id,status,metrics,error}], aggregate` as specified. Compare gives per-item headline-metric rows (better/worse/same), aggregate counts, and a verdict per panel; items missing from either file listed first. n_detected is informative only (no direction).
- Fe handling (decision to review): the npz is 7914 samples; samples 5848+ are the dead 4th channel (zeros; wavelengths jump 947.9 -> 960.2 nm; one stray non-zero at the last sample). I cut at the >5 nm wavelength gap and trim trailing zeros -> 5848 samples. ALSO the npz (2026-09-22) predates the Z300 PIXEL_OFFSET -18 fix (axis ~18 px too blue). Run raw, Fe was never detected in any of the 26 runs (JSON `baseline-20261103-rawfeaxis.json`). I therefore apply by default a +18 sample shift (`--fe-pixel-offset 18`, label of sample i+18 for sample i, last 18 samples dropped); set 0 for raw. This is an approximation at the 365/620 nm seams. A properly re-derived corrected npz would supersede it (the CORR_ROOT used by cal/ scripts lives in a vanished /private/tmp scratchpad, not in the repo).

## Tests
- Before: `ALIBZ_SKIP_ROUNDTRIP=1 .venv/bin/python -m pytest -q` at HEAD with my new test file not yet written-in-effect: 609 passed, 9 skipped (525 s). 
- After (both invocations, ALIBZ_SKIP_ROUNDTRIP=1, full tests/): venv: 636 passed, 9 skipped; PYTHONPATH=src+site-packages python3: 636 passed, 9 skipped. No failures. (Round-trip tests were skipped, not run; the +27 vs the before count includes my 6 tests plus tests added/collected differently at that moment; the earlier run overlapped with my file creation so the baseline count is approximate.)

## Baseline
JSON `reports/fit-panel/baseline-20261103.json`; summary `reports/2026-11-03-fit-panel-baseline.md`. 36 items, 4 workers, 902 s wall (single-item times: synthetic 130-320 s, real 37-274 s, Fe 25-43 s). One error row: synthetic/iron_rich_low_t (`ValueError: target stage abundances must sum to one, got 0.99999998884`, render). Synthetic and real panels were bit-identical between two independent runs.

## Unverified
- Console script not installed/tested via entry point.
- ne_at_bound threshold 0.05 dex unvalidated (all 5 real spectra flag it).
- Fe +18 sample shift is approximate; Fe baseline numbers are only as good as that.
- Timeout/hard-kill/worker-died paths were not exercised by a test (only the error-row path via exception).
- Round-trip test file not run (ALIBZ_SKIP_ROUNDTRIP=1).

## CAUTION: concurrent edits to source during the baseline
`git status` shows alibz/peaky_indexer_v3.py (mtime 17:05), alibz/pipeline.py (17:13), alibz/utils/sahaboltzmann.py (17:23), db/partition_barklem2016.npz (17:12), db/atomic_line_sources.json modified/new that I did NOT touch (another session). Run 1 (raw Fe axis) finished 17:05; the final baseline ran ~17:06-17:21, and each item spawns a fresh process, so items started after 17:12-17:13 imported the edited pipeline/db. Synthetic and real metrics were bit-identical between run 1 and the final run, which suggests the edits did not change them, but the baseline is NOT guaranteed to correspond to HEAD 620cc22. Recommend re-running the baseline on a quiet tree (about 15 min) before using it for A/B. The test runs (17:21-17:36) also overlapped sahaboltzmann.py edits.
