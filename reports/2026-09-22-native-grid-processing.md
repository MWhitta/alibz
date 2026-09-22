# All acquisition processing on the detector's native grid — 2026-09-22

## Outcome

Patched and tested in the Pantheum source; **deployment and the native re-fetch
are the user's steps** (`scripts/deploy-alibz-native-grid.sh`, then
`scripts/recover-alibz-awaiting-data.sh --refetch --run <ids> --apply`). No
physical command; analyzer access was read-only GETs; no provider switch.

## Audit: where the pipeline left the native grid

| Stage | Before | After |
|---|---|---|
| `Acquisition._sorted_unique_csv` (shots + average CSVs, dataset import) | `pixels_to_wavelength` (native, knot-trimmed) then `resample_uniform` to 1/30 nm (23,250 pts) | native grid, duplicate knot sample dropped, strictly increasing (7,914 pts), no interpolation |
| `test.json` / `manifest.json` | no grid record | `grid: native` |
| `optimization_metrics._line_measurement` | fixed ±0.12 nm line window, 0.25–0.5 nm sidebands, needs ≥3 samples each; on native pitch (0.089/0.129/0.179 nm UV/VIS/NIR) that is 1–3 samples and a 3-sample window makes any off-centre peak an "edge peak" | same nm windows, widened to ≥`MIN_CENTRAL_PX`=5 (±2 samples around the nearest sample) and ≥`MIN_SIDEBAND_PX`=3 with a 1-sample gap; unchanged on the 1/30 nm grid |
| `analyze_batch` result | no grid provenance | `metrics.grid = {points, median/min/max_step_nm, min_central_px, min_sideband_px, uniform}` |
| alibz engine (`pipeline.load_spectrum_csv`, `profiles`, `peak_window_pca`) | takes any strictly increasing grid; pitch = `median(np.diff(x))`; no 1/30 nm assumption in the analysis path (only `peaky_maker` synthetic corpora use 1/30) | unchanged; now receives native input |
| `pantheum.alibz.spectra.parse_spectrum` | requires strictly increasing, 2 columns | unchanged; native passes |
| `z300_calibration.resample_uniform` | used by acquisition | kept for tests/exports only |

Native grid facts from a real shot (test d45a1971…): 7,914 samples; UV 2,027 px at
0.078–0.096 nm, VIS 1,985 px at 0.113–0.142 nm, NIR 1,836 px at 0.156–0.198 nm,
plus the 960–961 nm stub. NIR coverage ends ~12 nm short of the 960 nm knot,
so the native grid has a real ~12 nm gap there that the 1/30 nm resampling had
been filling by linear interpolation.

## Real-data check (test 63f26855…, ten shots, read-only fetch)

| Grid | Metrics code | score | quality lines | median SNR | CV |
|---|---|---|---|---|---|
| resampled 1/30 nm | live (deployed) | 0.28640 | 12 | 12.26 | 0.128 |
| resampled 1/30 nm | new | 0.28640 | 12 | 12.26 | 0.128 |
| native, 3-sample window (rejected) | new | 0.2335 | 9 | 15.8 | 0.132 |
| native, 5/3-sample windows (shipped) | new | 0.4222 | 18 | 12.04 | 0.140 |

The new code reproduces the deployed score bit-for-bit on the resampled grid.
On the native grid the wider ±2-sample line window keeps lines whose peak sits
one sample off the reference wavelength (e.g. Fe 355.49 nm, SNR 26, 10/10
detections on the resampled grid, 0/10 with a 3-sample window). Native and
resampled scores are **not comparable**: the completed batch must be re-scored
natively before the session proposes from it (`--refetch`).

## Changes

- `pantheum/alibz/acquire.py` (live before `143b3d29…` → `aa6a9750…`): native
  CSV writer, `GRID = 'native'` recorded, `resample_uniform` import removed.
- `pantheum/alibz/optimization_metrics.py` (live before `e01eb1cf…` → `68d4d485…`):
  minimum-sample windows, grid provenance.
- Tests: `tests/test_alibz_acquire.py` (native CSV keeps the per-sample pitch;
  synthetic calibration given cubic curvature), `tests/test_alibz_optimization_metrics.py`
  (0.18 nm uneven grid measured with the minimum-sample windows; grid fields).
- `DECISIONS.md`: 2026-09-22 entry.
- `scripts/recover-alibz-awaiting-data.py --refetch`: re-fetches a succeeded run,
  re-stores it natively (new dataset id; old dataset kept), re-scores its
  completed batch and refreshes the session proposal/best.
- `scripts/deploy-alibz-native-grid.sh`: manifest-pinned deploy of the two modules.

## Verification

- Focused: `test_alibz_optimization_metrics`, `test_alibz_acquire`,
  `test_alibz_optimization`, `test_alibz_checkouts`: 113 tests OK.
- Full suite: `reports/2026-09-22-native-grid-full-tests-v3.log` **785 tests in 151.340 s, OK (skipped=26)**.
- Pre-edit module hashes reconstructed by reversing the edits equal the live
  runtime hashes, so the pinned deploy cannot clobber a concurrent change.

## Order of operations for the user

1. `scripts/deploy-alibz-native-grid.sh` (dry run), then `--apply`.
2. `scripts/recover-alibz-awaiting-data.sh --refetch --run run-eae9d54f22834928ae2f6af8d4a23779 --run run-e825d9f567204c5cbc6fc33c9c16bbf7 --run run-a50ac639cc7747b4af81a677c8e99312 --apply`
   (re-stores the three tests natively; re-scores the session batch; expect
   score ≈0.42, 18 lines, proposal recomputed).
3. Subsequent batches are native end to end.

## Caveats

- The parallel Opal ingest work decodes the instrument DB bundles with
  `scripts/z300_fb_decode.py` (`stitch`: half-open `[lo, hi)` per segment,
  PIXEL_OFFSET −18). `pixels_to_wavelength` trims inclusively on both ends.
  Both should yield the same native grid; the seam samples deserve one
  cross-check before the two archives are treated as identical.
- Old datasets `2679219a…`, `4f678d79…`, `0982aede…` (resampled) remain in the
  portal; `--refetch` points the runs at new native datasets.

## Applied by the user, 10:01 PDT

- `deploy-alibz-native-grid.sh --apply` at 10:01:32: installed hashes matched
  (`aa6a9750…`, `68d4d485…`), backup `~/pantheum-fire-fix-backup-20260922T100132`,
  both services active.
- `recover-alibz-awaiting-data.sh --refetch --apply` for the three runs at
  10:01:44 (backup `~/pantheum-recover-backup-20260922T100144`): each re-stored
  with 7,914 native samples per shot (`test.json` `grid: native`); new datasets
  `c0c80c93…` (d45a1971…), `9e26b731…` (afb5b8b9…), `d7cb3589…` (63f26855…);
  old resampled datasets retained but no longer referenced by the runs.
- Session `opt-b177bed6…` batch re-scored natively: score 0.422242, grid
  {points 7914, median step 0.0954 nm, max step 12.23 nm (NIR coverage gap)};
  session `ready`, best = 10/25 @ 0.4222, proposal delay 5 / period 25.
- Post-check 10:0x PDT: `live_allowed=true`, data API reachable, gantry Idle,
  checkout held, both services active.
