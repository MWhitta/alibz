# Showcase data manifest + loader (2026-11-04)

Files: scripts/showcase_data.py, manuscript/data/showcase.json, manuscript/data/README.md,
data/showcase/*.csv (local, gitignored: 5 staged). No file under alibz/ touched.

## Commands that produced numbers
- `ls <JC dir> | wc -l` -> 41; `find <JC dir> -name '*AverageSpectrum.csv' | wc -l` -> 38; non-Average: detections.csv, fit_inspection.ipynb, summary.csv
- `ls MW2-112/raw | wc -l` -> 929; `find MW2-112 -name '*.csv' | wc -l` -> 1856
- find csv: 20250529LIBSAuto 2918, mapping_01 6090, LIBS_data_paper_revision 6 (plan said 3), NETL 4
- `ls data/remote_samples` -> 5 files; npz run_ids -> 26
- JC name matches: exactly one each (MDD017-24-15 2nd #1 Kaleb_20260617_110533_AM, Alaskire MDD27 box 26_20260614_083522_AM)
- `PYTHONPATH=src .venv/bin/python scripts/showcase_data.py --build | --stage | --check | --list`

## --check output (after --stage), exit 0
OK mw2_1000 n=23431 / OK jc_mdd017_24_15_2nd_1 n=23431 / OK jc_alaskire n=23431 /
OK user_spec1 n=23431 / OK fe_fresh_d5p25 n=5830 / OK fe_reused_fail n=5830
(First --stage wrote Fe csv with numpy repr "np.float64(...)", which the loader could not parse; fixed with float(); re-staged; the final check is the one above.)

## Per-key stats (from manifest; source file sha256 prefix)
| key | size B | sha256 | n | wl min-max nm | max counts | neg frac | 1.4826*MAD |
|---|---|---|---|---|---|---|---|
| mw2_1000 | 868361 | 2ae55d99fc | 23431 | 180.0-961.0 | 8070.5 | 0.0006 | 45.16 |
| jc_mdd017_24_15_2nd_1 | 879540 | 68d8344a49 | 23431 | 180.0-961.0 | 7149.5 | 0.4352 | 38.45 |
| jc_alaskire | 869679 | c6d0c8199e | 23431 | 180.0-961.0 | 8068.3 | 0.0608 | 40.94 |
| user_spec1 | 865564 | f551977f85 | 23431 | 180.0-961.0 | 12911.9 | 0.0008 | 33.45 |
| fe_fresh_d5p25 | 1267240 (npz) | f0c0abe693 | 5830 | 187.784-947.936 | 12632.2 | 0.0492 | 116.21 |
| fe_reused_fail | 1267240 (npz) | f0c0abe693 | 5830 | 187.78-947.952 | 12386.8 | 0.1638 | 73.05 |

Anomaly: jc_mdd017_24_15_2nd_1 has 43.5% negative samples (offset/background-subtracted export); the noise MAD is still sane.
Fe staged-csv loads are bit-identical (np.array_equal x and y) to `fit_quality_panel.load_fe(run_id, pixel_offset=18)` for both runs.

## Drive-file full presence
All three Drive CSVs and the MW2 file were read to the end twice (sha256 over all bytes in --build, then again on --stage/--check; load_spectrum_csv parsed all 23431 rows). No lazy-sync failure. Note: --check now reads the staged copies, so it no longer exercises Drive; Drive was read in --build/--stage only. Error path tested: missing file with ALIBZ_SHOWCASE_ROOT=/nonexistent and staged copy absent -> "[jc_alaskire] file not found: <path>"; unknown key -> clear error. A sha256 mismatch path is coded but not exercised.

## Moissanite
`ssh -n moissanite 'ls -la /media/mwhittaker/; lsblk ...'`: /media/mwhittaker has Corpus_One (root, drwx------, Mar 24 2026), OS (nvme0n1p3), Samsung_T5 (root, drwx------, empty mount point), Samsung_T51 (sda1, Samsung_T5 label). robodata (sdb1) at /media/xc/robodata; sdc 0B. No device for Corpus_One: plan confirmed (could not list its contents as it is root-only).

## Unverified
- Mismatch branch; ALIBZ_SHOWCASE_ROOT success path (only the failure path with a bad root was run).
- V standard on pantheum not checked. The Fe delay/period/site fields are copied from the plan, not re-derived from the npz.
- `--check` after staging does not touch Drive (see above).
