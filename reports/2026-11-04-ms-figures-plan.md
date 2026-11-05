# Manuscript figures from real data — survey, selection, plan

Date 2026-11-04 (coordinator). Request: survey the available LIBS data, select representative
datasets for the manuscript, and make a few figures that show the pipeline applied to real data,
stage by stage, in the spirit of `notebooks/pipeline_walkthrough.ipynb` and
`notebooks/fit_inspection.ipynb`. The manuscript currently has **no figures** (`manuscript/figures/`
is empty; no `\includegraphics` in any section).

## 1. Survey of LIBS holdings (as reachable on 2026-11-04)

| set | where | reachable now | n | grid | in manuscript today |
|---|---|---|---|---|---|
| MW2-112 | Drive `Postdocs/Xuan Cao/Data/LIBS/MW2-112/raw` | yes (mounted) | 929 single shots, IDs 989–1917 | vendor 1/30 nm, 23 431 pts | yes: production gate; shot 1000 and spectrum 0 fit-fidelity numbers; id 1518 timing |
| JChristensen | Drive `CMI/Data/LIBS/JChristensen/2026-07-01` | yes | 38 AverageSpectrum + an old run's `summary.csv`/`detections.csv` (2026-07) | vendor 1/30 nm | yes: Mn/Mg confounder, Be/Fe mass guard, stimulated-emission A/B, Alaskire K collapse (memory) |
| Remote samples | `data/remote_samples/` (gitignored: `.gitignore` line 30 `data/`) | yes | 5 (REE_01, REE_44, scan9x9, user_spec1, argon_noAr) | vendor 1/30 nm | yes: thermometer paragraph; fit-panel real panel (user_spec1 r² 0.92, scan9x9 0.83) |
| Fe standard (Z300 native) | `data/triage-real/fe_aesar_9998_means_20260922.npz` (26 run means) + per-shot on moissanite pantheum (127 acquisitions) | yes (npz local; pantheum reachable) | 26 runs / 252 shots | native 7 914 px, needs the 18-px shift (`fit_quality_panel.load_fe`) | yes: calibration section; fit-panel Fe panel |
| V standard | pantheum only; `cal/data/*ledger*.json` | yes (remote) | run1 + run2 (59 batches) | native | yes: timing study |
| Z300 corpus (4 909) | `moissanite:/media/mwhittaker/Corpus_One*/All_LIBS_Till_20260319` | **NO** — the Corpus_One drive is not attached (no block device; only Samsung_T5 and robodata present; `/media/mwhittaker/Corpus_One` is a root-owned empty mount point) | 4 909 | vendor | yes: 620 nm response fallback (numbers already frozen in `corrections/detector.json`) |
| 20250529LIBSAuto | Drive `Postdocs/Xuan Cao/Data/LIBS/` | yes | 1 462 acquisitions (Average + Spectrum1 each) | vendor | no; no provenance in repo |
| mapping_01 | same | yes | 1 513 positions, multiple shots each (6 090 csv) | vendor | no; no provenance in repo |
| LIBS_data_paper_revision | same | yes | 3 samples (white / round / sharp), 6 CSVs | vendor | no |
| NETL | Drive `CMI/Data/LIBS/NETL` | yes | a few 2024-03 exports | ? | no |

The last four are usable but undocumented (no acquisition notes, no sample identity in the repo),
so they are not selected: a showcase figure must be traceable to a dataset the manuscript already
describes in Section 5.

## 2. Selection (one spectrum per role; each is already named in the manuscript text)

| key | spectrum | role in the figures | why this one |
|---|---|---|---|
| `mw2_1000` | MW2-112 shot 1000 (`1000-Test #1000 10 38, …csv`) | stage-by-stage walkthrough; refinement zooms (K I 766.5 phantom split, Li I 670.8 SA tag) | production-gate case; the manuscript already quotes its χ²/σ² and residual counts (07-results "Fit fidelity on MW2-112") |
| `jc_mdd017_24_15_2nd_1` | JChristensen "MDD017-24-15 2nd #1 Kaleb" | confounder contest: Mn 0.52 as-fit on two peaks inside the Mg II doublet → confounded(Mg) → resolved | the archetype of the Mn/Mg confound quoted in 07-results (old summary.csv: Mn 0.52, flags `Mn:confounded(Mg)`, r² 0.82) |
| `jc_alaskire` | JChristensen "Alaskire MDD27 box 26" | K-dominant, `dominant-weak-shape`: self-absorbed K I 766/770 doublet; segment-response sensitivity | the basin-collapse / K-doublet case in DECISIONS and memory (K 0.76→0.45 when the 620 nm step was wired) |
| `user_spec1` | `data/remote_samples/user_spec1.csv` | full-spectrum overview (data, background, model, residual/σ, three segments) | best-fit real item on the panel (r² 0.92); the one real spectrum a reader can re-run from the panel harness |
| `fe_fresh_d5p25` | Fe standard run index 17 = `run-07c25af6dd9b4fa6bc801e18a40661a4` (raster fresh site, delay 5, period 25) | known composition: Fe I / Fe II Saha–Boltzmann stage tie at the fitted (T, nₑ); composition bar (Fe 0.924 on the adopted defaults, one non-Fe species) | fresh site (no crater), well recovered |
| `fe_reused_fail` (optional) | run index 9 = `run-674f4feb6d2149c4bac365e395a4d327` (reused spot, delay 5, period 25) | counterexample: Fe fraction 0.000, r² −0.80 on the same defaults | shows what a failed native-grid fit looks like; use only if a panel has room |

## 3. Figures (target: five)

1. **F1 overview** — `user_spec1`: raw + background / model + peak marks / residual in σ, segment edges at 365 and 620 nm (= `plot_spectrum_overview`).
2. **F2 walkthrough** — `mw2_1000`: one or two windows across stages (blind → 3a → seed/recover/deblend → final), each row data−bg, stage model, residual/σ; right column: composition evolution (stacked bars per solve: pass 1, pass 2, deepening, SA recovery) and (T, log nₑ) per solve.
3. **F3 refinement decisions** — `mw2_1000`: K I 766.5 (phantom split merged) and Li I 670.8 (asymmetry tagged, not split): symmetric vs asymmetric component and residual RMS in the window (the manuscript quotes 101 vs 19 and 227 vs 59 on spectrum 0; the figure's own numbers go in the caption).
4. **F4 confounder contest** — `jc_mdd017_24_15_2nd_1`: Mg II 279.55/280.27 window with Mn and Mg predicted patterns and the independent Mg I 285.2 line; bars of as-fit vs resolved fractions for Mn and Mg.
5. **F5 known standard** — `fe_fresh_d5p25`: Saha–Boltzmann plot of fitted Fe I and Fe II line areas vs upper-level energy at the fitted T with the stage tie, plus the composition bar and (T, log nₑ).

## 4. Implementation

- **Stage capture in the production chain** (specialist): `analyze_spectrum(..., capture_stages=False)`;
  when True, `analysis["stages"]` holds ordered snapshots (schema in the brief). OFF by default and
  proven output-identical on a real spectrum. `scripts/fit_inspector.capture()` is a drifted copy of
  the chain (legacy overlap mode, no partition/IPD options, no SA recovery, no detections) and is
  **not** used for the figures.
- **Showcase data manifest + loader** (worker): `manuscript/data/showcase.json`,
  `manuscript/data/README.md`, `scripts/showcase_data.py` (sha256-verified loads; Drive root override).
- **Figure script** (specialist, after capture lands): `scripts/manuscript_figures.py` → cached
  analyses under `manuscript/figures/cache/` (gitignored) and `manuscript/figures/F*.pdf|png`;
  figure environments + captions with `% src:` comments added to the sections; `make` compiles.
- Pipeline settings for every figure: `analyze_spectrum` defaults (n_calls 40, grid search, adopted
  defaults conserve/barklem2016/debye). Numbers quoted in captions come from the cache, not from
  older reports.

## 5. Verification (coordinator)

Figures viewed; caption numbers cross-checked against the cache JSON; capture no-op proof re-run;
suite; `make` clean; STATUS/DECISIONS updated. Nothing committed (owner).

## Log
- 12:45 plan written; agents launched (w4-capture, w4-data).
- 13:00 w4-data done and coordinator-verified: `scripts/showcase_data.py --check` OK 6/6 (rc 0, no PYTHONPATH needed); manifest
  Fe delay/period/index equal the npz metadata (5/25/17 fresh, 5/25/9 reused) — upgraded from "taken from the plan";
  moissanite Corpus_One absence confirmed independently; paper_revision count corrected (3 samples, 6 CSVs).
  Staged copies in data/showcase/ (gitignored). Report reports/2026-11-04-ms-showcase-data.md.
- 13:35 w4-capture done and coordinator-verified: `_snap` gated on `capture_stages` (pipeline.py 747, 1497-1498); 13 stage keys; tests/test_capture_stages.py 7 passed (44 s, coordinator re-run); specialist no-op proof on user_spec1 PASS (coordinator re-run in progress); full suite started in background. Patch reports/patches/2026-11-04-w4-capture.patch. w4-figs (specialist) launched with the brief in scratchpad/w4-figs-brief.md.
- 13:50 coordinator re-run of the user_spec1 no-op proof: VERDICT PASS (True minus "stages" == False; 14 snapshots incl. 2 deepen bars). Full suite still running.
- 14:40 full suite on the capture tree: 691 passed, 10 skipped, 2 xfailed (21 min; = 684 + 7). w4-figs first pass verified: cache consistent with the fit panel (Fe fresh run T_rec 22.3 kK, ne at H-alpha bound, Fe 0.924 on the panel; 25/26 Fe runs at 20-25 kK), make clean (54 pp), captions honest. Rework sent (scratchpad/w4-figs-rework.md): F2 model hidden under data + exaggerated T axis + tick collisions; F3 clipped peaks; F4 contest does not fire on the 2026-07 archetype under current defaults -> JChristensen 38-spectrum survey launched (data/jchristensen_20261104, CLI, conserve) for re-selection; F5 reframed as the two-thermometer discrepancy with the cal Fe I VIS Boltzmann fit overlaid. 05-datasets corrected (remote_samples is gitignored, not committable).
- 15:05 JChristensen current-defaults survey done (CLI, conserve, n_calls 40, draws 32, 6 workers): 38/38 ok, 397 detection rows, median r2 0.782; Mn confounded(Mg) on 6 spectra (as-fit 0.032-0.102 -> resolved 0.000; top: MDD017-29-2c #1 Kaleb 0.102), Mn upper-limit on 10, blended-only 6, marginal 4, weak 3, single-line 2, detected 1. The 2026-07 40-53% hot-Mn cases no longer occur. Products in data/jchristensen_20261104/ (w4-figs copies them to reports/jchristensen-20261104/).
- 15:30 CLOSED: second pass verified (all 5 PNGs viewed; 18 caption numbers vs cache incl. F2 0.019 Al, F3 RMS/dBIC, F4 survey 38/6/10/6/0.78 and Mn 0.102->0 Mg 0.033->0.065, F5 112 Fe II / 22601 K / Fe I 10 lines 11.5+-0.4 kK); coordinator fixed F2 top-window tick labels and F1 legend frame in scripts/manuscript_figures.py and re-rendered from cache; make clean 54 pp; DECISIONS + STATUS + memory updated. Nothing committed.
