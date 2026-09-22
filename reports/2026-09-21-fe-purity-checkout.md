# Fe purity checkout (Z300, 4x4 grid x 3 reps) — 2026-09-21

**Provider:** Claude subscription (`sub`, opus[1m]); no provider switches.
**Status:** Data recovered and analysed. CF-LIBS quantification failed its own
QC and is **not** reported; conclusions rest on direct line measurements.

## Headline

The target behaves like clean Fe once the surface layer is gone. Every
non-Fe species with real signal is either **surface contamination that
ablates away** (Na, K, Li, Ca, Cu) or **ambient air in the plasma** (O, H).
**No alloying or bulk impurity element was detected in any of the 47
shots** — no C, Si, Mn, Al, Mg; Cr, Ni and Ti only at the detection floor.
That is consistent with the nominal 99.98% purity.

Repeatability is dominated by surface state, not by the instrument:

| | rep 1 | rep 2 | rep 3 |
|---|---|---|---|
| Fe line area (median) | 4 483 | 6 330 | 6 794 |
| **Fe line area RSD** | **52.7%** | **26.4%** | **14.1%** |
| peak intensity RSD | 49.6% | 25.2% | 13.3% |

Fe signal rises 52% and its shot-to-shot scatter falls almost 4x from first
to third exposure. The instrument's own precision floor on this target is
therefore **~14% RSD single-shot**, and anything worse is the surface.

## Contamination, and that it cleans up

Contaminant line area as a ratio to the Fe reference area, median per rep:

| species | rep 1 | rep 2 | rep 3 | change | r vs Fe |
|---|---|---|---|---|---|
| K I | 0.0236 | 0.0056 | 0.0028 | **−88%** | −0.71 |
| Cu I | 0.0102 | 0.0045 | 0.0032 | −69% | −0.36 |
| Na I | 0.2120 | 0.1145 | 0.0846 | −60% | −0.59 |
| Ca II | 0.0231 | 0.0124 | 0.0105 | −55% | −0.41 |
| Li I (detected shots) | 0.0198 (7/15) | 0.0022 (6/16) | 0.0005 (2/16) | −97% | −0.36 |
| O I | 0.1264 | 0.1563 | 0.1350 | +7% | **+0.94** |
| H I | 0.0530 | 0.1232 | 0.1111 | +110% | **+0.86** |

Two groups fall out cleanly, and the `r vs Fe` column is what separates them:

* **Surface layer** — Na, K, Li, Ca, Cu are *anti*-correlated with Fe
  (r = −0.36 to −0.71). Where the contaminant is thick the matrix signal is
  suppressed, which is exactly how an overlying layer behaves, and all five
  decline monotonically across reps.
* **Ambient, not sample** — O I and H I *track* Fe (r = +0.94, +0.86): they
  scale with plasma strength, i.e. entrained air and humidity. They do not
  clean up and should not be read as sample contamination.

Species disappearing outright: Ca I 4/15 → 0 → 0, Mg II 2/15 → 0 → 0,
Li I 7/15 → 6/16 → 2/16.

The dirtiest site is stark: `rep1_pos01` has Na/Fe = 2.4 with Fe area 463,
against 0.04 and ~7 000 for the cleanest rep-3 shots — a factor ~15 in Fe
signal between the worst first-exposure site and a cleaned one.

## Purity limits

No detection in any of 47 shots for C, Si, Mn, Al, Mg I. 3-sigma upper
limits on line area relative to the Fe reference area (median over shots,
strongest clean line per species):

| species | 3σ UL (area/Fe) | detected |
|---|---|---|
| Mn I | 0.0019 | 0/47 |
| Ti II | 0.0020 | 1/47 |
| Al I | 0.0028 | 0/47 |
| Cr I | 0.0030 | 5/47, SNR ≤ 3.5 |
| Cu I | 0.0046 | 38/47, SNR ≤ 9.7 |
| Mg II | 0.0089 | 2/47 |
| Ni I | 0.0095 | 38/47, SNR ≤ 4.1 |
| Si I | 0.0130 | 0/47 |
| Mg I | 0.0180 | 0/47 |
| C I | 0.0864 | 0/47 |

**These are line-area limits, not concentrations.** Converting them to mass
fractions needs relative sensitivity factors, which is precisely what the
CF-LIBS inversion would have supplied had it converged. Do not quote them
as ppm.

Ni and Cr deserve a caveat: both sit at the detection floor (SNR 4.1 and
3.5 against a 3σ threshold). I checked whether they were Fe lines leaking
into their windows — Fe I 352.617 is only 0.16 nm from Ni I 352.454 — but
neither tracks Fe (r = 0.03 and 0.12), so it is not simple interference.
They are best read as *not established*: too weak to claim, not excluded.
Cu is the one marginal species with a coherent story (SNR 9.7, declines 69%,
anti-correlates with Fe), which points at surface Cu from handling or the
holder rather than bulk alloy.

## Why the CF-LIBS numbers are not reported

`alibz-analyze` ran all 47 spectra to completion (47/47 ok, 23 elements,
269 detection records) on beryl with 24 workers. It then failed its own
gates:

* `qc_status`: **fail on 46/47**, warn on 1, pass on none
* r² median **0.42**, minimum **−0.72** (worse than a flat line)
* `electron-density-at-bound` on 43/47; `log_ne` railed at the 18.1 bound
* `physical-pattern-r2` on 27, `dominant-weak-shape` on 11,
  `composition-collapse` on 5; `T_K` railed at the 25 000 K bound on one
* `response_620_source`: **fallback on all 47** — no session response
  calibration for this instrument/day, so the corpus 3.9x stand-in was used
* `global-only-wavelength-shift` on 43/47 — the segment-wise shift stage
  never engaged

The composition it produced is the known basin-collapse signature and is
plainly wrong: Fe 0.13 median on a 99.98% Fe target, with Ar 0.84, Ag 0.65,
P 0.57. Reporting any of it would be reporting an artefact. This matches
[[alibz-basin-collapse-guard]] and the response/shift blockers already on
record.

The obvious suspect was S/N — these are **single-shot** spectra
(`numShotsToAvg = 1`), far noisier than the averaged spectra the pipeline is
tuned on. That was tested directly and ruled out; see the next section.

## Averaging tested — hypothesis falsified

The 47 single shots were stacked into 15 averaged spectra (3 per-rep averages
of 15/16/16, plus 4 quarter-group averages per rep of 3-4 shots) and refitted.
Averaging did **not** rescue the inversion:

| | single-shot (47) | averaged (15) |
|---|---|---|
| `qc_status` pass | 0 | **0** |
| r² median | 0.419 | **0.433** |
| `electron-density-at-bound` | 43/47 (91%) | **15/15 (100%)** |
| `detector-response-fallback` | 47/47 | 15/15 |

A 3-16x gain in S/N moved r² by 0.014 and left every gate failing. The
failure is therefore **not** S/N-limited, and "re-run with averaging" is
dead as a fix.

The residuals point somewhere specific instead. The *cleanest* spectra give
the *worst* compositions — `rep3avg` returns H 0.41 / P 0.20 / Fe 0.09 and
`rep3g3` returns H 0.88 — while several dirty rep-1 averages return a
plausible Fe 0.79-0.87. Combined with `electron-density-at-bound` on 100% of
averaged fits, that is the signature of **Hα driving n_e**: the fitter
over-assigns H I 656.28, pushes electron density to its bound, and the
composition follows it off a cliff. The two live suspects are therefore the
Hα/Stark n_e path and the `detector-response-fallback` (wrong for this
instrument and session on every single fit), not photon statistics.

Caveat on this test: averaging across *locations* is not identical to the
instrument's own `numShotsToAvg`, which stacks repeats at one spot. It is
equivalent for S/N, which is what the hypothesis was about, but it mixes
surface states. A same-location series would settle it without that mixing —
see below.

## How the data was recovered

The shots were never on the SD card as spectra. The Z300 keeps every test in
its internal Couchbase Lite database and only writes
`export/geochem_pro_spectra/Test …/Shot(N).csv` when an operator exports;
no export happened, and the card's newest test folder was 9-14-26 with its
auto-export CSVs stopping at 2026-06-16.

Recovery path, all read-only: pull `libzdata/libzdb.cblite` (20.6 MB) over
MTP, query it for today's tests, and follow each test's
`shotTable.all_fb` to a bundle in `libzdata/spectra/`.

Today's three tests, all `test_context: GeochemPro`:

| seq | time | locations | bundle |
|---|---|---|---|
| 35028 | 13:44:41 | **15** | `89899c99-…` |
| 35029 | 13:46:31 | 16 | `f5022c9d-…` |
| 35030 | 13:47:03 | 16 | `c544de3a-…` |

`numShotsPerLocation: 1`, `numCleaningShotsPerLocation: 0`, so each site got
exactly one shot and the rep-to-rep comparison really is first/second/third
exposure of that site.

**47 shots, not 48** — rep 1 returned 15 locations. Which site is missing is
not recoverable: the database holds no per-shot coordinates, and the Sep 14
test proves skips happen mid-raster (its exported shots were 1–9, 11–14, 16,
i.e. locations 10 and 15 dropped). Rep 1 is therefore excluded from per-site
pairing and used only for rep-level statistics. Reps 2 and 3 are fully
paired site-by-site.

### Decoding the bundle format

The bundles are schema-less FlatBuffers. Structure, reverse-engineered:
a root vector of N shot tables; each shot carries the segment edges
`(180, 365, 620, 960, 961)` and 4 segment tables; each segment holds a
4-coefficient cubic wavelength polynomial and **2066 float64 intensities**
— matching `z300.py`'s own `synthetic_shot(segments=4, pixels_per_segment=2066)`.

The one non-obvious parameter is a **−18 pixel offset** between the
polynomial's origin and the first stored sample (masked/dark columns the
firmware trims). It was solved for, not assumed: scanning the offset against
the vendor's export peaks the correlation at exactly −18.0 px in every
segment of every shot tested. Getting it wrong shifts lines by up to 3.5 nm
and silently mis-assigns elements.

Validated against the vendor's own export of the Sep 14 test:
correlation **0.94 and 0.98** on two shots, RMS difference ~1.2% of full
scale, and line centres agreeing to **±51 pm** — which is just the vendor's
0.1 nm grid quantisation. Before the offset was applied, correlation was 0.05.

## Instrument side effects — please read

Early on I probed the analyzer's `/data/*` HTTP endpoints. **Every `/data/*`
request kills the instrument's HTTP server**, including requests to
non-existent paths; `/instrument/*` is unaffected. I confirmed this twice —
once before and once after you enabled the remote service — and after each
attempt port 9000 stopped listening while the device still answered ping and
USB. It recovers when the app is reopened. `pantheum/alibz/z300.py`'s
`tests_since()` and `shot_spectrum()` cannot work against firmware
`v2.19.7-0-gca24454` and should be considered unusable.

Also: enabling the remote service **takes the instrument off USB/MTP**, so
the SD card is unreachable in that mode. The two access paths are exclusive.

Config bug worth fixing: `config/alibz.example.json` sets
`acquire.analyzer_url` to `192.168.50.65:9000`, dead since the 2026-09-13
renumber. The analyzer is at **192.168.60.65:9000**.

Nothing was changed on the deployment: verified afterwards that
`opal-pipeline.json` still has `mtp.enabled = False`, `destination_root =
C:\LIBS-Staging\SD`, both sources present. All MTP work ran from temporary
candidate configs, since deleted. ~32 entries of pre-existing Sep-14-or-older
SD content were staged by an authorised one-shot pull that I later stopped;
staging integrity verified afterwards at **7 578 entries, 0 incomplete, 0
partial files**. No motion, laser, or acquisition command was ever issued.

## Artefacts

* `scripts/z300_fb_decode.py` — decode `all_fb` bundles to per-shot CSVs;
  `--compare` validates against a vendor export.
* `scripts/fe_line_qc.py` — line-intensity purity/repeatability/cleanup QC;
  the path that does not depend on the inversion.
* `scripts/fe_grid_qc.py` — the same three analyses driven off
  `summary.csv`/`detections.csv`, for when CF-LIBS does converge. Verified on
  a synthetic 15+16+16 fixture with a known injected surface layer; handles
  an unpaired partial rep.
* Spectra, fit outputs and QC tables:
  beryl `~/fe_checkout_20260921/`, scratch `ferun/`, `fe_spectra/`.

## Next

Averaging is ruled out, so the remaining work is on the two gates that failed
on **100%** of fits:

1. **Fix the session detector response.** `response_620_source = fallback` on
   every fit means the corpus 3.9x stand-in is being used for an instrument
   and day it was not measured on. Pure Fe is the cleanest single-element
   case available and is the natural target for measuring it properly.
2. **Investigate the Hα/Stark n_e path.** `electron-density-at-bound` on
   100% of averaged fits, with H reaching 0.88 atom fraction on the cleanest
   spectra, says n_e is being driven off H I 656.28 into its bound. Note the
   line-intensity analysis independently shows H tracks Fe (r = +0.86), i.e.
   it is ambient, so treating it as a sample constituent is wrong from the
   start.
3. **A same-location shot series** would separate depth/cleaning from
   averaging and supply a steady-state pure-Fe spectrum for (1). Design and
   blockers below.
4. If you want per-site spatial maps, export on the instrument so the
   location indices survive — that is the one thing the database cannot
   give back.

## Proposed same-location experiment, and why it is not queued

**Design** (mode 4 GeochemPro, optics matched to today so results compare):
`numlocations = 4`, `numShotsPerLocation = 24`, `numCleaningShotsPerLocation
= 0`, `numShotsToAvg = 1`, `intergrationDelay = 19`, `useGating = false`,
`argonpreflush = 300`. 96 shots, against the spec's 600 cap
(`numlocations * numShotsPerLocation <= 600`). Every shot saved individually.

What it yields that today's data cannot:

* a **depth/cleaning curve at fixed location** — contaminant/Fe against shot
  index, which locates the steady-state onset from the data instead of
  assuming it
* a **confound-free averaging ladder** — average N = 1, 2, 4, 8, 16 drawn
  only from steady-state shots, so surface state is held constant; this is
  the rigorous version of the falsification above
* a **response standard** — a steady-state pure-Fe spectrum to replace the
  failing 3.9x fallback
* an **Hα discriminator** — whether H decays with depth (surface water) or
  persists (ambient), which tests the n_e-railing story directly

**Blockers, all verified:**

1. *The portal can fire but cannot retrieve.* `acquire.py` collects results
   via `tests_since()` (line 576, in a poll loop) and `shot_spectrum()`
   (lines 584-585) — both `/data/*`, both confirmed to kill the analyzer's
   HTTP server on `v2.19.7`. The preflight at line 189 calls `tests_since()`
   too. A live run would fire the laser, fail retrieval, and land in
   `uncertain` in the physical ledger. The shots would survive in the
   instrument DB and be recoverable by the MTP route used here, but the run
   itself cannot complete as designed.
2. *Laser firing is enabled in the live deployment* —
   `acquire.enabled_actions: ['z300.firetest']`, `acquire.enabled: true`.
   So this is a real, armed path, not a dry one.
3. *The hardware reservation has lapsed* (expired 2026-09-21T20:27:36Z).
4. *Mode conflict* — acquisition needs the network API, which needs remote
   service, which takes the instrument off USB/MTP. Retrieval then needs
   USB/MTP back. The two modes are mutually exclusive, so the run needs a
   bench-side mode switch between firing and recovery.

Given that the stated purpose — testing the averaging hypothesis — is already
answered, this run would be spending laser time and an `uncertain` ledger
entry to refine a negative result. It is worth running for the *response
standard* and *Hα* purposes instead, which is a different justification and
worth an explicit decision rather than an assumed one.
