# 2026-11-03 — alibz manuscript initialised (theory / assumptions / workflow / datasets / metrics)

Provider: Claude subscription (no switch). Coordinator: Fable; literature workers: Sonnet (×3);
theory/assumptions drafter: Opus 4.8 specialist; further section drafters: Sonnet.

## Deliverable

`manuscript/` — arXiv-ready LaTeX (`article` class, natbib numeric `unsrtnat`, latexmk).
Journal target recorded as *Applied Spectroscopy* (SAS/SAGE); the request said "journal of
applied spectroscopy" — if the Springer *Journal of Applied Spectroscopy* was meant, only the
author block and bibliography style change (`svjour3`). Skeleton compiled at 10:57 PDT
(3 pages: title, abstract draft, section stubs with drafting guidance in comments).

Fixes during skeleton compile: `\ne` clashes with LaTeX's not-equal (renamed `\nel`);
underscores in appendix `\todo{}` text escaped.

## Literature retrieval design

`literature-retrieval-agent` (`lra`, ~/.local/bin) FETCHES PDFs by DOI; it does not search.
Therefore: (1) three workers search the literature and write verified `.bib` files
(`manuscript/bib/{theory,cflibs,ml}.bib`) plus DOI manifests
(`manuscript/lit/sources-<topic>.txt`), every DOI checked against the Crossref API;
(2) `manuscript/lit/fetch.sh` runs `lra fetch --automate-portal --background --skip-existing
--retries 1 --throttle 3 --report` over all manifests using the lra repo's `strategies.json`
and `.chrome_session` profile. Smoke test 10:56 PDT: `lra fetch --auto-open-access --background`
captured arXiv 1706.03762 (2.2 MB, sha256 bdfaa68d…) unattended with a JSON report.
pdftotext + tesseract present, so `--fulltext` is available.

## Status log

- 10:52 skeleton written; 10:57 compiles.
- 10:53 lit workers launched (theory / cflibs / ml). Reports: `reports/2026-11-03-lit-<topic>.md`.
- 10:58 theory+assumptions specialist launched. Report: `reports/2026-11-03-ms-theory.md`.
- pending: workflow drafter, datasets+metrics+results drafter, intro drafter; fetch run;
  coordinator verification of every number in sections 05–07 against the cited report.

## Verification (coordinator) — to fill

### 11:10 theory + assumptions (specialist, reports/2026-11-03-ms-theory.md) — VERIFIED with corrections
Checked against code: `line_emissivity` energy convention (eV/sr/emitter, hc/λ) ✓; `partition` = union of
lower+upper levels ✓; Saha via thermal de Broglie ✓; `stage_disagreement` = (max−min)/sum ✓;
`halpha_log_ne` = 17 + log10(w/w_ref)/α ✓; Voigt FWHM 0.5346/0.2166 ✓; κ ∝ λ² gA e^{−E_low/kT}/U·f ✓;
`invert_doublet_tau` R = R0·f(sτ)/f(τ) ✓; `_stage_tie_cost` = Σ max(0, tied−free) ✓;
COLLAPSE_TOP_FRACTION 0.9 ✓; ELEMENT_DETECT_Z_MIN 3 ✓. Tie-cost numbers traced to
reports/2026-09-21-stage-consistency-thermometer.md (360 truth / 3.7e3 ridge / 2.6e6 data-chosen).
Corrections applied by coordinator: removed a self-citation standing in for Barklem & Collet; replaced the
"R formed consistently" overclaim (UV/VIS response is uncalibrated); doublet equation now distinguishes R0
(thin emissivity ratio) from s_r (absorption-strength ratio); 2e6 → 2.6e6; PSF modules named correctly
(peak_location, epsf_shapes); A6 no longer implies the 3.9× fallback is a UV/VIS response.
10 `\todo{cite:}` markers await the bib files. Compiles (11 pages).
- 11:12 workflow + parameter-appendix worker launched (reports/2026-11-03-ms-workflow.md).
- 11:25 theory.bib VERIFIED: 50 entries, 0 duplicate keys, manifest 46 well-formed `key=DOI` lines, 8/8 random
  DOIs re-checked against Crossref (title+year match), `bibtex` with `\nocite{*}` parses all 50 with no
  warnings. 3 no-DOI book/chapter entries (Griem 1964, Griem 1974, McWhirter 1965) written from memory —
  check at proof. Report: reports/2026-11-03-lit-theory.md.
- 11:26 datasets/metrics/results worker launched (reports/2026-11-03-ms-datasets-metrics.md); 4 in flight.
- 11:40 cflibs.bib (59) and ml.bib (59) delivered; random Crossref re-checks 8/8 each; the mislabelled
  `sciaps2021` renamed `urbinamedina2021branching` (Urbina Medina et al. 2021, branching-ratio optical-depth test).
  CROSS-FILE DUPLICATES found (10 identical keys + 6 same-DOI/different-key): deduplicated keeping the first
  occurrence in order local→theory→cflibs→ml → 156 unique entries; alias map manuscript/lit/key-aliases.txt
  (dropped→kept). Combined `bibtex` with `\nocite{*}`: 156 bibitems, no warnings after `@software`→`@misc`.
  Manifests regenerated from the deduplicated bibs (my first prune wrongly dropped 9 kept-key lines):
  theory 46 / cflibs 48 / ml 54 = 148 fetchable (DOIs + 6 arXiv); NIST ASD dataset DOI excluded.
- 11:45 theory/assumptions `\todo{cite:}` placeholders replaced by coordinator with verified keys (Saha,
  Griem 1997/1974/1968, McWhirter + Griem 1963 + Fujimoto 1990, Ecker–Kröll + Stewart–Pyatt, GGC 2003,
  Gigosos 1996, Konjević reviews, Olivero 1977 + Thompson 1987, Humlíček, Anderson–King ePSF, Cowan–Dieke,
  Irons, El Sherbini/Sun/Bulajic, Lawson–Hanson/Bro, Jones EGO, Schwarz BIC, spike-and-slab, Kramida/
  Ralchenko/Kurucz, Janesick, Edlén/Birch–Downs).
- 11:46 PDF fetch launched in background: manuscript/lit/fetch.sh --fulltext (lra --automate-portal
  --background --skip-existing --retries 1 --throttle 3); log manuscript/lit/fetch-<stamp>.log, JSON report
  manuscript/lit/report-<stamp>.json. Downloads/fulltext are gitignored.
- 11:55 introduction (worker, reports/2026-11-03-ms-intro.md) VERIFIED: 6 paragraphs, ~1,400 words, 56 keys all
  present in the bibs; design statements only for alibz. Coordinator restored three citations the dedupe had
  orphaned (Mermet 2008 LOQ, Villas-Boas 2023 CF-LIBS LOD, Urbina Medina 2021 branching ratios — meanings
  confirmed against reports/2026-11-03-lit-cflibs.md) and added Lawson–Hanson to the NNLS sentence. One
  honest `\todo{cite}` remains (LIBS synthetic-spectrum training with a physical forward model: no verified
  reference found). Appendix A1 notation table written by coordinator. `eilers2003smoother` (cited by the
  workflow draft) verified via Crossref (10.1021/ac034173t) and appended to ml.bib + manifest.
- 12:20 workflow + A2 (worker, reports/2026-11-03-ms-workflow.md) VERIFIED: DEEPEN_BARS (3.0, 2.0) matches
  pipeline.py (brief was stale); runtime 225→106 s traced to commit e715ff0 and vacuum→air 110–240 pm to
  commit 5daafa2 (src comments added); "inverted Ca/Mg" clause had no source → removed (code comment supports
  only "χ² misranks T"); duplicate sentence and an over-general "ten-shot averages" statement fixed.
- 12:30 datasets/metrics/results (worker, reports/2026-11-03-ms-datasets-metrics.md) VERIFIED by spot check of
  ~25 numbers against cited sources (round-trip docstring + tolerances; thermometer table; detector.json 4909/
  435/3.9/3.0–5.2; archive 8,213/97.9/99.1/96.6/97.3; Fe 26 runs/5 of 26/6873–13701/10805; ePSF 330 lines/24
  runs/5.3 vs 11.6–12.3; line matching 86/95/78/88 %; V period means and V II 311 slope; registration 107 runs;
  926/928 K pairs; MW2-112 IDs/Spearman; pipeline constants; BIC_AMBIGUITY 6). All matched. Worker corrections
  accepted: MW2-112 is a lithium MUDROCK (docs/mw2_112/provenance.md), only suite S exists, no CRM named, no
  measured Ar/air pair in repo. Coordinator wrote 08-discussion, 09-conclusions, A1 notation, final abstract;
  recorded the 2026-07-01 ~20 % major-element target in DECISIONS.md so the manuscript has a repo source;
  converted all remaining `\todo{}` markers to sourced statements or LaTeX comments.
- 12:40 FINAL COMPILE: 47 pages, 0 undefined citations, 0 undefined references, 101 cited of 157 bib entries,
  20 overfull boxes (cosmetic, wide tables), `make arxiv` bundle 74 kB. Fetch at 12:40: 14 PDFs + 14 fulltext
  .txt captured, 18 sources skipped on bot-detection challenges (5 s unattended timeout), 2 errors; still
  running. Re-run for the skipped sources needs a human at the browser:
  `manuscript/lit/fetch.sh --pause-on-challenge --challenge-timeout 180` (skip-existing keeps the captured ones).

## Open items (for the owner)
1. Confirm journal: *Applied Spectroscopy* (SAS/SAGE, assumed) vs Springer *Journal of Applied Spectroscopy*.
2. Co-authors, affiliations, funding, data-deposition DOI in main.tex / Acknowledgements.
3. Engine benchmark (sec. 7.2) cannot run: bench/ imports the removed cflibs prototype — re-run or drop.
4. Figures: none yet; candidates in reports/figures/ and cal/figures/ (thermometer scan, ePSF, Fe Boltzmann,
   V doublet, registration residuals, confounder example).
5. Interactive fetch pass for the challenge-skipped PDFs; then agents can read lit/fulltext/*.txt to tighten citations.
6. Commit: nothing committed (manuscript/, DECISIONS.md entry, reports/, STATUS.md, bib) — owner pushes.
- 12:55 FETCH COMPLETE (manuscript/lit/report-20261103-*.json; summary lit/fetch-outcome-summary.txt):
  148 sources → 69 captured (69 fulltext .txt, 0 text holes), 60 skipped on bot challenges (49 × Elsevier
  10.1016 = most of Spectrochim. Acta B; 4 × 10.1086; 3 × 10.1088), 18 no_pdf (6 arXiv — portal locator has
  no control to click on arXiv; 4 IEEE; books/chapters), 1 error (li2014spectralresponse, RSC).
  fetch.sh rewritten as two passes (arXiv via --auto-open-access, then portal); arXiv re-pass run separately.
  Elsevier + IOP + AAS need the interactive run with --pause-on-challenge (owner).
- 13:10 arXiv open-access pass: 6/6 captured (needs the bundled strategy map + its own profile; the lab
  strategies.json lacks the open_access flag for arxiv.org, and my first attempts passed the six refs as ONE
  argument — zsh does not word-split — giving a 404 on a URL with spaces). fetch.sh now runs that pass first
  with lit/profile/oa. Corpus: 75/148 PDFs, 75 fulltext .txt, 0 text holes. Remaining 73 = 60 bot-challenge
  skips (Elsevier/IOP/AAS; interactive re-run) + 12 no_pdf (IEEE, books/chapters) + 1 error.

## 13:50 literature-retrieval-agent structural fix (user request: challenge loop on ScienceDirect)
Diagnosis: `--automate-portal` opened the ScienceDirect landing page for each of 49 Elsevier DOIs; the
Cloudflare-style challenge never clears for the automated Chrome (even on a human click), each source
revealed the window, waited, skipped; nothing learned across sources; `--fallback` only fired on NO_PDF.
Fix (specialist, reports/2026-11-03-lra-challenge-breaker.md; on top of the repo's uncommitted in-flight
work, NOT committed): (1) per-(publisher, strategy) challenge breaker, `--challenge-strikes` (default 2),
new FetchStatus.BLOCKED, report `blocked` list; (2) challenge-skip escalates to `--fallback`; (3)
ResolvedUrlPdfLocator understands EZproxy dash-encoded hosts and re-proxies the derived PDF URL;
(4) ElsevierApiLocator (api.elsevier.com full-text, ELSEVIER_API_KEY / ELSEVIER_INSTTOKEN, browser-free,
dead-after-2) first in the portal chain; README sections. Tests: 126 passed + the SAME 3 pre-existing
failures (fakes lack goto_and_capture; from the in-flight work; untouched). No stealth/fingerprint evasion.
Coordinator LIVE TEST (scratch profile, 3 Elsevier DOIs, timeout 3 s, strikes 2): skipped 2 → ⛔ line →
blocked 1, no third navigation; report `blocked` entry correct. alibz manuscript/lit/fetch.sh now passes
--challenge-strikes 2 --fallback berkeley (env LRA_CHALLENGE_STRIKES / LRA_FALLBACK). The kill of the
user's stalled run was denied by the permission classifier — user must Ctrl-C it.
- 14:05 LIVE escalation test (scratch profile): challenge-skip → berkeley escalation fires; berkeley route hit the
  SAME challenge and was blocked after 2 strikes; 3rd source blocked on both, no navigation. ROOT CAUSE of the
  proxy no-op: from the LBNL network (AS16) `libproxy.berkeley.edu/login?url=…` 302s straight to the raw
  publisher URL with no CalNet login (curl: ScienceDirect 403 Cloudflare, IUCr "Just a moment"), i.e. EZproxy
  treats this network as on-campus. → fetch.sh fallback made opt-in (LRA_FALLBACK); READMEs note it.
  Effective Elsevier route from LBNL = Elsevier full-text API (reachable; needs ELSEVIER_API_KEY; entitlement
  by institutional IP). Repeated automated hits likely escalate Cloudflare's per-IP challenge level; the
  breaker now caps that at 2 per publisher per run.
- 14:20 Elsevier key storage: lra now resolves ELSEVIER_API_KEY / ELSEVIER_INSTTOKEN from the environment first,
  then the OS keychain (macOS `security find-generic-password -s elsevier-api-key -w`; Linux secret-tool);
  7 new tests (tests/test_secrets.py), suite 133 passed + same 3 pre-existing failures. Keychain round trip
  verified with a throwaway item (add → read without prompt → delete). READMEs updated. Nothing committed.
- 14:40 User stored a keychain item, but the value is 11 non-hex chars; Elsevier answers 401 APIKEY_INVALID
  (an Elsevier key is 32 hex chars). Live scratch test: API 401 → browser path → challenge → blocked after 1
  strike (as configured), no hang. Locator now parses Elsevier's error body (XML <error-code> / JSON
  statusCode) and prints the code with an actionable hint (APIKEY_INVALID → re-store; AUTHORIZATION/
  AUTHENTICATION_ERROR → entitlement; QUOTA_EXCEEDED); test_elsevier_errors.py added; the specialist's
  no-key test isolated from the keychain. Suite: 135 passed + same 3 pre-existing failures. User's second
  fetch.sh run (13:31, --challenge-strikes 2) is running; 76 PDFs so far.
- 14:55 Real key stored (32 hex; API returns a 1.06 MB PDF from the LBNL network → entitled). Run 14:41:
  25 captured (22 Elsevier via API, 0 browser pages), total 103 PDFs / 0 text holes. DEFECT FOUND: two
  consecutive DNS failures ("nodename nor servname provided") counted as API failures → locator retired
  the API for the run → 22 Elsevier sources fell to the browser → challenge → breaker blocked them.
  FIX: network errors retry once (2 s) and never retire the API; only 401/403 count, and never after a
  success in the run; 404/429/5xx are per-source. tests/test_elsevier_resilience.py (5 tests); the
  specialist's dies-after-two test aligned to 401-only. Suite 140 passed + same 3 pre-existing failures.
  Other blocked publishers this run: 10.1086 (AAS, IOP-hosted) and 10.1088 (IOP) — same challenge class;
  candidates for an ADS/Unpaywall open-access route (follow-up, not built). Re-run launched 14:56.
- 15:20 Run 15:08 (resilient API): 36 captured (27 Elsevier via API; 0 Elsevier browser pages), total 139/148 PDFs,
  140 fulltext, 0 text holes. Missing 9: 10.1086 ×4 (Irwin 1981, Karzas–Latter 1961, Stewart–Pyatt 1966,
  Anderson–King 2000 — AAS journals on IOP; challenge → blocked; free scans exist on NASA ADS), 10.1109 ×2
  (IEEE), cremers2013handbook (Wiley book), thompson1987voigt (IUCr; berkeley proxy bounces from LBNL),
  sweetapple2015lithium (Am. Mineral.; same). Follow-up candidate: ADS/Unpaywall OA locator for AAS/old papers.
- 15:22 USER: window foregrounding on every fetch is disruptive → specialist launched for quiet mode
  (reveal only with explicit --pause-on-challenge; lazy browser launch so API-only batches open no Chrome;
  macOS hide-by-pid under --background). Report reports/2026-11-03-lra-quiet-mode.md.
  Remote option checked: "serpentine" = tomoPC, Windows 10 (not suitable); moissanite = Linux 64c, LBNL
  network (API entitled), google-chrome present, no Xvfb/playwright yet (venv + `sudo apt install xvfb`
  would make a fully invisible headed run possible there).
- 15:35 REMOTE (user request "run lra on moissanite"): moissanite = Python 3.10.12 (no ensurepip/pip), no
  sudo (no Xvfb → headless), pdftotext yes / tesseract no, Chrome 147. Done: venv via `python3 -m venv
  --without-pip` + get-pip; `pip install -e ~/lra` (tree rsynced, uncommitted fixes included); corpus mirrored
  to ~/lra-corpus (140 files, 271 MB); smoke test headless arXiv fetch OK (2.2 MB captured). Wrapper
  manuscript/lit/fetch-remote.sh (push tree+corpus → run headless + API → pull downloads/fulltext/report).
  lra gained a 0600-file secret fallback (~/.config/lra/<service>) for keyring-less hosts (tests added).
  Key transfer to moissanite DENIED by the permission classifier (credential materialization) — user runs:
  `security find-generic-password -s elsevier-api-key -w | tr -d '\n' | ssh moissanite 'umask 077; install -m 600 /dev/stdin ~/.config/lra/elsevier-api-key'`.
  Remote full run deferred until the quiet-mode specialist finishes (retriever.py mid-edit).
- 15:45 User placed the key on moissanite (~/.config/lra/elsevier-api-key, mode 600, 32 hex); resolver finds it;
  Elsevier API from moissanite: HTTP 200, 1.06 MB PDF (entitled by LBNL IP); ElsevierApiLocator available.
  Remote full run pending the quiet-mode specialist (then fetch-remote.sh re-syncs the tree and runs).
- 15:55 QUIET MODE (specialist, reports/2026-11-03-lra-quiet-mode.md) VERIFIED: suite 156 passed + same 3
  pre-existing failures; cli passes reveal_on_challenge only for an explicit --pause-on-challenge; _LazySession
  + _try_api_capture before snapshot/new_page. Coordinator live test (2 Elsevier DOIs, scratch dirs): both
  captured via API, "browser was never launched", profile dir never created. Specialist's own macOS smoke:
  hidden-by-pid holds across pages; user's Chrome untouched; ~0.5 s flash at launch remains. Two bugs it
  caught itself: pgrep pattern must not start with "--"; a new page un-hides the app → per-page re-hide.
  Remote run launched via manuscript/lit/fetch-remote.sh (pushes tree, headless + API on moissanite, pulls back).
- 16:05 REMOTE RUN (moissanite, headless Chrome + API, report manuscript/lit/report-remote-20261103-154432.json):
  pipeline works end to end (push tree → run → pull); 140 exists, 0 new: the 9 stragglers all challenge headless
  Chrome — IEEE "unusual traffic" (10.1109, blocked after 2), AAS/IOP AWS WAF 202 interstitial (10.1086, blocked
  after 2), Wiley book + IUCr "performing security verification", Am. Mineral. no_pdf. 0 text holes. Follow-ups
  (not built): generalize the AWS-WAF wait-out (exists for De Gruyter) to any host, and an ADS open-access
  route for the 4 AAS papers. Corpus final today: 139/148 PDFs.

## 16:30 Phase 2 (user): ingest the retrieved papers; merge ideas by testing fit-quality improvements
Design: (A) 3 ingestion workers (theory 40 texts / cflibs 47 / ml 53; 13 keys have no text) → one note per paper
(manuscript/lit/notes/<key>.md: method, adoptable specifics, alibz module, fit-quality test, caveats) + ranked
candidate tables (reports/2026-11-03-lit-ingest-{theory,cflibs,ml}.md). (B) A/B harness worker →
scripts/fit_quality_panel.py (`run --variant … --config-json …`, `compare`): fixed panel = suite S (5 synthetic),
data/remote_samples (5 real), Fe Aesar 99.98 % run means (26, native grid); baseline JSON under
reports/fit-panel/ + reports/2026-11-03-fit-panel-baseline.md. (C) coordinator ranks candidates; each shipped
behind a flag/kwarg and scored before/after on the panel; adopt only measured improvements. Stragglers left.
- 17:05 INGEST theory + ml done (40 + 53 notes; reports/2026-11-03-lit-ingest-{theory,ml}.md). CORPUS QUALITY DEFECT
  found by the ingest: 54/140 "captured" PDFs are 1-page stubs and ≥5 are wrong documents (AIP login page,
  byte-identical twin, wrong proceedings, front matter). Root cause for the 49 Elsevier ones: the API PDF
  endpoint serves a first-page PREVIEW (praher 1 page/799 words) while `?view=FULL` JSON `originalText` has the
  complete article (60–142k chars, refs included) for the same key; the entitlement endpoint is 403
  AUTHENTICATION_ERROR (not enabled for the key) and is advisory only. lra tasks sent to the specialist:
  Elsevier full text via view=FULL → `<name>.elsevier.xml` + fulltext txt, new FetchStatus.TEXT; page-count vs
  Crossref/pageRange preview rejection; post-capture DOI/title/twin verification → FetchStatus.SUSPECT
  (`<name>.suspect.pdf`, not skip-existing-done); `lra verify` to re-check an existing corpus. After it lands:
  verify corpus → re-fetch → re-ingest the ~54 stub-based notes. Manuscript: IPD magnitude corrected
  (~0.07 eV per unit charge at 1e17/8 kK, Debye length ≈ 20 nm; coordinator-checked).
- 17:30 INGEST complete: 140 notes (40 theory / 53 ml / 47 cflibs); 3 ranked tables (30 candidates). More corpus
  impostors found by cflibs ingest: gaudiuso2010/gornushkin2022/harmon2023 = cover pages; saloman2004 = user
  guide; miziolek2006/noll2012 = front matter; li2014spectralresponse & palleschi2022 = byte-identical copies of
  korbel2024; harmon2019mineral = press release. Literature accuracy context: CF-LIBS 5–30 % on majors,
  30–200 % on minors; CSigma 3.2 % rel; handheld Li MAPE 11.75 % (Korbel 2024).
  WAVE 1 selection (coordinator; criteria: expected effect on panel metrics, cost, evidence from full texts,
  independence): W1 adaptive NN-LASSO + Huber IRLS in the concentration solve (spurious mass / phantom elements)
  → worker; W3 Barklem–Collet 2016 partition tables + Debye IPD (T, stage fractions) → worker; queued for the
  next free slots: W2 BH-FDR on detection z; W4 GP settings (Matern-5/2, normalised dims, EI early stop) +
  effective-n BIC on export-grid inputs. Deferred: NIST accuracy-grade weights (db has no grade column),
  two-temperature thermometer, oxide closure (changes reported quantity, not fit quality), SA discriminance /
  CSigma validity gate (wave 2, after SA-focused panel items exist), per-line McWhirter flag (risk).
- 18:05 lra capture verification landed (specialist; 168 tests + same 3): FetchStatus.TEXT (Elsevier view=FULL
  originalText → <name>.elsevier.xml + fulltext), preview guard (1 page vs pageRange), FetchStatus.SUSPECT
  (DOI/title identity, ≥2 pages, sha256 twin), `lra verify`. Coordinator ran `lra verify` on the corpus:
  74 ok, 65 suspect (46 Elsevier previews, 18 identity, 1 twin), 10 no_pdf. Inspection of the 19 non-preview
  suspects: 9 genuine impostors (AIP "Content Platform User Guide" ×2, MDPI journal flyers ×4, wrong J. Plasma
  Fusion Res. proceedings, wrong J. Phys. B paper, korbel twin) and 10 FALSE POSITIVES (6 arXiv: no DOI to
  match; 4 JSTOR scans: image-only pages). Sent to the specialist: arXiv-id / OCR identity fallbacks; MDPI
  `/pdf` rule; exclude guide/flyer links; journal-mismatch reason. Re-fetch running in background (Elsevier
  previews → full text via API). Then: re-verify, re-fetch impostors, re-ingest the stub-based notes.
- 18:40 ELSEVIER KEY RESTRICTED: all article endpoints (FULL, META_ABS, PDF) now return 403 AUTHENTICATION_ERROR
  "Requestor configuration settings insufficient" (worked at ~17:00: 60–142k-char originalText). No rate-limit
  headers → not quota; most likely Elsevier's usage review after ~150 full-text calls from a new key today.
  Owner action: check the key at dev.elsevier.com / contact Elsevier API support; until then the 46 Elsevier
  previews stay stubs (abstract-level notes only). Specialist verify follow-up landed (178 tests + 3): arXiv-id /
  OCR identity fallbacks, MDPI `/pdf` rule, guide/flyer link exclusion, journal-mismatch reason. Re-run of
  `lra verify` showed it ignores `.suspect.pdf` (74 ok / 75 no_pdf) → fix requested (re-examine + promote).
  Old-code re-fetch stopped (TaskStop); new re-fetch launched with --no-elsevier-api to re-capture the 10 false
  positives and the MDPI items under the corrected locators.
- 19:10 HARNESS + BASELINE landed (worker; reports/2026-11-03-fit-panel-harness.md, -baseline.md; JSON
  reports/fit-panel/baseline-20261103.json; 36 items, 902 s on 4 workers; 6 tests). Baseline: synthetic
  median rel err 0.20 (median), spurious mass 0, dominant wrong on binary_argon_ambient, log ne 16.5 (truth
  17.0) on 3/4, iron_rich_low_t render error (sum 0.99999998884 vs rtol 1e-8 → coordinator relaxed
  synthetic.py tolerance to rtol 1e-6; renders now); real: r² median 0.72, all 5 ne-at-bound + qc fail,
  REE_01 collapses to C 0.95; Fe standard (axis shifted +18 samples per segment — approximation at seams,
  Fe undetectable without it; proper fix = regenerate the npz from re-derived native shots): Fe fraction
  mean 0.42, non-Fe mass 0.58, T railed at 25 kK on many runs, r² median 0.28, spurious H/C/B — the
  pure-Fe standard is the hardest panel and the clearest target. Baseline to be RE-RUN on a quiet tree once
  the wave-1 workers finish (they were editing pipeline/indexer/sahaboltzmann during the run; synthetic and
  real metrics were bit-identical across two runs, so contamination is unlikely but unproven).
- 19:40 CORPUS after re-fetch (--no-elsevier-api) + `lra verify` with rehabilitation: 89 verified PDFs (+4 MDPI
  real papers via the new `/pdf` rule; 6 arXiv + 4 JSTOR scans rehabilitated by the arXiv-id/OCR identity
  fallbacks), 46 Elsevier previews (key restricted — owner), 3 impostors (AIP guide ×2, SPIE journal
  mismatch) + 1 twin, 10 never captured. fulltext 90. MDPI re-ingest launched (4 notes).
- 20:05 EARLY A/B READS (vs baseline-20261103, fast settings): SPARSE adaptive NN-LASSO — no gain: synthetic
  unchanged (spurious mass already 0), real stage_disagreement 2 worse/0 better, Fe fraction mean 0.419→0.407
  (3 better/5 worse) → not adopted on this evidence (Huber + combined still running). PARTITION (Barklem–Collet
  tables) — Fe standard IMPROVES: Fe fraction mean 0.419→0.544, non-Fe mass 0.581→0.456 (14 better/9 worse;
  one run 0.41→0.93); real neutral; synthetic mixed (feldspar worse, argon-ambient dominant now correct) BUT
  the synthetic scenes are rendered with the line-sum partition function, so that panel is an inverse-crime
  test against the tables — not diagnostic; a fair synthetic test needs the generator to take the same
  partition_source. Fe panel remains dominated by T railing near 25 kK on single-element spectra (thermometer
  identifiability with one element + uncalibrated response) — a wave-2 target. moissanite alibz setup running
  for fast panel runs (64 cores); laptop load 65–80 with three panels concurrent.
- 20:30 PANEL ON MOISSANITE: alibz tree (snapshot with sparse/partition/FDR code, flags off; db pickle, panel
  data) at moissanite:~/alibz, venv via get-pip; smoke item 4 min single-thread. Series launched with 32 workers:
  baseline-moissanite + huber, sparse-huber, ipd, partition-ipd, fdr05, fdr05-by, partition, sparse →
  reports/fit-panel/<variant>-moissanite.json (pulled back). All verdicts will be taken against the
  moissanite baseline (same CPU; GP trajectories may differ from the laptop's). Laptop nohup panels (huber,
  ipd, fdr05) left to finish; redundant.
- 21:00 WAVE-1 REPORTS: sparse/robust (reports/2026-11-03-ab-sparse-robust-solve.md): flags OFF bit-identical;
  5 tests; laptop panel: sparse neutral, huber hurts real r² 0.470→0.361 and silicate/feldspar; worker recommends
  rejecting all three (huber Fe rows invalid: transient mid-edit SyntaxError). partition/IPD
  (reports/2026-11-03-ab-partition-ipd.md): Barklem–Collet table downloaded (VizieR CfA TSV, sha256 5391d67c…,
  277 species, 42-point T grid → db/partition_barklem2016.npz, provenance in atomic_line_sources.json); 16 tests;
  laptop panel: ipd best (synthetic median_rel_err 0.273→0.199, real r² up 4/5, stage_disagreement 0.483→0.535,
  T error worse), partition Fe better by count, partition-ipd no clear win. Follow-up sent: generator honours
  partition_source/ipd (fair round trip) + --render-config-json + partition_extrapolated in panel JSON.
  Final verdicts deferred to the moissanite clean series (b95mg6uux).
- 21:40 BO/BIC implemented (reports/2026-11-03-ab-bo-bic.md; 649 tests; not panel-scored; on binary case both ON
  moved Ca/Mg 0.52/0.48 → 0.57/0.43 vs truth 0.60/0.40). Generator now honours partition_source/ipd
  (--render-config-json; partition_extrapolated recorded) but the CONSISTENT tables round trip was WORSE
  (0.216 vs 0.084 median rel err) → coordinator found the cause: pass-1 indexer (pipeline.py:924) was built
  without the shared `sb`, so pass 1 ran line-sum U while pass 2 ran tables. PATCHED (db=db, sb=sb); related
  tests 60 pass. Consequence: the moissanite `partition` / `partition-ipd` runs (pre-fix tree) must be re-run
  after syncing; `partition-fair`, `bo`, `bic-auto`, `bo-bic` queued for the same re-sync.
- 21:50 CORRECTION: idx_kwargs (pipeline.py:812) already carries db=db, sb=sb, so pass 1 DID receive the shared
  SahaBoltzmann; my explicit addition duplicated the keyword (TypeError, swallowed by the xfail). Reverted;
  tests pass. The moissanite partition runs are therefore valid as-is. Root cause of the worse tables/tables
  round trip (0.216 vs 0.084) remains OPEN → specialist debug task next.
- 22:30 DEBUG VERDICT (specialist, reports/2026-11-03-debug-partition-roundtrip.md): NOT plumbing. Forward and
  inverse agree in emissivity space for both partition sources; at the EXACT true state the peak-based inverse
  over-recovers Ca (lines 0.627, tables 0.711; truth 0.60). Tables raise Ca I U by ~45 % near 10 kK, which
  moves the stage-tie minimum ONTO the truth node (T 9873 / ne 17.0, maxdis 0.04) where the bias is fully
  expressed; line-sum lands cold (8922 / 16.5) where the bias is accidentally compensated. Seed 7 reverses
  (tables 0.062 vs lines 0.160). Accuracy is governed by basin, not partition correctness. Decision: do NOT
  flip the default yet; the real defect is the peak-inverse Ca-over-Mg bias at truth → wave-2 isolation task
  (specialist) launched: true areas vs fitted areas, weighted vs unweighted, SA channel, evidence rows.
- 23:05 ROOT CAUSE of the composition bias at truth (specialist, reports/2026-11-03-debug-composition-bias-at-
  truth.md): the whole-pattern DESIGN MATRIX is not emission-conserving — `_build_observed_overlap_map` spreads
  each db line over every peak within shift_tolerance by a Voigt height ratio, so Σ_i overlap(i,j) = 1.04 (Ca)
  vs 1.97 (Mg) on the binary, 2.2–5.3 on the silicate; dense-forest elements are over-predicted per unit
  concentration and shrunk by NNLS (oracle areas at exact truth → Ca 0.731; control A·c_true → 0.600 exactly;
  inert: partition source, thermometer, evidence/pseudo rows, SA, normalisation, aggregation; weighted_solve
  worse). Same mechanism on silicate (Ca over, Si/Fe/Mg under at truth). Predicts the pure-Fe panel failure
  (Fe = densest forest → under-recovered, light elements fill). WAVE-2 launched: `overlap_mode`
  ("legacy" | "conserve" = partition of unity per db line, never scale up | "nearest" = hard assignment),
  oracle-at-truth regression tests, then panel on moissanite.
- 23:30 CLEAN SERIES (moissanite) verdicts → reports/2026-11-03-ab-wave1-verdicts.md; DECISIONS.md entry added.
  Paired counts vs baseline: fdr05 Fe +0/−19 (Fe fraction 0.40→0.19) ; huber syn +2/−3, real r² +1/−4,
  Fe +7/−13 ; sparse Fe +6/−2 ; ipd real stage +4/−1, Fe +13/−6 ; partition Fe +15/−8 ; partition-ipd real r²
  +5/−0, Fe +12/−11.
- 00:20 (11-04) WAVE 2 landed (specialist, reports/2026-11-03-wave2-conserving-design.md; 660 tests, +9):
  `overlap_mode` legacy|conserve|nearest plumbed through pipeline, detections contest, cli, fit_inspector;
  pseudo (missing-line) map left raw by design. Oracle at exact truth: binary Ca legacy 0.730 → conserve 0.587
  → nearest 0.600; silicate (truth .40/.30/.20/.10): legacy .23/.40/.16/.06, conserve .43/.27/0/0 (Fe, Mg PRUNED
  by the prefilters, U/Pb hallucinated), nearest .33/.25/.25/.02. Default kept legacy. Specialist asked why
  nearest is not exact on the silicate oracle (prefilters vs merged peaks vs ties). Tree synced to moissanite;
  SERIES 2 launched as two 16-worker chains: A = baseline2 + conserve{,-partition,-ipd,-partition-ipd,-sparse};
  B = nearest{,-partition,-partition-ipd} + bo, bic-auto, bo-bic → reports/fit-panel/*-moissanite2.json.
- 00:40 (11-04) FOLLOW-UP VERDICT: `nearest` is the principled overlap mode — clean control obs=A·c_true is EXACT
  under nearest (.40/.30/.20/.10) while LEGACY gives .363/.274/.272/.091 on its own data (self-inconsistent
  replication). Silicate miss under nearest = candidate-builder defects: 54 % of Mg's rendered flux (Mg II)
  lands on peaks with NO candidate line within shift_tolerance (coverage gap; Si/Ca/Fe ~0 %), plus U/Pb
  hallucination. Specialist launched for the coverage fix (reports/2026-11-04-candidate-coverage.md).
- 01:00 (11-04) COVERAGE PREMISE FALSIFIED (specialist, reports/2026-11-04-candidate-coverage.md): every rendered
  Mg II line IS a candidate; the "orphaned 54 %" is flux on coincident TWIN oracle peaks (db duplicates at identical
  wavelength rendered as separate peaks) where `nearest` tie-breaks all co-located lines onto one twin. Real
  fitters never emit two components at one wavelength → oracle artefact. Decision: merge coincident lines in
  the oracle (≤ max(1e-3 nm, 0.25 × instrumental FWHM)), keep `nearest` semantics; re-run oracles under all
  modes with restricted and full candidates (U/Pb check). Specialist resumed.
- 01:30 (11-04) REALISTIC ORACLE (merged within max(1e-3 nm, 0.25×FWHM); tests/test_oracle_roundtrip.py):
  binary legacy .606 / conserve .578 / nearest .600; silicate legacy .309/.373/.226/.092, conserve
  .366/.225/.255/.101 (+Dy/Pb/U ~1.5 % each), nearest .364/.274/.271/.091 (no spurious). The earlier "Ca 0.73 at
  truth" was inflated by coincident twin peaks in the unmerged oracle → manuscript + memory CORRECTED. Still
  real: legacy misallocates Si/Ca on multi-element scenes and is inconsistent on its own clean control; nearest
  is exact there. NEW separate defect: `_aggregate_elements` gives Fe .27 (truth .20) even on the clean control
  (3-stage aggregation floor) → wave-3 item. Series 2 (incl. nearest family) still running on moissanite.
- 02:10 (11-04) AGGREGATION PREMISE FALSIFIED (specialist, reports/2026-11-04-aggregation-floor.md): clean control
  aggregates exactly (Si/Ca/Fe/Mg .4/.3/.2/.1 to 1e-7); hand-built 1/2/3-stage invariant holds in legacy; the
  silicate oracle has only Fe I/II. Real cause of Fe .27: design under-predicts Fe's merged-oracle areas by
  ~1.23× relative to Si/Ca/Mg (real/design 0.123 vs 0.100) → NNLS inflates Fe. Forward (synthetic._component_
  lines) vs design (_line_weights) parity gap, Fe-specific. Manuscript + memory corrected; specialist re-briefed
  to isolate/fix the parity gap with a uniform-ratio invariant test.
- 02:40 (11-04) PARITY RESOLVED as configuration (specialist, reports/2026-11-04-forward-design-parity.md): Fe III,
  line-set cuts, Saha/partition all ruled out (Fe I:II emissivity ratio forward 0.441 = design 0.440); Fe's
  1.23× gap is a 28 % SHAPE misfit (9k dense lines merged to centroids + per-line Stark widths vs the design's
  single median (σ,γ) kernel) absorbed by the UNWEIGHTED solve. Production WNNLS (weighted_solve) + nearest at
  truth → Si .401 / Ca .300 / Fe .198 / Mg .100 (max dev .002); weighting COLLAPSES legacy/conserve (Fe → 1.0).
  → weighted+nearest is the consistent pair; parity test being written for it; SERIES 3 queued on moissanite:
  wn, wn-partition, wn-ipd, wn-partition-ipd, wn-fdr05, weighted-legacy control. Wave-3 model target: per-peak
  (σ,γ) / per-line Stark widths in the overlap kernel.
- 03:20 (11-04) SERIES 2 chain A + early B/C (vs baseline s1 → see table in log): baseline2 (same code, options
  off) ≠ baseline s1: syn median err 0.207→0.156, Fe +6/−13, real r² +1/−4 — run-to-run or off-path drift of
  the same order as variant effects → baseline3 launched for a determinism check; verdicts to be taken vs the
  matching-code baseline2. Relative to baseline2: conserve-partition-ipd best balance (syn 0.136, real r²
  0.483 +4/−1, Fe 0.571, n_non_fe 0.885); conserve-ipd syn 0.122; bic-auto syn 0.116 / T err 0.030 but real r²
  0.437 and one dominant wrong; bo: real stage_disagreement +5/−0; nearest alone syn 0.208 / T err 0.054;
  wn (weighted+nearest) BAD end-to-end (syn 0.399, real r² +0/−5) despite exact oracle at truth → the weighted
  search misbehaves away from truth (consistent with the 2026-07 WNNLS notes).
- 03:40 (11-04) baseline s1 vs baseline2 per item: ALL 36 differ — most in the 3rd decimal (r², T), a few basin
  flips (Fe run 01105e 0.84→1.00, 90143c 0.79→0.51, eae9d5 0.28→0.54; silicate 0.50→0.33 with T 9.0→10.1 kK,
  ne 16.5→17.1). Either the option plumbings moved off-path numerics between the two syncs or the search is
  non-deterministic; baseline3 (identical code) running to decide. Chain B done (nearest family, bo, bic-auto,
  bo-bic). Verdicts held until baseline3.
- 04:10 (11-04) DETERMINISM: identical item twice locally → bit-identical (so s1→s2 baseline drift = off-path code
  drift between syncs; reference for series 2/3 = baseline2). Parity test landed (tests/test_forward_design_
  parity.py, 668 tests). Provisional verdicts → reports/2026-11-04-ab-wave2-verdicts.md: conserve+partition+ipd
  improves every headline mean on all panels (syn 0.156→0.136, real r² +4/−1, Fe 0.398→0.571, spurious
  1.15→0.89); nearest/wn not adopted end-to-end; bic-auto/bo mixed. PRODUCTION-SETTINGS confirmation
  (n_calls 40, draws 8) launched on moissanite: baseline-prod vs conserve-partition-ipd-prod. Still running:
  baseline3 (moissanite determinism), chain C (wn family).
- 05:00 (11-04) DETERMINISM ON MOISSANITE: baseline2 vs baseline3 (identical code) differ ONLY in runtime_s on all
  36 items → panel deterministic; baseline2 is the valid reference for series 2/3; the s1→s2 drift (all 36
  items, mostly 3rd decimal + a few basin flips) is OFF-PATH CODE DRIFT between syncs — the per-option
  "bit-identical when OFF" checks covered single cases, not the panel. Caveat recorded in the verdict report.
- 06:30 (11-04) PRODUCTION CONFIRMATION (n_calls 40, draws 8): conserve+partition+ipd vs baseline — syn median
  rel err 0.205→0.147 (+4/−1), |T err| 0.065→0.075 (+3/−2), real r² 0.477→0.486 (+4/−1), stage dis. 0.457→0.487
  (+1/−4), Fe 0.398→0.562 (+14/−8), n_non_fe 1.08→0.92 (+6/−4). DEFAULTS FLIPPED at pipeline/CLI level
  (AnalysisConfig, analyze_spectrum, analyze_directory, _get_sb, alibz-analyze --overlap-mode); library class
  defaults unchanged. DECISIONS.md entry; manuscript results/conclusions updated (49 pp). Full fast suite +
  fast round trip running with the new defaults.
- 07:30 (11-04) OFF-PATH DRIFT BISECTED (worker, reports/2026-11-04-offpath-drift-bisection.md): two UNFLAGGED
  default-path changes in peaky_indexer_v3.py — INNER_STAGE_COUPLING_WEIGHT = 0.15 soft stage-tie rows (always
  on) and run(stark_width_weight) 0.0→0.05 — reverting both reproduces HEAD bit-for-bit; all opt-in groups
  neutral when OFF. Locally: coupling HURTS binary (0.1474 → 0.1058 without it), Stark HELPS silicate (0.5114 →
  0.3802, T→10040 K, ne→17.08). Both now GATED (inner_stage_coupling_weight=0.0, stark_width_weight=0.0 in
  the indexer, AnalysisConfig, analyze_spectrum/_directory, run_kwargs); HEAD-equivalence verified (binary
  0.1701 with all options off). CONSEQUENCE: every series-2/3 verdict and the prod confirmation were measured
  with both ON → SERIES 3 launched on moissanite (true-HEAD baseline `head`, `head-stark`, `cpi` = new
  defaults, `cpi-stark`, `cpi-coupling`, `cpi-both`); the adoption decision will be re-confirmed on it.
  Full fast suite + fast round trip re-running locally on the gated tree.
- 10:30 (11-04) SERIES 3 (true HEAD): head = s1 baseline exactly; cpi improves every headline (syn 0.207→0.156,
  T 0.082→0.069, real r² 0.471→0.478 +4/−1, stage dis. 0.515→0.500 +4/−1, Fe 0.404→0.546 +13/−11); Stark
  0.05 mixed (syn 0.136 but T err/stage dis. worse); coupling 0.15 only helps the metric it manipulates →
  both stay OFF. Weighted family rejected in every combination (real r² collapses). FINAL ADOPTION recorded
  (DECISIONS addendum); verdict report + manuscript updated to the clean numbers; gated-tree suite 668 + 5/5.
  prod2 (n_calls 40) head vs cpi launched on moissanite for the paper's production-settings statement.
- 12:40 (11-04) PROD2 (clean, n_calls 40): head→cpi syn 0.213→0.170 (+3/−2), T err equal, one synthetic dominant
  flips wrong, real r² 0.472→0.488 (+4/−1), stage dis. 0.502→0.489, Fe 0.427→0.563 (+13/−11), n_non_fe 0.96→1.08
  (+5/−7). Adoption stands; manuscript results sentence finalised with the production numbers and both costs.
  ROUND CLOSED. Nothing committed (owner).
- 09:30 (11-04) WAVE 3 LAUNCHED (resume): remaining ranked ideas triaged — chosen: (A) per-peak/per-line-Stark width
  overlap kernel (model target; specialist), (B) outer-state polish after GP (ml ingest #7; worker), (C) two-T stage
  thermometer + LTE gap flag (cflibs #2 / theory #3,#5; worker). Deferred with reasons: oxide closure (no ground truth
  on the real panel), one-point calibration (overfit risk, needs CRM hold-out), SA Cσ/branching gates (suite S has no
  SA truth), LOD/LOQ reporting (FDR analogue rejected), fit_all speed (runtime only), p,T air-index audit (25 pm,
  absorbed by registration). Plumbing pre-added in the main tree so three agents can work in copies without merge
  conflicts; verified by import smoke test + 20 nearby tests. Copies: scratchpad w3k/w3p/w3t (db symlinked; jobs/
  17 GB and runs/ excluded after a first fat copy filled moissanite's disk to 100 % — fixed, 51 GB free).
- 09:35 (11-04) W3 THERMOMETER (C) VERIFIED by coordinator from reports/fit-panel/w3t-{base,offset,lte}.json: base = cpi-s3
  on 36/36; two-T offset (max 2000 K) REJECT — synthetic |T err| 0.069→0.138 (3/5 worse), Fe 0.546→0.523 (8/26 worse),
  real stage dis. 0.500→0.685 (5/5 worse), and δ rails at 2000 K on 3/5 single-T synthetic scenes (absorbs outer-state
  error, not a second temperature); LTE gap flag NO-OP on this panel (36/36 identical; only single-stage gas species
  flagged, ΔE proxy hits fine-structure levels for Fe). Suite on w3t tree 673 passed. w3t-both chained on moissanite
  (cannot win; recorded when it lands). Report thin → compare tables appended by coordinator.
- 09:50 (11-04) w3t-both landed (chain script): headline-identical to w3t-offset → thermometer chunk closed (offset
  REJECT, LTE flag NO-OP). Thermometer agent stopped (it re-sent the same hand-back on every leftover monitor).
- 09:55 (11-04) W3 WIDTH KERNEL (A) VERIFIED by coordinator from reports/fit-panel/w3k-*.json: base = cpi-s3 36/36; suite
  674 passed / 2 xfail. peak+stark+conserve: syn 0.156→0.127 (3/2), |T err| 0.069→0.073 (3/2), spurious 0→8.5e-5 (1 worse),
  real r² 0.478→0.482 (3/2), stage dis. 0.500→0.445 (3/2), Fe 0.546→0.563 (12/11), n_non_fe 1.154→1.000 (3/0); runtime not
  worse (median 70 s vs 93 s). peak alone mixed; both nearest variants worse. Oracle finding: merged-oracle peaks sit at
  Δλ≈0 where every width gives overlap 1 → the parity report's "28 % → <5 %" target is not a well-posed oracle quantity
  (agent measured ~4 % for median and peak); the panel is the only arbiter. Production confirmation AUTHORISED and launched
  by coordinator on ~/alibz-w3k (n_calls 40, draws 8, 32 workers) → compare vs cpi-prod2.
- 10:00 (11-04) PATCHES APPLIED to the main tree (patch -p0, no rejects): reports/patches/2026-11-04-w3k.patch (kernel impl,
  tests/test_overlap_kernel.py, lambda-signature fix in tests/test_peaky_indexer.py) and 2026-11-04-w3t.patch (thermometer
  options impl, tests/test_stage_thermometer_options.py, panel records δ/flags). All options remain OFF by default; the
  rejected thermometer options stay as documented flags (same policy as huber/sparse/fdr). w3p pending.
- 10:07 (11-04) [earlier wave-3 times above are coordinator estimates, clock-corrected] W3 KERNEL PRODUCTION (vs cpi-prod2): MIXED → not adopted. syn 0.170→0.174 (3/2), |T err| 0.077→0.068 (3/2),
  dominant 0.8→1.0 (binary flip repaired), real r² 0.489→0.509 (4/1), stage dis. 0.489→0.474 (2/3), Fe 0.563→0.545 (10/14 —
  fails the 11/26 limit), n_non_fe 1.08→1.12. Kernel stays an option (overlap_kernel="peak+stark"); appended to the report.
- 10:35 (11-04) MERGED TREE (plumbing + w3k + w3t + w3p patches, all options OFF) synced to moissanite ~/alibz and
  scored as `w3-merged-base`: 0/36 items differ from cpi-s3 (149 aggregate rows unchanged, 0 errors) → the default
  path is bit-identical after wave 3 (reports/fit-panel/w3-merged-base.{json,md}). Manuscript 07-results updated with
  the wave-3 paragraph (49 pp, compiles clean); DECISIONS.md wave-3 entry written; memory saved. Local full suite on
  the merged tree running.
- 11:05 (11-04) MERGED-TREE SUITE: 684 passed, 10 skipped, 2 xfailed, 71 subtests (29.5 min) = 668 + 6 kernel + 5 thermometer
  + 5 polish. WAVE 3 CLOSED. Nothing committed (owner).
- 12:20 (11-04) Owner asked for the moissanite scratch trees to go. Verified first: no process running from them; all 14
  panel JSONs present locally, 13 byte-identical, w3t-both differs only in created/runtime_s (a second run of the same
  config, every fit metric equal). Deleted ~/alibz-w3{k,p,t}; ~/alibz canonical tree intact; root disk 51 GB free.
- 15:30 (11-04) MANUSCRIPT FIGURES: F1-F5 from real data via capture_stages; see reports/2026-11-04-ms-figures-plan.md (log) and DECISIONS.md 2026-11-04. Nothing committed.

- 16:44 Fe hot-basin root cause (fit_peaks window gate on native grids) + VIS thermometer assessment: reports/2026-11-04-fe-hot-basin-root-cause.md
- 2026-11-05 11:30 native gate fixed (F5 cache regenerated; T 20.9 kK, Na 0.81 — caption/prose to re-derive after the solve fix): reports/2026-11-05-native-gate-fix.md; pipeline assumption survey: reports/2026-11-05-pipeline-assumption-survey.md
