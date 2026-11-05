# LRA capture verification — stop banking impostor PDFs

Date: 2026-11-03
Repo: `/Users/mwhittaker/Projects/github/literature-retrieval-agent` (branch `fix/z300-pixel-offset`, dirty tree, no commit)
Defect: the magic-byte check banked 54/140 single-page previews / wrong docs / twins.

## Outcome

Implemented items 1–3 plus the Elsevier amendment. Tests
(`.venv/bin/python -m pytest -q`): **before 3 failed / 156 passed → after 3
failed / 168 passed**. The 3 failures are the same pre-existing ones
(`test_direct_url_locator_arxiv`, `test_direct_url_locator_custom_rule`,
`test_autonomous_open_access_via_direct_url`); everything else green. +12 new
tests. Did NOT run `lra verify` on the alibz corpus (that is the coordinator's
run) and did not touch `manuscript/lit/`.

## Changes (file:line)

### New statuses — `models.py`
- `models.py:80` `FetchStatus.TEXT = "text"`, `:82` `SUSPECT = "suspect"`.
- `models.py:109` `ok` now true for CAPTURED **or** TEXT.

### Elsevier: view=FULL XML first, then a page-checked PDF — `locators.py`
- `locators.py:51` `TextCapture` dataclass (hook signal: full text, no PDF).
- `locators.py:735` `_xml_to_text` (dependency-free tag-strip, keeps paragraphs).
- `locators.py:891,913,917` `ElsevierApiLocator` gains `fulltext_dir` and
  `try_full_text` (default True; tests scripting only the PDF endpoint pass
  `try_full_text=False`).
- `locators.py:936` `pre_navigation_capture` now returns `TextCapture` | `True`
  | `False`. `:970` `_get_full` GETs `?view=FULL` JSON; if `originalText` is a
  str ≥ max(20000, 4×abstract) → `:1004` `_save_full_text` writes
  `<dl>/<name>.elsevier.xml` verbatim + `<fulltext>/<name>.txt`, returns
  `TextCapture`. `:1022` `_get_pdf` is the old PDF path **plus** a preview
  guard: keep a PDF only when its page count matches `prism:pageRange` (±1),
  else delete it and return False (never a strike). `:1085` `_page_count`
  (PyMuPDF, best-effort).
- `locators.py:1167` `ChainLocator.pre_navigation_capture` now propagates the
  child's return value (was coerced to `True`) so a `TextCapture` survives.
- Entitlement-endpoint gate: implemented as advisory-only by **not** adding a
  dedicated call — the PDF endpoint's 403 AUTHENTICATION_ERROR is already
  logged by `_report_non_200`, and a FULL non-200 is reported and never
  retires the API. (Deliberate deviation from the first message, consistent
  with the amendment demoting it to advisory; avoids a useless 403 round-trip
  every run. Noted for review.)

### Post-capture verification — new `verify.py`
- `verify.py` `CaptureVerifier` (injectable `extract_text` + `crossref_lookup`):
  (a) DOI string present (case-insensitive) OR ≥60% of 4+-char title words
  present in the first 2 pages (title from `Source.title` else Crossref);
  (b) ≥2 pages unless the Crossref `page` range says 1; (c) sha256 not identical
  to another capture this run or already in the download dir. `parse_page_count`
  turns a numeric range (`"671-679"`) into a count. Degrades to DOI/page checks
  when Crossref is unavailable.

### Wiring — `retriever.py`
- `retriever.py:123` ctor `verify=False` default + injectable `verifier`
  (CLI turns it on; direct-construction tests stay unverified).
- `:288` `fetch_batch` runs `_verify_capture` before `_ensure_fulltext`.
- `:578/607` `_try_api_capture` maps a `TextCapture` to a `FetchStatus.TEXT`
  result (fulltext already written, no PDF) and clears any stale suspect.
- `:356` `_fetch_one` treats `<name>.elsevier.xml` as done under
  `--skip-existing` (returns TEXT, never routed through the PDF/OCR path).
- `:764/773/811` `_get_verifier`, `_verify_capture` (renames a failing single
  PDF to `<name>.suspect.pdf`, status SUSPECT, drops its fulltext), `_clear_suspect`.

### CLI + report — `cli.py`, `stores.py`
- `cli.py:166` `--no-verify`; `:289` `verify=not --no-verify`; `:255`
  `ElsevierApiLocator(fulltext_dir=settings.fulltext_dir)`; `:302` `text` counts
  as success in `_summarize`; `:349` `cmd_verify` + `:550` the `verify`
  subparser (re-check existing files, rename failures, write JSON, optionally
  drop stale fulltext).
- `stores.py:93/103/105` report gains a top-level `text` count and a `suspect`
  list (`{name, reason}`); SUSPECT/TEXT appear in `summary` automatically.

## Tests (new)
- `tests/test_verify.py` — parse_page_count; DOI-present ok; title-words ok;
  1-page-of-12 → suspect; byte-identical twin → suspect; no-network degrade;
  retriever capture pass→CAPTURED, fail→SUSPECT (file renamed, stale fulltext
  dropped); `lra verify` CLI renames only the suspect and writes a report.
- `tests/test_elsevier_fulltext.py` — long `originalText` → TextCapture (xml +
  txt written, PDF endpoint never hit); short originalText → falls through to
  the PDF endpoint; 1-page PDF with pageRange 671-679 → deleted, not banked,
  `_dead` stays False.
- `tests/test_challenge_breaker.py`, `tests/test_elsevier_resilience.py` —
  existing Elsevier locators set `try_full_text=False` (they script only the
  PDF endpoint's HTTP sequence); intent unchanged, all green.

## Validated by hand
`lra verify` over two scratch PDFs with a stub verifier: suspect renamed to
`<name>.suspect.pdf`, good left intact, report JSON carries
`summary={suspect:1,captured:1}`, top-level `text:0`, and
`suspect=[{name, reason}]`.

## Unverified / caveats
- No live Elsevier key / Crossref / network here: the view=FULL flow, the real
  Crossref title/page lookup, and PyMuPDF page-counting on real PDFs are
  covered by injected stubs only. The amendment's measured behaviour (FULL
  returns `originalText`; PDF endpoint returns 1-page previews; entitlement
  endpoint 403) was taken as given.
- `_clear_suspect` removes `<name>.suspect.pdf` when text / a verified PDF
  lands; it does not delete a stale non-suspect `<name>.pdf` (could be a real
  other capture), so a leftover preview `<name>.pdf` from before this change
  would need the `lra verify` pass to quarantine it.
- The `lra verify` corpus run over `manuscript/lit/{downloads,fulltext}` is the
  coordinator's to run (I did not touch that tree).
- Provider: Claude subscription, no switch this task.

---

## Follow-up (2026-11-03): false-positive & root-cause fixes

Driven by the live `lra verify` over the corpus (74 captured / 65 suspect /
10 no_pdf). Tests: **3 failed / 168 passed → 3 failed / 178 passed** (same 3
pre-existing failures; +10 tests). No commit; manuscript/lit untouched.

### 1. Identity-check false positives (good files were flagged)
- **arXiv** (`verify.py:264` `arxiv_id` param, `:215` `_arxiv_identity_ok`,
  `:91` `_default_arxiv_lookup`; `locators.py:95` `arxiv_id_of`): identity now
  passes when the arXiv id (version-stripped) appears in the first pages, or
  when the arXiv Atom API title matches by the 60% rule. Wired in
  `retriever.py` (`arxiv_id=arxiv_id_of(src.url)`) and `cli.py cmd_verify`.
- **Image-only scans** (`verify.py:277` <50-word fallback, `:224`
  `_scan_fallback_text`, `:104` `_default_ocr_text`): when the first two pages
  yield < 50 words the verifier augments the identity text with an existing
  `<fulltext>/<name>.txt` and a fresh `pdftoppm`+`tesseract` OCR of the first
  pages (both injectable; skipped gracefully if the tools are absent) before
  deciding. `lra verify` still deletes a suspect's fulltext txt, but only after
  that read (verify runs first), and the verifier is built with
  `fulltext_dir=` in both the retriever and `cmd_verify`.

### 2. Genuine impostors — fixed at the cause
- **MDPI flyer** (`locators.py:121` `_mdpi_pdf`, `:298` rule): the article PDF
  is the landing URL + `/pdf`, added as a deterministic ResolvedUrl rule so
  MDPI never scrapes/clicks the 4-page "journal flyer" advert.
- **Flyer/help/platform/journal-index links never clicked or scraped**
  (`locators.py:114` `is_excluded_link`, `:438` ScrapedHref skip;
  `browser.py:30` import, `:847` `_element_is_excluded`, `:831` applied in
  `click_and_download`): matches `guide|help|platform|flyer|/journal/` in the
  href or link text. This is what banked the AIP "Content Platform User Guide".
  No AIP deterministic rule added (AIP/Silverchair needs the landing page +
  cookies; the exclusion makes those sources fall through to NO_PDF rather than
  bank the wrong doc).
- **Journal mismatch** (`verify.py:301`): when identity already fails and the
  Crossref `container-title` (now fetched, `verify.py:83`) is also absent from
  the first pages, a `journal mismatch (expected ...)` reason is appended. It
  only augments an already-suspect file, so it cannot create new false
  positives on the 74 good captures. (The janesick SPIE case is independently
  caught by the byte-identical-twin check.)
- **palleschi twin**: already caught by the twin check; no change.

### Tests added
`tests/test_locators.py`: MDPI `/pdf` rule; MDPI flyer path ignored;
ScrapedHref skips a flyer and captures the article PDF; `is_excluded_link`
patterns; `arxiv_id_of` variants. `tests/test_verify.py`: arXiv id-in-text ok;
arXiv API-title ok; scan→OCR fallback ok; scan→existing-fulltext-txt ok;
journal-mismatch reason appended. Existing stub verifiers updated for the new
`arxiv_id` kwarg.

### Unverified / caveats (follow-up)
- No live network/tools here: the arXiv API, Crossref `container-title`, and
  the real `pdftoppm`+`tesseract` OCR path are covered by injected stubs only
  (the default OCR skips gracefully when the tools are absent).
- The click-exclusion in `browser.py click_and_download` reads each candidate's
  href/inner_text via Playwright; it is unit-tested only at the pure-function
  (`is_excluded_link`) and locator (ScrapedHref) level, not against a live
  browser.
- No AIP deterministic PDF rule (needs the authenticated landing page); the
  exclusion prevents the wrong-doc capture but AIP will report NO_PDF until a
  landing-page rule exists.

---

## Follow-up (2026-11-03): rehabilitate existing suspects

Gap: `lra verify` reported existing `<name>.suspect.pdf` as no_pdf, so a
false positive could never recover without a re-fetch. Fix: both `lra verify`
(`cli.py cmd_verify`) and the retriever's `--skip-existing` path
(`retriever.py:_rehabilitate`, called from `_fetch_one`) now take
`<name>.suspect.pdf` as the candidate when `<name>.pdf` is absent, re-run the
full verification (identity incl. arXiv/OCR, page count, twin), and on PASS
rename it back to `<name>.pdf`, regenerate `<fulltext>/<name>.txt` when a
fulltext dir is given (`cli._regen_fulltext`; the retriever does it via
`_ensure_fulltext`), and report `captured` with note `rehabilitated`
(`_verify_capture` skips a rehabilitated result). On FAIL it stays suspect with
the updated reason. Tests: 3 failed / 178 → 3 failed / 181 passed (+3:
cli promote, cli stay-suspect, retriever skip-existing promote).
