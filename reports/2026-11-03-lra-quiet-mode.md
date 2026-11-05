# LRA quiet mode — browser never steals focus on a fetch

Date: 2026-11-03
Repo: `/Users/mwhittaker/Projects/github/literature-retrieval-agent` (branch `fix/z300-pixel-offset`, dirty tree, no commit)
Task: stop the browser window coming to the foreground on every fetch.

## Outcome

All four items implemented. Tests: **3 failed, 156 passed** — the 3 failures are
exactly the named pre-existing ones
(`tests/test_locators.py::test_direct_url_locator_arxiv`,
`::test_direct_url_locator_custom_rule`,
`tests/test_retriever.py::test_autonomous_open_access_via_direct_url`);
everything else green. Baseline before my changes: **3 failed, <N> passed**
(measured 142 on disk; the brief's "140" predates two of my earlier uncommitted
tests — the failing set is identical). I added 14 new tests (142 → 156).

Command (run before and after, from the repo root):
`.venv/bin/python -m pytest -q`

## Changes (file:line)

### 1. Reveal only when attended
- `src/.../retriever.py:118` — new ctor param `reveal_on_challenge: bool = False`.
- `src/.../retriever.py:164` — stored as `self.reveal_on_challenge`.
- `src/.../retriever.py:952` — `_resolve_challenge` now reveals only when
  `self.reveal_on_challenge and callable(reveal) and not self._is_blocked(key)`;
  tracks a local `revealed` flag.
- `src/.../retriever.py:976` — conceal only `if revealed and callable(conceal)`.
- Printed challenge lines are unchanged (still always printed).
- `src/.../cli.py:201,212,278` — `explicit_pause = getattr(args,
  "pause_on_challenge", False)`; `reveal_on_challenge = explicit_pause`; passed
  to the retriever. So ONLY an explicit `--pause-on-challenge` reveals; the
  implicit challenge handling from `--automate-portal` / `--auto-open-access`
  does not. `reveal_on_challenge=True` + no background reproduces the old
  behaviour exactly.
- Tests (`tests/test_retriever.py`): FakeSession gains `reveal_calls` /
  `conceal_calls` (+ `reveal_for_human`/`conceal_after_human`);
  `_challenge_retriever` gains `reveal`/`strikes`. New:
  `test_unattended_challenge_never_reveals` (0 reveals),
  `test_attended_challenge_reveals_and_conceals_once` (1/1),
  `test_attended_challenge_timeout_still_conceals` (1/1 on timeout),
  `test_attended_blocked_pair_is_not_revealed_again` (blocked pair → no reveal).

### 2. Lazy browser launch + page-free API capture first
- `src/.../retriever.py:54` — new `_LazySession` wrapping a factory; `__enter__`
  returns self WITHOUT launching; `__getattr__` builds+enters the real session
  on first use and prints `  launching browser (first source needing a page:
  <name>)`; `opened` property; `__exit__` tears down only if opened;
  `current_source` settable attr.
- `src/.../retriever.py:255` — `fetch_batch` uses
  `_LazySession(lambda: BrowserSession(self.settings))` (factory reads the
  module attr at call time → the `retriever_mod.BrowserSession` patch still
  works).
- `src/.../retriever.py:305` — prints `  browser was never launched` when
  `not session.opened`.
- `src/.../retriever.py:230` — `fetch_batch` now does
  `download_dir.mkdir(...)` itself (the browser used to create it; a page-free
  batch never enters the browser).
- `src/.../retriever.py:545` — new `_try_api_capture(session, item, target)`:
  runs the `pre_navigation_capture` hook before any page exists; on success
  applies `_reject_non_pdf` (if `require_pdf`), `_name_captured`,
  `_record_capture`, returns the result; else `None` to fall through.
- `src/.../retriever.py:592,598` — `_attempt_once` calls `_try_api_capture`
  FIRST, before `session.snapshot()`/`new_page()`; sets `session.current_source`
  just before the page-needing access.
- `src/.../retriever.py:_try_pre_navigation` — the capture-hook block removed;
  it now only does the page-needing deterministic-URL fast path. `_authenticate`
  already touches the session only for `needs_prelogin` (unchanged).
- Note on signature: the brief wrote `_try_api_capture(item, target)`; I kept a
  `session` param because the hook contract is `cap(session, item, target)`. The
  hook (Elsevier / the test stubs) ignores `session`, so passing the lazy
  session never triggers a launch. This is a faithful, minimal deviation.
- Tests: `test_api_only_batch_never_constructs_browser` (factory raises if
  called — never does), `test_mixed_batch_constructs_browser_once` (built
  exactly once, at the first page-needing source). Existing retriever tests
  unchanged in outcome.

### 3. macOS: hide the Chrome process in `--background`
- `src/.../browser.py:32` — `_run_argv(cmd) -> (rc, stdout)` default runner
  (short-timeout subprocess, swallows errors).
- `src/.../browser.py:148,156,161` — ctor gains injectable `runner`; stores
  `self._run_cmd` and `self._macos_pid`.
- `src/.../browser.py:411` `_should_hide_macos` (darwin + background + not
  headless + `LRA_NO_HIDE` unset); `:421` `_find_browser_pid` (pgrep on the
  unique profile arg); `:440` `_set_app_visible`; `:451` `_set_app_frontmost`;
  `:461` `_rehide_macos`; `:472` `_maybe_hide_macos`.
- `src/.../browser.py:360` — `__enter__` calls `_maybe_hide_macos()` after
  launch. `:499` reveal un-hides + fronts by pid. `:517` conceal re-hides by
  pid. `:528` `new_page` calls `_rehide_macos()`.
- **Two corrections found during the live smoke test (see below):**
  1. `browser.py:430` — the pgrep pattern must NOT start with `--`
     (`user-data-dir={profile}`, dashless): pgrep otherwise parses a
     dash-leading pattern as its own option and never matches — so the hide
     silently failed (`could not find the browser process to hide`).
  2. `browser.py:528` — `__enter__`-only hiding (as the brief specified) is
     undone the instant `new_page()` opens a window (macOS un-hides the whole
     app and places the new window on-screen). Verified directly: no page →
     `visible=false`; after `new_page()` → `visible=true`; explicit re-hide with
     a window open → `visible=false` and it sticks. So I re-hide after each page
     open (skipped while revealed for a human). This is the minimal change
     needed to make the brief's own acceptance criterion (b) actually hold.
- Tests (`tests/test_browser_overlay.py`): injected `RecordingRunner`; 8 new
  tests for pid/visibility/should-hide/maybe-hide/reveal+conceal/rehide/off-darwin
  — no real `osascript`/`pgrep`.

### 4. README
- `README.md` — new "Quiet operation" subsection under the blocked-hosts
  section: unattended modes never bring the window forward; `--pause-on-challenge`
  is the only mode that does; an API-only batch launches no browser; macOS
  `--background` hides the Chrome process (`LRA_NO_HIDE=1` to disable); Linux
  fully-invisible headed run via `xvfb-run -a lra fetch …`.

## Smoke test (real browser)

Scratch profile `…/scratchpad/quiet-test/profile`, command:
`lra fetch --auto-open-access --background --download-dir … --profile-dir …
--report … 1706.03762`. Orchestration + evidence under
`…/scratchpad/quiet-test/` (`smoke.sh`, `evidence.txt`).

- (a) PDF captured while hidden: YES — `1706-03762.pdf` 2 215 244 bytes,
  `fetch_exit=0`, log `✓ saved: 1706-03762.pdf`.
- (b) `get visible` reports false during the run: YES — samples over the run go
  `true, true, false, false`: the app reaches and holds hidden once the
  per-page re-hide settles; no hide-failure notes.
- (c) user Chrome untouched: YES — the user's Chrome (pid 785) read `visible=true`
  before, through every sample, and after the run.

## Unverified / caveats

- There is a ~0.5 s window at launch/first-page where the app is briefly
  `visible=true` before the re-hide settles (samples 1–2). It is far less
  disruptive than the window coming fully forward on every fetch, but it is not
  a perfectly invisible launch; reducing it further (re-hiding on more events)
  risks fighting Chrome's activation and was left out of scope.
- `_set_app_frontmost` / `bring_to_front` on reveal were exercised only in unit
  tests (the smoke test never hit a CAPTCHA), not against a real reveal.
- Smoke test ran only arXiv (DirectUrl in-page fetch). The page-free API path
  (Elsevier) was verified by unit tests only (no API key available here).
- A package fetch was running concurrently (different profile/.chrome_session);
  I used a separate scratch profile and did not touch `strategies.json`, the
  manuscript lit dir, or `.chrome_session`.
- Provider: Claude subscription (no usage-limit switch this task).
