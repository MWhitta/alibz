# LRA challenge breaker + proxied-host + Elsevier API

Repo: /Users/mwhittaker/Projects/github/literature-retrieval-agent
Branch: fix/captcha-timer-and-dedupe (working tree was already DIRTY; built on
top, no checkout/stash/reset, no commit).

## Test counts (`.venv/bin/python -m pytest -q`)
- Baseline (before): 3 failed, 112 passed.
- After: 3 failed, 126 passed (14 new tests, all pass).
- The 3 failures are EXACTLY the pre-existing ones, unchanged:
  - tests/test_locators.py::test_direct_url_locator_arxiv (AttributeError: FakeSess has no goto_and_capture)
  - tests/test_locators.py::test_direct_url_locator_custom_rule (same)
  - tests/test_retriever.py::test_autonomous_open_access_via_direct_url (FakeSession has no goto_and_capture)
  These stem from the in-flight uncommitted work (DirectUrlLocator `_capture`
  now falls through to `goto_and_capture`, which those two test fakes omit). Not
  touched per brief.

## Changes (file:line)

### Part 1 — host challenge breaker
- models.py:77 — `FetchStatus.BLOCKED = "blocked"`.
- retriever.py:65 — `__init__(..., challenge_strikes: int = 2, ...)`.
- retriever.py:~113 — stores `self.challenge_strikes`; inits per-run state
  (`_strikes`, `blocked`, `_key_remaining`, `_last_challenge_reason`).
- retriever.py:~149 — `fetch_batch` resets that state and fills `_key_remaining`
  (Counter of (publisher_key, strategy) pairs still to start).
- retriever.py:~245 — `_fetch_one` rewritten: decrements `_key_remaining`;
  blocked short-circuit (escalate to fallback if its key isn't blocked, else
  return BLOCKED); escalation now triggers on NO_PDF OR a challenge-skip; a
  fallback that is also NO_PDF/challenge-skipped leaves the primary standing.
- retriever.py:325 `_key`, 334 `_is_blocked`, 339 `_is_challenge_skip`,
  343 `_blocked_result`, 350 `_record_challenge_strike` (blocks the pair once
  strikes reach the budget, prints the one `⛔` line), 374 `blocked_report`.
- retriever.py:793 `_resolve_challenge` — captures the reason from
  `detect_challenge`, records a strike on timeout, resets strikes on clear,
  suppresses `reveal_for_human` for an already-blocked pair.
- retriever.py (pause_on_challenge block in `_attempt_on_page`) — a challenge
  SKIP now returns `SKIPPED, error="challenge: <reason>"`; stays bare
  (`error=None`) when `challenge_strikes=0` (today's behaviour exactly).
- cli.py:~90 — `--challenge-strikes N` (default 2) and `--no-elsevier-api`.
- cli.py:272 — wires `challenge_strikes`.
- cli.py `_maybe_report`/call sites — passes `retriever.blocked_report()`.
- stores.py:42 `write_report(..., blocked=None)`; stores.py:99 emits top-level
  `blocked` list; `blocked` is auto-counted in `summary` and in `_summarize`.

### Part 2 — proxied-host support
- locators.py:238 `_PROXY_DOMAIN_MARKERS`; :244 `unproxy_host`; `_reproxy_host`;
  `_with_host`.
- locators.py `ResolvedUrlPdfLocator.locate` — resolves the rule on the origin
  host, runs it on the host-substituted URL, re-proxies the derived pdf_url's
  host when a proxy suffix exists; unproxied path byte-identical to before.

### Part 3 — ElsevierApiLocator
- locators.py (imports) — add `os`, `Path`.
- locators.py `PdfLocator` protocol docstring — documents the optional
  `pre_navigation_url` / `pre_navigation_capture` hooks.
- locators.py:~718 `_urllib_http_get` (default, 60 s); :736 `ElsevierApiLocator`
  (prefixes=("10.1016",), key from arg/ELSEVIER_API_KEY, insttoken from
  arg/ELSEVIER_INSTTOKEN, injectable http_get, dead-after-2, never logs key,
  `pre_navigation_capture` writes %PDF to target on 200, `locate`->False).
- locators.py:874 `ChainLocator.pre_navigation_capture` delegates to first child.
- retriever.py:434 `_try_pre_navigation` — calls `pre_navigation_capture` FIRST
  (before the proxied-strategy guard and the pre_navigation_url path).
- __init__.py — exports `ElsevierApiLocator`, `unproxy_host`.
- cli.py automate_portal chain — ElsevierApiLocator wired FIRST when a key is
  present and `--no-elsevier-api` not set (verified by smoke test).

### Part 4 — README.md
- New "Blocked hosts and challenges" and "Elsevier full-text API" sections under
  "Running unsupervised for longer", matching the existing responsible-use tone.

## New tests (tests/test_challenge_breaker.py, 14, all pass)
Breaker: blocks_pair_after_strikes, escalates_challenge_skip_to_fallback,
challenge_strikes_zero_reproduces_old_skip. Proxy: unproxy_host_decodes_ezproxy,
unproxy_host_passes_through_unproxied_and_non_dash_proxy,
resolved_locator_reproxies_derived_pdf_url, resolved_locator_unproxied_unchanged.
Elsevier: captures_pdf, 401_falls_back_without_file, dies_after_two_failures,
not_engaged_without_key, not_engaged_for_non_elsevier_doi,
chain_pre_navigation_capture_delegates_to_first_hit,
pre_navigation_capture_runs_before_proxy_guard.

## Evidence (commands run)
- Smoke test: `_build_retriever` wires challenge_strikes; automate_portal chain =
  [DeGruyter, ResolvedUrl, ScrapedHref, PortalClick] with no key, and
  [Elsevier, ...] when ELSEVIER_API_KEY set and not disabled.
- write_report: summary {'blocked': 1}; top-level `blocked` list present;
  BLOCKED result `error` = "blocked: 10.1016 via direct: captcha".

## Unverified
- Live Elsevier API and the EZproxy/ScienceDirect proxy route CANNOT be
  exercised without a real ELSEVIER_API_KEY / institutional entitlement and a
  live CalNet proxy session; both are covered only by fakes (injected http_get,
  recorded capture URL).
- `challenge_strikes=0` "reproduces today exactly" verified by test (bare
  SKIPPED, no error tag, no block, both sources navigated); the one visible
  additive change to report format is the always-present `"blocked": []` key,
  which the brief requested.
