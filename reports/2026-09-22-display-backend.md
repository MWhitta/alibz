# Analyzer display backend

## Outcome

Implemented the guarded analyzer display API in the staged checkout
`/private/tmp/pantheum-display-controls-20260922`.

- `GET /api/analyzer/display` returns a TTL-cached helper observation plus
  checkout-aware Sleep and Wake permissions. A status helper failure degrades
  to `unknown` so the current hardware-checkout holder can still attempt Wake.
- `POST /api/analyzer/display` accepts exactly `{"action":"sleep"}` or
  `{"action":"wake"}`. Sleep requires a fresh known-awake observation and is
  refused for active/uncertain physical work or any live `awaiting_data` run.
  Wake is a recovery action, so the same holder may use it through a pinned
  safety hold.
- The reservation guard remains held from blocker evaluation through helper
  completion, serializing Sleep with acquisition submission. The configured
  argv has exactly one `{action}` placeholder, uses `shell=False`, discards
  stderr, reads at most 16 KiB of stdout through a deadline-bound pipe, and
  never publishes configured paths or helper diagnostics.
- Helper observations require exact, internally consistent fields and a
  timezone-aware timestamp no more than 60 seconds old or 5 seconds in the
  future. Sleep must verify off and Wake must verify on. A dispatched command
  with no trustworthy verification is audited as `uncertain` in the separate
  `display_actions` table and does not poison the physical-operation ledger.
- Example configuration keeps the feature disabled and documents a fixed
  command, a 45-second timeout, and a 10-second status TTL.

## Files

- `pantheum/alibz/display.py` (new)
- `pantheum/alibz/service.py`
- `pantheum/alibz/__main__.py`
- `tests/test_alibz_display.py` (new)
- `config/alibz.example.json`

## Verification

- `python3 -m unittest -v tests.test_alibz_display.DisplayTests`: 12 passed.
- `python3 -m unittest -v tests.test_alibz_display.DisplayHTTPTests`: 3 passed
  with localhost permission enabled (the restricted sandbox blocks socket
  binding).
- `python3 -m py_compile` passed for the display, service, and route modules.
- `python3 -m json.tool config/alibz.example.json` passed.

No live display mutation was made. No provider switch occurred.
