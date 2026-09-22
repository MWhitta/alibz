# Optimize Acquisition: composition instead of Fe-only (2026-09-22)

Repo: `/Users/mwhittaker/Projects/github/pantheum-I`. Removed the delay/period
Fe-only restriction, renamed the panel to **Optimize Acquisition**, added a
composition (element/formula/weighted-list) entry, and used the parsed elemental
composition to drive per-element scoring.

## Files changed

### New: `pantheum/alibz/composition.py`
- Stdlib-only parser. `ELEMENTS` (92 symbols H..U), `parse_composition(text)`,
  `format_composition(items)`. Formula parser (`_parse_formula`, parentheses +
  multipliers), list parser (`_parse_list`, `Sym`/`Sym:n`/`Sym=n`/`Sym n`/`Sym n%`),
  `_finalise` (fractions>0 sum to 1, ≤12 elements). Errors on unknown symbol,
  empty, >200 chars, zero/negative/non-finite weight, duplicate symbol, mixed
  weighted/unweighted.
- Primary element = first entered (formula cation-first order; list typed order),
  e.g. `Fe2O3` → primary `Fe` 0.4, `O` 0.6. See "Design note" below.

### New: `tests/test_alibz_composition.py` (11 tests)
- Element table, single element, formula variants (SiO2/CaCO3/Ca(OH)2/talc),
  weighted/unweighted lists, separators, sum-to-one, format, error cases,
  non-string input.

### `pantheum/alibz/optimization_metrics.py`
- L15-27: `REFERENCE` → `REFERENCE_DIR` module const + `reference_path(symbol)`;
  added `BLEND_NM = 0.25`.
- `_response_pairs` (~L120): pairs now require same `element` (not just upper_key).
- `analyze_batch` (~L157): new signature `analyze_batch(run_dir, composition=None,
  element='Fe', adc_max=None, min_shots=6)`; per-constituent measurement, cross-
  element blend flag `potential_blend` + `blend_elements`, per-element `elements`
  summary, `unscored_elements`, batch `score` = atomic-fraction-weighted mean over
  eligible elements, batch `eligible` = primary eligible & not saturated. Kept
  every top-level key; added `composition`, `elements`, `unscored_elements`,
  `references`, and kept `reference` = primary's entry. `method` =
  `composition-window-quality-v1`. Renamed flags `potential_Fe_blend`→
  `potential_blend`, `insufficient_Fe_line_windows`→`insufficient_line_windows`.
  Limitation string "non-Fe blends"→"cross-element blends". `_normalise_composition`
  helper added.

### `tests/test_alibz_optimization_metrics.py`
- setUp now builds a temp `REFERENCE_DIR` (`refs/`) with an fe-v1.json and patches
  `metrics.REFERENCE_DIR`; `write_reference`/`write_blocks` helpers added. Flag
  assertion renamed. Added 5 tests: single-element default, two-element weighted,
  cross-element blend, missing reference → unscored, response pairs never cross.

### `pantheum/alibz/optimization.py`
- `_migrate` (~L173): idempotent `ALTER TABLE` adding `composition`,
  `composition_text` columns.
- `defaults()` (~L207): added `'composition': 'Fe'` (kept `'element': 'Fe'`).
- `_clean_create` (~L282): allowed keys add `composition`; parse `composition`
  (else legacy `element`) via `parse_composition`; store primary as `element`;
  delay/period requires `reference_path(primary).exists()` else ValueError
  ("No bundled line reference for element X..."). Return adds `composition`,
  `composition_text`.
- `create` INSERT (~L455): writes the two new columns.
- `_composition_fields` staticmethod (~L462) derives (list, text) from a row,
  falling back to `element` for old rows.
- `_session_from_row` (~L500): sets `composition`/`composition_text`.
- `list_sessions` (~L520): projection adds `composition`, `composition_text`.
- `_analyze(run_dir, composition)` (~L580) passes `composition=` to analyze_batch.
- `_reconcile` scoring call (~L810) passes `self._composition_fields(session)[0]`.
- Detail/export strings: "usable Fe lines"→"usable primary-element lines";
  provenance method "Fe metrics"→"composition metrics".

### `tests/test_alibz_optimization.py`
- Removed `request(element='Cu')` from the generic bad list; added two composition
  bad cases. New `test_delay_period_requires_a_bundled_reference_for_the_primary_element`
  (empty temp REFERENCE_DIR refuses; synthetic cu-v1.json accepts) and
  `test_compound_composition_sets_primary_element_and_fractions` (Fe2O3 → element
  Fe, comp Fe 0.4/O 0.6).

### `web/alibz/index.html`
- Eyebrow "ACQUISITION STUDY"→"OPTIMIZE ACQUISITION"; h2 "Stepped acquisition
  study"→"Optimize Acquisition"; section description mentions composition scoring;
  Element input replaced by `#optimization-composition` (value `Fe`, placeholder
  + title explaining the two forms).

### `web/alibz/app.js`
- `applyOptimizationDefaults`: uses `defaults.composition` (fallback `element`) →
  `#optimization-composition`.
- Title row: `session.composition_text || session.element`.
- `renderOptimizationInspection`: generic summary text; per-element table
  (element/fraction/lines/median SNR/score/eligible) + unscored-elements note.
- `createOptimizationSession`: reads composition; validation "Sample and
  composition are required."; payload sends `composition` (no `element`).

### `tests/test_alibz_ui.cjs`
- `optimization-element`→`optimization-composition` (3 spots); create test asserts
  `body.composition === 'Fe'` and `body.element === undefined`.

### Docs / decisions
- `docs/acquisition-optimization.md`: removed Fe-only statements, describes
  composition input, primary element, per-element weighted scoring, blend flag,
  reference-availability rule.
- `docs/alibz-architecture.md`: optimization section updated (composition.py,
  per-element references, weighted score, primary-reference requirement).
- `DECISIONS.md`: appended 2026-09-22 entry (appended only; existing edits
  untouched).
- `deploy/alibz/README.md`: no change — it contains no Fe-only statement or the
  old panel name (grep confirmed).

## Tests

Baseline (HEAD working tree, before changes):
- `python3 -m unittest discover -s tests` → **Ran 873, OK (skipped=26)**, exit 0.
- `node --check web/alibz/app.js` → OK; `node --test tests/test_alibz_ui.cjs` → 42 pass.

After changes (targeted):
- `python3 -m unittest tests.test_alibz_composition` → **11 pass**.
- `python3 -m unittest tests.test_alibz_optimization_metrics` → **13 pass**.
- `python3 -m unittest tests.test_alibz_optimization` → **29 pass**.
- `node --check web/alibz/app.js` → OK.
- `node --test tests/test_alibz_ui.cjs` → **42 pass, 0 fail**.
- Full `python3 -m unittest discover -s tests` → **Ran 891, OK (skipped=26)**, exit 0 (baseline 873; +18 new tests, no regressions).

End-to-end (mirrors test service, simulate delay_period, composition `Fe2O3`):
```
created: True
element: Fe
composition: [{"element": "Fe", "fraction": 0.4}, {"element": "O", "fraction": 0.6}]
composition_text: Fe2O3
study_type: delay_period
```

## sha256 (BEFORE `git show HEAD:path` → AFTER working tree)
- pantheum/alibz/composition.py: (new) → 3a2c5251d88844d0efc21161aa2c02b26961cf433d46d62ce53a10703e980751
- pantheum/alibz/optimization.py: 601fb175b12075d8632f61f710f23136b4e573a8443bf491503f8c93649511f9 → e2b31880ab3b9c4499b4282cd918f5239f06d92fa8bf39a4f51a29d192c57dee
- pantheum/alibz/optimization_metrics.py: 37e45156c66bddc72493e65c39e42839bf4dd65d9b3f097efffa121e3ff83da9 → 6dc6bbd215ad0874818e9428f9f4cdadc1ecce7e231912a0c7938ece971cb5af
- web/alibz/app.js: 7518f5ad48ff5b3fcf53eae7728bcd3462a7bd172aeb06c3302f416f86d79ee1 → c2607bfbc0525bbfdb721dfb3ca0486c2b64f3f58206f0eb99ba671db8bd7156
- web/alibz/index.html: 26585c26bc078b754ae766002e05aaaf840ba40938359acbf36bad398f8c208b → c399b375a67a5179753355d72e88d0bf3aad310e33afe2b4a9f3a4a7f1953eac

## Design note / flag for review (single deviation)
Section A of the brief says `parse_composition` sorts "by fraction desc then input
order (so [0] is the primary element)", but section F's example and the required
E2E both expect `Fe2O3` → primary `Fe` (Fe 0.4 < O 0.6). Sorting by fraction
would make **O** primary, which would (a) contradict the F test and (b) make the
delay/period create for `Fe2O3` fail (no `o-v1.json`), breaking the required E2E.
I implemented **input-order primacy** (primary = first entered; formula keeps
cation-first order) so `Fe2O3`→`Fe`, matching F, the E2E, and the domain intent
(oxide analyte = cation). Consequence: a *weighted list* typed out of magnitude
order (e.g. `Cr 20, Fe 70`) keeps `Cr` primary rather than the larger `Fe`; no
test exercises that case. If fraction-desc primacy is preferred for lists, this
is the one spot to revisit.

## Not verified
- Sibling-added `references/<el>-v1.json` files: not created/edited here; nothing
  in this change depends on them existing (tests use temp REFERENCE_DIR or Fe).
- No live-hardware path exercised (simulate only).
