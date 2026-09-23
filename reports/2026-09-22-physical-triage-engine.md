# Physical triage engine implementation report

Date: 2026-09-22

## Outcome

Implemented a standalone, conservative physical-evidence triage engine and a
focused adversarial test suite.  The engine computes candidate rankings and
evidence only; pipeline integration and policy remain caller-controlled.
Scores are explicitly labelled uncalibrated evidence scores, not probabilities.
Rejection suggestions are conditional on the supplied peaks, coverage, and
noise model.

No provider switch occurred during this task.

## Files and changes

- `alibz/triage.py:30` adds the validated `TriageConfig` dataclass, including
  the integration fields `match_tolerance_nm` and `max_ion_stage`, a broad
  4,000--25,000 K grid, and diagnostic-only companion contradictions by
  default.  Enabling companion rejection is documented as an explicit caller
  assertion about bounded detector response, optical thinness, saturation,
  masks, and isolation.
- `alibz/triage.py:85` adds JSON-safe ratio, species, element, and result
  evidence dataclasses.  `TriageResult` exposes `keep_elements`,
  `rejected_elements`, `deferred_elements`, per-element/per-species evidence,
  `to_dict()`, and scope/score interpretation text.
- `alibz/triage.py:291` builds independent-stage strengths proportional to
  `gA / wavelength * exp(-Ek/kT)`, ranks the strongest groups over the full
  temperature envelope, merges only truly unresolved same-species lines, and
  caps each group span so dense line forests cannot merge transitively.
  Ranking occurs after wavelength-coverage filtering, so stronger VUV lines
  cannot displace observable lines.
- `alibz/triage.py:356` performs binary-search wavelength matching, counts one
  fitted peak only once per species, recognizes low-excitation bracketing
  profiles as one tentative self-reversal feature, and downweights peaks
  claimed by multiple elements.
- `alibz/triage.py:424` adds an all-line (not top-N-only) low-excitation
  reversal guard before any `no_wavelength_support` deferral.
- `alibz/triage.py:450` uses the separate NIST observed/classified catalog as
  wavelength-only weak retention evidence when quantitative lines provide no
  match.  It does not invent oscillator strengths, ratios, scores, or absence
  evidence from non-quantitative rows.
- `alibz/triage.py:504` adds local resolved-pair diagnostics.  General pairs
  use robust temperature-envelope bounds; shared-upper branching additionally
  requires matching configuration/term/J identity and upper energy within
  `1e-5 eV`.  Low-Ei, blended, reversed, distant, or uncertain-strength
  (Se/Th/U by database metadata) pairs abstain.  Ratios never hard reject.
- `alibz/triage.py:567` implements the optional missing-companion diagnostic.
  It requires fitted-reference significance from `peak_sigma`, independent
  local absent-line noise, at least two covered/local/isolated/non-resonant
  companions above threshold at every configured temperature, and trusted
  strengths.  It cannot reject unless the caller explicitly enables the
  experimental contract; no `local_noise` means absence always abstains.
- `alibz/triage.py:615` adds the public
  `triage_candidates(peak_array, db, *, config=None, wavelength_range=None,
  coverage_intervals=None, peak_sigma=None, protected_elements=(),
  local_noise=None)` interface, strict input validation, protected-element
  retention, natural-analysis exclusions, stage-limited weak evidence,
  coverage-hole handling, deferred/out-of-range/unavailable distinctions, and
  JSON-safe results.
- `tests/test_triage.py:48` through `tests/test_triage.py:318` add 24 focused
  tests for unresolved doublets and anti-chaining, temperature-envelope
  selection, in-band ranking, stage-III survival, cross-stage support,
  ambiguity, coverage holes, narrow bands, protected/uncertain elements,
  observed-only lines, shifted/missing patterns, self-reversal, ratio
  abstention and upper-level identity, diagnostic-only missing companions,
  strict invalid inputs, and JSON safety.

Only these new implementation/test files and this report were written for the
engine task.  Existing source files were not modified, so pre-existing behavior
outside explicit caller imports is byte-identical.  `git diff --check` reported
no whitespace errors.

## Verification and evidence

Baseline, before implementation:

- `PYTHONPATH=src python3 -m pytest tests/ -q` -- failed during collection:
  30 collection errors, all rooted in `ModuleNotFoundError: No module named
  'scipy'` from the system Python environment; no tests ran.
- `PYTHONPATH=src:$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])") python3 -m pytest tests/ -q`
  -- 455 passed, 4 skipped, 71 subtests passed in 298.51 s.

Focused final verification:

- `MPLCONFIGDIR=/private/tmp/alibz-mpl PYTHONPATH=src .venv/bin/python -m pytest tests/test_triage.py -q`
  -- 24 passed in 1.40 s.
- `python3 -m py_compile alibz/triage.py` -- passed.
- Actual project database, 200 deterministic random peaks over 250--900 nm:
  cold 0.949 s, warm 0.620 s; 166 species evaluated, 83 kept, 8 deferred,
  0 rejected.  Database construction was outside the timed region.

Final whole-project verification:

- `PYTHONPATH=src python3 -m pytest tests/ -q` -- failed during collection:
  33 collection errors, again rooted in missing SciPy in system Python; the
  increase reflects concurrently added gas/triage test modules, not a new
  failure mode.
- `PYTHONPATH=src:$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])") python3 -m pytest tests/ -q`
  -- 497 passed, 4 skipped, 71 subtests passed in 284.16 s.  One existing
  joblib CPU-count warning was emitted.
- `MPLCONFIGDIR=/private/tmp/alibz-mpl .venv/bin/python -m pytest tests/ -q`
  -- 498 passed, 4 skipped, 71 subtests passed in 273.46 s.  One existing
  joblib CPU-count warning was emitted.  This later run includes one
  concurrently added test that was not present during the 497-pass augmented
  run.

The baseline-to-final whole-suite count includes tests added concurrently by
the main integration and gas-detection work.  The isolated engine contribution
is the 24-test focused suite above.

## Limits and items not verified

- The system-Python command cannot run tests until SciPy is installed in that
  interpreter.  The required augmented and project-venv environments exercise
  the code with project dependencies.
- Scores are heuristic evidence summaries and were not calibrated as
  probabilities on a labelled corpus in this implementation task.
- Missing-companion hard rejection was implemented behind a default-off
  experimental contract and was not enabled for the uncalibrated production
  corpus.
- The deterministic 200-peak timing is a bounded performance check, not a
  full production-corpus benchmark.  The separate benchmark workflow owns
  corpus truth/recall validation.
