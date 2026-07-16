# MW2-112 LIBS workflow

This directory is the documentation entry point for the MW2-112 stratigraphic
LIBS profile. The production result is a within-element relative chemical
profile used to anchor correlative X-ray and neutron diffraction; it is not an
absolute or cross-element composition.

## Documents

- [Analysis plan](analysis_plan.md): scientific scope, assumptions, gates, and
  execution design.
- [Provenance and regeneration](provenance.md): complete data lineage,
  physical priors, integrity model, and tested reproduction procedure.
- [Peak-window PCA](peak_window_pca.md): corpus peak-shape prior,
  deconflicted multi-line quantification, detector-shift validation, results,
  and regeneration.
- [TODO](TODO.md): prioritized scientific, integration, and maintenance work.

## Command-line entry point

Install the development checkout, then use the consolidated command:

```bash
mw2-112 --help
```

The primary stages are:

```bash
# Primary within-element profiles from raw spectra
mw2-112 relative RAW_DIR RUN_DIR --db DB_DIR --workers 6

# Independent corpus-prior, peak-window PCA profiles
mw2-112 peak-pca RAW_DIR PCA_RUN_DIR \
  --db DB_DIR --basis corrections/corpus_peak_shape_pca_05x_10pc.npz \
  --calibration RUN_DIR/session_calibration.csv \
  --vendor-dir VENDOR_DIR --previous-run RUN_DIR

# Deterministic downstream rebuild and report
mw2-112 reassemble RUN_DIR
mw2-112 report RUN_DIR --vendor-dir VENDOR_DIR

# Verify and reproduce a frozen result bundle
mw2-112 provenance verify BUNDLE \
  --input-dir RAW_DIR --vendor-dir VENDOR_DIR
mw2-112 regenerate BUNDLE RAW_DIR NEW_OUTPUT \
  --vendor-dir VENDOR_DIR --workers 6
```

The older `run-mw2-112`, `build-mw2-112-relative-profiles`, and direct
`python -m scripts.*` commands remain available for compatibility. New
automation should use `mw2-112`.

## Maintained boundaries

- `alibz.mw2_112` owns shared geometry, table I/O, frozen-calibration loading,
  and relative-contrast assembly.
- `scripts/mw2_112.py` is the single user-facing dispatcher.
- Stage-specific scripts retain orchestration only and remain independently
  callable for recovery.
- The dated bundle under `runs/mw2_112_profile/` is immutable provenance, not
  working source code.
