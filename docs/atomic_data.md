# Atomic-line data scope and provenance

The composition schema has 92 fixed positions in atomic-number order from H
through U.  For synthetic development, 87 positions are supported.  Pm, Po,
At, Rn, and Pa retain their positions but are explicitly unsupported; an
unsupported position is not a measured zero. Tc, Fr, Ra, and Ac are a distinct
policy class: their available line data remain technically supported, but they
are excluded from targets and background mixtures in the current training
round. Thus, the output schema has 92 positions, the atomic support mask has
87, and the current training mask has 83. Exclusion does not assert that an
element is absent from a measured spectrum; a future radioactive-sample
workflow must opt in and establish separate calibration before quantifying it.

## Se, Th, and U repair

The original `el_lines92.pickle` contained only hydrogenic, very-high-stage
records for Se, Th, and U.  Those stages are not representative of ordinary
LIBS plasmas and cannot form a contiguous low-stage Saha chain.  The local
database is rebuilt by `scripts/update_heavy_atomic_lines.py` from:

- NIST Atomic Spectra Database line queries for Se I/II, Th I/II, and U I/II,
  180--962 nm, retrieved 2026-07-13.  NIST supplies preferred observed/Ritz
  wavelengths and level classifications.  It supplies quantitative `gA` for
  only a subset of the matched U I records in this wavelength range.
- R. L. Kurucz's public GFALL element files `gf3400`, `gf9000`, `gf9001`,
  `gf9200`, and `gf9201`.  These supply log(gf), lower/upper levels, and level
  angular momenta for quantitative forward synthesis.

The importer verifies pinned SHA-256 hashes, converts the Kurucz/NIST mixed
vacuum-below-200-nm and air-above-200-nm convention to the repository's stored
vacuum convention, converts log(gf) to `g_k A_ki`, and cross-matches NIST before
atomically replacing the pickle.  `db/atomic_line_sources.json` records URLs,
hashes, match counts, and reference-code counts.

The normalized merged rows are retained as reviewable text in
`db/quantitative_lines_se_th_u.tsv`; the 227 MB pickle is generated from those
same normalized records and remains ignored by Git.

Current quantitative coverage is:

| Element | Stages | Lines in 180--962 nm | NIST matches | NIST `gA` overrides |
|---|---:|---:|---:|---:|
| Se | I | 8 | 5 | 0 |
| Th | I--II | 2,069 | 1,912 | 0 |
| U | I--II | 1,156 | 299 | 26 |

## Scientific limitations

NIST lists additional observed wavelengths without the transition
probabilities and classified levels needed for deterministic
Saha--Boltzmann synthesis.  All 15,740 exported NIST rows are retained in
`db/observed_lines_nist.tsv` with a `quantitative_ready` flag and are available
through `Database.observed_lines()`, but they are not assigned invented
oscillator strengths.

Se II remains detection-only in the searched public sources: NIST ASD has
observed lines but no directly exportable quantitative records in this band,
and the public Kurucz GFALL set has no `gf3401` file.  Se I can be synthesized;
Se should not yet be claimed quantitatively valid across plasma states where
Se II dominates.

Most Th/U Kurucz oscillator strengths are inherited from older
intensity-derived compilations (`MC` reference code), and the small Se set
includes Corliss--Bozman, multiplet-estimated, and guessed values.  Synthetic
training must therefore sample source-aware strength uncertainty, and final
accuracy claims require certified standards.

## Rebuild

```bash
python scripts/update_heavy_atomic_lines.py \
    --source-dir /path/to/atomic-source-cache
```

The script downloads missing snapshots, refuses changed source content unless
its pinned hash is deliberately reviewed, validates the 14-column database
schema, and preserves the 92-element dictionary.
