# alibz manuscript

Theory, assumptions, workflow, datasets and fit metrics behind alibz.
Target: arXiv preprint (physics.ins-det / physics.plasm-ph / astro-ph.IM
cross-list to be decided), then *Applied Spectroscopy* (Society for Applied
Spectroscopy, SAGE). The journal has no mandatory LaTeX template; the `article`
class here is arXiv-ready and converts to the SAGE template at submission.

- `main.tex` — preamble, author block, section includes. `\arxivtrue` adds line numbers.
- `sections/` — one file per section; `\todo{}` marks unwritten text.
- `bib/` — `theory.bib`, `cflibs.bib`, `ml.bib` (literature workers), `local.bib` (software/data).
- `lit/` — DOI manifests and the retrieval-agent fetch script; PDFs are gitignored.
- `figures/` — generated figures (PDF/PNG); scripts that make them belong in `cal/scripts` or `scripts/`.

Build: `make` (latexmk). `make check` lists TODO counts and bibliography warnings.
`make arxiv` bundles sources + `.bbl` for arXiv upload.

Provenance rule: every number in `sections/07-results.tex` cites the repo report,
notebook, or test that produced it (path in a LaTeX comment on the same line).
