# Literature corpus for the alibz manuscript

Two stages, deliberately separate:

1. **Search** (done by Claude workers, not by the retrieval agent): each topic
   worker writes a verified BibTeX file `manuscript/bib/<topic>.bib` and a
   manifest `manuscript/lit/sources-<topic>.txt` with one `bibkey=DOI-or-arXiv`
   per line. Every DOI is checked against the Crossref API before it is listed.
   Evidence: `reports/2026-11-03-lit-<topic>.md`.
2. **Fetch** (the literature-retrieval-agent, `lra`): `fetch.sh` reads every
   manifest and drives the user's authenticated Chrome session to capture PDFs
   into `lit/downloads/` (gitignored), with a machine-readable outcome report.
   Open-access sources are captured autonomously; subscription sources use the
   institution's access through the persisted profile. See
   `~/Projects/github/literature-retrieval-agent/README.md`.

Topics: `theory` (plasma emission, broadening, self-absorption, atomic data),
`cflibs` (calibration-free and quantitative LIBS, geological/handheld
applications), `ml` (statistical inference, AI/ML and numerical methods used for
LIBS quantification and in the alibz pipeline).

## Re-running the fetch

`fetch.sh` is resumable (`--skip-existing`). The unattended run skips any source
that shows a bot-detection challenge after 5 s. To collect those, run it with a
person at the browser window (it is parked off-screen; bring Chrome forward from
the Dock when prompted):

```
manuscript/lit/fetch.sh --pause-on-challenge --challenge-timeout 180
```

Pass `--fallback berkeley` to retry failures through the library proxy (needs a
live CalNet session in the profile). Outcomes per source are in the newest
`report-*.json`; `text_holes` lists PDFs with no extractable text.

## Outcome of the first unattended run (2026-11-03)

148 sources (final tally after the Elsevier API route, 2026-11-03 15:08): 139 captured with extractable text; 9 missing (4 AAS/IOP, 2 IEEE, 1 Wiley book, 1 IUCr, 1 Am. Mineral.) — see `fetch-outcome-summary.txt`. The first unattended run had: 75 captured, 60 skipped on bot-detection
challenges (49 Elsevier/ScienceDirect, which hosts most of *Spectrochimica Acta B*;
also IOP and AAS journals), 18 no PDF (6 arXiv, fixed by the open-access pass the
script now runs first; 4 IEEE; books and chapters), 1 error. Summary in
`fetch-outcome-summary.txt`. The Elsevier set needs the interactive re-run above.

## Bot challenges (fixed 2026-11-03)

ScienceDirect shows a challenge page to the automated browser that does not
clear even when clicked. The retrieval agent now has a per-publisher circuit
breaker: after two unclearable challenges on one (publisher, route) pair it
stops opening that publisher for the rest of the run, escalates the remaining
sources to the library proxy (`--fallback berkeley`, set by `fetch.sh`), and
reports anything it could not route as `blocked` in the JSON report. Two
sanctioned routes avoid the challenge entirely:

- **Library proxy (off-network only).** From the LBNL network
  `libproxy.berkeley.edu` answers with an immediate redirect to the raw
  publisher site and no login (verified 2026-11-03 for ScienceDirect and
  IUCr), so the proxy cannot help here. Off-network, export
  `LRA_FALLBACK=berkeley` and sign in to CalNet once in the agent's profile.
- **Elsevier full-text API.** Register a free key at https://dev.elsevier.com
  and store it once in the macOS Keychain (`security add-generic-password -a
  "$USER" -s elsevier-api-key -w`, which prompts for the secret); the agent
  reads it from there, or from `ELSEVIER_API_KEY` if exported. Full text needs
  institutional entitlement (LBNL network/VPN, or an institutional token under
  the keychain service `elsevier-insttoken`). The agent then fetches Elsevier
  DOIs without opening a page at all.

Neither route attempts to defeat the publisher's bot detection.
