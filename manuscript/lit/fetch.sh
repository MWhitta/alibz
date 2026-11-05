#!/usr/bin/env bash
# Fetch every PDF listed in sources-*.txt with the literature-retrieval-agent.
# Usage: manuscript/lit/fetch.sh [extra lra flags]
# Requires: lra on PATH, Google Chrome, an authenticated profile at $PROFILE.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LRA_REPO="${LRA_REPO:-$HOME/Projects/github/literature-retrieval-agent}"
PROFILE="${LRA_PROFILE:-$LRA_REPO/.chrome_session}"
STRATEGIES="${LRA_STRATEGIES:-$LRA_REPO/strategies.json}"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$HERE/downloads"
refs=()
while IFS= read -r line; do
  line="${line%%#*}"; line="${line// /}"
  [ -z "$line" ] && continue
  refs+=("$line")
done < <(cat "$HERE"/sources-*.txt)
arxiv=(); other=()
for r in "${refs[@]}"; do
  case "$r" in *=arXiv:*|*=arxiv:*) arxiv+=("$r");; *) other+=("$r");; esac
done
COMMON=(--download-dir "$HERE/downloads" --profile-dir "$PROFILE"
        --background --skip-existing --fulltext "$HERE/fulltext")
if [ "${#arxiv[@]}" -gt 0 ]; then
  # Bundled strategy map: it marks arxiv.org open-access, the lab map does not,
  # and without that flag the open-access locator falls back to a human click.
  echo "pass 1: ${#arxiv[@]} arXiv sources via the open-access locator"
  mkdir -p "$HERE/profile/oa"
  lra fetch "${COMMON[@]}" --profile-dir "$HERE/profile/oa" --auto-open-access --throttle 2 \
    --report "$HERE/report-$STAMP-oa.json" "${arxiv[@]}" || true
fi
echo "pass 2: ${#other[@]} publisher sources via portal automation -> $HERE/report-$STAMP.json"
# Challenge breaker: a publisher that shows an unclearable bot challenge twice
# is not opened again this run; its remaining sources go straight to the
# library proxy when LRA_FALLBACK=berkeley is exported (off-network only: from
# the LBNL network libproxy.berkeley.edu redirects to the raw site without
# proxying, verified 2026-11-03) or are reported "blocked". With ELSEVIER_API_KEY exported, Elsevier
# DOIs are fetched through Elsevier's full-text API before any page is opened.
lra fetch "${COMMON[@]}" --strategies "$STRATEGIES" --automate-portal --retries 1 --retry-backoff 5 --throttle 3 \
  --challenge-strikes "${LRA_CHALLENGE_STRIKES:-2}" ${LRA_FALLBACK:+--fallback "$LRA_FALLBACK"} \
  --report "$HERE/report-$STAMP.json" "$@" "${other[@]}"
