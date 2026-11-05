#!/usr/bin/env bash
# Run the literature fetch on a Linux lab host (default: moissanite) and pull
# the results back. Nothing is shown on any desktop: the Elsevier API needs no
# browser, and everything else uses headless Chrome on the remote.
# Usage: manuscript/lit/fetch-remote.sh [extra lra flags]
# Env:   LRA_REMOTE (ssh host, default moissanite), LRA_REMOTE_DIR (~/lra),
#        LRA_REMOTE_CORPUS (~/lra-corpus), LRA_SYNC_TREE=0 to skip pushing code.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LRA_REPO="${LRA_REPO:-$HOME/Projects/github/literature-retrieval-agent}"
REMOTE="${LRA_REMOTE:-moissanite}"
RDIR="${LRA_REMOTE_DIR:-lra}"
RCORPUS="${LRA_REMOTE_CORPUS:-lra-corpus}"
STAMP="$(date +%Y%m%d-%H%M%S)"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE")

if [ "${LRA_SYNC_TREE:-1}" = 1 ]; then
  echo "pushing agent tree -> $REMOTE:$RDIR"
  rsync -az --delete --exclude .venv --exclude .chrome_session --exclude downloads \
    --exclude '__pycache__' --exclude '*.pyc' --exclude .pytest_cache --exclude .git \
    "$LRA_REPO/" "$REMOTE:$RDIR/"
  "${SSH[@]}" "cd $RDIR && .venv/bin/python -m pip install -q -e . >/dev/null"
fi

echo "pushing manifests + existing corpus -> $REMOTE:$RCORPUS"
"${SSH[@]}" "mkdir -p $RCORPUS/downloads $RCORPUS/fulltext $RCORPUS/profile"
rsync -az "$HERE"/sources-*.txt "$LRA_REPO/strategies.json" "$REMOTE:$RCORPUS/"
rsync -az "$HERE/downloads/" "$REMOTE:$RCORPUS/downloads/"
rsync -az "$HERE/fulltext/" "$REMOTE:$RCORPUS/fulltext/"

# Refs as separate arguments (comments stripped), via a file on the remote.
grep -hvE '^\s*(#|$)' "$HERE"/sources-*.txt | tr -d ' ' > "/tmp/lra-refs-$STAMP.txt"
rsync -az "/tmp/lra-refs-$STAMP.txt" "$REMOTE:$RCORPUS/refs.txt"

echo "running on $REMOTE (headless Chrome + Elsevier API) -> report-$STAMP.json"
"${SSH[@]}" "cd $RCORPUS && xargs -a refs.txt ~/$RDIR/.venv/bin/lra fetch \
  --download-dir downloads --profile-dir profile --strategies strategies.json \
  --headless --automate-portal --skip-existing --retries 1 --retry-backoff 5 --throttle 3 \
  --challenge-strikes ${LRA_CHALLENGE_STRIKES:-2} ${LRA_FALLBACK:+--fallback $LRA_FALLBACK} \
  --fulltext fulltext --report report-$STAMP.json $*" 2>&1 | tee "$HERE/fetch-remote-$STAMP.log" | grep -vE '^\s*$' | tail -40

echo "pulling results <- $REMOTE:$RCORPUS"
rsync -az "$REMOTE:$RCORPUS/downloads/" "$HERE/downloads/"
rsync -az "$REMOTE:$RCORPUS/fulltext/" "$HERE/fulltext/"
rsync -az "$REMOTE:$RCORPUS/report-$STAMP.json" "$HERE/report-remote-$STAMP.json"
echo "done: $(ls "$HERE"/downloads/*.pdf | wc -l) PDFs locally; report $HERE/report-remote-$STAMP.json"
