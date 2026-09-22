#!/usr/bin/env bash
# Deploy the 2026-09-22 sequence-mode + fresh-site change (four alibz files) to
# Moissanite's live alibz runtime. Default is a dry run; pass --apply to install.
#
# The bundle is exported from a pantheum-I COMMIT (default: the commit recorded
# below), never from the working tree, so concurrent uncommitted work in that
# tree (e.g. the D455 camera-power feature) is not deployed. The live baseline
# must be commit 393741d (composition change, applied 15:23 PDT) for every
# bundled file; the wrapper reads the live hashes over ssh and refuses otherwise.
# Uses the reservation-locked, manifest-pinned scripts/deploy-alibz-fire-fix.py,
# which refuses if any acquisition/job/hardware operation is active or unresolved,
# backs up files/config/SQLite, restarts both alibz services, and verifies hashes.
#
#   scripts/deploy-alibz-sequence-mode.sh            dry run
#   scripts/deploy-alibz-sequence-mode.sh --apply    install + restart services
#   PANTHEUM_COMMIT=<sha> ...                        override the source commit
set -euo pipefail
SRC=${PANTHEUM_SRC:-"$HOME/Projects/github/pantheum-I"}
HERE=$(cd "$(dirname "$0")" && pwd)
COMMIT=${PANTHEUM_COMMIT:-278d525}
BASELINE=393741d
REMOTE_DIR=pantheum-sequence-mode-bundle-20260922
FILES=(pantheum/alibz/optimization.py pantheum/alibz/__main__.py web/alibz/app.js web/alibz/index.html)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git -C "$SRC" rev-parse --verify "$COMMIT^{commit}" >/dev/null || { echo "commit $COMMIT not found in $SRC" >&2; exit 1; }
before=(); after=()
for f in "${FILES[@]}"; do
    mkdir -p "$tmp/$(dirname "$f")"
    git -C "$SRC" show "$COMMIT:$f" > "$tmp/$f"
    after+=("$(shasum -a 256 "$tmp/$f" | cut -d' ' -f1)")
    before+=("$(git -C "$SRC" show "$BASELINE:$f" | shasum -a 256 | cut -d' ' -f1)")
done
echo "source commit $(git -C "$SRC" rev-parse --short "$COMMIT"); baseline $BASELINE"
live=$(ssh -o BatchMode=yes moissanite "cd ~/pantheum-I && sha256sum ${FILES[*]} | cut -d' ' -f1")
i=0
for h in $live; do
    [[ "$h" == "${before[$i]}" ]] || { echo "live ${FILES[$i]} hash $h != baseline $BASELINE (${before[$i]}); refusing" >&2; exit 1; }
    i=$((i+1))
done
echo "live files match baseline $BASELINE"
cfg=$(ssh -o BatchMode=yes moissanite 'sha256sum ~/.config/pantheum/alibz.json | cut -d" " -f1')
python3 - "$tmp" "$cfg" "${#FILES[@]}" "${FILES[@]}" "${before[@]}" "${after[@]}" <<'PY'
import json, sys
tmp, cfg, n, *rest = sys.argv[1:]
n = int(n); files, before, after = rest[:n], rest[n:2*n], rest[2*n:3*n]
json.dump({'files': dict(zip(files, after)), 'before': dict(zip(files, before)), 'config_sha256': cfg},
          open(f'{tmp}/manifest.json', 'w'), indent=1)
PY
cp "$HERE/deploy-alibz-fire-fix.py" "$tmp/deploy.py"
ssh -o BatchMode=yes moissanite "rm -rf ~/$REMOTE_DIR && mkdir -p ~/$REMOTE_DIR"
scp -q -r "$tmp/." "moissanite:~/$REMOTE_DIR/"
if [[ "${1:-}" == "--apply" ]]; then
    ssh -o BatchMode=yes moissanite "cd ~ && python3 ~/$REMOTE_DIR/deploy.py --bundle ~/$REMOTE_DIR --apply"
    ssh -o BatchMode=yes moissanite "cd ~/pantheum-I && sha256sum ${FILES[*]} && systemctl --user is-active pantheum-alibz.service pantheum-alibz-worker.service && journalctl --user -u pantheum-alibz.service --since '-1 min' --no-pager | grep -iE 'auto-step|sequence|error' | tail -5 || true"
else
    ssh -o BatchMode=yes moissanite "cd ~ && python3 ~/$REMOTE_DIR/deploy.py --bundle ~/$REMOTE_DIR"
    echo "dry run only; rerun with --apply to install and restart the alibz services"
fi
