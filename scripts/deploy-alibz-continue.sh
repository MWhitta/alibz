#!/usr/bin/env bash
# Deploy the 2026-09-22 continue-after-failed-batch + retrieval-cap change (optimization.py, retrieval.py) to
# Moissanite's live alibz runtime. Default is a dry run; pass --apply to install.
# Uses the reservation-locked, manifest-pinned scripts/deploy-alibz-fire-fix.py,
# which refuses if any acquisition/job/hardware operation is active or unresolved,
# backs up files/config/SQLite, restarts both alibz services, and verifies hashes.
set -euo pipefail
SRC=${PANTHEUM_SRC:-"$HOME/Projects/github/pantheum-I"}
HERE=$(cd "$(dirname "$0")" && pwd)
REMOTE_DIR=pantheum-continue-bundle-20260922
FILES=(pantheum/alibz/optimization.py pantheum/alibz/retrieval.py)
EXPECTED_BEFORE=(2c96f4a6b9dd3a1dd6f35bb468aff3ae7af4be81b409ab593da22665b1b1e5cd \
                 fbd2cd965ca4858f63a553f58abe4564d216c835d652f4f681afdc2f14735b02)
EXPECTED_AFTER=(c61d9a38913a4661d0a45caa4067afc1c62345d92052c6fc080bdc7af1ad1e47 \
                4ff0fee97205d6b121a078b08de3909a1ba895915f58fb3e6f3d79036c3836ec)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for i in "${!FILES[@]}"; do
    f=${FILES[$i]}; mkdir -p "$tmp/$(dirname "$f")"; cp "$SRC/$f" "$tmp/$f"
    h=$(shasum -a 256 "$tmp/$f" | cut -d' ' -f1)
    [[ "$h" == "${EXPECTED_AFTER[$i]}" ]] || { echo "source $f hash $h != expected ${EXPECTED_AFTER[$i]}" >&2; exit 1; }
done
cfg=$(ssh -o BatchMode=yes moissanite 'sha256sum ~/.config/pantheum/alibz.json | cut -d" " -f1')
python3 - "$tmp" "$cfg" "${FILES[@]}" <<'PY'
import json, sys
tmp, cfg, *files = sys.argv[1:]
before = dict(zip(files, """2c96f4a6b9dd3a1dd6f35bb468aff3ae7af4be81b409ab593da22665b1b1e5cd
fbd2cd965ca4858f63a553f58abe4564d216c835d652f4f681afdc2f14735b02""".split()))
after = dict(zip(files, """c61d9a38913a4661d0a45caa4067afc1c62345d92052c6fc080bdc7af1ad1e47
4ff0fee97205d6b121a078b08de3909a1ba895915f58fb3e6f3d79036c3836ec""".split()))
json.dump({'files': after, 'before': before, 'config_sha256': cfg}, open(f'{tmp}/manifest.json', 'w'), indent=1)
PY
cp "$HERE/deploy-alibz-fire-fix.py" "$tmp/deploy.py"
ssh -o BatchMode=yes moissanite "rm -rf ~/$REMOTE_DIR && mkdir -p ~/$REMOTE_DIR"
scp -q -r "$tmp/." "moissanite:~/$REMOTE_DIR/"
if [[ "${1:-}" == "--apply" ]]; then
    ssh -o BatchMode=yes moissanite "cd ~ && python3 ~/$REMOTE_DIR/deploy.py --bundle ~/$REMOTE_DIR --apply"
    ssh -o BatchMode=yes moissanite "cd ~/pantheum-I && sha256sum ${FILES[*]} && systemctl --user is-active pantheum-alibz.service pantheum-alibz-worker.service"
else
    ssh -o BatchMode=yes moissanite "cd ~ && python3 ~/$REMOTE_DIR/deploy.py --bundle ~/$REMOTE_DIR"
    echo "dry run only; rerun with --apply to install and restart the alibz services"
fi
