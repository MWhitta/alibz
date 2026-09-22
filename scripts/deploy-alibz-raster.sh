#!/usr/bin/env bash
# Deploy the 2026-09-22 raster/verified-conditions/refusal/retrieval-backoff change (optimization.py, acquire.py, retrieval.py) to
# Moissanite's live alibz runtime. Default is a dry run; pass --apply to install.
# Uses the reservation-locked, manifest-pinned scripts/deploy-alibz-fire-fix.py,
# which refuses if any acquisition/job/hardware operation is active or unresolved,
# backs up files/config/SQLite, restarts both alibz services, and verifies hashes.
set -euo pipefail
SRC=${PANTHEUM_SRC:-"$HOME/Projects/github/pantheum-I"}
HERE=$(cd "$(dirname "$0")" && pwd)
REMOTE_DIR=pantheum-raster-bundle-20260922
FILES=(pantheum/alibz/optimization.py pantheum/alibz/acquire.py pantheum/alibz/retrieval.py)
EXPECTED_BEFORE=(c61d9a38913a4661d0a45caa4067afc1c62345d92052c6fc080bdc7af1ad1e47 \
                 4a23a6915c1bc88454c13dacdce5e1de9cbea6a358fb06c9276e324c7d6f147b \
                 4ff0fee97205d6b121a078b08de3909a1ba895915f58fb3e6f3d79036c3836ec)
EXPECTED_AFTER=(fe07dc9037408845960ec8a67b54d20c1d8f3268d29b1893f4b8419c7a755683 \
                1c642acddc755201ce0393576ef1763a5985ddf4672eb10a2b909d6a568fc69a \
                714185c10518825f5941699514d688bdd88204d07a6193182996ca5b0acedd7d)
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
before = dict(zip(files, """c61d9a38913a4661d0a45caa4067afc1c62345d92052c6fc080bdc7af1ad1e47
4a23a6915c1bc88454c13dacdce5e1de9cbea6a358fb06c9276e324c7d6f147b
4ff0fee97205d6b121a078b08de3909a1ba895915f58fb3e6f3d79036c3836ec""".split()))
after = dict(zip(files, """fe07dc9037408845960ec8a67b54d20c1d8f3268d29b1893f4b8419c7a755683
1c642acddc755201ce0393576ef1763a5985ddf4672eb10a2b909d6a568fc69a
714185c10518825f5941699514d688bdd88204d07a6193182996ca5b0acedd7d""".split()))
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
