#!/usr/bin/env bash
# Deploy the 2026-09-22 raster-step fix (z300.py, optimization.py, app.js) to
# Moissanite's live alibz runtime. Default is a dry run; pass --apply to install.
# Uses the reservation-locked, manifest-pinned scripts/deploy-alibz-fire-fix.py,
# which refuses if any acquisition/job/hardware operation is active or unresolved,
# backs up files/config/SQLite, restarts both alibz services, and verifies hashes.
set -euo pipefail
SRC=${PANTHEUM_SRC:-"$HOME/Projects/github/pantheum-I"}
HERE=$(cd "$(dirname "$0")" && pwd)
REMOTE_DIR=pantheum-raster-step-bundle-20260922
FILES=(pantheum/alibz/z300.py pantheum/alibz/optimization.py web/alibz/app.js)
EXPECTED_BEFORE=(a01ac05967253592d9c7a8c521e7ab50740ba3f0679f227e2b1604453b8d658c \
                 3f199210426405a97a9094faf69465c2da90b70d9bcce32b4eeda35e3c33c921 \
                 a8afa00467e26ea6f93bcb0e07c460a2372d8e396db8514d87e881ae6cbdbe0b)
EXPECTED_AFTER=(eeb08470ac3b9a219ade72a984f3f235340dcb07d7347d88ffd9e3679bf5c07d \
                f4f0fced9536c1946238694cb98c774851fc1b3a228899401a2bc5085f925a05 \
                f2f9bb51f1239309d9b8e650ce84194476dc1eae545b055b45604699bec12427)
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
before = dict(zip(files, """a01ac05967253592d9c7a8c521e7ab50740ba3f0679f227e2b1604453b8d658c
3f199210426405a97a9094faf69465c2da90b70d9bcce32b4eeda35e3c33c921
a8afa00467e26ea6f93bcb0e07c460a2372d8e396db8514d87e881ae6cbdbe0b""".split()))
after = dict(zip(files, """eeb08470ac3b9a219ade72a984f3f235340dcb07d7347d88ffd9e3679bf5c07d
f4f0fced9536c1946238694cb98c774851fc1b3a228899401a2bc5085f925a05
f2f9bb51f1239309d9b8e650ce84194476dc1eae545b055b45604699bec12427""".split()))
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
