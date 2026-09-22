#!/usr/bin/env bash
# Deploy the 2026-09-22 10 Hz study-default fix (optimization.py, app.js) to
# Moissanite's live alibz runtime. Default is a dry run; pass --apply to install.
# Uses the reservation-locked, manifest-pinned scripts/deploy-alibz-fire-fix.py,
# which refuses if any acquisition/job/hardware operation is active or unresolved,
# backs up files/config/SQLite, restarts both alibz services, and verifies hashes.
set -euo pipefail
SRC=${PANTHEUM_SRC:-"$HOME/Projects/github/pantheum-I"}
HERE=$(cd "$(dirname "$0")" && pwd)
REMOTE_DIR=pantheum-rate-10hz-bundle-20260922
FILES=(pantheum/alibz/optimization.py web/alibz/app.js)
EXPECTED_BEFORE=(f4f0fced9536c1946238694cb98c774851fc1b3a228899401a2bc5085f925a05 \
                 f2f9bb51f1239309d9b8e650ce84194476dc1eae545b055b45604699bec12427)
EXPECTED_AFTER=(be02da8dc57026463fc1db545947e39a48c62f2cb954aae5d6ccced688f2fe89 \
                f7252cf9cde3c18a6bdaf4e3fc1e69b5b6ef005667855440d655731a338b519e)
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
before = dict(zip(files, """f4f0fced9536c1946238694cb98c774851fc1b3a228899401a2bc5085f925a05
f2f9bb51f1239309d9b8e650ce84194476dc1eae545b055b45604699bec12427""".split()))
after = dict(zip(files, """be02da8dc57026463fc1db545947e39a48c62f2cb954aae5d6ccced688f2fe89
f7252cf9cde3c18a6bdaf4e3fc1e69b5b6ef005667855440d655731a338b519e""".split()))
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
