#!/usr/bin/env bash
# Deploy the 2026-09-22 native-grid change (acquire.py, optimization_metrics.py) to
# Moissanite's live alibz runtime. Default is a dry run; pass --apply to install.
# Uses the reservation-locked, manifest-pinned scripts/deploy-alibz-fire-fix.py,
# which refuses if any acquisition/job/hardware operation is active or unresolved,
# backs up files/config/SQLite, restarts both alibz services, and verifies hashes.
set -euo pipefail
SRC=${PANTHEUM_SRC:-"$HOME/Projects/github/pantheum-I"}
HERE=$(cd "$(dirname "$0")" && pwd)
REMOTE_DIR=pantheum-native-grid-bundle-20260922
FILES=(pantheum/alibz/acquire.py pantheum/alibz/optimization_metrics.py)
EXPECTED_BEFORE=(143b3d29ee975ff638480659fa478d360139b91e7157d2feeec5ef235a6ff25a \
                 e01eb1cfd7fbef328d67082f37c84b1560e3c53c43d96b0454808c284cc40c60)
EXPECTED_AFTER=(aa6a97501557b2589b1c3b9bb59f9c41bbf8683fb435f00423d2794525a9791a \
                68d4d4859d349f7629fa6d6a451569632c89a85729247a0f298269735eee850a)
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
before = dict(zip(files, """143b3d29ee975ff638480659fa478d360139b91e7157d2feeec5ef235a6ff25a
e01eb1cfd7fbef328d67082f37c84b1560e3c53c43d96b0454808c284cc40c60""".split()))
after = dict(zip(files, """aa6a97501557b2589b1c3b9bb59f9c41bbf8683fb435f00423d2794525a9791a
68d4d4859d349f7629fa6d6a451569632c89a85729247a0f298269735eee850a""".split()))
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
