#!/usr/bin/env bash
# Deploy the 2026-09-24 alibz portal UI changes to Moissanite's live runtime.
# Default is a dry run; pass --apply to install.
#
#   web/alibz/index.html -- Optimize Acquisition description now reads "Find the
#     signal-to-noise optimum delay and acquisition period."
#   web/alibz/app.js     -- CONDITION FAULTS shows only ACTIVE (still-blocking)
#     faults plus faults cleared while the page is open; cleared/superseded
#     history disappears on reload. RUNS shows the 10 most recent acquisitions.
#   web/alibz/styles.css -- .runs-scroll: the Recent acquisitions table scrolls
#     (max-height 420px, sticky header).
#
# Uses the reservation-locked, manifest-pinned scripts/deploy-alibz-fire-fix.py,
# which refuses if any acquisition/job/hardware operation is active or unresolved,
# backs up files/config/SQLite, restarts both alibz services, and verifies hashes.
set -euo pipefail
SRC=${PANTHEUM_SRC:-"$HOME/Projects/github/pantheum-I"}
HERE=$(cd "$(dirname "$0")" && pwd)
REMOTE_DIR=pantheum-portal-ui-bundle-20260924
FILES=(web/alibz/index.html web/alibz/app.js web/alibz/styles.css)
EXPECTED_BEFORE=(64334b06ac873fa4c06c7405ace9d2740da39266f3fbbb1a2ab78023f1d1833a \
                 ce49669ff429c7c804c990dc872f09795b5a6d77813132fc7b44bf4f4ea8018b \
                 8458aeca48d01fb52352caca8ad70e59dec0247c8110a7c5b3cee87f05847fe3)
EXPECTED_AFTER=(2e281ddc387aaa4ac26e2fb4a75f9f59204369ecd3a45dc3ba4f21bf56ee4548 \
                86416c406f1524ad31260d916f1ae1000c6bb18e919074c5071a8b9293e88e77 \
                b713b3ad162d8f00a2e9d8774bb60ee3b27187f120d695904c6bf196c8f3b408)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for i in "${!FILES[@]}"; do
    f=${FILES[$i]}; mkdir -p "$tmp/$(dirname "$f")"; cp "$SRC/$f" "$tmp/$f"
    h=$(shasum -a 256 "$tmp/$f" | cut -d' ' -f1)
    [[ "$h" == "${EXPECTED_AFTER[$i]}" ]] || { echo "source $f hash $h != expected ${EXPECTED_AFTER[$i]}" >&2; exit 1; }
done
for i in "${!FILES[@]}"; do
    live=$(ssh -o BatchMode=yes moissanite "cd ~/pantheum-I && sha256sum '${FILES[$i]}'" | cut -d' ' -f1)
    [[ "$live" == "${EXPECTED_BEFORE[$i]}" ]] || {
        echo "LIVE ${FILES[$i]} hash $live != EXPECTED_BEFORE ${EXPECTED_BEFORE[$i]}" >&2
        echo "the live runtime is not at the expected version; do not deploy -- reconcile first" >&2; exit 1; }
done
cfg=$(ssh -o BatchMode=yes moissanite 'sha256sum ~/.config/pantheum/alibz.json | cut -d" " -f1')
python3 - "$tmp" "$cfg" "${FILES[@]}" "${EXPECTED_BEFORE[@]}" "${EXPECTED_AFTER[@]}" <<'PY'
import json, sys
tmp, cfg, *rest = sys.argv[1:]
n = len(rest) // 3
files, before, after = rest[:n], rest[n:2 * n], rest[2 * n:]
json.dump({'files': dict(zip(files, after)), 'before': dict(zip(files, before)), 'config_sha256': cfg},
          open(f'{tmp}/manifest.json', 'w'), indent=1)
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
