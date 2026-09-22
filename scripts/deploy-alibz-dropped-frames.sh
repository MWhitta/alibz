#!/usr/bin/env bash
# Deploy the 2026-09-22 dropped-frame tolerance change (five alibz modules) to
# Moissanite's live alibz runtime. Default is a dry run; pass --apply to install.
# Uses the reservation-locked, manifest-pinned scripts/deploy-alibz-fire-fix.py,
# which refuses if any acquisition/job/hardware operation is active or unresolved,
# backs up files/config/SQLite, restarts both alibz services, and verifies hashes.
set -euo pipefail
SRC=${PANTHEUM_SRC:-"$HOME/Projects/github/pantheum-I"}
HERE=$(cd "$(dirname "$0")" && pwd)
REMOTE_DIR=pantheum-dropped-frames-bundle-20260922
FILES=(pantheum/alibz/optimization.py pantheum/alibz/acquire.py pantheum/alibz/retrieval.py pantheum/alibz/z300.py pantheum/alibz/optimization_metrics.py)
EXPECTED_BEFORE=(fe07dc9037408845960ec8a67b54d20c1d8f3268d29b1893f4b8419c7a755683 \
                 1c642acddc755201ce0393576ef1763a5985ddf4672eb10a2b909d6a568fc69a \
                 714185c10518825f5941699514d688bdd88204d07a6193182996ca5b0acedd7d \
                 eeb08470ac3b9a219ade72a984f3f235340dcb07d7347d88ffd9e3679bf5c07d \
                 68d4d4859d349f7629fa6d6a451569632c89a85729247a0f298269735eee850a)
EXPECTED_AFTER=(5d0a8a8a3efad92d7092ef70597c729cdfd7f2a247b3dcee1c7c94c653adc218 \
                df63466b4f41269fa08a57bb4a3ba66871a0fbf69dbbd516db28c7a404385484 \
                64f4370667dc893e7510b734e5645443dab989829006e2c9f20d92a94f99f38a \
                efc6adf7b2ebb32ae364f5ef938c96cda09c5b55131daa1c171d16d8cf67ca44 \
                37e45156c66bddc72493e65c39e42839bf4dd65d9b3f097efffa121e3ff83da9)
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
before = dict(zip(files, """fe07dc9037408845960ec8a67b54d20c1d8f3268d29b1893f4b8419c7a755683
1c642acddc755201ce0393576ef1763a5985ddf4672eb10a2b909d6a568fc69a
714185c10518825f5941699514d688bdd88204d07a6193182996ca5b0acedd7d
eeb08470ac3b9a219ade72a984f3f235340dcb07d7347d88ffd9e3679bf5c07d
68d4d4859d349f7629fa6d6a451569632c89a85729247a0f298269735eee850a""".split()))
after = dict(zip(files, """5d0a8a8a3efad92d7092ef70597c729cdfd7f2a247b3dcee1c7c94c653adc218
df63466b4f41269fa08a57bb4a3ba66871a0fbf69dbbd516db28c7a404385484
64f4370667dc893e7510b734e5645443dab989829006e2c9f20d92a94f99f38a
efc6adf7b2ebb32ae364f5ef938c96cda09c5b55131daa1c171d16d8cf67ca44
37e45156c66bddc72493e65c39e42839bf4dd65d9b3f097efffa121e3ff83da9""".split()))
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
