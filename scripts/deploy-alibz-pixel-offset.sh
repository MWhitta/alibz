#!/usr/bin/env bash
# Deploy the 2026-09-24 Z300 pixel->wavelength offset fix (step 1 of 3) to
# Moissanite's live alibz runtime. Default is a dry run; pass --apply to install.
#
# What this changes on the live (acquire.retrieval=data_api) path:
#   pantheum/alibz/z300_calibration.py  -- pixels_to_wavelength now evaluates the
#     per-segment cubic at (stored_index + PIXEL_OFFSET), PIXEL_OFFSET = -18, so
#     API/native spectra are no longer labelled ~18 px too blue; the inactive
#     960-961 nm 4th-channel placeholder is excluded; adds argon_axis_offset_px.
#   pantheum/alibz/z300.py              -- FakeZ300Server synthetic calibrations
#     made offset-consistent so tests still map pixel 0 to the intended nm.
#   pantheum/alibz/acquire.py           -- records an Ar I NIR axis_check in the
#     dataset/test provenance and warns in the batch detail when |offset| > 2 px
#     or the lock is ambiguous (never rejects data).
#
# Uses the reservation-locked, manifest-pinned scripts/deploy-alibz-fire-fix.py,
# which refuses if any acquisition/job/hardware operation is active or unresolved,
# backs up files/config/SQLite, restarts both alibz services, and verifies hashes.
#
# The Opal ingest producer (z300_opal_ingest.py, used only when
# acquire.retrieval=opal_database, which Moissanite is NOT currently in --
# retrieval=data_api as of 2026-09-24) also carries the -18 fix on its legacy
# ZIP/gzip-JSON path. It lives on the Windows Opal box, not in ~/pantheum-I, and
# is staged by scripts/stage_z300_opal.py. Phase B below runs that stager in dry
# run so its new hashes are shown; re-stage it with `stage_z300_opal.py --apply`
# (a Windows/powershell path) when the opal_database retrieval path is next used.
set -euo pipefail
SRC=${PANTHEUM_SRC:-"$HOME/Projects/github/pantheum-I"}
HERE=$(cd "$(dirname "$0")" && pwd)
REMOTE_DIR=pantheum-pixel-offset-bundle-20260924
FILES=(pantheum/alibz/z300_calibration.py pantheum/alibz/z300.py pantheum/alibz/acquire.py)
# EXPECTED_BEFORE: the live ~/pantheum-I/<path> hashes on Moissanite, confirmed
# on 2026-09-24 to equal `git show HEAD:<path>` for all three files.
EXPECTED_BEFORE=(8a6135d058f09c8de62542695c7a1ac923d577906fdc88b8db2e8ded89bd156d \
                 efc6adf7b2ebb32ae364f5ef938c96cda09c5b55131daa1c171d16d8cf67ca44 \
                 58ab146f2854ecc9422c3e6647fa71fd7d9f2039744c4b30fa4b4985178d37d4)
EXPECTED_AFTER=(449dfb4981ecfc993c07489b65ee8241b794dc87894e921be4d0371c16217beb \
                fb4e4f672df51be8e993a8e6558c68ab594ad874e1cf03cb011f2f9bb1e1bc9a \
                d67efa0d3d71e41628c98dabe5acafa2b780039c3cd2a134f912ba32d6537ae5)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for i in "${!FILES[@]}"; do
    f=${FILES[$i]}; mkdir -p "$tmp/$(dirname "$f")"; cp "$SRC/$f" "$tmp/$f"
    h=$(shasum -a 256 "$tmp/$f" | cut -d' ' -f1)
    [[ "$h" == "${EXPECTED_AFTER[$i]}" ]] || { echo "source $f hash $h != expected ${EXPECTED_AFTER[$i]}" >&2; exit 1; }
done
# Confirm the live files still equal EXPECTED_BEFORE before pinning the manifest.
for i in "${!FILES[@]}"; do
    live=$(ssh -o BatchMode=yes moissanite "cd ~/pantheum-I && sha256sum '${FILES[$i]}'" | cut -d' ' -f1)
    [[ "$live" == "${EXPECTED_BEFORE[$i]}" ]] || {
        echo "LIVE ${FILES[$i]} hash $live != EXPECTED_BEFORE ${EXPECTED_BEFORE[$i]}" >&2
        echo "the live runtime is not at HEAD; do not deploy -- reconcile first" >&2; exit 1; }
done
cfg=$(ssh -o BatchMode=yes moissanite 'sha256sum ~/.config/pantheum/alibz.json | cut -d" " -f1')
python3 - "$tmp" "$cfg" "${FILES[@]}" <<'PY'
import json, sys
tmp, cfg, *files = sys.argv[1:]
before = dict(zip(files, """8a6135d058f09c8de62542695c7a1ac923d577906fdc88b8db2e8ded89bd156d
efc6adf7b2ebb32ae364f5ef938c96cda09c5b55131daa1c171d16d8cf67ca44
58ab146f2854ecc9422c3e6647fa71fd7d9f2039744c4b30fa4b4985178d37d4""".split()))
after = dict(zip(files, """449dfb4981ecfc993c07489b65ee8241b794dc87894e921be4d0371c16217beb
fb4e4f672df51be8e993a8e6558c68ab594ad874e1cf03cb011f2f9bb1e1bc9a
d67efa0d3d71e41628c98dabe5acafa2b780039c3cd2a134f912ba32d6537ae5""".split()))
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

echo
echo "== Phase B: Opal ingest producer (z300_opal_ingest.py) =="
echo "Moissanite retrieval is data_api (2026-09-24), so the Opal producer is NOT"
echo "on the live path; re-stage it before switching to opal_database retrieval."
echo "Dry-run of scripts/stage_z300_opal.py (no Opal contact on dry run):"
python3 "$HERE/stage_z300_opal.py"
echo "to install the -18 producer on Opal: python3 scripts/stage_z300_opal.py --apply"
