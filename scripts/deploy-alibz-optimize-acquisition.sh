#!/usr/bin/env bash
# Deploy the 2026-09-22 "Optimize Acquisition / composition" change to
# Moissanite's live alibz runtime. Default is a dry run; pass --apply to install.
#
# Ships: pantheum/alibz/optimization.py, pantheum/alibz/optimization_metrics.py,
# web/alibz/app.js, web/alibz/index.html (modified), plus the new
# pantheum/alibz/composition.py and every pantheum/alibz/references/*-v1.json
# file EXCEPT fe-v1.json (82 files, enumerated by glob at run time).
#
# This SUPERSEDES scripts/deploy-alibz-verify-min-shots.sh: the live
# optimization.py is currently at commit e4e79ee, and this bundle carries
# the 97433bb verified-condition change on top of the uncommitted
# composition-optimizer diff, so verify-min-shots.sh must not be run after
# this one (its pinned before/after hashes are for the e4e79ee -> 97433bb
# step only and would refuse -- or worse, target stale content -- once this
# bundle is live).
#
# WARNING: pantheum-I/reports/2026-09-22-alibz-field-deploy.md documents a
# separate PENDING deploy that rsyncs ~/pantheum-I-stage-6f7c654/ into
# ~/pantheum-I on Moissanite. That stage tree is HEAD (6f7c654) and does NOT
# contain pantheum/alibz/composition.py or the references/*.json files (they
# are uncommitted in this working tree) and has optimization.py/app.js at
# their committed (pre-composition-optimizer) content. If that rsync runs
# AFTER this script, it will overwrite optimization.py, optimization_metrics.py,
# app.js and index.html back to their HEAD versions, silently regressing the
# composition-optimizer change (the untracked composition.py/references files
# would survive since plain rsync has no --delete, but the code that uses them
# would be gone). Run the stage-6f7c654 rsync BEFORE this script, or commit the
# composition-optimizer change to HEAD first, so both trees agree.
#
# Uses the reservation-locked, manifest-pinned scripts/deploy-alibz-fire-fix.py
# (extended here to support a manifest `before` of null, meaning "this file
# must not exist live yet"), which refuses if any acquisition/job/hardware
# operation is active or unresolved, backs up files/config/SQLite, restarts
# both alibz services, and verifies hashes.
#
# Compatible with bash 3.2 (macOS default): no associative arrays, no mapfile.
set -euo pipefail

SRC=${PANTHEUM_SRC:-"$HOME/Projects/github/pantheum-I"}
HERE=$(cd "$(dirname "$0")" && pwd)
REMOTE_DIR=pantheum-optimize-acquisition-bundle-20260922
REMOTE_ROOT=pantheum-I
KNOWN_COMMITS=(6f7c654 97433bb e4e79ee)
MODIFIED_FILES=(pantheum/alibz/optimization.py pantheum/alibz/optimization_metrics.py web/alibz/app.js web/alibz/index.html)

if [[ ! -d "$SRC/.git" ]]; then
    echo "PANTHEUM_SRC ($SRC) is not a git checkout; cannot verify known-good commits" >&2
    exit 1
fi

REF_FILES=($(cd "$SRC" && ls pantheum/alibz/references/*-v1.json 2>/dev/null | grep -v '/fe-v1\.json$' | sort))
if [[ ${#REF_FILES[@]} -ne 82 ]]; then
    echo "Expected 82 reference files (excluding fe-v1.json), found ${#REF_FILES[@]}" >&2
    exit 1
fi
NEW_FILES=(pantheum/alibz/composition.py "${REF_FILES[@]}")
ALL_FILES=("${MODIFIED_FILES[@]}" "${NEW_FILES[@]}")

echo "== verifying local sources exist ==" >&2
for f in "${ALL_FILES[@]}"; do
    [[ -f "$SRC/$f" ]] || { echo "missing source file: $SRC/$f" >&2; exit 1; }
done

echo "== reading live hashes for modified files over ssh ==" >&2
LIVE_HASH_OUT=$(ssh -o BatchMode=yes moissanite "cd ~/$REMOTE_ROOT && sha256sum ${MODIFIED_FILES[*]}")
LIVE_HASHES=()
LIVE_PATHS=()
while read -r hash path; do
    [[ -n "$path" ]] || continue
    LIVE_HASHES+=("$hash")
    LIVE_PATHS+=("$path")
done <<< "$LIVE_HASH_OUT"
if [[ ${#LIVE_HASHES[@]} -ne ${#MODIFIED_FILES[@]} ]]; then
    echo "expected ${#MODIFIED_FILES[@]} live hashes, got ${#LIVE_HASHES[@]}: $LIVE_HASH_OUT" >&2
    exit 1
fi
for i in "${!MODIFIED_FILES[@]}"; do
    [[ "${LIVE_PATHS[$i]}" == "${MODIFIED_FILES[$i]}" ]] || {
        echo "live sha256sum path order mismatch: expected ${MODIFIED_FILES[$i]}, got ${LIVE_PATHS[$i]}" >&2
        exit 1
    }
done

echo "== checking new files do not already exist live ==" >&2
EXISTING_LIVE=$(ssh -o BatchMode=yes moissanite "cd ~/$REMOTE_ROOT && for f in ${NEW_FILES[*]}; do [ -e \"\$f\" ] && echo \"\$f\"; done" || true)
if [[ -n "$EXISTING_LIVE" ]]; then
    echo "Live file already exists (refusing): " >&2
    echo "$EXISTING_LIVE" >&2
    exit 1
fi

echo "== matching each live modified-file hash against known-good commits ${KNOWN_COMMITS[*]} ==" >&2
for i in "${!MODIFIED_FILES[@]}"; do
    f=${MODIFIED_FILES[$i]}
    live_hash=${LIVE_HASHES[$i]}
    matched=""
    for c in "${KNOWN_COMMITS[@]}"; do
        known=$(cd "$SRC" && git show "$c:$f" 2>/dev/null | shasum -a 256 | cut -d' ' -f1) || known=""
        if [[ "$known" == "$live_hash" ]]; then
            matched=$c
            break
        fi
    done
    if [[ -z "$matched" ]]; then
        echo "Live $f (hash $live_hash) does not match any known-good commit (${KNOWN_COMMITS[*]}); refusing" >&2
        exit 1
    fi
    echo "  live $f matches commit $matched" >&2
done

echo "== reading private config hash over ssh ==" >&2
cfg=$(ssh -o BatchMode=yes moissanite 'sha256sum ~/.config/pantheum/alibz.json | cut -d" " -f1')

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for f in "${ALL_FILES[@]}"; do
    mkdir -p "$tmp/$(dirname "$f")"
    cp "$SRC/$f" "$tmp/$f"
done

after_list="$tmp/.after.tsv"
before_list="$tmp/.before.tsv"
: > "$after_list"; : > "$before_list"
for f in "${ALL_FILES[@]}"; do
    h=$(shasum -a 256 "$SRC/$f" | cut -d' ' -f1)
    printf '%s\t%s\n' "$f" "$h" >> "$after_list"
done
for i in "${!MODIFIED_FILES[@]}"; do
    printf '%s\t%s\n' "${MODIFIED_FILES[$i]}" "${LIVE_HASHES[$i]}" >> "$before_list"
done
for f in "${NEW_FILES[@]}"; do
    printf '%s\tNULL\n' "$f" >> "$before_list"
done

python3 - "$tmp" "$cfg" "$after_list" "$before_list" <<'PY'
import json, sys
tmp, cfg, after_list, before_list = sys.argv[1:]
def load(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if not line:
                continue
            name, _, value = line.partition('\t')
            out[name] = None if value == 'NULL' else value
    return out
after = load(after_list)
before = load(before_list)
json.dump({'files': after, 'before': before, 'config_sha256': cfg}, open(f'{tmp}/manifest.json', 'w'), indent=1)
PY
rm -f "$after_list" "$before_list"

cp "$HERE/deploy-alibz-fire-fix.py" "$tmp/deploy.py"
ssh -o BatchMode=yes moissanite "rm -rf ~/$REMOTE_DIR && mkdir -p ~/$REMOTE_DIR"
scp -q -r "$tmp/." "moissanite:~/$REMOTE_DIR/"

if [[ "${1:-}" == "--apply" ]]; then
    ssh -o BatchMode=yes moissanite "cd ~ && python3 ~/$REMOTE_DIR/deploy.py --bundle ~/$REMOTE_DIR --apply"
    ssh -o BatchMode=yes moissanite "cd ~/$REMOTE_ROOT && sha256sum ${MODIFIED_FILES[*]} pantheum/alibz/composition.py && echo 'reference_json_count:' \$(ls pantheum/alibz/references/*-v1.json | wc -l) && systemctl --user is-active pantheum-alibz.service pantheum-alibz-worker.service"
    echo "== GET /api/optimization over the alibz gateway unix socket (status check; no writes) ==" >&2
    ssh -o BatchMode=yes moissanite "curl -s --unix-socket ~/.local/state/pantheum/alibz-gateway/alibz.sock http://alibz/api/optimization" || true
else
    ssh -o BatchMode=yes moissanite "cd ~ && python3 ~/$REMOTE_DIR/deploy.py --bundle ~/$REMOTE_DIR"
    echo "dry run only; rerun with --apply to install and restart the alibz services"
fi
