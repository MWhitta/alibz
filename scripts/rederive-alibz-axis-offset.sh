#!/usr/bin/env bash
# Stage and run scripts/rederive-alibz-axis-offset.py on Moissanite (step 2 of the
# Z300 -18 px pixel->wavelength fix). Default is a DRY RUN (read-only): it
# inventories/classifies every stored dataset, computes the Ar I NIR axis check
# BEFORE (offset 0) and AFTER (offset -18) per run, predicts score changes for a
# sample of batches, and writes a JSON report -- and touches nothing.
#
#   scripts/rederive-alibz-axis-offset.sh                 # dry run (all live runs)
#   scripts/rederive-alibz-axis-offset.sh --sample run-...  # add a run to the preview
#   scripts/rederive-alibz-axis-offset.sh --apply          # re-derive + re-score (owner)
#
# ORDER: the owner deploys step 1 first (scripts/deploy-alibz-pixel-offset.sh
# --apply) so the LIVE ~/pantheum-I/pantheum/alibz/z300_calibration.py is the
# fixed module; THEN this tool's --apply, which refuses unless that live module
# hashes to the fixed version and uses it for the conversion (one source of
# truth). For a dry-run preview BEFORE deploy, this wrapper stages the local
# pantheum-I fixed module into a bundle dir under ~ on Moissanite and passes it as
# --preview-module (reporting only; never written to live state).
set -euo pipefail
SRC=${PANTHEUM_SRC:-"$HOME/Projects/github/pantheum-I"}
HERE=$(cd "$(dirname "$0")" && pwd)
FIXED_HASH=449dfb4981ecfc993c07489b65ee8241b794dc87894e921be4d0371c16217beb
BUNDLE=pantheum-rederive-bundle-20260924
CAL=pantheum/alibz/z300_calibration.py

# Verify the local fixed module before staging it as the preview module.
h=$(shasum -a 256 "$SRC/$CAL" | cut -d' ' -f1)
if [[ "$h" != "$FIXED_HASH" ]]; then
    echo "local $SRC/$CAL hash $h != fixed $FIXED_HASH; check out fix/z300-pixel-offset" >&2
    exit 1
fi

ssh -o BatchMode=yes moissanite "rm -rf ~/$BUNDLE && mkdir -p ~/$BUNDLE"
scp -q "$HERE/rederive-alibz-axis-offset.py" "moissanite:~/$BUNDLE/rederive.py"
scp -q "$SRC/$CAL" "moissanite:~/$BUNDLE/z300_calibration_fixed.py"

# Report whether the live module is already the fixed one (single source of truth).
live=$(ssh -o BatchMode=yes moissanite "sha256sum ~/pantheum-I/$CAL | cut -d' ' -f1")
echo "live ~/pantheum-I/$CAL : $live"
if [[ "$live" == "$FIXED_HASH" ]]; then
    echo "  -> fix IS deployed; the tool will convert with the live module."
else
    echo "  -> fix NOT deployed; the tool will PREVIEW with the staged fixed module."
fi

args=""
(( $# )) && args=$(printf '%q ' "$@")
# cwd ~/pantheum-I so the tool imports the deployed pantheum package.
ssh -o BatchMode=yes moissanite \
    "cd ~/pantheum-I && python3 ~/$BUNDLE/rederive.py --preview-module ~/$BUNDLE/z300_calibration_fixed.py $args"
