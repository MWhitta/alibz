#!/usr/bin/env bash
# Copy recover-alibz-awaiting-data.py to Moissanite and run it there.
#   scripts/recover-alibz-awaiting-data.sh --all-awaiting        # dry run, all awaiting_data runs
#   scripts/recover-alibz-awaiting-data.sh --all-awaiting --apply   # store spectra, rescore, restart services
#   scripts/recover-alibz-awaiting-data.sh --apply --enable-data-api   # only switch retrieval to data_api
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ssh -o BatchMode=yes moissanite 'mkdir -p ~/pantheum-recover-20260922'
scp -q "$HERE/recover-alibz-awaiting-data.py" moissanite:~/pantheum-recover-20260922/recover.py
# Arguments pass through unchanged (quoted for the remote shell). Use
# --all-awaiting or --run to select runs; --apply --enable-data-api alone only
# switches the retrieval mode.
args=$(printf '%q ' "$@")
ssh -o BatchMode=yes moissanite "cd ~/pantheum-I && python3 ~/pantheum-recover-20260922/recover.py $args"
