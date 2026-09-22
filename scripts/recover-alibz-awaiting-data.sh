#!/usr/bin/env bash
# Copy recover-alibz-awaiting-data.py to Moissanite and run it there.
#   scripts/recover-alibz-awaiting-data.sh                       # dry run, all awaiting_data runs
#   scripts/recover-alibz-awaiting-data.sh --apply               # store spectra, rescore, restart services
#   scripts/recover-alibz-awaiting-data.sh --apply --enable-data-api   # ...and switch retrieval to data_api
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ssh -o BatchMode=yes moissanite 'mkdir -p ~/pantheum-recover-20260922'
scp -q "$HERE/recover-alibz-awaiting-data.py" moissanite:~/pantheum-recover-20260922/recover.py
ssh -o BatchMode=yes moissanite "cd ~/pantheum-I && python3 ~/pantheum-recover-20260922/recover.py --all-awaiting $*"
