#!/usr/bin/env bash
# Copy unblock-alibz-session.py to Moissanite and run it there (args passed through).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ssh -o BatchMode=yes moissanite 'mkdir -p ~/pantheum-recover-20260922'
scp -q "$HERE/unblock-alibz-session.py" moissanite:~/pantheum-recover-20260922/unblock.py
# Quote every argument for the remote shell so a multi-word --reason survives.
args=""
(( $# )) && args=$(printf '%q ' "$@")
ssh -o BatchMode=yes moissanite "python3 ~/pantheum-recover-20260922/unblock.py $args"
