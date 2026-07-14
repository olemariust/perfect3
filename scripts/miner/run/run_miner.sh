#!/bin/bash
# poker44-aquila — miner startup script (legacy path kept for parity).
# All identity (wallet/hotkey/port/pm2 name/model name) comes from the
# repo-local .env via model/ecosystem.config.js — nothing is hardcoded here,
# and both launch paths therefore publish the same manifest identity.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if ! command -v pm2 >/dev/null 2>&1; then
    echo "Error: PM2 is not installed" >&2
    exit 1
fi
if [ ! -f "$REPO/.env" ]; then
    echo "Error: $REPO/.env missing — copy .env.example and fill it in" >&2
    exit 1
fi

exec pm2 start "$REPO/model/ecosystem.config.js"
