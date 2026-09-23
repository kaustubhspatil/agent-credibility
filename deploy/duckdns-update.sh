#!/usr/bin/env bash
# Keep a DuckDNS name pointed at this host (spot IP changes on restart).
#
#   sudo install -m 700 duckdns-update.sh /usr/local/bin/
#   printf 'DUCKDNS_DOMAIN=your-name\nDUCKDNS_TOKEN=...\n' | sudo tee /etc/freeboard-duckdns.env
#   sudo chmod 600 /etc/freeboard-duckdns.env
#   sudo systemctl enable --now duckdns.timer
#
# keep the token in that file only, not in the repo or on the command line
set -euo pipefail
# shellcheck disable=SC1091
. /etc/freeboard-duckdns.env

response=$(curl -fsS --max-time 20 \
  "https://www.duckdns.org/update?domains=${DUCKDNS_DOMAIN}&token=${DUCKDNS_TOKEN}&ip=")

if [ "$response" = "OK" ]; then
  echo "duckdns: ${DUCKDNS_DOMAIN} updated"
else
  echo "duckdns: update failed (${response})" >&2
  exit 1
fi
