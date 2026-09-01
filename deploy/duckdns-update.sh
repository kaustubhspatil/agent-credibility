#!/usr/bin/env bash
# Keep a DuckDNS name pointing at this host.
#
# A Spot instance loses its external IP every time it is preempted, so any DNS
# name pointing at it goes stale on the next start. This is the cheap fix; the
# alternative is a reserved static IP, which bills while the instance is
# stopped and a Spot instance is stopped often.
#
#   sudo install -m 700 duckdns-update.sh /usr/local/bin/
#   printf 'DUCKDNS_DOMAIN=your-name\nDUCKDNS_TOKEN=...\n' | sudo tee /etc/freeboard-duckdns.env
#   sudo chmod 600 /etc/freeboard-duckdns.env
#   sudo systemctl enable --now duckdns.timer
#
# The token lives only in that file, mode 600, root-owned. Do not put it in a
# repo, a unit file, or a command line -- command lines are world-readable.
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
