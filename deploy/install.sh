#!/usr/bin/env bash
# Install or update the Freeboard bureau on a Debian/Ubuntu host.
# Idempotent: safe to re-run for a deploy.
#
#   sudo bash install.sh [git-ref]
#
# Deliberately no TLS here. The service binds to 127.0.0.1 and a reverse proxy
# in front terminates TLS -- see deploy/caddy.example. Putting certificate
# handling inside a stdlib http.server would be the wrong place for it.
set -euo pipefail

REF="${1:-main}"
REPO="https://github.com/kaustubhspatil/agent-credibility.git"
APP_DIR=/opt/freeboard
DATA_DIR=/var/lib/freeboard

echo "==> user and directories"
id -u freeboard >/dev/null 2>&1 || useradd --system --home "$APP_DIR" --shell /usr/sbin/nologin freeboard
mkdir -p "$APP_DIR" "$DATA_DIR"

# git refuses to operate on a tree it does not own, and after the first install
# this one belongs to the service user while the installer runs as root. Without
# this the UPDATE path fails ("dubious ownership") while a fresh install
# succeeds -- so the script looks idempotent and is not.
git config --global --add safe.directory "$APP_DIR" 2>/dev/null || true

echo "==> source at ref ${REF}"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" fetch --depth 1 origin "$REF"
  git -C "$APP_DIR" checkout -f FETCH_HEAD
else
  apt-get update -qq && apt-get install -y -qq git python3
  git clone --depth 1 --branch "$REF" "$REPO" "$APP_DIR"
fi

# The bureau needs no third-party packages. That is the point: nothing to
# install means nothing to break on an unattended reboot after a preemption.
python3 - <<'PY'
import sys
assert sys.version_info >= (3, 10), f"need python 3.10+, found {sys.version}"
print("python", sys.version.split()[0], "ok")
PY

# Record the deployed revision before handing the tree to the service user:
# afterwards git refuses to read it as root ("dubious ownership").
REV="$(git -C "$APP_DIR" rev-parse --short HEAD)"

chown -R freeboard:freeboard "$APP_DIR" "$DATA_DIR"
chmod 750 "$DATA_DIR"

echo "==> service"
install -m 644 "$APP_DIR/deploy/bureau.service" /etc/systemd/system/bureau.service
systemctl daemon-reload
systemctl enable bureau
systemctl restart bureau
sleep 2

echo "==> health"
systemctl is-active --quiet bureau && echo "    service active" || { journalctl -u bureau -n 30 --no-pager; exit 1; }
curl -fsS --max-time 5 http://127.0.0.1:8080/v1/health && echo
echo "==> done (${REV})"
