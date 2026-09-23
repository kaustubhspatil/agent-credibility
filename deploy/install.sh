#!/usr/bin/env bash
# Install or update the bureau on Debian/Ubuntu. Safe to re-run.
#
#   sudo bash install.sh [git-ref]
#
# fetch this script by commit SHA, branch URLs on raw.githubusercontent.com get cached:
#
#   SHA=$(git rev-parse HEAD)
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/$SHA/deploy/install.sh
#
# no TLS here, the reverse proxy handles it (see deploy/caddy.example)
set -euo pipefail

REF="${1:-main}"
REPO="https://github.com/kaustubhspatil/agent-credibility.git"
APP_DIR=/opt/freeboard
DATA_DIR=/var/lib/freeboard

echo "==> user and directories"
id -u freeboard >/dev/null 2>&1 || useradd --system --home "$APP_DIR" --shell /usr/sbin/nologin freeboard
mkdir -p "$APP_DIR" "$DATA_DIR"

# mark the repo as safe or updates fail with "dubious ownership"
# use --system since sudo keeps the user's HOME
git config --system --add safe.directory "$APP_DIR" 2>/dev/null || true

echo "==> source at ref ${REF}"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" fetch --depth 1 origin "$REF"
  git -C "$APP_DIR" checkout -f FETCH_HEAD
else
  apt-get update -qq && apt-get install -y -qq git python3
  git clone --depth 1 --branch "$REF" "$REPO" "$APP_DIR"
fi

# no third-party packages needed
python3 - <<'PY'
import sys
assert sys.version_info >= (3, 10), f"need python 3.10+, found {sys.version}"
print("python", sys.version.split()[0], "ok")
PY

# record the revision before chown (git refuses after)
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
