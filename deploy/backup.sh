#!/usr/bin/env bash
# Snapshot the bureau db.
#
#   sudo bash backup.sh /var/backups/freeboard
#
# use VACUUM INTO, not cp - the db is in WAL mode so copying bureau.db alone
# gives you an empty db. copy the snapshot off the VM afterwards.
set -euo pipefail

DEST="${1:-/var/backups/freeboard}"
DB="${2:-/var/lib/freeboard/bureau.db}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${DEST}/bureau-${STAMP}.db"

mkdir -p "$DEST"

python3 - "$DB" "$OUT" <<'PY'
import sqlite3, sys
src, out = sys.argv[1], sys.argv[2]
con = sqlite3.connect(src)
con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
con.execute("VACUUM INTO ?", (out,))
con.close()

check = sqlite3.connect(out)
episodes = check.execute("select count(*) from episodes").fetchone()[0]
deployments = check.execute("select count(*) from deployments").fetchone()[0]
check.close()
# verify the backup
assert episodes > 0 or deployments == 0, "snapshot has deployments but no episodes"
print(f"  {out}: {deployments} deployments, {episodes} episodes")
PY

chmod 600 "$OUT"
find "$DEST" -name 'bureau-*.db' -mtime +14 -delete
echo "==> backup complete"
