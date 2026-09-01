#!/usr/bin/env bash
# Snapshot the bureau database safely.
#
#   sudo bash backup.sh /var/backups/freeboard
#
# WHY THIS EXISTS: the database runs in WAL mode, so recent writes live in
# `bureau.db-wal`, not `bureau.db`. Copying the main file alone -- which is what
# `cp`, `scp` and most backup tools do by default -- silently produces an EMPTY
# database. That is not hypothetical: it happened during the migration from the
# Spot host, where 240 episodes sat in a 304 KB WAL beside a 4 KB main file, and
# the copy looked like it succeeded.
#
# `VACUUM INTO` is the correct primitive. It takes a consistent snapshot of the
# whole database, WAL included, into a single new file, without stopping the
# service.
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
# A backup nobody verified is a backup nobody has.
assert episodes > 0 or deployments == 0, "snapshot has deployments but no episodes"
print(f"  {out}: {deployments} deployments, {episodes} episodes")
PY

chmod 600 "$OUT"
find "$DEST" -name 'bureau-*.db' -mtime +14 -delete
echo "==> backup complete"
