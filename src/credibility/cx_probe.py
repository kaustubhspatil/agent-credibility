"""Probe the gated cx-cmu/agent_trajectories corpus without downloading it.

Six domains -- mathhay, mcpbench, search, swebench, tau2bench, terminalbench --
is the role spread the web study lacked: customer service, ops, research and
coding are genuinely different jobs, not four flavours of browser task.

Reading a parquet footer over HfFileSystem pulls kilobytes rather than the ~2 GB
the corpus weighs, so this answers "what is in here and can we use it" before
committing to a download. It reports, per domain: row count, schema, and which
columns are plausible candidates for the three things the credibility estimator
needs -- a role, a deployment, and an outcome.
"""

from __future__ import annotations

import re

REPO = "cx-cmu/agent_trajectories"

# Column-name shapes worth flagging, in the vocabulary these corpora tend to use.
CANDIDATES = {
    "outcome": re.compile(
        r"(reward|success|solved|resolved|passed|correct|score|is_correct|label)",
        re.I,
    ),
    "deployment": re.compile(
        r"(model|agent|policy|run|config|scaffold|harness|env|environment|"
        r"website|site|domain|task_?set|split|customer|tenant)",
        re.I,
    ),
    "role": re.compile(r"(benchmark|domain|suite|task_?type|category|role)", re.I),
    "trace": re.compile(
        r"(messages|trajectory|steps|actions|tool_?calls|turns|history|transcript)",
        re.I,
    ),
    "identity": re.compile(r"(id$|_id|uuid|instance|episode|traj)", re.I),
}


def probe() -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    root = f"datasets/{REPO}"

    try:
        entries = fs.ls(root, detail=True)
    except Exception as exc:  # gated, unauthenticated, or renamed
        print(f"cannot list {REPO}: {type(exc).__name__}: {exc}")
        print("\nIf this says 403/GatedRepo, access has not been granted yet.")
        print("If it says 401, run:  hf auth login")
        return

    parquets = sorted(
        e["name"] for e in entries if e["name"].endswith(".parquet")
    )
    print(f"{REPO}: {len(parquets)} parquet files\n")

    for path in parquets:
        name = path.rsplit("/", 1)[-1]
        try:
            with fs.open(path, "rb") as fh:
                pf = pq.ParquetFile(fh)
                n = pf.metadata.num_rows
                schema = pf.schema_arrow
                # one row group is enough to see real values
                head = pf.read_row_group(0).slice(0, 1).to_pylist()
        except Exception as exc:
            print(f"### {name}: unreadable -- {type(exc).__name__}: {exc}")
            continue

        print(f"### {name}  ({n:,} rows, {len(schema)} columns)")
        hits: dict[str, list[str]] = {k: [] for k in CANDIDATES}
        for field in schema:
            for kind, pattern in CANDIDATES.items():
                if pattern.search(field.name):
                    hits[kind].append(field.name)
        for kind, cols in hits.items():
            if cols:
                print(f"    {kind:11s}: {', '.join(cols[:8])}")

        # show a few short scalar values so the encoding is obvious
        if head:
            row = head[0]
            shown = 0
            for key, val in row.items():
                if isinstance(val, (bool, int, float)) or (
                    isinstance(val, str) and len(val) < 60
                ):
                    print(f"      e.g. {key} = {val!r}")
                    shown += 1
                if shown >= 5:
                    break
        print()


if __name__ == "__main__":
    probe()
