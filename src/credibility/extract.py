"""Extract per-episode behavioural features from SWE-smith agent trajectories.

One row in, one *episode* out. An episode is the unit the trace-economic
underwriting paper prices: a monitored customer-task-trace episode under a
defined role. Here the role is "SWE-agent coding agent", the customer is the
repository under test, and the deployment is (scaffold, repo).

Nothing here reads a payload it would not be willing to hand an auditor: the
output is counts, rates and entropies, never source code or prompt text.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass

# --- action parsing -------------------------------------------------------
# The three splits are the same agent behind three different action encodings.
# We normalise all of them to a list of (command, raw) pairs.

_XML_FN = re.compile(r"<function=([A-Za-z0-9_\-]+)")
_TICKS = re.compile(r"```(?:bash|sh|python)?\n(.*?)```", re.S)
_ERROR_MARKERS = re.compile(
    r"Traceback \(most recent call last\)|"
    r"command not found|No such file or directory|"
    r"^Error:|^ERROR:|SyntaxError|ImportError|ModuleNotFoundError|"
    r"AssertionError|FAILED |exit code [1-9]",
    re.M,
)
# Actions that can mutate the customer's asset. A coarse exposure proxy: we see
# the action name but not its arguments, so `str_replace_editor` counts even
# when it was only used to view a file. Coarse is acceptable here -- the
# question is whether deployments separate, not whether the label is perfect.
_MUTATING_NAMES = frozenset({
    "str_replace_editor", "insert", "create", "edit", "write", "apply_patch",
    "sed", "rm", "mv", "mkdir", "chmod", "git", "tee", "patch",
})


def _text(content) -> str:
    """Message content is either a string or a list of typed blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict):
                out.append(block.get("text") or "")
            elif isinstance(block, str):
                out.append(block)
        return "\n".join(out)
    return ""


def _command_head(raw: str) -> str:
    """First meaningful token of a shell command, used as the 'tool' name."""
    raw = raw.strip()
    if not raw:
        return "<empty>"
    # strip leading env assignments and sudo
    for token in raw.split():
        if "=" in token and not token.startswith("-"):
            continue
        if token in ("sudo", "!"):
            continue
        return token.split("/")[-1][:40]
    return raw.split()[0][:40]


def parse_actions(messages: list[dict], scaffold: str) -> list[str]:
    """Return the normalised action name for each assistant action, in order."""
    actions: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue

        if scaffold == "tool":
            # Native tool calls: prefer the structured `action` field.
            calls = msg.get("tool_calls") or []
            if calls:
                for call in calls:
                    fn = (call or {}).get("function") or {}
                    name = fn.get("name") or "<tool>"
                    args = fn.get("arguments") or ""
                    if name in ("bash", "execute_bash") and args:
                        try:
                            parsed = json.loads(args)
                            cmd = parsed.get("command") or ""
                            actions.append(f"bash:{_command_head(cmd)}")
                            continue
                        except (ValueError, AttributeError):
                            pass
                    actions.append(name)
                continue
            action = msg.get("action")
            if action:
                actions.append(f"bash:{_command_head(action)}")
            continue

        body = _text(msg.get("content"))
        if scaffold == "xml":
            for name in _XML_FN.findall(body):
                if name == "bash":
                    m = re.search(
                        r"<parameter=command>(.*?)</parameter>", body, re.S
                    )
                    actions.append(f"bash:{_command_head(m.group(1) if m else '')}")
                else:
                    actions.append(name)
        else:  # ticks
            for block in _TICKS.findall(body):
                actions.append(f"bash:{_command_head(block)}")
    return actions


def _normalise(action: str) -> str:
    """`bash:cd` and `cd` are the same action seen through two scaffolds.

    Without this the action vocabularies differ by encoding rather than by
    behaviour, which would manufacture between-deployment variance that is a
    pure artefact of the harness.
    """
    return action[5:] if action.startswith("bash:") else action


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    h = -sum((c / total) * math.log2(c / total) for c in counts.values() if c)
    return h


@dataclass
class Episode:
    traj_id: str
    instance_id: str
    scaffold: str
    model: str
    repo: str
    task_family: str
    resolved: bool
    # behavioural signals -- the sufficient statistics that would leave a tenant
    n_turns: int
    n_actions: int
    n_distinct_actions: int
    action_entropy: float
    repeat_rate: float
    error_rate: float
    n_mutating: int
    mutating_share: float
    obs_chars: int
    assistant_chars: int


def episode_from_row(row: dict, scaffold: str) -> Episode | None:
    try:
        messages = json.loads(row["messages"])
    except (ValueError, TypeError):
        return None
    if not isinstance(messages, list) or not messages:
        return None

    actions = [_normalise(a) for a in parse_actions(messages, scaffold)]
    n_actions = len(actions)
    counts = Counter(actions)

    repeats = sum(1 for a, b in zip(actions, actions[1:]) if a == b)
    repeat_rate = repeats / max(n_actions - 1, 1)

    observations = [
        _text(m.get("content"))
        for m in messages
        if m.get("role") in ("tool", "user") and m.get("message_type") != "instruction"
    ]
    n_err = sum(1 for o in observations if _ERROR_MARKERS.search(o))
    error_rate = n_err / max(len(observations), 1)

    n_mutating = sum(1 for a in actions if a in _MUTATING_NAMES)
    obs_chars = sum(len(o) for o in observations)
    assistant_chars = sum(
        len(_text(m.get("content"))) for m in messages if m.get("role") == "assistant"
    )

    instance_id = row["instance_id"]
    repo = instance_id.split(".")[0]
    # instance ids look like  owner__repo.<commit>.<mutation-family>__<hash>,
    # except the PR-derived ones, which end `.pr_<number>`. Left as-is every PR
    # number becomes its own "family" and the covariate is useless, so collapse
    # them to a single `pr` family.
    parts = instance_id.split(".")
    task_family = parts[2].split("__")[0] if len(parts) > 2 else "unknown"
    if re.fullmatch(r"pr_\d+", task_family):
        task_family = "pr"

    return Episode(
        traj_id=row["traj_id"],
        instance_id=instance_id,
        scaffold=scaffold,
        model=row["model"],
        repo=repo,
        task_family=task_family,
        resolved=bool(row["resolved"]),
        n_turns=len(messages),
        n_actions=n_actions,
        n_distinct_actions=len(counts),
        action_entropy=_entropy(counts),
        repeat_rate=repeat_rate,
        error_rate=error_rate,
        n_mutating=n_mutating,
        mutating_share=n_mutating / max(n_actions, 1),
        obs_chars=obs_chars,
        assistant_chars=assistant_chars,
    )


def run(out_path: str, limit_per_split: int | None = None) -> None:
    import pandas as pd
    from datasets import load_dataset

    from pathlib import Path

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for scaffold in ("tool", "xml", "ticks"):
        # Write each split as it finishes. A long stream that dies (or gets
        # killed) two splits in should not lose two splits of work.
        part = out.with_name(f"{out.stem}.{scaffold}.parquet")
        if part.exists():
            print(f"{scaffold}: reusing {part.name}", flush=True)
            frames.append(pd.read_parquet(part))
            continue

        ds = load_dataset(
            "SWE-bench/SWE-smith-trajectories", split=scaffold, streaming=True
        )
        rows: list[dict] = []
        for row in ds:
            ep = episode_from_row(row, scaffold)
            if ep is not None:
                rows.append(asdict(ep))
                if len(rows) % 2000 == 0:
                    print(f"  {scaffold}: {len(rows)} episodes", flush=True)
            if limit_per_split and len(rows) >= limit_per_split:
                break

        frame = pd.DataFrame(rows)
        frame.to_parquet(part, index=False)
        print(f"{scaffold}: kept {len(frame)} episodes -> {part.name}", flush=True)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} episodes -> {out}", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/episodes.parquet")
    ap.add_argument("--limit-per-split", type=int, default=None)
    args = ap.parse_args()
    run(args.out, args.limit_per_split)
