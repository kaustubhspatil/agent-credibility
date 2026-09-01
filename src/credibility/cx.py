"""Six agent roles from cx-cmu/agent_trajectories.

The AgentRewardBench study could only compare four flavours of browser task.
This corpus spans genuinely different jobs -- customer service (tau2bench),
terminal operations (terminalbench), web research (search), coding (swebench),
MCP tool use (mcpbench) and maths reasoning (mathhay) -- which is the spread a
role registry has to survive.

The structure is unusually well suited to credibility:

    role        benchmark            six of them
    deployment  benchmark x domain x source_model
    episode     one (task, pass) attempt, with `reward` as the outcome

`pass` is the part that matters most: each task is attempted up to four times
by the same model, so repeated attempts at identical work give a clean read on
process variance -- the EPV term -- rather than having to infer it from task
mix.

Only scalar columns are pulled. `messages`, `eval_details` and `trace_meta` are
the bulk of the ~2 GB and are not needed to estimate a frequency.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = "cx-cmu/agent_trajectories"
DOMAINS = (
    "mathhay",
    "mcpbench",
    "search",
    "swebench",
    "tau2bench",
    "terminalbench",
)
SCALARS = [
    "id",
    "benchmark",
    "domain",
    "task_id",
    "source_model",
    "pass",
    "num_turns",
    "reward",
    "num_passes_available",
]


def load(cache: str | Path = "data/cx_episodes.parquet") -> pd.DataFrame:
    """Scalar columns for all six roles, cached locally after the first pull."""
    cache = Path(cache)
    if cache.exists():
        return pd.read_parquet(cache)

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    frames = []
    for name in DOMAINS:
        path = f"datasets/{REPO}/{name}.parquet"
        with fs.open(path, "rb") as fh:
            table = pq.ParquetFile(fh).read(columns=SCALARS)
        frame = table.to_pandas()
        print(f"  {name}: {len(frame):,} episodes", flush=True)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the columns the credibility code expects.

    `reward` is kept as-is *and* thresholded. Rewards are not on a common scale
    across benchmarks -- a maths score and a customer-service task outcome are
    not the same quantity -- so the loss used for pooling is the binary "did
    this episode fail", which is comparable in a way the raw number is not.
    """
    out = df.copy()
    out["role"] = out["benchmark"]
    out["model"] = out["source_model"]
    out["site"] = out["domain"]
    out["loss"] = (out["reward"] <= 0).astype(float)
    return out


def describe(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("benchmark").agg(
        episodes=("reward", "size"),
        tasks=("task_id", "nunique"),
        domains=("domain", "nunique"),
        models=("source_model", "nunique"),
        mean_reward=("reward", "mean"),
        loss_rate=("loss", "mean"),
        binary_reward=("reward", lambda s: bool(set(s.unique()) <= {0.0, 1.0})),
        max_pass=("pass", "max"),
    )
    return g.round(4)


def main(out_dir: str = "out") -> None:
    import json

    from .buhlmann import bootstrap_k, components

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = analysable(prepare(load()))
    print(f"{len(frame):,} episodes | {frame.role.nunique()} roles | "
          f"{frame._dep.nunique()} deployments | {frame.model.nunique()} models")

    pooled = components(frame["loss"].to_numpy(float), frame["_dep"].to_numpy())
    lo, hi = bootstrap_k(frame["loss"].to_numpy(float), frame["_dep"].to_numpy(), 300)
    print(f"\npooled across roles: K = {pooled.k:.2f} [{lo:.2f}, {hi:.2f}]")

    roles = per_role_k(frame)
    roles.to_csv(out / "cx_per_role_k.csv", index=False)
    print("\n--- one K per role ---")
    print(roles.to_string(index=False))

    dec = four_level_decomposition(frame)
    (out / "cx_decomposition.json").write_text(json.dumps(dec, indent=2))
    print("\n--- where risk variation actually lives ---")
    print(f"  pass-to-pass noise (same task) : {dec['var_pass_noise']:.5f}")
    print(f"  task mix within deployment     : {dec['var_task_mix']:.5f}"
          f"   {dec['share_task_mix']:.1%} of systematic")
    print(f"  deployment within role         : {dec['var_deployment']:.5f}"
          f"   {dec['share_deployment']:.1%}")
    print(f"  role                           : {dec['var_role']:.5f}"
          f"   {dec['share_role']:.1%}")


# --- analysis --------------------------------------------------------------

def analysable(df: pd.DataFrame, min_episodes: int = 8) -> pd.DataFrame:
    """Roles with a comparable outcome, deployments with enough exposure.

    mcpbench is dropped: its reward is not binary (261 distinct values, mean
    3.17) and is on a scale the other five do not share. Thresholding it into a
    loss would pool two different quantities and quietly flatter the result.
    """
    out = df[df["role"] != "mcpbench"].copy()
    out["_dep"] = out["role"] + "|" + out["site"] + "|" + out["model"]
    counts = out["_dep"].value_counts()
    out = out[out["_dep"].isin(counts[counts >= min_episodes].index)]
    out["_task"] = out["_dep"] + "||" + out["task_id"]
    return out


def per_role_k(df: pd.DataFrame, min_deployments: int = 3) -> pd.DataFrame:
    """One credibility constant per role. There is no universal K."""
    from .buhlmann import bootstrap_k, components

    rows = []
    for role, g in df.groupby("role"):
        if g["_dep"].nunique() < min_deployments:
            rows.append({"role": role, "k": float("nan"),
                         "n_deployments": g["_dep"].nunique(),
                         "n_episodes": len(g), "note": "too few deployments"})
            continue
        comp = components(g["loss"].to_numpy(float), g["_dep"].to_numpy())
        lo, hi = bootstrap_k(g["loss"].to_numpy(float), g["_dep"].to_numpy(), 300)
        rows.append({
            "role": role, "k": comp.k, "k_lo": lo, "k_hi": hi, "mu": comp.mu,
            "n_deployments": comp.n_risks, "n_episodes": comp.n_total, "note": "",
        })
    return pd.DataFrame(rows).sort_values("k")


def four_level_decomposition(df: pd.DataFrame) -> dict:
    """Split risk variation into pass, task, deployment and role components.

    This is the question the earlier studies could not ask. Every task here is
    attempted up to four times by the same model, so pass-to-pass variance
    measures pure run-to-run noise on *identical work*. Everything above it is
    systematic, and can be attributed.

    Each level is debiased for the sampling noise of the means being compared,
    the same correction Bühlmann-Straub applies to VHM.
    """
    import numpy as np

    task_mean = df.groupby("_task")["loss"].mean()
    task_n = df.groupby("_task")["loss"].size()
    repeated = task_n[task_n >= 2].index
    sub = df[df["_task"].isin(repeated)]
    ss = ((sub["loss"] - sub["_task"].map(task_mean)) ** 2).sum()
    epv_pass = float(ss / (len(sub) - len(repeated)))

    def debiased(values, weights, centres) -> float:
        raw = float(np.average((values - centres) ** 2, weights=weights))
        return max(raw - float(np.mean(epv_pass / weights)), 0.0)

    dep_mean = df.groupby("_dep")["loss"].mean()
    tm = df.groupby(["_dep", "_task"])["loss"].agg(["mean", "size"]).reset_index()
    v_task = debiased(tm["mean"], tm["size"], tm["_dep"].map(dep_mean))

    role_mean = df.groupby("role")["loss"].mean()
    dep = df.groupby(["role", "_dep"])["loss"].agg(["mean", "size"]).reset_index()
    v_dep = debiased(dep["mean"], dep["size"], dep["role"].map(role_mean))

    rl = df.groupby("role")["loss"].agg(["mean", "size"]).reset_index()
    mu = float(np.average(rl["mean"], weights=rl["size"]))
    v_role = debiased(rl["mean"], rl["size"], mu)

    total = v_task + v_dep + v_role
    return {
        "var_pass_noise": epv_pass,
        "var_task_mix": v_task,
        "var_deployment": v_dep,
        "var_role": v_role,
        "share_task_mix": v_task / total if total else float("nan"),
        "share_deployment": v_dep / total if total else float("nan"),
        "share_role": v_role / total if total else float("nan"),
        "n_tasks_with_repeats": int(len(repeated)),
        "n_episodes": int(len(df)),
    }


if __name__ == "__main__":
    main()
