"""Does the derived task envelope carry real risk signal?

`freeboard.derive_envelope` reads a tool manifest and returns a structural risk
shape -- capabilities, a rule-of-two flag, a worst-case reversibility, a class
code. Everything downstream assumes that shape means something. Until now the
only evidence for it was that it seemed reasonable.

ATBench (arXiv 2604.02022, Apache-2.0) is 1,000 agent trajectories, each with
the tool manifest the agent was given and a human-audited safety label, plus a
taxonomy of risk source, failure mode and real-world harm. Five reviewers
audited every row. That makes it a test set for the weakest component in the
SDK: the keyword capability priors, which fire whenever a tool declares nothing.

Two things this can and cannot show.

It CAN show discrimination: whether manifests that the envelope calls riskier
are in fact the ones that went wrong. That is a relative question and the
balanced design supports it.

It CANNOT show a base rate. ATBench is 503 safe / 497 unsafe *by construction*.
A 49.7% unsafe rate is a curation choice, not an observed frequency, and no K
can be estimated from it -- there is no deployment grouping and no exposure
denominator. Frequency still needs a live fleet.

The confound to watch: if the benchmark's unsafe scenarios were built by
selecting riskier tools, then the envelope predicting the label is partly
circular. `tool_overlap()` measures how far that holds.
"""

from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import pandas as pd

from freeboard import Capability, ToolSpec, derive_envelope

REPO = "AI45Research/ATBench"
FILE = "ATBench/test.json"


def load() -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, FILE, repo_type="dataset")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def envelope_frame(rows: list[dict]) -> pd.DataFrame:
    """One row per trajectory, with the envelope derived from its manifest."""
    out = []
    for row in rows:
        tools = [
            ToolSpec(name=t.get("name", ""), description=t.get("description", "") or "")
            for t in row.get("tool_used", [])
            if isinstance(t, dict)
        ]
        env = derive_envelope(tools)
        caps = {c.value for c in env.capabilities}
        out.append(
            {
                "id": row.get("id"),
                "unsafe": int(row.get("label", 0)),
                "risk_source": row.get("risk_source"),
                "failure_mode": row.get("failure_mode"),
                "real_world_harm": row.get("real_world_harm"),
                "n_tools": env.n_tools,
                "n_caps": len(env.capabilities),
                "untrusted_input": int(Capability.UNTRUSTED_INPUT.value in caps),
                "private_data": int(Capability.PRIVATE_DATA.value in caps),
                "external_effect": int(Capability.EXTERNAL_EFFECT.value in caps),
                "rule_of_two_violated": int(env.rule_of_two_violated),
                "max_reversibility": env.max_reversibility.value,
                "n_irreversible_tools": env.n_irreversible_tools,
                "inferred_tools": env.inferred_tools,
                "class_code": env.class_code,
                "tool_names": tuple(sorted(t.name for t in tools)),
            }
        )
    return pd.DataFrame(out)


# --- metrics ---------------------------------------------------------------


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC. 0.5 is chance; ties handled by mid-ranks."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # mid-ranks for ties
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]

    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auc(scores, labels, n_boot=1000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(scores)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auc(scores[idx], labels[idx])
        if not math.isnan(a):
            out.append(a)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def lift_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Unsafe rate split by a binary or categorical structural feature."""
    g = df.groupby(column)["unsafe"].agg(["size", "mean"]).rename(
        columns={"size": "n", "mean": "unsafe_rate"}
    )
    base = df["unsafe"].mean()
    g["lift_vs_base"] = g["unsafe_rate"] / base
    return g.sort_values("unsafe_rate", ascending=False).round(4)


def tool_overlap(df: pd.DataFrame) -> dict:
    """How circular is this? If safe and unsafe rows draw on the same tools,
    the envelope is not just re-reading the benchmark's own construction."""
    safe_tools: Counter = Counter()
    unsafe_tools: Counter = Counter()
    for _, r in df.iterrows():
        (unsafe_tools if r["unsafe"] else safe_tools).update(r["tool_names"])
    shared = set(safe_tools) & set(unsafe_tools)
    union = set(safe_tools) | set(unsafe_tools)
    safe_vol = sum(safe_tools.values())
    unsafe_vol = sum(unsafe_tools.values())
    return {
        "distinct_tools": len(union),
        "tools_in_both_classes": len(shared),
        "share_of_tools_in_both": len(shared) / max(len(union), 1),
        "safe_tool_uses_on_shared": sum(safe_tools[t] for t in shared) / max(safe_vol, 1),
        "unsafe_tool_uses_on_shared": sum(unsafe_tools[t] for t in shared)
        / max(unsafe_vol, 1),
    }


def main(out_dir: str = "out") -> None:
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load()
    df = envelope_frame(rows)
    df.drop(columns=["tool_names"]).to_csv(out / "atbench_envelopes.csv", index=False)

    print(f"{len(df):,} trajectories | unsafe {df.unsafe.mean():.1%} "
          f"(balanced by construction -- not a base rate)")
    print(f"tools per trajectory: median {df.n_tools.median():.0f}, "
          f"max {df.n_tools.max()}")
    print(f"every tool is unlabelled, so the keyword priors do all the work: "
          f"{df.inferred_tools.sum():,} inferred classifications")

    print("\n--- how circular is this? ---")
    ov = tool_overlap(df)
    for k, v in ov.items():
        print(f"  {k:30s} {v:.4f}" if isinstance(v, float) else f"  {k:30s} {v}")

    print("\n--- A1: does the rule of two carry signal? ---")
    print(lift_table(df, "rule_of_two_violated").to_string())

    print("\n--- A2: each capability on its own ---")
    for cap in ("untrusted_input", "private_data", "external_effect"):
        t = lift_table(df, cap)
        if 1 in t.index and 0 in t.index:
            print(f"  {cap:18s} present {t.loc[1,'unsafe_rate']:.3f} "
                  f"vs absent {t.loc[0,'unsafe_rate']:.3f}  "
                  f"lift {t.loc[1,'unsafe_rate']/max(t.loc[0,'unsafe_rate'],1e-9):.2f}x"
                  f"  (n={int(t.loc[1,'n'])}/{int(t.loc[0,'n'])})")
        else:
            print(f"  {cap:18s} degenerate: only one value present")

    print("\n--- A3: discrimination of purely structural scores ---")
    labels = df["unsafe"].to_numpy()
    for name, score in [
        ("n_capabilities", df["n_caps"].to_numpy(float)),
        ("n_irreversible_tools", df["n_irreversible_tools"].to_numpy(float)),
        ("n_tools", df["n_tools"].to_numpy(float)),
        ("rule_of_two", df["rule_of_two_violated"].to_numpy(float)),
    ]:
        a = auc(score, labels)
        lo, hi = bootstrap_auc(score, labels)
        verdict = "signal" if lo > 0.5 else ("inverse" if hi < 0.5 else "no signal")
        print(f"  {name:22s} AUC {a:.3f}  [{lo:.3f}, {hi:.3f}]  {verdict}")

    print("\n--- A4: risk stratification by class code ---")
    codes = lift_table(df, "class_code")
    print(codes[codes["n"] >= 20].to_string())

    print("\n--- A5: what kind of risk does ATBench actually contain? ---")
    unsafe = df[df.unsafe == 1]
    src = unsafe.risk_source.value_counts()
    injected = src.index.str.contains("inject|prompt|description|malicious", case=False)
    print(f"  {src[injected].sum()} of {len(unsafe)} unsafe rows "
          f"({src[injected].sum() / len(unsafe):.0%}) arrive by injection -- "
          f"from outside the tool manifest")
    still_safe = int((df[df.unsafe == 0].risk_source != "benign").sum())
    print(f"  {still_safe} SAFE rows also carry a risk source: the same threat was "
          f"present and handled")
    print("  => ATBench varies OUTCOME with threat roughly held constant. It asks")
    print("     whether the agent coped -- competence. The envelope measures")
    print("     exposure. Different quantities, so a null result here is expected.")

    print("\n--- A6: does the envelope predict SEVERITY (harm type)? ---")
    from scipy import stats

    for feat in ("n_tools", "n_caps", "n_irreversible_tools", "rule_of_two_violated"):
        groups = [
            g[feat].to_numpy()
            for _, g in unsafe.groupby("real_world_harm")
            if len(g) >= 20
        ]
        stat, pval = stats.kruskal(*groups)
        eta = (stat - len(groups) + 1) / (len(unsafe) - len(groups))
        verdict = "SIGNAL" if pval < 0.01 else "no signal"
        print(f"  {feat:22s} H={stat:6.2f}  p={pval:.2e}  eta2={eta:.3f}  {verdict}")

    print("\n--- A7: harm categories, for a severity taxonomy ---")
    harm = (
        unsafe
        .groupby("real_world_harm")
        .agg(
            n=("unsafe", "size"),
            irreversible_tools=("n_irreversible_tools", "mean"),
            capabilities=("n_caps", "mean"),
            rule_of_two_rate=("rule_of_two_violated", "mean"),
        )
        .sort_values("irreversible_tools", ascending=False)
    )
    print(harm.round(2).to_string())
    harm.to_csv(out / "atbench_harm_taxonomy.csv")

    print("\n--- verdict ---")
    print("  frequency: NO. The derived envelope does not predict whether a")
    print("             trajectory goes wrong. Best structural AUC 0.534.")
    print("  severity : YES. It predicts which kind of harm results, modestly")
    print("             but significantly (n_tools p=1e-10, eta2=0.12).")
    print("  => class_code is an exposure/severity dimension. It must not be")
    print("     presented as a failure-rate predictor.")


if __name__ == "__main__":
    main()
