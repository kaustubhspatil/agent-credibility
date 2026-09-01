"""Do roles separate? The registry's whole claim, tested two ways.

A role class registry is only worth maintaining if the class you price against
changes the price. Two tests, in increasing order of how much they should
count:

  R1  Three-level variance decomposition. Split deployment-level risk variation
      into a between-role part and a within-role part. If role explains little,
      the registry is decoration.

  R2  Cross-role cold start. Price a held-out deployment at n = 0 from its own
      role's prior, from the pooled all-roles prior, and from a deliberately
      wrong role's prior. R1 is a description; this is the decision. If the
      wrong-role prior prices about as well as the right one, nobody needs the
      registry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .buhlmann import components


def deployment_table(
    df: pd.DataFrame, channel: str, dep_cols: list[str], min_episodes: int = 8
) -> pd.DataFrame:
    d = df.copy()
    d["_dep"] = d[dep_cols].astype(str).agg("|".join, axis=1)
    g = d.groupby(["_dep", "role"])[channel].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= min_episodes]
    return g.rename(columns={"mean": "rate", "size": "n"})


def r1_decomposition(
    df: pd.DataFrame, channel: str, dep_cols: list[str], min_episodes: int = 8
) -> dict:
    """How much of deployment-level risk variation does role explain?

    Uses the same debiasing idea as Bühlmann-Straub at each level: observed
    spread between group means overstates true spread by the sampling noise of
    those means, so the noise term is subtracted before taking a ratio.
    """
    d = df.copy()
    d["_dep"] = d[dep_cols].astype(str).agg("|".join, axis=1)
    counts = d["_dep"].value_counts()
    d = d[d["_dep"].isin(counts[counts >= min_episodes].index)]

    # level 1: episode noise within a deployment
    epv = components(d[channel].to_numpy(float), d["_dep"].to_numpy()).epv

    tab = deployment_table(df, channel, dep_cols, min_episodes)
    mu = float(np.average(tab["rate"], weights=tab["n"]))

    # level 3: between roles, debiased by the sampling noise of a role mean
    role_stats = (
        tab.assign(w=tab["n"])
        .groupby("role")
        .apply(
            lambda g: pd.Series(
                {
                    "mean": np.average(g["rate"], weights=g["w"]),
                    "n": g["n"].sum(),
                    "k": len(g),
                }
            ),
            include_groups=False,
        )
    )
    n_roles = len(role_stats)
    between_raw = float(
        np.average((role_stats["mean"] - mu) ** 2, weights=role_stats["n"])
    )
    between_noise = float(np.mean(epv / role_stats["n"]))
    between = max(between_raw - between_noise, 0.0)

    # level 2: deployments within their role, debiased likewise
    merged = tab.merge(
        role_stats["mean"].rename("role_mean"), left_on="role", right_index=True
    )
    within_raw = float(
        np.average((merged["rate"] - merged["role_mean"]) ** 2, weights=merged["n"])
    )
    within_noise = float(np.mean(epv / merged["n"]))
    within = max(within_raw - within_noise, 0.0)

    total = between + within
    return {
        "channel": channel,
        "n_deployments": int(len(tab)),
        "n_roles": int(n_roles),
        "epv": epv,
        "var_between_roles": between,
        "var_within_roles": within,
        "share_explained_by_role": (between / total) if total > 0 else np.nan,
        "role_means": {r: round(float(v), 4) for r, v in role_stats["mean"].items()},
    }


def r2_cross_role_cold_start(
    df: pd.DataFrame,
    channel: str,
    dep_cols: list[str],
    min_episodes: int = 8,
    seed: int = 41,
) -> pd.DataFrame:
    """Leave one deployment out, then price it three ways at n = 0."""
    rng = np.random.default_rng(seed)
    tab = deployment_table(df, channel, dep_cols, min_episodes)

    rows = []
    for _, target in tab.iterrows():
        others = tab[tab["_dep"] != target["_dep"]]
        same = others[others["role"] == target["role"]]
        other_roles = others[others["role"] != target["role"]]
        if len(same) < 2 or other_roles["role"].nunique() < 1:
            continue

        own_role_prior = float(np.average(same["rate"], weights=same["n"]))
        pooled_prior = float(np.average(others["rate"], weights=others["n"]))

        # a deliberately wrong role: the one whose mean is furthest away, i.e.
        # the worst case a mis-specified registry entry could produce
        role_means = other_roles.groupby("role").apply(
            lambda g: np.average(g["rate"], weights=g["n"]), include_groups=False
        )
        wrong_role = role_means.sub(own_role_prior).abs().idxmax()
        wrong_prior = float(role_means[wrong_role])
        # worst case is the headline, but a registry mistake is not always the
        # most distant entry, so report the average wrong entry too
        wrong_avg = float(role_means.mean())

        truth = float(target["rate"])
        rows.append(
            {
                "deployment": target["_dep"],
                "role": target["role"],
                "n": int(target["n"]),
                "truth": truth,
                "err_own_role": abs(own_role_prior - truth),
                "err_pooled": abs(pooled_prior - truth),
                "err_wrong_role": abs(wrong_prior - truth),
                "err_wrong_role_avg": abs(wrong_avg - truth),
                "wrong_role_used": wrong_role,
            }
        )
    return pd.DataFrame(rows)


def power_at_scale(
    n_deployments: int,
    episodes_per: int,
    mean: float,
    k_true_grid=(2, 5, 10, 25, 50, 100, 250),
    n_reps: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Same power question as before, re-asked at this study's much smaller n."""
    rng = np.random.default_rng(seed)
    rows = []
    for k_true in k_true_grid:
        a, b = mean * k_true, (1 - mean) * k_true
        ests, collapsed = [], 0
        for _ in range(n_reps):
            theta = rng.beta(a, b, size=n_deployments)
            vals = np.concatenate(
                [rng.binomial(1, t, episodes_per) for t in theta]
            ).astype(float)
            ids = np.repeat(np.arange(n_deployments), episodes_per)
            comp = components(vals, ids)
            if comp.vhm_was_truncated:
                collapsed += 1
            else:
                ests.append(comp.k)
        arr = np.array(ests)
        rows.append(
            {
                "k_true": k_true,
                "k_median": float(np.median(arr)) if len(arr) else np.inf,
                "k_p05": float(np.percentile(arr, 5)) if len(arr) else np.inf,
                "k_p95": float(np.percentile(arr, 95)) if len(arr) else np.inf,
                "false_kill_rate": collapsed / n_reps,
                "rel_bias": (float(np.median(arr)) / k_true - 1) if len(arr) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main(out_dir: str) -> None:
    import json
    from pathlib import Path

    from .webagents import annotator_agreement, load_episodes

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ep, dup = load_episodes()
    dep_cols = ["role", "model", "site"]
    channels = ["loss_failure", "loss_side_effect", "loss_looping"]

    print(f"{len(ep):,} web-agent episodes | {ep.role.nunique()} roles | "
          f"{ep.model.nunique()} agent models")

    agree = annotator_agreement(dup)
    agree.to_csv(out / "r0_annotator_agreement.csv", index=False)
    print("\n--- R0: expert agreement (the label-noise floor inside EPV) ---")
    print(agree.round(3).to_string(index=False))

    tab = deployment_table(ep, "loss_failure", dep_cols)
    n_dep, med_n = len(tab), int(tab["n"].median())
    print(f"\ndeployments with >= 8 episodes: {n_dep} (median {med_n} episodes)")

    pw = power_at_scale(n_dep, med_n, mean=float(ep.loss_failure.mean()))
    pw.to_csv(out / "r_power_web.csv", index=False)
    print("\n--- power at THIS scale (not the SWE-smith scale) ---")
    print(pw.round(3).to_string(index=False))
    ok = pw[(pw.false_kill_rate < 0.05) & (pw.rel_bias.abs() < 0.25)]
    if len(ok):
        print(f"=> reliably detectable up to K ~ {ok.k_true.max()}")

    print("\n--- R1: does role explain deployment-level risk? ---")
    r1 = [r1_decomposition(ep, c, dep_cols) for c in channels]
    pd.DataFrame([{k: v for k, v in r.items() if k != "role_means"} for r in r1]).to_csv(
        out / "r1_role_decomposition.csv", index=False
    )
    for r in r1:
        print(f"\n  {r['channel']}  ({r['n_deployments']} deployments, "
              f"{r['n_roles']} roles)")
        print(f"    role means            : {r['role_means']}")
        print(f"    var between roles     : {r['var_between_roles']:.5f}")
        print(f"    var within roles      : {r['var_within_roles']:.5f}")
        share = r["share_explained_by_role"]
        print(f"    share explained by role: "
              f"{share:.1%}" if np.isfinite(share) else "    n/a")

    print("\n--- R2: cross-role cold start (the registry's actual claim) ---")
    summary = {}
    for c in channels:
        r2 = r2_cross_role_cold_start(ep, c, dep_cols)
        r2.to_csv(out / f"r2_cross_role_{c}.csv", index=False)
        if not len(r2):
            continue
        med = r2[["err_own_role", "err_pooled", "err_wrong_role",
                  "err_wrong_role_avg"]].median()
        summary[c] = {k: round(float(v), 4) for k, v in med.items()}
        print(f"\n  {c}  ({len(r2)} deployments)")
        print(f"    median |err| own-role prior   : {med.err_own_role:.4f}")
        print(f"    median |err| pooled prior     : {med.err_pooled:.4f}")
        print(f"    median |err| wrong role, avg  : {med.err_wrong_role_avg:.4f}")
        print(f"    median |err| wrong role, worst: {med.err_wrong_role:.4f}")
        print(f"    loss events in the data       : {int(ep[c].sum())}")
        penalty = med.err_wrong_role / med.err_own_role if med.err_own_role else np.nan
        print(f"    mis-specification penalty     : {penalty:.2f}x")

    (out / "role_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    main(args.out)
