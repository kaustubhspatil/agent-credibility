"""The kill-switch experiment.

Four questions, in the order that decides whether the business exists:

  E1  How large is K, and does defining the risk class more tightly shrink it?
      If tighter class definitions lower K, a role registry has measurable
      economic value. If they do not, the registry is decoration.

  E2  Is K small enough that Z rises within realistic exposure? This is the
      kill switch. K in the thousands means every deployment is priced at the
      class average forever -- exactly the industry average insurers reject.

  E3  Out of sample, does the credibility blend actually beat both of the
      things it blends? A convex combination that beats neither endpoint is
      not worth selling.

  E4  Cold start: fit on every other deployment, then price an unseen one at
      n = 0. This is the day-zero claim, measured.

Honest scope. Episodes are shuffled with a fixed seed to form the past/future
split, because the dataset carries no timestamps. So E3 measures
generalisation to unseen episodes of the same deployment, not resistance to
temporal drift. The drift half of the thesis needs timestamped fleet data and
is not testable here -- say so rather than implying otherwise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .buhlmann import bootstrap_k, components

MIN_EPISODES = 12  # a deployment needs some exposure before it is a risk


# --- class / deployment factorisations ------------------------------------

FACTORISATIONS: dict[str, dict[str, list[str] | None]] = {
    # name: {class: [cols] or None for one class, deployment: [cols]}
    "one class (deployment = repo)": {"cls": None, "dep": ["repo"]},
    "class = task family (deployment = repo)": {
        "cls": ["task_family"],
        "dep": ["repo"],
    },
    "one class (deployment = repo x task family)": {
        "cls": None,
        "dep": ["repo", "task_family"],
    },
}


def prepare(df: pd.DataFrame, scaffold: str = "tool") -> pd.DataFrame:
    """Reduce the raw extract to one honest population.

    The dataset's three splits are NOT three independent populations: xml and
    ticks cover the same 13,500 task instances and agree on the outcome 97% of
    the time, because they are largely the same trajectories re-serialised into
    three action encodings. Treating scaffold as a deployment dimension would
    triple the apparent number of deployments while adding almost no
    independent information -- it inflates between-deployment structure and
    makes K look far better than it is.

    So: analyse one scaffold at a time. Within a single split, repeated
    attempts at the same instance have distinct trajectory ids and are genuine
    re-runs, which is exactly the process variance EPV should be measuring.
    """
    out = df[df["scaffold"] == scaffold].copy()
    out = out.drop_duplicates(subset=["traj_id"])
    return out


def _key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).agg("|".join, axis=1)


def _filtered(df: pd.DataFrame, dep_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["_dep"] = _key(out, dep_cols)
    counts = out["_dep"].value_counts()
    keep = counts[counts >= MIN_EPISODES].index
    return out[out["_dep"].isin(keep)]


@dataclass
class ClassResult:
    name: str
    class_label: str
    k: float
    k_lo: float
    k_hi: float
    mu: float
    epv: float
    vhm: float
    n_episodes: int
    n_deployments: int
    n_for_z50: float
    n_for_z90: float


def e1_variance_components(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """K under each candidate definition of the risk class."""
    rows: list[ClassResult] = []
    for name, spec in FACTORISATIONS.items():
        sub = _filtered(df, spec["dep"])
        if spec["cls"] is None:
            groups = [("all", sub)]
        else:
            sub = sub.copy()
            sub["_cls"] = _key(sub, spec["cls"])
            groups = list(sub.groupby("_cls"))

        for class_label, g in groups:
            if g["_dep"].nunique() < 5 or len(g) < 200:
                continue
            comp = components(g[target].to_numpy(float), g["_dep"].to_numpy())
            lo, hi = bootstrap_k(
                g[target].to_numpy(float), g["_dep"].to_numpy(), n_boot=300
            )
            rows.append(
                ClassResult(
                    name=name,
                    class_label=str(class_label),
                    k=comp.k,
                    k_lo=lo,
                    k_hi=hi,
                    mu=comp.mu,
                    epv=comp.epv,
                    vhm=comp.vhm,
                    n_episodes=comp.n_total,
                    n_deployments=comp.n_risks,
                    n_for_z50=comp.episodes_for_z(0.5),
                    n_for_z90=comp.episodes_for_z(0.9),
                )
            )
    return pd.DataFrame([r.__dict__ for r in rows])


def e3_holdout_sweep(
    df: pd.DataFrame,
    target: str,
    dep_cols: list[str],
    # n_train must be >= 2: with one episode per deployment there are no
    # within-deployment degrees of freedom and EPV is undefined.
    train_sizes: tuple[int, ...] = (2, 3, 5, 10, 20, 40, 80, 160),
    seed: int = 17,
    k_from_pool: bool = True,
) -> pd.DataFrame:
    """Does the blend beat both of the things it blends, out of sample?

    `k_from_pool` decides where the credibility constant comes from, and it is
    not a cosmetic choice. Estimating K from the same thin training slice used
    to price is badly biased downward when that slice is 2-3 episodes, which
    makes Z far too aggressive and can leave the blend worse than the prior
    alone. A bureau does not work that way: K is a property of the pooled class,
    estimated once from every deployment's full history, and handed to a new
    deployment on day zero. `k_from_pool=True` is therefore the product-realistic
    setting; `False` is the pessimistic one, kept because the gap between them
    is itself a finding.
    """
    sub = _filtered(df, dep_cols)

    # One global shuffle, then a per-deployment running index. Every train size
    # nests inside the next, so the sweep is a single consistent ordering
    # rather than a fresh random split at each point.
    order = sub.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    order["_i"] = order.groupby("_dep").cumcount()

    # K as the bureau would supply it: estimated once from the whole pool.
    pool = components(sub[target].to_numpy(float), sub["_dep"].to_numpy())

    rows = []
    for n_train in train_sizes:
        train = order[order["_i"] < n_train]
        test = order[order["_i"] >= n_train]
        # a deployment must have enough held-out episodes to score against
        ok = test["_dep"].value_counts()
        ok = ok[ok >= 8].index
        train, test = train[train["_dep"].isin(ok)], test[test["_dep"].isin(ok)]
        if train["_dep"].nunique() < 5:
            continue

        comp = components(train[target].to_numpy(float), train["_dep"].to_numpy())
        own = train.groupby("_dep")[target].mean()
        n = train.groupby("_dep")[target].size()
        truth = test.groupby("_dep")[target].mean()
        weight = test.groupby("_dep")[target].size()

        idx = truth.index
        # the class prior itself is always pooled; only K's source varies
        source = pool if k_from_pool else comp
        z = np.asarray(source.z(n.reindex(idx).to_numpy(float)), dtype=float)
        pred_prior = np.full(len(idx), comp.mu)
        pred_own = own.reindex(idx).to_numpy(float)
        pred_cred = z * pred_own + (1 - z) * comp.mu
        w = weight.reindex(idx).to_numpy(float)
        y = truth.to_numpy(float)

        def wmse(p):
            return float(np.sum(w * (p - y) ** 2) / np.sum(w))

        rows.append(
            {
                "n_train": n_train,
                "k_used": source.k,
                "k_refit_on_slice": comp.k,
                "mean_z": float(np.mean(z)),
                "n_deployments": len(idx),
                "mse_prior_only": wmse(pred_prior),
                "mse_own_only": wmse(pred_own),
                "mse_credibility": wmse(pred_cred),
            }
        )
    return pd.DataFrame(rows)


def e4_cold_start(
    df: pd.DataFrame,
    target: str,
    dep_cols: list[str],
    own_sizes: tuple[int, ...] = (1, 2, 3, 5, 8, 12, 20, 35, 60),
    seed: int = 23,
) -> pd.DataFrame:
    """Leave one deployment out; price it at n = 0 from the others' prior.

    The comparison that matters is not against an invented "guess 0.5"
    baseline. It is against the only real alternative: waiting. So alongside
    the prior's day-zero error we measure the error of pricing off n episodes
    of the deployment's own history, which converts the pooled prior into the
    unit a buyer understands -- how many episodes of waiting it replaces.
    """
    sub = _filtered(df, dep_cols)
    rng = np.random.default_rng(seed)

    rows = []
    for dep in sub["_dep"].unique():
        others = sub[sub["_dep"] != dep]
        held = sub[sub["_dep"] == dep]
        if others["_dep"].nunique() < 5 or len(held) < 2 * max(own_sizes):
            continue
        comp = components(others[target].to_numpy(float), others["_dep"].to_numpy())

        vals = held[target].to_numpy(float).copy()
        rng.shuffle(vals)
        cut = len(vals) // 2
        seen, future = vals[:cut], vals[cut:]
        truth = float(future.mean())

        row = {
            "deployment": dep,
            "n_episodes": len(held),
            "truth": truth,
            "pred_prior": comp.mu,
            "abs_err_prior": abs(comp.mu - truth),
        }
        for n in own_sizes:
            row[f"abs_err_own_{n}"] = abs(float(seen[:n].mean()) - truth)
        rows.append(row)
    return pd.DataFrame(rows)


def e5_behavioural(df: pd.DataFrame, dep_cols: list[str]) -> pd.DataFrame:
    """Do the behavioural signals separate deployments at all?"""
    sub = _filtered(df, dep_cols)
    feats = [
        "n_actions",
        "n_distinct_actions",
        "action_entropy",
        "repeat_rate",
        "error_rate",
        "mutating_share",
        "n_turns",
    ]
    rows = []
    for f in feats:
        comp = components(sub[f].to_numpy(float), sub["_dep"].to_numpy())
        lo, hi = bootstrap_k(sub[f].to_numpy(float), sub["_dep"].to_numpy(), n_boot=200)
        rows.append(
            {
                "feature": f,
                "k": comp.k,
                "k_lo": lo,
                "k_hi": hi,
                "epv": comp.epv,
                "vhm": comp.vhm,
                "n_for_z50": comp.episodes_for_z(0.5),
                "separation_ratio": (comp.vhm / comp.epv) if comp.epv else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("k")


def main(episodes_path: str, out_dir: str) -> None:
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(episodes_path)
    raw["loss"] = (~raw["resolved"]).astype(float)
    # tolerate parquet written before the pr_<n> family collapse landed
    raw["task_family"] = raw["task_family"].str.replace(
        r"^pr_\d+$", "pr", regex=True
    )

    df = prepare(raw, scaffold="tool")
    print(f"raw {len(raw):,} rows -> analysing the 'tool' scaffold only: "
          f"{len(df):,} episodes | {df.repo.nunique()} repos (deployments) | "
          f"{df.instance_id.nunique():,} distinct tasks | "
          f"loss rate {df.loss.mean():.4f}")

    # Robustness: the same headline on each scaffold, run independently.
    print("\n--- headline K per scaffold (independent populations) ---")
    for scaffold in ("tool", "xml", "ticks"):
        s = prepare(raw, scaffold=scaffold)
        s = _filtered(s, ["repo"])
        comp = components(s["loss"].to_numpy(float), s["_dep"].to_numpy())
        lo, hi = bootstrap_k(s["loss"].to_numpy(float), s["_dep"].to_numpy(), 300)
        print(f"  {scaffold:6s} K = {comp.k:6.2f}  [{lo:5.2f}, {hi:6.2f}]  "
              f"mu = {comp.mu:.3f}  deployments = {comp.n_risks}  n = {comp.n_total:,}")

    e1 = e1_variance_components(df, "loss")
    e1.to_csv(out / "e1_variance_components.csv", index=False)
    print("\n--- E1: credibility constant by class definition ---")
    print(e1[["name", "class_label", "k", "k_lo", "k_hi", "mu",
              "n_deployments", "n_episodes", "n_for_z50"]].to_string(index=False))

    dep_cols = ["repo"]
    e3 = e3_holdout_sweep(df, "loss", dep_cols)
    e3.to_csv(out / "e3_holdout_sweep.csv", index=False)
    print("\n--- E3: out-of-sample MSE by experience size ---")
    print(e3.to_string(index=False))

    e4 = e4_cold_start(df, "loss", dep_cols)
    e4.to_csv(out / "e4_cold_start.csv", index=False)
    print("\n--- E4: cold start (leave-one-deployment-out) ---")
    prior_med = e4.abs_err_prior.median()
    print(f"deployments scored: {len(e4)}")
    print(
        f"pooled prior at n = 0 : median |err| {prior_med:.4f}  "
        f"(within 0.10 for {(e4.abs_err_prior <= 0.10).mean():.0%} of deployments)"
    )
    matched = None
    for col in [c for c in e4.columns if c.startswith("abs_err_own_")]:
        n = int(col.rsplit("_", 1)[1])
        med = e4[col].median()
        flag = ""
        if matched is None and med <= prior_med:
            matched, flag = n, "   <-- own data finally matches the prior here"
        print(f"  own history, n = {n:3d} : median |err| {med:.4f}{flag}")
    if matched:
        print(f"\n=> the pooled prior is worth about {matched} episodes of waiting")
    else:
        print("\n=> own history never matched the prior in the range tested")

    e5 = e5_behavioural(df, dep_cols)
    e5.to_csv(out / "e5_behavioural.csv", index=False)
    print("\n--- E5: behavioural separation ---")
    print(e5.to_string(index=False))

    summary = {
        "n_episodes": int(len(df)),
        "scaffold": "tool",
        "n_deployments": int(_filtered(df, dep_cols)["_dep"].nunique()),
        "overall_loss_rate": float(df.loss.mean()),
        "headline_k": float(e1[e1.name == "one class (deployment = repo)"].k.iloc[0]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="data/episodes.parquet")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    main(args.episodes, args.out)
