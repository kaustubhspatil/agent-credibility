"""A second role: web agents, from AgentRewardBench expert annotations.

The SWE-smith study could only measure heterogeneity *within* one role, because
every episode there is the same coding agent. The registry claim -- that the
class you price against has to be the right role -- needs at least two roles.

AgentRewardBench supplies four environments (webarena, visualwebarena,
workarena, assistantbench) crossed with four agent models, and every trajectory
carries a human expert's judgement rather than an automatic score. It also
carries three distinct loss channels, which matters because they are not the
same risk:

    failure      the task was not accomplished          -- service credit
    side effect  the agent changed something it should  -- the claimable one
                 not have
    looping      the agent cycled repetitively          -- runaway cost

The dataset is small (1,302 trajectories). Everything here is reported against
a power analysis run at this scale, not the SWE-smith one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REPO = "McGill-NLP/agent-reward-bench"


def _load(path: str) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    return pd.read_csv(hf_hub_download(REPO, path, repo_type="dataset"))


def load_episodes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (episodes, duplicate_pairs).

    `duplicate_pairs` holds trajectories annotated by more than one expert; they
    are the label-noise floor and are reported rather than silently dropped.
    """
    ann = _load("data/annotations.csv")
    ann["annotator_name"] = ann["annotator_name"].str.strip()

    # --- a 'site' column per benchmark: the customer-like grouping ---------
    site_maps: dict[str, pd.Series] = {}

    wa = _load("data/webarena.csv")
    site_maps["webarena"] = wa.set_index("task_name")["sites"]

    vwa = _load("data/visualwebarena.csv")
    site_maps["visualwebarena"] = vwa.set_index("task_name")["sites"]

    # workarena has no sites; its category is the equivalent grouping
    wk = _load("data/workarena.csv")
    wk_key = wk.set_index("task_name")["category"]
    site_maps["workarena"] = wk_key

    # assistantbench is a set of open web questions with no site structure;
    # difficulty is the only grouping available and is a weak stand-in
    ab = _load("data/assistantbench.csv")
    site_maps["assistantbench"] = ab.set_index("task_name")["difficulty"]

    def site_of(row) -> str:
        m = site_maps.get(row["benchmark"])
        if m is None:
            return "unknown"
        val = m.get(row["task_id"])
        if isinstance(val, str):
            # webarena tasks can list several sites; take the combination as
            # the grouping rather than exploding the row
            return val.strip("[]").replace("'", "").replace(", ", "+")
        return "unknown"

    ann["site"] = ann.apply(site_of, axis=1)

    # workarena task_ids carry an l1/l2 level; keep it, it is a real risk driver
    lvl = wk.set_index("task_name")["level"]
    ann["level"] = ann["task_id"].map(lvl).fillna("na")

    # --- loss channels ----------------------------------------------------
    ann = ann[ann["trajectory_success"].isin(["Successful", "Unsuccessful"])].copy()
    ann["loss_failure"] = (ann["trajectory_success"] == "Unsuccessful").astype(float)
    ann["loss_side_effect"] = (ann["trajectory_side_effect"] == "Yes").astype(float)
    ann["loss_looping"] = (ann["trajectory_looping"] == "Yes").astype(float)

    ann["role"] = ann["benchmark"]
    ann["model"] = ann["model_name"].str.replace("GenericAgent-", "", regex=False)

    key = ["benchmark", "task_id", "model_name"]
    dup_mask = ann.duplicated(key, keep=False)
    duplicates = ann[dup_mask].copy()

    # one row per trajectory: keep the first annotator deterministically
    episodes = ann.sort_values("annotator_name").drop_duplicates(key, keep="first")
    return episodes.reset_index(drop=True), duplicates


def annotator_agreement(duplicates: pd.DataFrame) -> pd.DataFrame:
    """How much of the within-deployment noise is just labelling disagreement?"""
    key = ["benchmark", "task_id", "model_name"]
    rows = []
    for channel in ("loss_failure", "loss_side_effect", "loss_looping"):
        g = duplicates.groupby(key)[channel]
        pairs = g.agg(["mean", "size"])
        pairs = pairs[pairs["size"] == 2]
        agree = (pairs["mean"].isin([0.0, 1.0])).mean()
        base = duplicates[channel].mean()
        # chance agreement for a binary label at this base rate
        chance = base**2 + (1 - base) ** 2
        kappa = (agree - chance) / (1 - chance) if chance < 1 else np.nan
        rows.append(
            {
                "channel": channel,
                "double_annotated": int(len(pairs)),
                "raw_agreement": float(agree),
                "chance_agreement": float(chance),
                "cohens_kappa": float(kappa),
            }
        )
    return pd.DataFrame(rows)
