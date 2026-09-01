"""Figures for the credibility experiment.

One chart per question. The credibility curve is the one that either sells the
company or kills it, so it gets the bootstrap band rather than a bare line.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#1E4C7A"
OCHRE = "#A87A17"
AQUA = "#3E8C93"
RED = "#A03A32"
GREEN = "#2E6F52"
GREY = "#6B7683"
RULE = "#D3D9E1"

mpl.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": "#E8ECF1",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _title(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="600", pad=28 if subtitle else 8)
    if subtitle:
        ax.text(
            0.0,
            1.012,
            subtitle,
            transform=ax.transAxes,
            fontsize=9,
            color=GREY,
            va="bottom",
        )


def credibility_curve(k: float, k_lo: float, k_hi: float, out: Path,
                      markers: dict[str, float] | None = None) -> None:
    """Z(n) = n / (n + K), with the bootstrap interval on K as a band."""
    n = np.logspace(0, 4.2, 400)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    if np.isfinite(k_lo) and np.isfinite(k_hi):
        ax.fill_between(n, n / (n + k_hi), n / (n + k_lo), color=BLUE, alpha=0.16, lw=0)
    ax.plot(n, n / (n + k), color=BLUE, lw=2.2)

    ax.axhline(0.5, color=RULE, lw=1, ls="--", zorder=0)
    if np.isfinite(k):
        ax.plot([k], [0.5], "o", color=OCHRE, ms=6, zorder=5)
        ax.annotate(
            f"Z = 0.5 at {k:,.0f} episodes",
            xy=(k, 0.5),
            xytext=(8, -16),
            textcoords="offset points",
            fontsize=9,
            color=OCHRE,
            fontweight="600",
        )

    for label, x in (markers or {}).items():
        ax.axvline(x, color=GREY, lw=1, ls=":", zorder=0)
        ax.text(x, 1.01, label, rotation=0, ha="center", va="bottom",
                fontsize=8, color=GREY)

    ax.set_xscale("log")
    ax.set_xlim(1, 16000)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Episodes observed for this deployment (log scale)")
    ax.set_ylabel("Z — weight on own experience")
    _title(
        ax,
        "How fast a deployment stops being priced at the class average",
        "Bühlmann credibility weight. Band is the 5–95% bootstrap interval on K "
        "(deployments resampled).",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def holdout_sweep(e3: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(e3.n_train, e3.mse_prior_only, "o-", color=OCHRE, lw=1.8, ms=4,
            label="class prior only  (Z = 0)")
    ax.plot(e3.n_train, e3.mse_own_only, "s-", color=RED, lw=1.8, ms=4,
            label="own experience only  (Z = 1)")
    ax.plot(e3.n_train, e3.mse_credibility, "^-", color=BLUE, lw=2.4, ms=5,
            label="credibility blend")
    ax.set_xscale("log")
    ax.set_xlabel("Episodes of experience used to price (log scale)")
    ax.set_ylabel("Weighted MSE on held-out episodes")
    ax.legend(loc="upper right")
    _title(
        ax,
        "Does the blend beat both of the things it blends?",
        "Predicting each deployment's held-out failure rate. Lower is better.",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def k_by_class(e1: pd.DataFrame, out: Path) -> None:
    """Does defining the risk class more tightly shrink K?"""
    agg = (
        e1.groupby("name")
        .apply(
            lambda g: pd.Series(
                {
                    "k": np.average(g.k.replace(np.inf, np.nan).dropna())
                    if g.k.replace(np.inf, np.nan).notna().any()
                    else np.nan,
                    "n": g.n_deployments.sum(),
                }
            ),
            include_groups=False,
        )
        .dropna()
        .sort_values("k")
    )
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(agg) + 1.9))
    colors = [BLUE if i == 0 else GREY for i in range(len(agg))]
    ax.barh(range(len(agg)), agg.k, color=colors, height=0.6)
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels(agg.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("K  (episodes to reach Z = 0.5; lower is better)")
    for i, v in enumerate(agg.k):
        ax.text(v, i, f"  {v:,.0f}", va="center", fontsize=9, color=GREY)
    ax.grid(axis="y", visible=False)
    _title(
        ax,
        "Does a tighter class definition buy anything?",
        "Mean K across classes under each factorisation. If tighter classes "
        "lower K, the role registry has measurable value.",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def behavioural(e5: pd.DataFrame, out: Path) -> None:
    d = e5.replace([np.inf, -np.inf], np.nan).dropna(subset=["separation_ratio"])
    d = d.sort_values("separation_ratio")
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(d) + 1.9))
    ax.barh(range(len(d)), d.separation_ratio, color=AQUA, height=0.6)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.feature, fontsize=9)
    ax.set_xlabel("VHM / EPV  — between-deployment variance per unit of within-deployment noise")
    for i, v in enumerate(d.separation_ratio):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=9, color=GREY)
    ax.grid(axis="y", visible=False)
    _title(
        ax,
        "Which behavioural signals actually separate deployments?",
        "Higher means the signal carries deployment identity rather than noise.",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main(out_dir: str) -> None:
    out = Path(out_dir)
    e1 = pd.read_csv(out / "e1_variance_components.csv")
    e3 = pd.read_csv(out / "e3_holdout_sweep.csv")
    e5 = pd.read_csv(out / "e5_behavioural.csv")

    head = e1[e1.name == "one class (deployment = repo)"].iloc[0]
    credibility_curve(
        head.k,
        head.k_lo,
        head.k_hi,
        out / "fig1_credibility_curve.png",
        # No "one day of a support agent" markers: this dataset is one coding
        # agent and carries no timestamps, so any wall-clock annotation would
        # be imported from elsewhere and dressed up as a measurement.
    )
    holdout_sweep(e3, out / "fig2_holdout_sweep.png")
    k_by_class(e1, out / "fig3_k_by_class.png")
    behavioural(e5, out / "fig4_behavioural.png")
    role_cold_start(out)
    print(f"figures -> {out}")




def role_cold_start(out: Path) -> None:
    """Cross-role cold start: does pricing from the wrong role hurt?"""
    channels = [
        ("loss_failure", "task failure"),
        ("loss_looping", "runaway looping"),
        ("loss_side_effect", "side effect"),
    ]
    frames = []
    for col, label in channels:
        f = out / f"r2_cross_role_{col}.csv"
        if f.exists():
            d = pd.read_csv(f)
            frames.append((label, d))
    if not frames:
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    width, gap = 0.26, 0.06
    xs = np.arange(len(frames))
    series = [
        ("err_own_role", "own role's prior", BLUE),
        ("err_pooled", "pooled across roles", GREY),
        ("err_wrong_role", "wrong role (worst case)", RED),
    ]
    for j, (col, label, color) in enumerate(series):
        vals = [d[col].median() for _, d in frames]
        ax.bar(xs + (j - 1) * (width + gap), vals, width, label=label, color=color)
        for x, v in zip(xs + (j - 1) * (width + gap), vals):
            ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=GREY)

    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for lbl, _ in frames])
    ax.set_ylabel("Median |error| pricing a new deployment at n = 0")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    _title(
        ax,
        "Does the registry entry have to be right?",
        "Leave-one-deployment-out across 4 web-agent roles. Lower is better.",
    )
    fig.tight_layout()
    fig.savefig(out / "fig5_role_cold_start.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    main(args.out)
