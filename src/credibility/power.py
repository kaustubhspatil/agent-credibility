"""How small a K could this dataset detect?

If the experiment returns "K is enormous", that reads as a kill signal. But a
noisy estimator returns enormous K when there is nothing to see *and* when
there is something to see and not enough data to see it. Those are opposite
business conclusions, so the difference has to be measured before the verdict
is read, not after.

This sweeps the true credibility constant at the design's actual scale and
reports (a) how well K is recovered and (b) how often VHM collapses to zero --
the false kill rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .buhlmann import components


def _beta_params(mean: float, k_true: float) -> tuple[float, float]:
    """Beta(a, b) with the given mean whose credibility constant is k_true.

    For a Beta-Bernoulli class, K = a + b exactly, so this is just a
    reparameterisation of the concentration.
    """
    return mean * k_true, (1.0 - mean) * k_true


def sweep(
    k_true_grid=(2, 5, 10, 25, 50, 100, 250, 500, 1000),
    n_deployments: int = 180,
    episodes_per: int = 130,
    mean: float = 0.35,
    n_reps: int = 60,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k_true in k_true_grid:
        a, b = _beta_params(mean, k_true)
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
        ests = np.array(ests)
        rows.append(
            {
                "k_true": k_true,
                "k_median": float(np.median(ests)) if len(ests) else np.inf,
                "k_p05": float(np.percentile(ests, 5)) if len(ests) else np.inf,
                "k_p95": float(np.percentile(ests, 95)) if len(ests) else np.inf,
                "false_kill_rate": collapsed / n_reps,
                "rel_bias": (float(np.median(ests)) / k_true - 1) if len(ests) else np.nan,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-deployments", type=int, default=180)
    ap.add_argument("--episodes-per", type=int, default=130)
    ap.add_argument("--out", default="out/power.csv")
    args = ap.parse_args()

    df = sweep(n_deployments=args.n_deployments, episodes_per=args.episodes_per)
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    detectable = df[(df.false_kill_rate < 0.05) & (df.rel_bias.abs() < 0.25)]
    if len(detectable):
        print(
            f"\nreliably detectable up to K ~= {detectable.k_true.max():,} "
            f"at {args.n_deployments} deployments x {args.episodes_per} episodes"
        )
