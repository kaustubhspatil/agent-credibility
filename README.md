# Credibility for agents

**Does a pooled prior across agent deployments earn its keep, or does it collapse into the industry average that insurers reject?**

It earns its keep. On 24,100 real agent episodes across 125 deployments, the credibility constant is **K ≈ 7.6 [4.8, 12.2]** — a deployment reaches equal weight between the pooled class prior and its own experience after **about eight episodes**, and 90% weight after roughly 68.

This is the kill-switch experiment for an agent-insurance "base rate bureau". It was built to fail loudly if the idea does not work.

![credibility curve](out/fig1_credibility_curve.png)

## Why this is the question

Insurability for AI agents attaches to *monitored roles*, not to agents ([arXiv 2606.16465](https://arxiv.org/pdf/2606.16465), Theorem 1: "General-purpose agents are not insurable objects by themselves"). Pricing a role needs a failure frequency, and the industry's own blueprint ([*Underwriting the Agent Economy*](https://www.underwriting-agents.com/report.pdf), July 2026) calls developing that frequency data "a priority for the industry" while rejecting industry-wide actuarial tables, because averages "gloss over these differences" between systems.

Bühlmann credibility resolves the contradiction:

```
p̂ = Z · (own experience) + (1 − Z) · (class prior),    Z = n / (n + K)
```

Nothing is overridden by an average — the average is only the fallback for what has not been observed yet. The whole proposition reduces to one estimated constant. `K = EPV / VHM`: expected process variance over variance of hypothetical means. **If deployments are noisy but alike, K explodes, Z stays near zero, and every deployment is priced at the class average forever.** That is the failure mode this repo was built to detect.

## Results

### The blend beats both of the things it blends

![holdout sweep](out/fig2_holdout_sweep.png)

Predicting each deployment's held-out failure rate, weighted MSE:

| episodes of experience | class prior only | own experience only | credibility blend |
|---:|---:|---:|---:|
| 2 | 0.0282 | 0.1420 | **0.0266** |
| 5 | 0.0285 | 0.0555 | **0.0188** |
| 10 | 0.0295 | 0.0286 | **0.0142** |
| 20 | 0.0283 | 0.0169 | **0.0104** |
| 40 | 0.0277 | 0.0059 | **0.0053** |
| 160 | 0.0242 | 0.0032 | **0.0031** |

The blend wins at every size, and by roughly 2× at n = 10–20 where the two endpoints cross.

One finding worth its own line: **K must come from the pool, not from the deployment being priced.** Re-estimating K on each thin training slice biases it low (K̂ = 2.2 at n = 2 against a true 7.6), makes Z far too aggressive, and leaves the blend *worse than the prior alone* at n = 2–3. Both configurations are reported — `k_used` vs `k_refit_on_slice` in `out/e3_holdout_sweep.csv`.

### The pooled prior is worth about eight episodes of waiting

Leave-one-deployment-out, priced at n = 0 from the other deployments:

- pooled prior, day zero: median absolute error **0.1046**, within 0.10 for 46% of deployments
- own history needs **n = 8** before its median error (0.1023) matches that

Two independent computations landing on the same number — E4's eight episodes and K = 7.6 — is the internal consistency check, since `Z = 0.5` at `n = K` is exactly the claim that the prior is worth K episodes of own data.

### A tighter class definition halves K

![k by class](out/fig3_k_by_class.png)

Conditioning on task family takes mean K from 7.6 to 3.8, and some families far lower (`func_pm_remove_loop` 0.94, `combine_file` 1.94) while `lm_rewrite` is the stubborn outlier at 14.9. This is the quantitative case for a role class registry: defining the class more tightly measurably speeds up experience rating. Scaffold, by contrast, buys nothing.

### Which behavioural signals carry deployment identity

`error_rate` (K = 5.9) and `n_turns` (K = 8.2) separate deployments; action-mix diversity does not (`action_entropy` K = 30.7, `n_actions` K = 34.7). Instrument the first two.

## A second role: do roles actually separate?

The SWE-smith study has one agent and one model, so it can only measure heterogeneity between *customers*. The registry claim — that the class you price against must be the right **role** — needs more than one role. [AgentRewardBench](https://huggingface.co/datasets/McGill-NLP/agent-reward-bench) supplies four web-agent environments crossed with four agent models, 1,302 trajectories, each judged by a **human expert** rather than an automatic scorer, and carrying three distinct loss channels:

| channel | meaning | base rate | events |
|---|---|---:|---:|
| failure | task not accomplished | 0.72 | 948 |
| looping | agent cycled repetitively — runaway cost | 0.51 | 675 |
| side effect | agent changed something it should not — **the claimable one** | 0.065 | 87 |

Roles differ plainly in base rate: failure runs 0.62 (webarena) to 0.92 (assistantbench), looping 0.38 to 0.76.

### Role explains about half of deployment-level risk

A three-level decomposition (episode → deployment → role), debiasing each level for the sampling noise of its own means:

| channel | var between roles | var within roles | share explained by role |
|---|---:|---:|---:|
| failure | 0.0107 | 0.0093 | **53.5%** |
| looping | 0.0267 | 0.0212 | **55.7%** |
| side effect | 0.0003 | 0.0010 | 21.7% |

### Pricing from the wrong role costs 2.6–2.9×

![role cold start](out/fig5_role_cold_start.png)

Leave-one-deployment-out, priced at n = 0, median absolute error:

| channel | own role's prior | pooled across roles | wrong role (avg) | wrong role (worst) |
|---|---:|---:|---:|---:|
| failure | **0.107** | 0.156 | 0.191 | 0.279 |
| looping | **0.136** | 0.200 | 0.212 | 0.398 |
| side effect | 0.053 | 0.066 | 0.056 | *0.039* |

For task failure and runaway looping the registry is load-bearing: knowing the role beats pooling across roles by ~1.5×, and a mis-specified role entry costs 2.6–2.9× in day-zero pricing error. That is the registry's value, measured.

**The side-effect channel inverts, and it should not be explained away.** The wrong-role prior prices side effects *better* than the right one. The reason is that all four role means sit between 0.023 and 0.095 — close together and near zero — so "furthest role" is picking noise, and any small number scores well. With 87 events across 54 deployments there are roughly 1.6 events per deployment. **The loss channel that matters most for insurance is exactly the one this data cannot establish role separation for**, because catastrophic events are rare. That is a data requirement, not a result: pricing the claimable channel needs a corpus with far more side-effect events than any public dataset currently has.

### Power at this scale is much weaker

Re-running the power analysis at the web study's actual size (54 deployments, median 19 episodes) rather than the SWE-smith size: K is recovered reliably only up to **K ≈ 25**, with a 24% false-kill rate by K = 100. The web results are directional and should not be quoted as point estimates. The SWE-smith K ≈ 7.6 is the one with an envelope behind it.

### Label noise is real and measured

105 trajectories were annotated twice. Expert-vs-expert agreement: κ = 0.755 (failure), 0.769 (looping), 0.646 (side effect). So roughly 11% of failure labels are disagreements between qualified humans — a floor on EPV that no model can go below, and a reminder that "ground truth" here is a judgement.

## Trusting the number

The estimator is validated against a closed-form answer before use. For a Beta(a,b)-Bernoulli class the credibility constant is exactly `a + b`, so there is a known target: `pytest` checks recovery across three concentration regimes and four seeds (16 tests).

A power analysis (`credibility/power.py`) settles interpretation in advance, because a noisy estimator returns "K is enormous" both when there is nothing to see and when there is something to see and too little data to see it — opposite business conclusions. At this design's scale, K is recovered unbiased up to **K ≈ 500 with a 0% false-kill rate**. The measured 7.6 sits two orders of magnitude inside that envelope.

## Honest limits

- **The SWE-smith study has one agent and one model.** Every episode there is SWE-agent + Claude 3.7 Sonnet, so its heterogeneity is between *customers* (repositories). The role question is answered separately, on AgentRewardBench, at much smaller scale.
- **Outcome is task-driven.** 13,501 distinct tasks are attempted up to 21 times each, and 82% of repeated tasks resolve identically. Much of the between-deployment variance is task-mix difficulty rather than behavioural difference.
- **No timestamps, so no drift test.** The past/future split is a seeded shuffle. This measures generalisation to unseen episodes, *not* resistance to temporal drift — the half of the thesis that says a decaying prior beats a stale table remains untested and needs timestamped fleet data.
- **`resolved` is a proxy for a claimable loss**, not a real loss. No dollar severity is modelled here.
- **The three dataset splits are not three populations.** `xml` and `ticks` cover the same 13,500 instances and agree on outcome 97% of the time — they are largely the same trajectories re-serialised. Treating scaffold as a deployment dimension would have inflated the result; the analysis uses one scaffold at a time and replicates across all three (K = 7.59 / 8.28 / 8.08).

## Running it

```bash
pip install -e ".[data,plots,dev]"
python -m credibility.extract --out data/episodes.parquet   # ~45 min, streams 4.2 GB
python -m credibility.experiment --episodes data/episodes.parquet --out out
python -m credibility.plots --out out
pytest
```

Data: [`SWE-bench/SWE-smith-trajectories`](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) — 76,002 trajectories from SWE-agent + Claude 3.7 Sonnet. Extraction emits counts, rates and entropies per episode; no prompt or source text leaves the trajectory.

## Layout

| path | what |
|---|---|
| `src/credibility/extract.py` | trajectories → per-episode behavioural features |
| `src/credibility/buhlmann.py` | Bühlmann-Straub variance components, Z, bootstrap CI |
| `src/credibility/experiment.py` | E1 class definitions, E3 holdout sweep, E4 cold start, E5 signals |
| `src/credibility/webagents.py` | AgentRewardBench expert annotations -> web-agent episodes |
| `src/credibility/role_separation.py` | R1 role decomposition, R2 cross-role cold start |
| `src/credibility/power.py` | how small a K this design could detect |
| `src/credibility/plots.py` | the four figures |
| `tests/` | estimator validation against the Beta-Bernoulli closed form |
