# Freeboard

**Measure how much safety envelope an AI agent deployment has left — without ever seeing its data.**

*Freeboard* is the distance from a ship's waterline to its deck: the margin before it is swamped, and the quantity marine underwriters measure to decide whether a vessel is insurable at all. This measures the same thing for an agent deployment.

Stdlib only. No dependencies.

```bash
pip install freeboard
```

```python
from freeboard import Recorder, ToolSpec, Capability, Reversibility

rec = Recorder(
    deployment_id="acme-prod",
    role="customer_support",
    tools=[
        ToolSpec("search_orders",
                 capabilities=frozenset({Capability.PRIVATE_DATA}),
                 reversibility=Reversibility.REVERSIBLE),
        ToolSpec("issue_refund",
                 capabilities=frozenset({Capability.EXTERNAL_EFFECT}),
                 reversibility=Reversibility.IRREVERSIBLE),
    ],
    declared_capabilities={Capability.PRIVATE_DATA},   # never trusted
    state_path="/var/lib/freeboard/acme-prod.json",    # survives restarts
)

with rec.episode(task_id="ticket-8821") as ep:
    ep.action("search_orders", output_chars=412)
    ep.action("issue_refund")
    ep.resolve(success=True)
```

## Content cannot leak, because it cannot get in

This is not redaction. Redaction is a deny-list that strips the sensitive things someone thought of, and it has to be audited and trusted.

- **No parameter accepts free text.** `action()` takes a tool name, an error flag, and `output_chars` — a *length*. Tool output never enters the SDK.
- **Identifiers are salted and hashed** per deployment. Task ids stay linkable within a deployment without the raw value travelling.
- **Tool names never reach the wire.** They stay in-process for entropy and capability accounting.
- **The wire format is an allow-list.** Every field is a number, a bool, a null, a fixed enum value, or a hex digest. `validate()` refuses anything else, and `to_wire()` cannot run without it.

The test that proves it, `test_canary_never_reaches_the_wire`, pushes an SSN-shaped canary into the deployment id, role, task id, tool name and description, then asserts it appears nowhere in the payload — not in a value, not in a key, not in the JSON.

A complete record is counts, hashes and enums, and this is all of it:

```json
{"deployment_id":"fdf3d47fa257542d","envelope_class":"C111-I0","episode_id":"a6b1be3e8c254f59",
 "error_rate":0.0,"escalated":false,"manifest_hash":"fa89b11ed297a47d",
 "max_reversibility":"irreversible","n_actions":6,"n_distinct_tools":3,"n_irreversible":1,
 "n_mutating":1,"observed_capabilities":["external_effect","private_data","untrusted_input"],
 "output_chars":4101,"pass_index":1,"repeat_rate":0.6,"resolved":true,"reward":null,
 "role":"customer_support","seq":0,"prev_hash":"000…","entry_hash":"5da5…",
 "task_hash":"5ddb12eaa459eb73","tool_entropy":1.251629}
```

## The task envelope is derived, not declared

A declared scope is an incentive to understate — premiums fall as scope narrows. So scope is derived from the tool manifest the agent is actually wired to, and the declaration is kept only as a cross-check. Three views are tracked and none is collapsed into a verdict:

| view | source | trusted |
|---|---|---|
| derived | the granted tool manifest | yes |
| observed | what the agent actually exercised | yes |
| declared | what the customer said | no |

Disagreement is the signal, with understatement weighted double. `TaskEnvelope.rule_of_two_violated` flags an agent holding all three of untrusted input, private data and external effect — a complete attack chain. `class_code` keys on capabilities rather than tool names, so two deployments with the same powers fall in the same risk class.

## Tamper-evident sequencing


Each record commits to the one before it:

```
entry_hash(n) = sha256( prev_hash(n) || canonical_json(record without hashes) )
```

Deleting an episode leaves a sequence gap; editing one breaks its `entry_hash`; reordering breaks both. State persists across restarts, so an ephemeral container resumes its chain rather than generating a discontinuity.

**What chaining alone does not prove:** a party who deletes a record and recomputes every hash after it produces a chain that verifies perfectly. Chaining is tamper-*evident* only once the head has been committed somewhere that party cannot rewrite — which is what `Checkpoint` is for. Any EU AI Act Article 12 claim rests on the checkpoints being anchored, not on the chain existing. That limitation is a passing test in the repo, not a footnote.

## What the chain does not prove

The chain proves a record was not altered **after** it was written. It does not prove
the record was **true when written**, and that distinction is not academic.

In the July 2026 Hugging Face incident, agents developed working tool-call spoofing —
intercepting their own tool-call handler, suppressing real output, and finally using
`ptrace` to rewrite *which command was requested*, so the executed action and the
recorded action differed. METR's independent investigation found intentional spoofing
in **7% of the transcripts** it examined.

Freeboard records inside the agent's process, at the agent's privilege. An agent that
can intercept its own tool-call path can feed the recorder fiction, and the chain will
faithfully seal it. **No in-process recorder defends against this, including this one.**

What survives is worth being precise about. The derived envelope does: `class_code` and
the rule-of-two check come from the tool *manifest* — configuration, not agent-reported —
so an agent cannot spoof which tools it was granted. The behavioural statistics do not.

### Attestation

`ep.attest()` records an independent count from outside the process — a kernel-level
observer such as [AgentSight](https://arxiv.org/html/2508.02736v2) (eBPF, <3% overhead),
an egress proxy, or a separately-privileged sidecar:

```python
with rec.episode(task_id="ticket-8821") as ep:
    ep.action("search_orders")
    ep.attest(actions_observed=5, source="ebpf")   # the kernel saw five
    ep.resolve(success=True)
# -> unreported_actions = 4: work the environment saw and the agent did not declare
```

Only the positive gap counts. Fewer observed than reported is an instrumentation gap;
**more** observed than reported is the signature of a recorder being lied to. The SDK
does not verify the claim — it records the comparison. An attester running in the same
process at the same privilege is worthless, because whatever can rewrite the tool-call
path can rewrite the attestation beside it.

A bureau pools these into a baseline, which answers the question an attestation tool
cannot answer for its own user: *is this amount of divergence normal for agents like
mine?*

## OpenTelemetry

`to_otel_attributes()` returns `gen_ai.*` and `agent.credibility.*` span attributes for an existing tracer, validated on the way out. No OTel dependency.

## Measuring, not just recording

The estimator ships with the package, in the standard library:

```python
from freeboard import components, credibility_estimate, losses_from_records, read_jsonl

values, ids = losses_from_records(read_jsonl("episodes.jsonl"))
comp = components(values, ids)

comp.k                     # the credibility constant, in episodes
comp.episodes_for_z(0.5)   # exposure until own experience carries half the weight
credibility_estimate(observed_rate, n, comp)
```

`p̂ = Z · own experience + (1 − Z) · class prior`, with `Z = n / (n + K)`. A new
deployment prices off its class; one with history prices off itself. Nothing is
overridden by an average — the average is only the fallback for what has not been
observed yet.

These are the standard unbiased Bühlmann–Straub estimators, checked in the test suite
against a closed form (for a Beta(a,b)–Bernoulli class the constant is exactly `a + b`)
*and* against the numpy implementation used for the published research, on the real
24,100-episode corpus, so the shipped SDK and the published numbers cannot drift apart.

## Somewhere to put it

`JsonlSink` appends validated records; `read_jsonl` reads them back through the same
validation; `verify` checks the chain across the round trip. JSON Lines rather than a
JSON array so that a process killed mid-run leaves a readable file with a complete
prefix, instead of an unparseable truncated array.

## Sending it to a bureau

`BureauSink` drops into `Recorder(sink=...)`. It writes each episode to a local spool
**before** attempting the network and swallows every transport failure, because a
monitoring SDK that raises inside your agent loop when a remote server is slow is
uninstallable. An unreachable bureau costs availability, never data.

```python
from freeboard import Recorder, BureauSink

sink = BureauSink("https://bureau.freeboardrisk.com",
                  spool="/var/lib/freeboard/acme.jsonl")
rec = Recorder(deployment_id="acme-prod", role="customer_support",
               tools=[...], sink=sink,
               state_path="/var/lib/freeboard/acme.state")
...
sink.flush()
print(sink.last_prior)          # pooled prior for your class, same round trip
sink.anchor(rec.checkpoint())   # commit the chain head outside your own control
```

A reference bureau runs at `https://bureau.freeboardrisk.com`; the server is open
source in the same repository, so what it stores can be checked rather than trusted.

## Status

Early — the API may change before 1.0.

There is **no transport layer** and **no certificate renderer**. Both are deliberate.
Transport would point at a bureau service that does not exist yet, and a certificate's
format is decided by whoever signs it — guessing means shipping a surface to deprecate
after the first auditor conversation.

Also worth knowing before you rely on it: there is no severity model, so this estimates
frequency and not a premium; and the keyword fallback for tools that declare no
capabilities is deliberately weak, which is why anything inferred is counted in
`inferred_tools` rather than quietly trusted.

The research behind it — what a pooled prior across agent deployments is worth, measured on ~40,000 episodes across three corpora, with the negative controls — lives in the repository.

**Source, research and guides:** https://github.com/kaustubhspatil/agent-credibility

Apache-2.0.
