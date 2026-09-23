# freeboard

Record how an AI agent deployment behaves, as privacy-safe counts, so it can be priced
and audited without anyone seeing its data.

(Freeboard is the height from a ship's waterline to its deck, which marine underwriters
measure before insuring a vessel.)

Standard library only, no dependencies.

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
    declared_capabilities={Capability.PRIVATE_DATA},   # cross-check only
    state_path="/var/lib/freeboard/acme-prod.json",    # survives restarts
)

with rec.episode(task_id="ticket-8821") as ep:
    ep.action("search_orders", output_chars=412)
    ep.action("issue_refund")
    ep.resolve(success=True)
```

## Privacy

- No parameter accepts free text. `output_chars` is a length. Tool output never enters
  the SDK.
- Ids are salted and hashed per deployment.
- Tool names stay in-process.
- The wire format is an allow-list (numbers, bools, nulls, fixed enums, hex digests) and
  `validate()` runs on every record.

`test_canary_never_reaches_the_wire` pushes an SSN-shaped value into every input and
checks it appears nowhere in the output.

A complete record:

```json
{"deployment_id":"fdf3d47fa257542d","envelope_class":"C111-I0","episode_id":"a6b1be3e8c254f59",
 "error_rate":0.0,"escalated":false,"manifest_hash":"fa89b11ed297a47d",
 "max_reversibility":"irreversible","n_actions":6,"n_distinct_tools":3,"n_irreversible":1,
 "n_mutating":1,"observed_capabilities":["external_effect","private_data","untrusted_input"],
 "output_chars":4101,"pass_index":1,"repeat_rate":0.6,"resolved":true,"reward":null,
 "role":"customer_support","seq":0,"prev_hash":"000…","entry_hash":"5da5…",
 "task_hash":"5ddb12eaa459eb73","tool_entropy":1.251629}
```

## Task envelope

Scope comes from the tools the agent is actually given. What the customer declares is only
a cross-check.

| View | Source | Trusted |
|---|---|---|
| derived | granted tool manifest | yes |
| observed | what the agent used | yes |
| declared | what the customer said | no |

`TaskEnvelope.rule_of_two_violated` flags an agent that has untrusted input, private data
and external effects all at once. `class_code` is based on capabilities, not tool names.

## Tamper evidence

```
entry_hash(n) = sha256( prev_hash(n) || canonical_json(record without hashes) )
```

Deleting, editing or reordering a record breaks the chain. That only holds once the chain
head is committed somewhere the writer can't change, which is what `Checkpoint` is for.

The chain also can't prove a record was true when written. An agent that controls its own
tool-call path can feed the recorder bad data. `ep.attest()` records an independent action
count from outside the process (eBPF, an egress proxy, a sidecar):

```python
with rec.episode(task_id="ticket-8821") as ep:
    ep.action("search_orders")
    ep.attest(actions_observed=5, source="ebpf")
    ep.resolve(success=True)
# unreported_actions = 4
```

## Estimating

```python
from freeboard import components, credibility_estimate, losses_from_records, read_jsonl

values, ids = losses_from_records(read_jsonl("episodes.jsonl"))
comp = components(values, ids)

comp.k                     # credibility constant, in episodes
comp.episodes_for_z(0.5)   # episodes until own data gets half the weight
credibility_estimate(observed_rate, n, comp)
```

`p̂ = Z · own experience + (1 − Z) · class prior`, `Z = n / (n + K)`. Standard
Bühlmann–Straub estimators, tested against a closed form and against the numpy version
used in the research.

## Storage and transport

- `JsonlSink` / `read_jsonl` / `verify`: append-only JSON Lines with chain checks.
- `BureauSink`: spools locally first and never raises into the agent loop.

```python
from freeboard import BureauSink

sink = BureauSink("https://bureau.freeboardrisk.com",
                  spool="/var/lib/freeboard/acme.jsonl")
sink.flush()
print(sink.last_prior)
sink.anchor(rec.checkpoint())
```

- `to_otel_attributes()`: `gen_ai.*` and `agent.credibility.*` span attributes, no OTel
  dependency.

## Status

Early, and the API may change before 1.0. There's no severity model, so this estimates
frequency, not a premium. The keyword fallback for tools with no declared capabilities is
rough, and inferred tools are counted in `inferred_tools`.

Source, research and the bureau: https://github.com/kaustubhspatil/agent-credibility

Apache-2.0
