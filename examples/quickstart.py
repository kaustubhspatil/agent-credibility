"""End to end, entirely inside the SDK: instrument, persist, reload, price.

Run:  PYTHONPATH=src python examples/quickstart.py

Nothing here imports the research package. `freeboard` records episodes as
sufficient statistics, writes them to a file a design partner could hand back,
reads them again through the same validation, checks the chain, and prices the
fleet -- with no dependencies at all.

The agent is simulated so the file runs anywhere. Every other path is the real
one.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

from freeboard import (
    Capability,
    JsonlSink,
    Recorder,
    Reversibility,
    ToolSpec,
    components,
    credibility_estimate,
    losses_from_records,
    read_checkpoint,
    read_jsonl,
    verify,
    write_checkpoint,
)

SUPPORT_TOOLS = [
    ToolSpec(
        "search_orders",
        capabilities=frozenset({Capability.PRIVATE_DATA}),
        reversibility=Reversibility.REVERSIBLE,
    ),
    ToolSpec(
        "read_ticket_email",
        capabilities=frozenset({Capability.UNTRUSTED_INPUT}),
        reversibility=Reversibility.REVERSIBLE,
    ),
    ToolSpec(
        "issue_refund",
        capabilities=frozenset({Capability.EXTERNAL_EFFECT}),
        reversibility=Reversibility.IRREVERSIBLE,
    ),
]

# three customers running the same agent, with genuinely different risk
FLEET = {"northwind": 0.18, "globex": 0.31, "initech": 0.52}


def run_deployment(name, true_failure_rate, n, rng, workdir):
    """Instrument one customer's fleet, writing to its own log and state."""
    sink = JsonlSink(workdir / f"{name}.jsonl")
    rec = Recorder(
        deployment_id=name,
        role="customer_support",
        tools=SUPPORT_TOOLS,
        declared_capabilities={Capability.PRIVATE_DATA},  # read-only, they said
        state_path=str(workdir / f"{name}.state.json"),
        sink=sink,
    )
    for i in range(n):
        with rec.episode(task_id=f"ticket-{i}") as ep:
            ep.action("read_ticket_email", output_chars=rng.randint(200, 3000))
            for _ in range(rng.randint(1, 4)):
                ep.action(
                    "search_orders",
                    error=rng.random() < 0.12,
                    output_chars=rng.randint(50, 800),
                )
            failed = rng.random() < true_failure_rate
            if not failed and rng.random() < 0.4:
                ep.action("issue_refund")
            if failed and rng.random() < 0.5:
                ep.escalate()
            ep.resolve(success=not failed)
    sink.close()
    write_checkpoint(workdir / f"{name}.checkpoint.json", rec.checkpoint())
    return rec


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="freeboard-quickstart-"))

    # a fresh stream per deployment, and enough episodes that the observed rate
    # lands near the true one -- otherwise the last column reads as though the
    # estimator were wrong when it is only the simulation being noisy
    recorders = {
        name: run_deployment(name, rate, 250, random.Random(seed), workdir)
        for seed, (name, rate) in enumerate(FLEET.items())
    }

    # 1. what actually leaves the customer's process
    first = next(iter(read_jsonl(workdir / "northwind.jsonl")))
    print("one wire record -- every field a number, an enum or a hash:")
    for key in sorted(first):
        print(f"    {key:24s} {first[key]}")

    # 2. the derived envelope, and where it disagrees with the declaration
    rec = recorders["northwind"]
    env, div = rec.envelope, rec.divergence()
    print(f"\nenvelope class        : {env.class_code}")
    print(f"rule of two           : "
          f"{'VIOLATED' if env.rule_of_two_violated else 'ok'}")
    print("declared              : ['private_data']")
    print(f"granted but undeclared: "
          f"{sorted(c.value for c in div.granted_but_not_declared)}")
    print(f"divergence score      : {div.score}  (understated: {div.understated})")

    # 3. the log, read back and checked -- this is the auditor's path
    print("\nreading the logs back and verifying each chain:")
    replayed = []
    for name in FLEET:
        records = list(read_jsonl(workdir / f"{name}.jsonl"))
        head = read_checkpoint(workdir / f"{name}.checkpoint.json")["head"]
        result = verify(records, expected_head=head)
        print(f"    {name:10s} {len(records):4d} records  chain ok: {bool(result)}")
        replayed.extend(records)

    # 4. price the fleet from the reloaded file, not from memory
    values, ids = losses_from_records(replayed)
    comp = components(values, ids)
    print(f"\npooled across the fleet: mu = {comp.mu:.3f}, K = {comp.k:.2f}")
    print(f"a new deployment reaches Z = 0.5 after "
          f"{comp.episodes_for_z(0.5):.0f} episodes\n")

    print(f"{'deployment':12s} {'n':>4s} {'observed':>9s} {'Z':>6s} "
          f"{'priced':>7s}  {'true':>5s}")
    for name in FLEET:
        own = [v for v, i in zip(values, ids) if i == recorders[name].deployment_hash]
        n = len(own)
        observed = sum(own) / n
        print(f"{name:12s} {n:4d} {observed:9.3f} {comp.z(n):6.2f} "
              f"{credibility_estimate(observed, n, comp):7.3f}  {FLEET[name]:5.2f}")

    print(f"\nlogs, state and checkpoints written to: {workdir}")


if __name__ == "__main__":
    main()
