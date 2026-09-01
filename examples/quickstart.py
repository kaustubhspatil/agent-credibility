"""End to end: instrument a fleet, emit wire records, price them.

Run:  PYTHONPATH=src python examples/quickstart.py

The point is that nothing bridges the two halves of this repo. The records the
recorder emits are already the input the credibility estimator takes -- the
sufficient statistics were chosen to be exactly what `components()` consumes,
which is why the SDK can be dependency-free and still produce something an
underwriter can price.

The agent here is simulated so the file runs anywhere. The recording path is
the real one.
"""

from __future__ import annotations

import random

from credibility.buhlmann import components, credibility_estimate
from freeboard import (
    Capability,
    Recorder,
    Reversibility,
    ToolSpec,
    to_wire,
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


def run_deployment(name: str, true_failure_rate: float, n: int, rng: random.Random):
    rec = Recorder(
        deployment_id=name,
        role="customer_support",
        tools=SUPPORT_TOOLS,
        # the customer declared read-only; the wiring says otherwise
        declared_capabilities={Capability.PRIVATE_DATA},
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
    return rec


def main() -> None:
    # a fresh stream per deployment, and enough episodes that the observed
    # rate lands near the true one -- otherwise the last column reads as though
    # the estimator were wrong when it is only the simulation being noisy
    recorders = {
        name: run_deployment(name, rate, 250, random.Random(seed))
        for seed, (name, rate) in enumerate(FLEET.items())
    }

    # 1. what leaves the customer's process
    sample = to_wire(recorders["northwind"].records[0])
    print("one wire record (every field a number, enum or hash):")
    for key in sorted(sample):
        print(f"    {key:24s} {sample[key]}")

    # 2. the derived envelope, and where it disagrees with the declaration
    rec = recorders["northwind"]
    env = rec.envelope
    div = rec.divergence()
    print(f"\nenvelope class      : {env.class_code}")
    print(f"rule of two         : {'VIOLATED' if env.rule_of_two_violated else 'ok'}")
    print("declared            : ['private_data']")
    print(f"granted but undeclared: "
          f"{sorted(c.value for c in div.granted_but_not_declared)}")
    print(f"divergence score    : {div.score}  (understated: {div.understated})")

    # 3. price the fleet -- the records are already the estimator's input
    values, ids = [], []
    for name, r in recorders.items():
        for record in r.records:
            values.append(0.0 if record.resolved else 1.0)
            ids.append(name)

    comp = components(values, ids)
    print(f"\npooled across the fleet: mu = {comp.mu:.3f}, K = {comp.k:.2f}")
    print(f"a new deployment reaches Z = 0.5 after "
          f"{comp.episodes_for_z(0.5):.0f} episodes\n")

    print(f"{'deployment':12s} {'n':>4s} {'observed':>9s} {'Z':>6s} "
          f"{'priced':>7s}  {'true':>5s}")
    for name, r in recorders.items():
        own = sum(0.0 if x.resolved else 1.0 for x in r.records) / len(r.records)
        n = len(r.records)
        z = float(comp.z(n))
        priced = float(credibility_estimate(own, n, comp))
        print(f"{name:12s} {n:4d} {own:9.3f} {z:6.2f} {priced:7.3f}  "
              f"{FLEET[name]:5.2f}")


if __name__ == "__main__":
    main()
