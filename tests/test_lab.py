"""The severity lab.

Two of these are regressions for bugs that made the experiment meaningless
while still producing confident-looking numbers, which is the dangerous kind.
"""

import random

import pytest

from lab.hazards import BY_NAME, CONTROL, Objective, Vector, requires_capability
from lab.runner import SimulatedExecutor, run_scenario, run_suite
from lab.scenario import (
    DELETE_RECORD,
    FETCH_PAGE,
    READ_CASE,
    Scenario,
    default_suite,
    matched_arms,
)
from lab.severity import pair_results, report, summarise


def suite(n=200):
    return default_suite(n_episodes=n)


# --- the experiment is actually an experiment ------------------------------


def test_three_arms_not_two():
    """constrained vs exposed alone is a tautology: an agent without the tool
    cannot use the tool. The baseline arm is what makes it a comparison."""
    arms = matched_arms(
        "x", "role", "task", [READ_CASE, FETCH_PAGE], DELETE_RECORD,
        BY_NAME["indirect_injection_destructive"], n_episodes=10,
    )
    assert [a.arm for a in arms] == ["constrained", "baseline", "exposed"]
    assert arms[1].hazard is CONTROL          # baseline has no adversary
    assert arms[1].tools == arms[2].tools     # but the same powers as exposed


def test_the_baseline_arm_can_register_harm(tmp_path):
    """REGRESSION: harm_tool was derived from the hazard's objective, so the
    baseline -- which has no hazard -- could never record anything, and the
    inducement figure silently collapsed into the exposed rate."""
    arms = matched_arms(
        "x", "role", "task", [READ_CASE, FETCH_PAGE], DELETE_RECORD,
        BY_NAME["indirect_injection_destructive"], n_episodes=400,
    )
    baseline = arms[1]
    assert baseline.harm_tool == DELETE_RECORD.name
    assert baseline.harm_is_reachable

    outcomes, _ = run_scenario(baseline, SimulatedExecutor(accident_rate=0.25), seed=1)
    assert summarise(outcomes).p_harm > 0.0


def test_ordinary_work_never_reaches_the_harm_tool():
    """REGRESSION: the benign tool pool included the harm tool, so the agent hit
    it by chance and the measurement became 'did it ever touch this' rather than
    'was it induced to'."""
    scenario = matched_arms(
        "x", "role", "task", [READ_CASE, FETCH_PAGE], DELETE_RECORD,
        CONTROL, n_episodes=1,
    )[2]
    ex = SimulatedExecutor(accident_rate=0.0, comply_base=0.0, comply_per_turn=0.0)
    rng = random.Random(0)
    for _ in range(300):
        steps = ex.run(scenario, rng)
        assert all(s.tool != DELETE_RECORD.name for s in steps)


def test_the_constrained_arm_cannot_realise_the_harm():
    arms = matched_arms(
        "x", "role", "task", [READ_CASE, FETCH_PAGE], DELETE_RECORD,
        BY_NAME["indirect_injection_destructive"], n_episodes=200,
    )
    outcomes, _ = run_scenario(arms[0], SimulatedExecutor(), seed=2)
    s = summarise(outcomes)
    assert s.p_harm == 0.0
    assert not s.harm_reachable
    assert "refusal untested" in s.note   # not counted as a safety win


def test_inducement_is_exposed_minus_baseline():
    out = run_suite(suite(300), SimulatedExecutor(), seed=5)
    for pair in pair_results(out):
        assert pair.inducement == pytest.approx(
            pair.exposed.p_harm - pair.baseline.p_harm
        )
        assert pair.exposed.p_harm > pair.baseline.p_harm


def test_persistence_raises_inducement():
    """A boundary that holds once often does not hold eight times; single-shot
    probes therefore overstate safety."""
    base = [READ_CASE, FETCH_PAGE]
    once = matched_arms("a", "r", "t", base, DELETE_RECORD,
                        BY_NAME["skill_md_exfiltration"], n_episodes=600)[2]
    many = matched_arms("b", "r", "t", base, DELETE_RECORD,
                        BY_NAME["skill_md_exfiltration_persistent"],
                        n_episodes=600)[2]
    ex = SimulatedExecutor()
    p_once = summarise(run_scenario(once, ex, seed=3)[0]).p_harm
    p_many = summarise(run_scenario(many, ex, seed=3)[0]).p_harm
    assert p_many > p_once


# --- the boundary the lab must not cross -----------------------------------


def test_nothing_returns_an_unconditional_rate():
    """A sandbox cannot honestly produce a population frequency: the workload
    is ours, and task mix is 65.7% of systematic variance."""
    import lab.severity as sev

    for name in dir(sev):
        assert "frequency" not in name.lower()
        assert "base_rate" not in name.lower()
    assert "conditional on a stated hazard" in sev.__doc__


def test_the_report_says_what_it_is_not():
    out = run_suite(suite(50), SimulatedExecutor(), seed=1)
    text = report(out)
    assert "None of it is a population frequency" in text
    assert "P(harm | the stated hazard)" in text


def test_hazard_reachability_maps_to_capability():
    assert requires_capability(Objective.EXFILTRATE_SECRET) == "external_effect"
    assert requires_capability(Objective.SILENT_MISREPORT) is None
    assert CONTROL.is_control


# --- the lab dogfoods the SDK ----------------------------------------------


def test_episodes_are_recorded_through_the_real_sdk():
    scenario = suite(20)[2]
    outcomes, rec = run_scenario(scenario, SimulatedExecutor(), seed=4)
    assert len(rec.records) == len(outcomes) == 20
    from freeboard import to_wire, verify

    wire = [to_wire(r) for r in rec.records]
    assert verify(wire, expected_head=rec.checkpoint().head)
