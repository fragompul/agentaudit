import json
from pathlib import Path

import numpy as np
import pytest

from agentaudit.agents.synthetic import NullAgent, OracleAgent, SkillAgent
from agentaudit.bench.suite import generate_suite
from agentaudit.runner import RunConfig, pending_attempts, run_attempt, run_evaluation
from agentaudit.schema import read_jsonl, to_resultset

TASKS = generate_suite(12, seed=0)  # 72 tasks


def test_oracle_solves_every_task_and_null_none() -> None:
    assert all(run_attempt(OracleAgent(), t, 0).success == 1.0 for t in TASKS)
    assert all(run_attempt(NullAgent(), t, 0).success == 0.0 for t in TASKS)


def test_skill_agent_matches_its_latent_probability() -> None:
    agent = SkillAgent("a", theta=0.5)
    expected = np.mean([agent.success_probability(t) for t in TASKS])
    got = np.mean([run_attempt(agent, t, r).success for t in TASKS for r in range(15)])
    assert got == pytest.approx(expected, abs=0.04)


def _calls(traj):  # type: ignore[no-untyped-def]
    return [(c.name, str(c.arguments)) for s in traj.steps for c in s.tool_calls]


def test_attempts_are_deterministic_and_runs_differ() -> None:
    agent = SkillAgent("a", theta=0.0)
    t = next(x for x in TASKS if 0.3 < agent.success_probability(x) < 0.7)
    a, b = run_attempt(agent, t, 3), run_attempt(agent, t, 3)
    assert a.success == b.success and _calls(a) == _calls(b)
    assert {run_attempt(agent, t, r).success for r in range(40)} == {0.0, 1.0}


def test_fault_injection_hurts_nonrobust_agents_and_is_agent_independent() -> None:
    robust = SkillAgent("robust", theta=3.0, retry_prob=1.0)
    brittle = SkillAgent("brittle", theta=3.0, retry_prob=0.0)
    cfg = RunConfig(fault_rate=0.3)
    sr = np.mean([run_attempt(robust, t, r, cfg).success for t in TASKS for r in range(4)])
    sb = np.mean([run_attempt(brittle, t, r, cfg).success for t in TASKS for r in range(4)])
    clean = np.mean([run_attempt(brittle, t, 0).success for t in TASKS])
    assert sr > sb + 0.2 and clean > 0.8
    # same (task, run) -> same injected fault pattern regardless of the agent
    t0 = TASKS[0]
    fa = run_attempt(OracleAgent("o1"), t0, 2, cfg).metadata["faults_injected"]
    fb = run_attempt(OracleAgent("o2"), t0, 2, cfg).metadata["faults_injected"]
    assert fa == fb


def test_trajectory_fields_are_populated() -> None:
    traj = run_attempt(OracleAgent(), TASKS[0], 0)
    assert traj.steps and traj.version.startswith("agentaudit-bench")
    assert traj.metadata["family"] == TASKS[0].family
    assert traj.termination == "completed"
    capped = run_attempt(OracleAgent(), TASKS[0], 0, RunConfig(max_steps=1))
    assert len(capped.steps) == 1


def test_resumable_and_balanced_prefix(tmp_path: Path) -> None:
    out = tmp_path / "t.jsonl"
    agent = SkillAgent("a", theta=1.0)
    # simulate an interruption after 100 attempts of a 3-run evaluation
    todo = list(pending_attempts("a", TASKS, 3, out))
    assert len(todo) == 3 * len(TASKS)
    assert [r for _, r in todo[: len(TASKS)]] == [0] * len(TASKS)  # run-major
    with out.open("w") as fh:
        for task, run in todo[:100]:
            fh.write(run_attempt(agent, task, run).to_json() + "\n")
        fh.write('{"truncated": ')  # crash mid-write
    n = run_evaluation(agent, TASKS, 3, out)
    assert n == 3 * len(TASKS) - 100
    assert run_evaluation(agent, TASKS, 3, out) == 0
    keys = [(t.task_id, t.run) for t in _read(out)]
    assert len(keys) == len(set(keys)) == 3 * len(TASKS)


def _read(path: Path) -> list:  # type: ignore[type-arg]
    lines = [line for line in path.read_text().splitlines() if line.startswith('{"')]
    keep = []
    for line in lines:
        try:
            keep.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    from agentaudit.schema import Trajectory

    return [Trajectory.from_dict(r) for r in keep]


def test_parallel_workers_equal_serial(tmp_path: Path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    agent = SkillAgent("a", theta=0.3)
    seen: list[tuple[int, int]] = []
    run_evaluation(agent, TASKS[:20], 2, a, workers=1)
    run_evaluation(agent, TASKS[:20], 2, b, workers=4, progress=lambda i, n: seen.append((i, n)))
    sa = to_resultset(read_jsonl(a)).scores
    sb = to_resultset(read_jsonl(b)).scores
    np.testing.assert_array_equal(sa, sb)
    assert seen[-1] == (40, 40)


class Exploding(NullAgent):
    def act(self, messages, tools, *, seed):  # type: ignore[no-untyped-def]
        raise RuntimeError("backend down")


def test_backend_errors_are_recorded_not_raised() -> None:
    traj = run_attempt(Exploding("boom"), TASKS[0], 0)
    assert traj.success == 0.0 and traj.termination.startswith("error")
