from pathlib import Path

import pytest

from agentaudit.agents.base import Agent, AgentAction, Message
from agentaudit.agents.cache import CachedAgent, CacheMissError, Store
from agentaudit.agents.synthetic import SkillAgent
from agentaudit.bench.base import ToolInvocation, ToolSpec
from agentaudit.bench.suite import generate_suite
from agentaudit.runner import run_attempt

TASKS = generate_suite(6, seed=0)


class Counting(Agent):
    def __init__(self) -> None:
        self.name = "counting"
        self.calls = 0

    def act(self, messages, tools, *, seed):  # type: ignore[no-untyped-def]
        self.calls += 1
        return AgentAction("x", (ToolInvocation("finish", {}),), 7, 3, 0.25, {"k": 1})


def test_record_then_replay_is_exact_and_calls_model_once(tmp_path: Path) -> None:
    inner = Counting()
    agent = CachedAgent(inner, Store(tmp_path / "c.jsonl"))
    msgs = [Message("system", "s"), Message("user", "u")]
    a = agent.act(msgs, [], seed=1)
    b = agent.act(msgs, [], seed=1)
    assert a == b and inner.calls == 1
    agent.act(msgs, [], seed=2)  # different seed => different key
    assert inner.calls == 2
    # a fresh process re-reads the file
    store2 = Store(tmp_path / "c.jsonl")
    assert len(store2) == 2
    replay = CachedAgent(Counting(), store2, mode="replay")
    assert replay.act(msgs, [], seed=1) == a
    assert replay.inner.calls == 0  # type: ignore[attr-defined]
    assert store2.hits == 1
    with pytest.raises(CacheMissError):
        replay.act(msgs, [], seed=99)
    rec = CachedAgent(inner, store2, mode="record")
    rec.act(msgs, [], seed=1)
    assert inner.calls == 3  # record mode always calls


def test_full_evaluation_replays_identically_without_the_model(tmp_path: Path) -> None:
    store = Store(tmp_path / "c.jsonl")
    live = CachedAgent(SkillAgent("a", 0.5), store)
    first = [run_attempt(live, t, r) for t in TASKS for r in range(3)]
    replay = CachedAgent(SkillAgent("a", 0.5), Store(tmp_path / "c.jsonl"), mode="replay")
    second = [run_attempt(replay, t, r) for t in TASKS for r in range(3)]
    assert [x.success for x in first] == [y.success for y in second]
    assert [x.total_tokens for x in first] == [y.total_tokens for y in second]


def test_store_tolerates_corrupt_lines(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text('{"key": "a", "value": {"x": 1}}\n{broken\n')
    s = Store(p)
    assert s.get("a") == {"x": 1} and s.get("b") is None and len(s) == 1
    assert ToolSpec("t", "d").name == "t"
