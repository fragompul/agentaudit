import numpy as np
import pytest

from agentaudit.bench.audit import audit
from agentaudit.bench.base import FaultInjector
from agentaudit.bench.envs.fs import FileSystemEnv, FileSystemFamily


def gen(n: int = 40, seed: int = 0, horizon: int = 3, distractors: int = 2):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    fam = FileSystemFamily()
    return fam, [fam.generate(rng, i, 1 + (i % horizon), distractors) for i in range(n)]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_oracle_solves_and_null_fails(seed: int) -> None:
    fam, tasks = gen(seed=seed)
    rep = audit(fam, tasks)
    assert rep.ok, rep


def test_generation_is_deterministic() -> None:
    _, a = gen(seed=5)
    _, b = gen(seed=5)
    assert a == b
    assert a != gen(seed=6)[1]


def test_wrong_actions_fail_checker() -> None:
    fam, tasks = gen(n=5)
    move = next(t for t in tasks if t.spec["template"] == "move")
    env = fam.build(move)
    for path in list(move.spec["initial"]):
        env.call("delete", {"path": path})  # destroys data instead of moving
    assert not fam.check(move, env)


def test_tool_errors_are_observations() -> None:
    env = FileSystemEnv({"/a/b.txt": "x"})
    assert env.call("read_file", {"path": "/nope"}).startswith("ERROR")
    assert env.call("read_file", {}).startswith("ERROR: missing")
    assert env.call("read_file", {"path": "/a/b.txt", "zzz": 1}).startswith("ERROR: unexpected")
    assert env.call("fly", {}).startswith("ERROR: unknown tool")
    assert env.call("read_file", {"path": "relative"}).startswith("ERROR")
    assert env.call("ls", {"path": "/a"}) == "b.txt"
    assert env.call("ls", {"path": "/"}) == "a/"
    assert env.call("ls", {"path": "/zzz"}).startswith("ERROR")
    assert env.call("move", {"src": "/a/b.txt", "dst": "/c/d.txt"}).startswith("Moved")
    assert env.call("move", {"src": "/a/b.txt", "dst": "/c/e.txt"}).startswith("ERROR")
    assert env.call("append_file", {"path": "/c/d.txt", "content": "y"}).startswith("Appended")
    assert env.call("append_file", {"path": "/nope", "content": "y"}).startswith("ERROR")
    assert env.call("delete", {"path": "/c/d.txt"}).startswith("Deleted")
    assert env.call("delete", {"path": "/c/d.txt"}).startswith("ERROR")
    assert env.call("finish", {}) and env.finished
    assert len(env.tools[0].json_schema()["function"]["parameters"]["required"]) == 1


def test_fault_injection_is_seeded_and_does_not_mutate() -> None:
    def run(seed: int) -> tuple[list[int], list[str]]:
        inj = FaultInjector(FileSystemEnv({"/a.txt": "x"}), 0.5, seed)
        outs = [inj.call("read_file", {"path": "/a.txt"}) for _ in range(40)]
        return inj.log, outs

    log1, out1 = run(3)
    log2, out2 = run(3)
    assert log1 == log2 and out1 == out2
    assert 8 < len(log1) < 32
    assert all(out1[i] == FaultInjector.TRANSIENT for i in log1)
    assert run(4)[0] != log1
    inj = FaultInjector(FileSystemEnv({"/a.txt": "x"}), 0.999, 0)
    before = inj.inner.state_hash()
    for _ in range(5):
        inj.call("delete", {"path": "/a.txt"})
    assert inj.inner.state_hash() == before or "/a.txt" not in inj.inner.state()["files"]
    with pytest.raises(ValueError):
        FaultInjector(FileSystemEnv({}), 1.0, 0)
    assert FaultInjector(FileSystemEnv({}), 0.0, 0).call("fly", {}).startswith("ERROR: unknown")
