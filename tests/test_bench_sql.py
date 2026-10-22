import numpy as np
import pytest

from agentaudit.bench.audit import audit
from agentaudit.bench.envs.sql import SqlEnv, SqlFamily


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_oracle_solves_and_null_fails(seed: int) -> None:
    rng = np.random.default_rng(seed)
    fam = SqlFamily()
    tasks = [fam.generate(rng, i, 1 + i % 3, i % 3) for i in range(30)]
    assert audit(fam, tasks).ok


def test_tool_semantics_and_guards() -> None:
    env = SqlEnv({"employees": [[1, "A B", "ops", 30000, "Rome"]]}, decoys=2)
    assert "employees_archive_1" in env.call("list_tables", {})
    assert "salary" in env.call("describe", {"table": "employees"})
    assert env.call("describe", {"table": "zzz"}).startswith("ERROR")
    assert env.call("query", {"sql": "DELETE FROM employees"}).startswith("ERROR")
    assert env.call("execute", {"sql": "SELECT 1"}).startswith("ERROR")
    assert env.call("query", {"sql": "SELECT * FROM nope"}).startswith("ERROR: SQL error")
    assert env.call("query", {"sql": "SELECT name FROM employees;"}) == "A B"
    assert env.call("query", {"sql": "SELECT 1 WHERE 0"}) == "(no rows)"
    assert env.call("execute", {"sql": "UPDATE employees SET salary=1"}).startswith("OK, 1")
    assert env.call("submit_answer", {"answer": " 7 "}) and env.answer == "7"


def test_wrong_state_and_wrong_answer_fail() -> None:
    fam = SqlFamily()
    rng = np.random.default_rng(1)
    t_raise = fam.generate(rng, 2, 1, 0)
    env = fam.build(t_raise)
    env.call("execute", {"sql": "UPDATE employees SET salary=salary+1"})
    assert not fam.check(t_raise, env)
    t_count = fam.generate(rng, 0, 1, 0)
    env = fam.build(t_count)
    env.call("submit_answer", {"answer": str(int(t_count.spec["answer"]) + 1)})
    assert not fam.check(t_count, env)
