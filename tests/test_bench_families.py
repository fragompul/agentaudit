"""Generic audit applied to every registered family (oracle solves, null fails, deterministic)."""

import numpy as np
import pytest

from agentaudit.bench.audit import audit
from agentaudit.bench.base import Family
from agentaudit.bench.envs.crm import CrmEnv, CrmFamily
from agentaudit.bench.envs.fs import FileSystemFamily
from agentaudit.bench.envs.sql import SqlFamily

FAMILIES: list[Family] = [FileSystemFamily(), SqlFamily(), CrmFamily()]


@pytest.mark.parametrize("fam", FAMILIES, ids=lambda f: f.name)
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_family_audit(fam: Family, seed: int) -> None:
    rng = np.random.default_rng(seed)
    tasks = [fam.generate(rng, i, 1 + i % 4, i % 4) for i in range(40)]
    rep = audit(fam, tasks)
    assert rep.ok, rep
    assert len({t.task_id for t in tasks}) == len(tasks)
    assert all(t.instruction and t.horizon >= 1 for t in tasks)


@pytest.mark.parametrize("fam", FAMILIES, ids=lambda f: f.name)
def test_generation_deterministic(fam: Family) -> None:
    a = [fam.generate(np.random.default_rng(9), i, 2, 1) for i in range(10)]
    b = [fam.generate(np.random.default_rng(9), i, 2, 1) for i in range(10)]
    assert a == b


def test_crm_semantics() -> None:
    env = CrmEnv(
        [
            {"id": 1, "name": "Ana Silva", "email": "a@x", "tier": "free", "city": "Rome"},
            {"id": 2, "name": "Bea Silva", "email": "b@x", "tier": "pro", "city": "Oslo"},
        ],
        [],
    )
    assert env.call("search_customers", {"name": "silva"}).count("\n") == 1
    assert env.call("search_customers", {"name": "zzz"}) == "(no matches)"
    assert env.call("get_customer", {"customer_id": 9}).startswith("ERROR")
    assert env.call("update_customer", {"customer_id": 1, "field": "x", "value": "y"}).startswith(
        "ERROR"
    )
    assert env.call(
        "update_customer", {"customer_id": 1, "field": "tier", "value": "gold"}
    ).startswith("ERROR")
    assert env.call(
        "update_customer", {"customer_id": 1, "field": "tier", "value": "pro"}
    ).startswith("Updated")
    assert env.call("create_ticket", {"customer_id": 1, "subject": "s"}) == "Created ticket 101"
    assert "open" in env.call("list_tickets", {"customer_id": 1})
    assert env.call("list_tickets", {"customer_id": 2}) == "(none)"
    assert env.call("close_ticket", {"ticket_id": 101}).startswith("Closed")
    assert env.call("close_ticket", {"ticket_id": 7}).startswith("ERROR")
