from pathlib import Path

import pytest

from agentaudit.bench.suite import (
    FAMILIES,
    audit_suite,
    difficulty_for,
    generate_suite,
    get_family,
    load_suite,
    save_suite,
)


def test_suite_size_balance_and_determinism() -> None:
    a = generate_suite(60, seed=0)
    assert len(a) == 360 and len({t.task_id for t in a}) == 360
    assert a == generate_suite(60, seed=0)
    assert a != generate_suite(60, seed=1)
    for fam in FAMILIES:
        sub = [t for t in a if t.family == fam]
        assert len(sub) == 60
        assert {t.distractors for t in sub} == {0, 1, 2}


def test_difficulty_schedule_is_full_factorial() -> None:
    grid = {difficulty_for(i) for i in range(60)}
    assert grid == {(h, d) for h in (1, 2, 3, 4) for d in (0, 1, 2)}


def test_full_suite_passes_checker_audit() -> None:
    reports = audit_suite(generate_suite(60, seed=0))
    bad = {k: r for k, r in reports.items() if not r.ok}
    assert not bad, bad
    assert all(r.n_tasks == 60 for r in reports.values())


def test_roundtrip_and_lookup(tmp_path: Path) -> None:
    tasks = generate_suite(5, seed=3)
    save_suite(tasks, tmp_path / "s.jsonl")
    loaded = load_suite(tmp_path / "s.jsonl")
    assert [t.task_id for t in loaded] == [t.task_id for t in tasks]
    assert loaded[0].instruction == tasks[0].instruction
    assert get_family("sql").name == "sql"
    with pytest.raises(KeyError, match="unknown family"):
        get_family("nope")
