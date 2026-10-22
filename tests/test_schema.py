import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from agentaudit.schema import (
    SchemaError,
    Step,
    ToolCall,
    Trajectory,
    read_jsonl,
    to_resultset,
    write_jsonl,
)

steps = st.lists(
    st.builds(
        Step,
        index=st.integers(0, 50),
        thought=st.text(max_size=20),
        tool_calls=st.lists(
            st.builds(ToolCall, name=st.text(min_size=1, max_size=8), arguments=st.just({"a": 1})),
            max_size=3,
        ).map(tuple),
        observation=st.text(max_size=20),
        prompt_tokens=st.integers(0, 5000),
        completion_tokens=st.integers(0, 500),
        latency_s=st.floats(0, 60),
    ),
    max_size=4,
).map(tuple)

trajs = st.builds(
    Trajectory,
    model=st.text(min_size=1, max_size=6),
    task_id=st.text(min_size=1, max_size=6),
    run=st.integers(0, 9),
    seed=st.integers(0, 2**31),
    success=st.sampled_from([0.0, 1.0]),
    steps=steps,
)


@given(st.lists(trajs, max_size=5))
def test_jsonl_roundtrip(items: list[Trajectory]) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.jsonl"
        assert write_jsonl(items, p) == len(items)
        assert list(read_jsonl(p)) == items


def test_derived_fields() -> None:
    t = Trajectory(
        "m",
        "t",
        0,
        1,
        1.0,
        steps=(
            Step(
                0,
                tool_calls=(ToolCall("x"), ToolCall("y")),
                prompt_tokens=10,
                completion_tokens=5,
                latency_s=1.5,
            ),
            Step(1, prompt_tokens=20, completion_tokens=5, latency_s=0.5),
        ),
    )
    assert (t.total_tokens, t.n_tool_calls, t.latency_s) == (40, 2, 2.0)


def test_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(SchemaError):
        Trajectory("m", "t", 0, 0, 1.5)
    with pytest.raises(SchemaError):
        Trajectory("", "t", 0, 0, 1.0)
    with pytest.raises(SchemaError):
        Trajectory("m", "t", -1, 0, 1.0)
    with pytest.raises(SchemaError, match="schema_version"):
        Trajectory.from_dict({"schema_version": 99})
    with pytest.raises(SchemaError, match="missing"):
        Trajectory.from_dict({"model": "m"})
    p = tmp_path / "bad.jsonl"
    p.write_text('{"model": "m", "task_id": "t", "run": 0, "seed": 0, "success": 1}\n\n{oops\n')
    with pytest.raises(SchemaError, match=":3:"):
        list(read_jsonl(p))
    p.write_text(json.dumps({"model": "m", "task_id": "t", "run": 0, "seed": 0, "success": 9}))
    with pytest.raises(SchemaError, match=":1:"):
        list(read_jsonl(p))


def test_to_resultset_pivot_and_missing() -> None:
    rows = [
        Trajectory("a", "t1", 0, 0, 1.0),
        Trajectory("a", "t1", 1, 0, 0.0),
        Trajectory("a", "t2", 0, 0, 1.0),
        Trajectory("b", "t1", 0, 0, 0.0),
    ]
    rs = to_resultset(rows)
    assert rs.models == ("a", "b") and rs.tasks == ("t1", "t2") and rs.n_runs == 2
    assert np.isnan(rs.scores[1, 1]).all() and np.isnan(rs.scores[0, 1, 1])
    common = to_resultset(rows, drop_incomplete_tasks=True)
    assert common.tasks == ("t1",)
    with pytest.raises(SchemaError, match="duplicate"):
        to_resultset(rows + [rows[0]])
    with pytest.raises(SchemaError, match="no trajectories"):
        to_resultset([])
    with pytest.raises(SchemaError, match="shared"):
        to_resultset(
            [Trajectory("a", "x", 0, 0, 1.0), Trajectory("b", "y", 0, 0, 1.0)],
            drop_incomplete_tasks=True,
        )


def test_from_resultset_roundtrip() -> None:
    from agentaudit.schema import from_resultset

    arr = np.array([[[1, 0], [np.nan, 1]], [[0, 0], [1, 1]]])
    from agentaudit.results import ResultSet

    rs = ResultSet.from_array(arr, ["a", "b"], ["t1", "t2"])
    back = to_resultset(from_resultset(rs))
    np.testing.assert_array_equal(back.scores, rs.scores)
