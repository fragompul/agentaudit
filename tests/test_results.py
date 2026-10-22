from pathlib import Path

import numpy as np
import pytest

from agentaudit.results import ResultSet


def test_roundtrip_with_nan(tmp_path: Path) -> None:
    arr = np.array([[[1, 0, np.nan], [1, 1, 0]], [[0, 0, 1], [np.nan, 1, 1]]], dtype=float)
    rs = ResultSet.from_array(arr, ["a", "b"], ["t1", "t2"])
    rs.save(tmp_path / "r.json")
    back = ResultSet.load(tmp_path / "r.json")
    assert back.models == rs.models and back.tasks == rs.tasks
    np.testing.assert_array_equal(back.scores, rs.scores)
    assert not rs.is_balanced()
    assert (rs.n_models, rs.n_tasks, rs.n_runs) == (2, 2, 3)


def test_validation_and_helpers() -> None:
    with pytest.raises(ValueError):
        ResultSet(("a",), ("t",), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        ResultSet(("a", "b"), ("t",), np.zeros((1, 1, 2)))
    with pytest.raises(ValueError):
        ResultSet(("a", "a"), ("t",), np.zeros((2, 1, 2)))
    with pytest.raises(ValueError):
        ResultSet.from_array(np.full((1, 1, 2), 2.0))
    rs = ResultSet.from_array(np.array([[1, 0], [1, 1]]))  # 2-D -> single model
    assert rs.models == ("model0",) and rs.n_runs == 2
    assert rs.model("model0").shape == (2, 2)
    two = ResultSet.from_array(np.zeros((2, 3, 4)), ["x", "y"])
    assert two.select(["y"]).models == ("y",)
    assert two.task_means().shape == (2, 3)
