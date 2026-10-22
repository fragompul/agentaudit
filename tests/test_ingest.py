import json
from pathlib import Path

import pytest

from agentaudit.ingest import swebench_resultset, taubench_resultset, taubench_scores


def write(path: Path, rewards: dict[str, list[float]]) -> Path:
    rows = [
        {"task_id": t, "reward": r, "info": {}, "traj": [], "trial": k}
        for t, rs in rewards.items()
        for k, r in enumerate(rs)
    ]
    path.write_text(json.dumps(rows))
    return path


def test_taubench_pivot_common_tasks_and_min_runs(tmp_path: Path) -> None:
    a = write(
        tmp_path / "a.json", {"0": [1.0, 0.0, 1.0], "1": [0.0, 0.0, 0.0], "2": [1.0, 1.0, 1.0]}
    )
    b = write(tmp_path / "b.json", {"0": [1.0, 1.0], "1": [1.0, 0.0], "10": [0.0, 0.0]})
    rs = taubench_resultset({"A": a, "B": b})
    assert rs.models == ("A", "B") and rs.tasks == ("0", "1") and rs.n_runs == 2
    assert rs.scores[0].tolist() == [[1, 0], [0, 0]] and rs.scores[1].tolist() == [[1, 1], [1, 0]]
    assert taubench_scores(a)["2"] == {0: 1.0, 1: 1.0, 2: 1.0}


def test_taubench_errors(tmp_path: Path) -> None:
    dup = tmp_path / "d.json"
    dup.write_text(json.dumps([{"task_id": 1, "reward": 1, "trial": 0}] * 2))
    with pytest.raises(ValueError, match="duplicate"):
        taubench_scores(dup)
    x = write(tmp_path / "x.json", {"0": [1.0]})
    y = write(tmp_path / "y.json", {"1": [1.0]})
    with pytest.raises(ValueError, match="no tasks"):
        taubench_resultset({"x": x, "y": y})


def test_swebench_resultset() -> None:
    rs = swebench_resultset({"b": {"t1"}, "a": {"t1", "t2"}}, ["t1", "t2", "t3"])
    assert rs.models == ("a", "b") and rs.n_runs == 1
    assert rs.scores[:, :, 0].tolist() == [[1, 1, 0], [1, 0, 0]]
