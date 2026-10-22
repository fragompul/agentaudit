"""Adapters that turn public benchmark result files into :class:`ResultSet` objects.

Adapters only read local files (download scripts live in ``scripts/``). Supported:

* tau-bench ``historical_trajectories/*.json`` (several trials per task, reward in {0, 1}),
* SWE-bench ``experiments`` ``results/results.json`` (one run, list of resolved instance ids).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from agentaudit.results import ResultSet


def taubench_scores(path: str | Path) -> dict[str, dict[int, float]]:
    """``{task_id: {trial: reward}}`` from one tau-bench trajectories file."""
    rows = json.loads(Path(path).read_text())
    out: dict[str, dict[int, float]] = {}
    for r in rows:
        tid, trial = str(r["task_id"]), int(r["trial"])
        if trial in out.setdefault(tid, {}):
            raise ValueError(f"duplicate trial {trial} for task {tid} in {path}")
        out[tid][trial] = float(r["reward"])
    return out


def taubench_resultset(files: Mapping[str, str | Path]) -> ResultSet:
    """Pivot several agents' tau-bench files (same domain) into a ResultSet.

    Only tasks present for every agent are kept; the number of runs is the smallest trial count.
    Rewards are binarised at 1.0 (tau-bench rewards are 0 or 1).
    """
    per_model = {m: taubench_scores(p) for m, p in files.items()}
    common = set.intersection(*(set(v) for v in per_model.values()))
    if not common:
        raise ValueError("agents share no tasks")
    tasks = sorted(common, key=lambda s: (len(s), s))
    n_runs = min(len(v[t]) for v in per_model.values() for t in tasks)
    arr = np.zeros((len(per_model), len(tasks), n_runs))
    for i, v in enumerate(per_model.values()):
        for j, t in enumerate(tasks):
            vals = [v[t][k] for k in sorted(v[t])][:n_runs]
            arr[i, j] = np.asarray(vals) >= 1.0
    return ResultSet(tuple(per_model), tuple(tasks), arr)


def swebench_resultset(resolved: Mapping[str, set[str]], instances: list[str]) -> ResultSet:
    """Single-run ResultSet from ``{submission: set(resolved instance ids)}``."""
    names = sorted(resolved)
    arr = np.array([[float(t in resolved[n]) for t in instances] for n in names])[:, :, None]
    return ResultSet(tuple(names), tuple(instances), arr)
