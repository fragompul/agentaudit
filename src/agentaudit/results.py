"""The ``ResultSet`` container: outcomes of several agents on shared tasks with repeats."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ResultSet:
    """Scores in ``[0, 1]`` (usually binary) with shape ``(models, tasks, runs)``.

    ``NaN`` marks a missing attempt. All statistics in AgentAudit consume either a
    ``ResultSet`` or one of its ``(tasks, runs)`` slices.
    """

    models: tuple[str, ...]
    tasks: tuple[str, ...]
    scores: FloatArray

    def __post_init__(self) -> None:
        if self.scores.ndim != 3:
            raise ValueError("scores must have shape (models, tasks, runs)")
        if self.scores.shape[:2] != (len(self.models), len(self.tasks)):
            raise ValueError("scores shape does not match models/tasks")
        if len(set(self.models)) != len(self.models):
            raise ValueError("duplicate model names")
        finite = self.scores[~np.isnan(self.scores)]
        if np.any(finite < 0) or np.any(finite > 1):
            raise ValueError("scores must lie in [0, 1]")

    @classmethod
    def from_array(
        cls,
        scores: ArrayLike,
        models: Sequence[str] | None = None,
        tasks: Sequence[str] | None = None,
    ) -> ResultSet:
        arr = np.asarray(scores, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[None]
        m, t, _ = arr.shape
        return cls(
            tuple(models) if models is not None else tuple(f"model{i}" for i in range(m)),
            tuple(tasks) if tasks is not None else tuple(f"task{i}" for i in range(t)),
            arr,
        )

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def n_runs(self) -> int:
        return int(self.scores.shape[2])

    def model(self, name: str) -> FloatArray:
        """The ``(tasks, runs)`` matrix of one model."""
        out: FloatArray = self.scores[self.models.index(name)]
        return out

    def select(self, models: Sequence[str]) -> ResultSet:
        idx = [self.models.index(m) for m in models]
        return ResultSet(tuple(models), self.tasks, self.scores[idx])

    def is_balanced(self) -> bool:
        return not bool(np.any(np.isnan(self.scores)))

    def task_means(self) -> FloatArray:
        """Per-model, per-task mean success: shape ``(models, tasks)``."""
        out: FloatArray = np.nanmean(self.scores, axis=2)
        return out

    def save(self, path: str | Path) -> None:
        payload = {
            "models": list(self.models),
            "tasks": list(self.tasks),
            "scores": np.where(np.isnan(self.scores), None, self.scores).tolist(),
        }
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> ResultSet:
        raw = json.loads(Path(path).read_text())
        scores = np.array(raw["scores"], dtype=object)
        arr = np.where(scores == None, np.nan, scores).astype(np.float64)  # noqa: E711
        return cls(tuple(raw["models"]), tuple(raw["tasks"]), arr)
