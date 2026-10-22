"""Versioned trajectory schema (JSONL) and conversion to :class:`ResultSet`.

One JSON object per line, one line per agent attempt. The schema is validated with
plain dataclasses (no dependency); unknown versions are rejected explicitly.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agentaudit.results import ResultSet

SCHEMA_VERSION = 1


class SchemaError(ValueError):
    """Raised when a trajectory record does not conform to the schema."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str | None = None


@dataclass(frozen=True)
class Step:
    """One agent turn: optional text, tool calls, and the resulting observation."""

    index: int
    thought: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    observation: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


@dataclass(frozen=True)
class Trajectory:
    """A complete attempt of one agent on one task."""

    model: str
    task_id: str
    run: int
    seed: int
    success: float  # verifiable outcome in [0, 1] (binary for AgentAudit-Bench)
    steps: tuple[Step, ...] = ()
    final_answer: str = ""
    termination: str = "completed"  # completed | step_limit | error | context_limit
    version: str = ""  # agent / scaffold / benchmark version
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not 0.0 <= self.success <= 1.0:
            raise SchemaError(f"success must be in [0, 1], got {self.success}")
        if self.run < 0:
            raise SchemaError("run must be >= 0")
        if not self.model or not self.task_id:
            raise SchemaError("model and task_id must be non-empty")

    @property
    def total_tokens(self) -> int:
        return sum(s.prompt_tokens + s.completion_tokens for s in self.steps)

    @property
    def latency_s(self) -> float:
        return sum(s.latency_s for s in self.steps)

    @property
    def n_tool_calls(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Trajectory:
        ver = raw.get("schema_version", SCHEMA_VERSION)
        if ver != SCHEMA_VERSION:
            raise SchemaError(f"unsupported schema_version {ver} (expected {SCHEMA_VERSION})")
        for key in ("model", "task_id", "run", "seed", "success"):
            if key not in raw:
                raise SchemaError(f"missing required field {key!r}")
        steps = tuple(
            Step(
                index=int(s["index"]),
                thought=s.get("thought", ""),
                tool_calls=tuple(ToolCall(**tc) for tc in s.get("tool_calls", ())),
                observation=s.get("observation", ""),
                prompt_tokens=int(s.get("prompt_tokens", 0)),
                completion_tokens=int(s.get("completion_tokens", 0)),
                latency_s=float(s.get("latency_s", 0.0)),
            )
            for s in raw.get("steps", ())
        )
        return cls(
            model=str(raw["model"]),
            task_id=str(raw["task_id"]),
            run=int(raw["run"]),
            seed=int(raw["seed"]),
            success=float(raw["success"]),
            steps=steps,
            final_answer=raw.get("final_answer", ""),
            termination=raw.get("termination", "completed"),
            version=raw.get("version", ""),
            metadata=dict(raw.get("metadata", {})),
            schema_version=int(ver),
        )


def write_jsonl(trajectories: Iterable[Trajectory], path: str | Path) -> int:
    n = 0
    with Path(path).open("w") as fh:
        for tr in trajectories:
            fh.write(tr.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Trajectory]:
    with Path(path).open() as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield Trajectory.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                raise SchemaError(f"{path}:{lineno}: {exc}") from exc
            except SchemaError as exc:
                raise SchemaError(f"{path}:{lineno}: {exc}") from exc


def to_resultset(
    trajectories: Iterable[Trajectory], *, drop_incomplete_tasks: bool = False
) -> ResultSet:
    """Pivot trajectories into a ``(models, tasks, runs)`` :class:`ResultSet`.

    Missing ``(model, task, run)`` cells become NaN. With ``drop_incomplete_tasks`` the result
    keeps only tasks that every model attempted at least once.
    """
    cells: dict[tuple[str, str, int], float] = {}
    models: dict[str, None] = {}
    tasks: dict[str, None] = {}
    seen: dict[str, set[str]] = defaultdict(set)
    for tr in trajectories:
        key = (tr.model, tr.task_id, tr.run)
        if key in cells:
            raise SchemaError(f"duplicate attempt {key}")
        cells[key] = tr.success
        models[tr.model] = None
        tasks[tr.task_id] = None
        seen[tr.model].add(tr.task_id)
    if not cells:
        raise SchemaError("no trajectories")
    task_list = list(tasks)
    if drop_incomplete_tasks:
        common = set.intersection(*seen.values())
        task_list = [t for t in task_list if t in common]
        if not task_list:
            raise SchemaError("no task is shared by all models")
    model_list = list(models)
    n_runs = max(run for _, _, run in cells) + 1
    arr = np.full((len(model_list), len(task_list), n_runs), np.nan)
    mi = {m: i for i, m in enumerate(model_list)}
    ti = {t: i for i, t in enumerate(task_list)}
    for (model, task, run), value in cells.items():
        if task in ti:
            arr[mi[model], ti[task], run] = value
    return ResultSet(tuple(model_list), tuple(task_list), arr)


def from_resultset(results: ResultSet, *, version: str = "") -> Iterator[Trajectory]:
    """Minimal trajectories (outcome only) for every non-missing cell of a ``ResultSet``."""
    for mi, model in enumerate(results.models):
        for ti, task in enumerate(results.tasks):
            for run in range(results.n_runs):
                val = results.scores[mi, ti, run]
                if not np.isnan(val):
                    yield Trajectory(model, task, run, 0, float(val), version=version)
