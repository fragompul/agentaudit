"""AgentAudit-Bench task suite: registry, procedural generation, (de)serialisation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from agentaudit.bench.audit import AuditReport, audit
from agentaudit.bench.base import Family, Task, stable_seed
from agentaudit.bench.envs.booking import BookingFamily
from agentaudit.bench.envs.crm import CrmFamily
from agentaudit.bench.envs.forms import FormsFamily
from agentaudit.bench.envs.fs import FileSystemFamily
from agentaudit.bench.envs.sql import SqlFamily
from agentaudit.bench.envs.tabular import TabularFamily

BENCH_VERSION = "agentaudit-bench-0.1"

FAMILIES: dict[str, Family] = {
    f.name: f
    for f in (
        FileSystemFamily(),
        SqlFamily(),
        CrmFamily(),
        BookingFamily(),
        FormsFamily(),
        TabularFamily(),
    )
}


def get_family(name: str) -> Family:
    try:
        return FAMILIES[name]
    except KeyError:
        raise KeyError(f"unknown family {name!r}; known: {', '.join(FAMILIES)}") from None


def difficulty_for(i: int) -> tuple[int, int]:
    """Knob schedule within a family: horizon 1-4 (period 20) x distractors 0-2 (period 60)."""
    return 1 + (i // 5) % 4, (i // 20) % 3


def generate_suite(n_per_family: int = 60, seed: int = 0) -> list[Task]:
    """Generate ``6 * n_per_family`` tasks; deterministic in ``seed``.

    Each family cycles through its templates (index mod 5) while horizon and distractor
    knobs sweep a full factorial grid, so difficulty is balanced across families.
    """
    tasks: list[Task] = []
    for name, fam in FAMILIES.items():
        for i in range(n_per_family):
            horizon, distractors = difficulty_for(i)
            rng = np.random.default_rng(stable_seed(BENCH_VERSION, seed, name, i))
            tasks.append(fam.generate(rng, i, horizon, distractors))
    return tasks


def audit_suite(tasks: list[Task]) -> dict[str, AuditReport]:
    """Run the checker self-audit per family."""
    return {
        name: audit(fam, [t for t in tasks if t.family == name]) for name, fam in FAMILIES.items()
    }


def save_suite(tasks: list[Task], path: str | Path) -> None:
    with Path(path).open("w") as fh:
        for t in tasks:
            fh.write(json.dumps(asdict(t), sort_keys=True) + "\n")


def load_suite(path: str | Path) -> list[Task]:
    with Path(path).open() as fh:
        return [Task(**json.loads(line)) for line in fh if line.strip()]
