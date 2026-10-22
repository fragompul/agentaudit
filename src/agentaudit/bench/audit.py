"""Self-audit of checkers: each task must be solved by its oracle and not by doing nothing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from agentaudit.bench.base import Family, Task


@dataclass(frozen=True)
class AuditReport:
    n_tasks: int
    oracle_failures: tuple[str, ...]
    null_passes: tuple[str, ...]
    nondeterministic: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (self.oracle_failures or self.null_passes or self.nondeterministic)


def audit(family: Family, tasks: Sequence[Task]) -> AuditReport:
    """Check oracle success, null-agent failure and state-hash determinism for each task."""
    oracle_fail: list[str] = []
    null_pass: list[str] = []
    nondet: list[str] = []
    for task in tasks:
        env = family.build(task)
        for call in family.oracle(task):
            env.call(call.name, call.arguments)
        if not family.check(task, env):
            oracle_fail.append(task.task_id)
        env2 = family.build(task)
        for call in family.oracle(task):
            env2.call(call.name, call.arguments)
        if env.state_hash() != env2.state_hash():
            nondet.append(task.task_id)
        if family.check(task, family.build(task)):
            null_pass.append(task.task_id)
    return AuditReport(len(tasks), tuple(oracle_fail), tuple(null_pass), tuple(nondet))
