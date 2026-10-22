"""Deterministic failure taxonomy and failing-step localisation for trajectories."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from agentaudit.schema import Trajectory

CATEGORIES = (
    "backend_error",
    "no_action",
    "malformed_call",
    "gave_up_after_transient_error",
    "loop_or_step_limit",
    "premature_stop",
    "wrong_answer",
    "wrong_final_state",
)
TRANSIENT = "transient failure"
MALFORMED = ("missing required", "unexpected argument", "unknown tool", "_unparsed")


@dataclass(frozen=True)
class FailureReport:
    category: str
    step: int | None  # index of the step where the failure is first visible
    detail: str = ""


def _signature(step_calls: Iterable[tuple[str, dict[str, object]]]) -> str:
    return json.dumps(list(step_calls), sort_keys=True, default=str)


def classify(traj: Trajectory, requires_answer: bool = False) -> FailureReport | None:
    """Classify a failed trajectory; ``None`` for successes.

    Rules are applied in priority order and depend only on the recorded trajectory.
    """
    if traj.success >= 1.0:
        return None
    if traj.termination.startswith("error"):
        return FailureReport("backend_error", len(traj.steps) - 1 or None, traj.termination)
    calls = [(s.index, c) for s in traj.steps for c in s.tool_calls]
    if not calls:
        return FailureReport("no_action", 0, "the model never called a tool")
    for idx, c in calls:
        text = (c.error or "").lower()
        if not c.ok and (any(k in text for k in MALFORMED) or "_unparsed" in str(c.arguments)):
            return FailureReport("malformed_call", idx, c.error or "")
    last_step = traj.steps[-1]
    if last_step.tool_calls and all(TRANSIENT in (c.error or "") for c in last_step.tool_calls):
        return FailureReport("gave_up_after_transient_error", last_step.index)
    sigs = [_signature([(c.name, c.arguments)]) for _, c in calls]
    for i in range(2, len(sigs)):
        if sigs[i] == sigs[i - 1] == sigs[i - 2]:
            return FailureReport("loop_or_step_limit", calls[i][0], "same call 3x in a row")
    if traj.termination == "step_limit":
        return FailureReport("loop_or_step_limit", last_step.index, "step budget exhausted")
    mutating = [c for _, c in calls if c.name not in {"finish", "submit_answer"}]
    if requires_answer:
        if any(c.name == "submit_answer" for _, c in calls):
            return FailureReport("wrong_answer", last_step.index)
        return FailureReport("premature_stop", last_step.index, "no answer was submitted")
    if len(mutating) <= 1 and all(c.name in {"finish"} for _, c in calls[-1:]):
        return FailureReport("premature_stop", last_step.index, "stopped before acting enough")
    return FailureReport("wrong_final_state", last_step.index)


def failure_table(
    trajs: Iterable[Trajectory], answer_tasks: set[str] | None = None
) -> dict[str, Counter[str]]:
    """Per-model counts of failure categories (successes excluded)."""
    answer_tasks = answer_tasks or set()
    out: dict[str, Counter[str]] = {}
    for t in trajs:
        rep = classify(t, t.task_id in answer_tasks)
        if rep is not None:
            out.setdefault(t.model, Counter())[rep.category] += 1
    return out
