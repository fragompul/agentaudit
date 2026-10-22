"""Run agents on benchmark tasks: single attempts and resumable, balanced evaluations."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from agentaudit.agents.base import SYSTEM_PROMPT, Agent, Message
from agentaudit.bench.base import FaultInjector, Task, stable_seed
from agentaudit.bench.suite import BENCH_VERSION, get_family
from agentaudit.schema import Step, ToolCall, Trajectory

MAX_OBS_CHARS = 1500


@dataclass(frozen=True)
class RunConfig:
    max_steps: int = 12
    fault_rate: float = 0.0
    base_seed: int = 0


def attempt_seed(cfg: RunConfig, model: str, task_id: str, run: int) -> int:
    return stable_seed("attempt", cfg.base_seed, model, task_id, run)


def fault_seed(cfg: RunConfig, task_id: str, run: int) -> int:
    """Independent of the agent, so every model sees the same failures for the same (task, run)."""
    return stable_seed("fault", cfg.base_seed, task_id, run)


def run_attempt(agent: Agent, task: Task, run: int, cfg: RunConfig | None = None) -> Trajectory:
    """Execute one attempt and return its verified trajectory."""
    cfg = cfg or RunConfig()
    fam = get_family(task.family)
    inner = fam.build(task)
    host = FaultInjector(inner, cfg.fault_rate, fault_seed(cfg, task.task_id, run))
    seed = attempt_seed(cfg, agent.name, task.task_id, run)
    agent = agent.session(task, fam, seed)
    messages = [Message("system", SYSTEM_PROMPT), Message("user", task.instruction)]
    steps: list[Step] = []
    termination = "step_limit"
    final = ""
    for i in range(cfg.max_steps):
        t0 = time.perf_counter()
        try:
            action = agent.act(messages, host.tools, seed=seed + i)
        except Exception as exc:  # backend failure: record, do not crash the whole evaluation
            termination = f"error: {type(exc).__name__}"
            final = str(exc)[:200]
            break
        latency = action.latency_s or (time.perf_counter() - t0)
        calls: list[ToolCall] = []
        obs_parts: list[str] = []
        messages.append(Message("assistant", action.content, action.tool_calls))
        for inv in action.tool_calls:
            obs = host.call(inv.name, inv.arguments)
            calls.append(
                ToolCall(
                    inv.name,
                    inv.arguments,
                    ok=not obs.startswith("ERROR"),
                    error=obs[:200] if obs.startswith("ERROR") else None,
                )
            )
            obs_parts.append(obs[:MAX_OBS_CHARS])
            messages.append(Message("tool", obs[:MAX_OBS_CHARS], tool_name=inv.name))
        steps.append(
            Step(
                i,
                action.content,
                tuple(calls),
                "\n".join(obs_parts),
                action.prompt_tokens,
                action.completion_tokens,
                latency,
            )
        )
        if not action.tool_calls:
            termination, final = "completed", action.content
            break
        if inner.finished or (task.requires_answer and inner.answer is not None):
            termination = "completed"
            break
    success = 1.0 if fam.check(task, inner) else 0.0
    return Trajectory(
        model=agent.name,
        task_id=task.task_id,
        run=run,
        seed=seed,
        success=success,
        steps=tuple(steps),
        final_answer=inner.answer or final,
        termination=termination,
        version=BENCH_VERSION,
        metadata={
            "family": task.family,
            "horizon": task.horizon,
            "distractors": task.distractors,
            "fault_rate": cfg.fault_rate,
            "faults_injected": len(host.log),
            "max_steps": cfg.max_steps,
        },
    )


def _done_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open() as fh:
        for line in fh:
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated last line after a crash; the attempt is simply redone
                keys.add((r["model"], r["task_id"], int(r["run"])))
    return keys


def pending_attempts(
    agent_name: str, tasks: Sequence[Task], n_runs: int, out_path: Path
) -> Iterator[tuple[Task, int]]:
    """Run-major order (run 0 over all tasks, then run 1, ...): any prefix is balanced."""
    done = _done_keys(out_path)
    for run in range(n_runs):
        for task in tasks:
            if (agent_name, task.task_id, run) not in done:
                yield task, run


def run_evaluation(
    agent: Agent,
    tasks: Sequence[Task],
    n_runs: int,
    out_path: str | Path,
    cfg: RunConfig | None = None,
    *,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Evaluate ``agent`` and append trajectories to ``out_path`` (JSONL); resumable.

    Returns the number of attempts executed in this call. Re-running skips attempts already
    present, so interrupted evaluations continue where they stopped.
    """
    cfg = cfg or RunConfig()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    todo = list(pending_attempts(agent.name, tasks, n_runs, path))
    total, finished = len(todo), 0
    if path.exists() and path.stat().st_size:
        with path.open("rb+") as raw:  # a crash can leave a partial last line: end it cleanly
            raw.seek(-1, 2)
            if raw.read(1) != b"\n":
                raw.write(b"\n")
    with path.open("a") as fh:
        if workers <= 1:
            for task, run in todo:
                fh.write(run_attempt(agent, task, run, cfg).to_json() + "\n")
                fh.flush()
                finished += 1
                if progress:
                    progress(finished, total)
        else:
            with ThreadPoolExecutor(workers) as ex:
                for traj in ex.map(lambda tr: run_attempt(agent, tr[0], tr[1], cfg), todo):
                    fh.write(traj.to_json() + "\n")
                    fh.flush()
                    finished += 1
                    if progress:
                        progress(finished, total)
    return finished
