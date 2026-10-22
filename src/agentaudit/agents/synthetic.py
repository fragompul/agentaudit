"""Synthetic agents of controllable skill that run inside the real benchmark pipeline.

They follow the task's oracle plan but make errors at a rate set by a latent skill and by the
task's difficulty (horizon, distractors, a seeded task effect). They need no model, so they
validate the whole stack (runner, checkers, fault injection, statistics) end to end and give
ground truth for the statistical studies on the *actual* benchmark.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from agentaudit.agents.base import Agent, AgentAction, Message
from agentaudit.bench.base import Family, Task, ToolInvocation, ToolSpec, stable_seed


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class OracleAgent(Agent):
    """Executes the reference solution exactly (an upper bound that audits the checkers)."""

    def __init__(self, name: str = "oracle") -> None:
        self.name = name
        self._plan: list[ToolInvocation] = []

    def begin(self, task: Task, family: Family, seed: int) -> None:
        self._plan = list(family.oracle(task))

    def act(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, seed: int
    ) -> AgentAction:
        n_prev = sum(1 for m in messages if m.role == "assistant")
        if n_prev >= len(self._plan):
            return AgentAction("done")
        return AgentAction("", (self._plan[n_prev],))


class NullAgent(Agent):
    """Does nothing; the checkers must reject it."""

    def __init__(self, name: str = "null") -> None:
        self.name = name

    def act(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, seed: int
    ) -> AgentAction:
        return AgentAction("I have nothing to do.")


class SkillAgent(Agent):
    """Oracle-following agent with task-dependent per-attempt success probability.

    Per-attempt success probability: ``p = sigmoid(theta - 0.45 (horizon - 1) - 0.35 distractors
    - b_task + g_agent_task)`` where ``b_task ~ N(0, sigma_task)`` is shared by all agents and
    ``g ~ N(0, sigma_interaction)`` is agent-specific (both seeded by hashing, so they are fixed
    properties of the benchmark). Each attempt succeeds by following the oracle; otherwise the agent
    derails at a random step by issuing a harmful or premature action. After a transient tool
    error it retries with probability ``retry_prob`` and otherwise gives up.
    """

    def __init__(
        self,
        name: str,
        theta: float,
        *,
        sigma_task: float = 1.0,
        sigma_interaction: float = 0.5,
        retry_prob: float = 0.9,
    ) -> None:
        self.name = name
        self.theta, self.sigma_task = theta, sigma_task
        self.sigma_interaction, self.retry_prob = sigma_interaction, retry_prob
        self._plan: list[ToolInvocation] = []
        self._fail_at: int | None = None
        self._giveup = False
        self._rng = np.random.default_rng(0)
        self._pending_retry: ToolInvocation | None = None

    def success_probability(self, task: Task) -> float:
        b = np.random.default_rng(stable_seed("task-effect", task.task_id)).normal(
            0, self.sigma_task
        )
        g = np.random.default_rng(stable_seed("interaction", self.name, task.task_id)).normal(
            0, self.sigma_interaction
        )
        return _sigmoid(self.theta - 0.45 * (task.horizon - 1) - 0.35 * task.distractors - b + g)

    def begin(self, task: Task, family: Family, seed: int) -> None:
        self._plan = list(family.oracle(task))
        self._rng = np.random.default_rng(seed)
        self._giveup, self._pending_retry = False, None
        ok = self._rng.random() < self.success_probability(task)
        self._fail_at = None if ok else int(self._rng.integers(0, max(1, len(self._plan) - 1)))
        self._idx = 0

    def act(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, seed: int
    ) -> AgentAction:
        last = messages[-1]
        if (
            last.role == "tool"
            and last.content.startswith("ERROR: transient")
            and self._pending_retry
        ):
            if self._rng.random() < self.retry_prob:
                return AgentAction("retrying", (self._pending_retry,))
            return AgentAction("The service seems down; I give up.")
        if self._idx >= len(self._plan):
            return AgentAction("done")
        step = self._plan[self._idx]
        if self._fail_at is not None and self._idx == self._fail_at:
            self._idx = len(self._plan)  # derail: harmful call, then stop following the plan
            self._pending_retry = None
            return AgentAction("", (ToolInvocation("finish", {}),))
        self._idx += 1
        self._pending_retry = step
        return AgentAction("", (step,))
