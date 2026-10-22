"""Core abstractions of AgentAudit-Bench: tools, environments, tasks and fault injection."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

JSONDict = dict[str, Any]


@dataclass(frozen=True)
class Param:
    type: str  # "string" | "integer" | "number" | "boolean"
    description: str
    required: bool = True


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: dict[str, Param] = field(default_factory=dict)

    def json_schema(self) -> JSONDict:
        """OpenAI/Ollama-style function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        k: {"type": p.type, "description": p.description}
                        for k, p in self.params.items()
                    },
                    "required": [k for k, p in self.params.items() if p.required],
                },
            },
        }


@dataclass(frozen=True)
class ToolInvocation:
    name: str
    arguments: JSONDict


class Environment(ABC):
    """A deterministic, fully offline simulated world with callable tools.

    Tool results are plain strings; errors are returned as ``"ERROR: ..."`` strings (never
    raised) so that an agent can observe and recover from them.
    """

    def __init__(self) -> None:
        self.answer: str | None = None
        self.finished = False
        self._tools: dict[str, ToolSpec] = {}

    @property
    def tools(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tools.values())

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def call(self, name: str, arguments: JSONDict) -> str:
        spec = self._tools.get(name)
        if spec is None:
            return f"ERROR: unknown tool {name!r}. Available: {', '.join(self._tools)}"
        missing = [k for k, p in spec.params.items() if p.required and k not in arguments]
        if missing:
            return f"ERROR: missing required argument(s): {', '.join(missing)}"
        extra = [k for k in arguments if k not in spec.params]
        if extra:
            return f"ERROR: unexpected argument(s): {', '.join(extra)}"
        try:
            return str(self._dispatch(name, arguments))
        except (KeyError, ValueError, TypeError, SyntaxError, ArithmeticError) as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    @abstractmethod
    def _dispatch(self, name: str, args: JSONDict) -> str: ...

    @abstractmethod
    def state(self) -> JSONDict:
        """JSON-serialisable snapshot of the mutable world state (for checkers/hashing)."""

    def state_hash(self) -> str:
        blob = json.dumps(self.state(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Task:
    """A procedurally generated task instance (fully described by ``family`` and ``spec``)."""

    task_id: str
    family: str
    instruction: str
    spec: JSONDict
    horizon: int  # number of tool calls an optimal solution needs
    distractors: int
    requires_answer: bool = False


class Family(ABC):
    """A task family: generates tasks, builds fresh environments, checks final state."""

    name: str

    @abstractmethod
    def generate(
        self, rng: np.random.Generator, index: int, horizon: int, distractors: int
    ) -> Task: ...

    @abstractmethod
    def build(self, task: Task) -> Environment: ...

    @abstractmethod
    def check(self, task: Task, env: Environment) -> bool: ...

    @abstractmethod
    def oracle(self, task: Task) -> list[ToolInvocation]:
        """A reference solution (used to audit that every checker accepts a correct run)."""


class FaultInjector:
    """Wraps an environment; each tool call independently fails with prob. ``rate``.

    A failed call is *not* applied to the wrapped environment and returns a transient error
    message, so a robust agent should retry. Failures are drawn from a generator seeded by
    ``seed`` and are therefore exactly reproducible; every injection is logged in ``log``.
    The draw happens on every call, so the failure pattern does not depend on the agent's
    choices (only on how many calls it makes).
    """

    TRANSIENT = "ERROR: transient failure (HTTP 503 Service Unavailable). Please retry."

    def __init__(self, inner: Environment, rate: float, seed: int) -> None:
        if not 0.0 <= rate < 1.0:
            raise ValueError("rate must be in [0, 1)")
        self.inner, self.rate = inner, rate
        self._rng = np.random.default_rng(seed)
        self.log: list[int] = []  # indices of calls that were failed
        self._n = 0

    @property
    def tools(self) -> tuple[ToolSpec, ...]:
        return self.inner.tools

    def call(self, name: str, arguments: JSONDict) -> str:
        idx = self._n
        self._n += 1
        failed = self._rng.random() < self.rate
        if failed and name in self.inner._tools:
            self.log.append(idx)
            return self.TRANSIENT
        return self.inner.call(name, arguments)


def stable_seed(*parts: object) -> int:
    """Deterministic 32-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:4], "big")


COMMON_TOOLS = (
    ToolSpec(
        "submit_answer",
        "Submit the final answer to the question. Call this exactly once when you know the answer.",
        {"answer": Param("string", "The final answer, as plainly as possible")},
    ),
    ToolSpec("finish", "Call this when the task is complete and no more actions are needed."),
)


def install_common_tools(env: Environment) -> None:
    for spec in COMMON_TOOLS:
        env.register(spec)


def handle_common(env: Environment, name: str, args: JSONDict) -> str | None:
    if name == "submit_answer":
        env.answer = str(args["answer"]).strip()
        return "Answer recorded."
    if name == "finish":
        env.finished = True
        return "Task marked as finished."
    return None
