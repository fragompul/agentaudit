"""Agent interface shared by all backends (local LLMs, scripted and synthetic agents)."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from agentaudit.bench.base import Family, Task, ToolInvocation, ToolSpec


@dataclass(frozen=True)
class Message:
    """A chat message. ``role`` is system | user | assistant | tool."""

    role: str
    content: str = ""
    tool_calls: tuple[ToolInvocation, ...] = ()
    tool_name: str = ""


@dataclass(frozen=True)
class AgentAction:
    """One agent turn: free text plus zero or more tool calls (none = the agent stopped)."""

    content: str = ""
    tool_calls: tuple[ToolInvocation, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    meta: dict[str, object] = field(default_factory=dict)


class Agent(ABC):
    """Backend-agnostic agent. ``name`` identifies the agent in result files."""

    name: str

    def begin(self, task: Task, family: Family, seed: int) -> None:  # noqa: B027
        """Hook called once per attempt (used by synthetic agents; ignored by LLM backends)."""

    def session(self, task: Task, family: Family, seed: int) -> Agent:
        """A per-attempt copy of the agent, so attempts can run concurrently without sharing state.

        Mutable per-attempt state must live on the copy (set it in :meth:`begin`); resources that
        must be shared (HTTP sessions, caches) are shared by the shallow copy.
        """
        clone = copy.copy(self)
        clone.begin(task, family, seed)
        return clone

    @abstractmethod
    def act(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, seed: int
    ) -> AgentAction:
        """Return the next action given the conversation so far."""

    def identity(self) -> dict[str, object]:
        """Configuration that determines the agent's behaviour (used for cache keys/provenance)."""
        return {"name": self.name}


SYSTEM_PROMPT = (
    "You are an autonomous assistant that completes tasks by calling the provided tools. "
    "Work step by step, use the tools to inspect the current state before changing it, and "
    "make exactly the requested changes and no others. If a tool returns an error that looks "
    "temporary (for example a 503), retry the call. When the task is done, call finish; if the "
    "task asks a question, call submit_answer with the answer."
)
