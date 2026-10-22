"""Record/replay cache: makes expensive model interactions reproducible and re-analysable.

Every agent turn is keyed by a hash of the agent identity, the full message history, the tool
names and the seed. In ``auto`` mode a hit is replayed (including the originally measured
tokens and latency) and a miss calls the wrapped agent and records the result; ``replay`` mode
raises on a miss so that published results can be regenerated with no model at all.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from agentaudit.agents.base import Agent, AgentAction, Message
from agentaudit.bench.base import Family, Task, ToolInvocation, ToolSpec

Mode = Literal["auto", "record", "replay"]


class CacheMissError(KeyError):
    """Raised in replay mode when an interaction was not recorded."""


class Store:
    """Append-only JSONL key-value store, safe for use from several threads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._data[rec["key"]] = rec["value"]
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            val = self._data.get(key)
            if val is not None:
                self.hits += 1
            return val

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self.misses += 1
            self._data[key] = value
            with self.path.open("a") as fh:
                fh.write(json.dumps({"key": key, "value": value}, separators=(",", ":")) + "\n")

    def __len__(self) -> int:
        return len(self._data)


def _key(
    identity: dict[str, object], messages: Sequence[Message], tools: Sequence[ToolSpec], seed: int
) -> str:
    blob = json.dumps(
        {
            "id": identity,
            "msgs": [
                [m.role, m.content, [[c.name, c.arguments] for c in m.tool_calls], m.tool_name]
                for m in messages
            ],
            "tools": [t.name for t in tools],
            "seed": seed,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class CachedAgent(Agent):
    def __init__(self, inner: Agent, store: Store, mode: Mode = "auto") -> None:
        self.inner, self.store, self.mode = inner, store, mode
        self.name = inner.name

    def identity(self) -> dict[str, object]:
        return self.inner.identity()

    def begin(self, task: Task, family: Family, seed: int) -> None:
        self.inner.begin(task, family, seed)

    def session(self, task: Task, family: Family, seed: int) -> Agent:
        return CachedAgent(self.inner.session(task, family, seed), self.store, self.mode)

    def act(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, seed: int
    ) -> AgentAction:
        key = _key(self.inner.identity(), messages, tools, seed)
        if self.mode != "record":
            hit = self.store.get(key)
            if hit is not None:
                return AgentAction(
                    str(hit["content"]),
                    tuple(
                        ToolInvocation(str(c["name"]), dict(c["arguments"]))
                        for c in hit["tool_calls"]
                    ),
                    int(hit["prompt_tokens"]),
                    int(hit["completion_tokens"]),
                    float(hit["latency_s"]),
                    dict(hit.get("meta") or {}),
                )
            if self.mode == "replay":
                raise CacheMissError(key)
        act = self.inner.act(messages, tools, seed=seed)
        self.store.put(
            key,
            {
                "content": act.content,
                "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in act.tool_calls],
                "prompt_tokens": act.prompt_tokens,
                "completion_tokens": act.completion_tokens,
                "latency_s": act.latency_s,
                "meta": act.meta,
            },
        )
        return act
