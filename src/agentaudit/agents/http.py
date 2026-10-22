"""Local-LLM backends over HTTP using only the standard library.

* :class:`OllamaAgent` talks to a local Ollama server (``/api/chat`` with native tool calling).
* :class:`OpenAICompatAgent` talks to any OpenAI-compatible endpoint. It is intended for **local**
  servers (llama.cpp ``llama-server``, ``mlx_lm.server``, LM Studio, vLLM). Pointing it at a paid
  hosted API is possible in principle but is outside the scope of this project and is never done
  by any code or experiment in this repository.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from agentaudit.agents.base import Agent, AgentAction, Message
from agentaudit.bench.base import ToolInvocation, ToolSpec


class BackendError(RuntimeError):
    """The model server failed or returned an unusable response."""


def _post(
    url: str, payload: dict[str, Any], timeout: float, headers: dict[str, str]
) -> dict[str, Any]:
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json", **headers}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (local servers)
            out: dict[str, Any] = json.loads(resp.read())
            return out
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise BackendError(f"{url}: {exc}") from exc


def _args(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a dict (Ollama) or a JSON string (OpenAI); be lenient."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            val = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {"_unparsed": raw}
        return val if isinstance(val, dict) else {"_unparsed": raw}
    return {}


class OllamaAgent(Agent):
    def __init__(
        self,
        model: str,
        *,
        name: str | None = None,
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.7,
        num_ctx: int = 8192,
        num_predict: int = 512,
        timeout: float = 300.0,
        keep_alive: str = "30m",
    ) -> None:
        self.model, self.host = model, host.rstrip("/")
        self.name = name or f"ollama/{model}"
        self.temperature, self.num_ctx = temperature, num_ctx
        self.num_predict, self.timeout, self.keep_alive = num_predict, timeout, keep_alive

    def identity(self) -> dict[str, object]:
        return {
            "backend": "ollama",
            "model": self.model,
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }

    @staticmethod
    def _wire(m: Message) -> dict[str, Any]:
        if m.role == "tool":
            return {"role": "tool", "content": m.content, "tool_name": m.tool_name}
        out: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            out["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
            ]
        return out

    def act(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, seed: int
    ) -> AgentAction:
        payload = {
            "model": self.model,
            "messages": [self._wire(m) for m in messages],
            "tools": [t.json_schema() for t in tools],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "seed": seed % (2**31),
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        t0 = time.perf_counter()
        data = _post(f"{self.host}/api/chat", payload, self.timeout, {})
        wall = time.perf_counter() - t0
        msg = data.get("message") or {}
        calls = tuple(
            ToolInvocation(str(tc["function"]["name"]), _args(tc["function"].get("arguments")))
            for tc in msg.get("tool_calls") or []
        )
        return AgentAction(
            content=str(msg.get("content") or ""),
            tool_calls=calls,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            latency_s=wall,
            meta={"done_reason": data.get("done_reason")},
        )


class OpenAICompatAgent(Agent):
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:8080/v1",
        name: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout: float = 300.0,
        api_key: str | None = None,
    ) -> None:
        self.model, self.base_url = model, base_url.rstrip("/")
        self.name = name or f"openai-compat/{model}"
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def identity(self) -> dict[str, object]:
        return {
            "backend": "openai-compat",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    @staticmethod
    def _wire(i: int, messages: Sequence[Message], m: Message) -> dict[str, Any]:
        if m.role == "tool":
            return {"role": "tool", "content": m.content, "tool_call_id": f"call_{i}"}
        out: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            out["tool_calls"] = [
                {
                    "id": f"call_{i + 1 + j}",
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for j, c in enumerate(m.tool_calls)
            ]
        return out

    def act(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, seed: int
    ) -> AgentAction:
        wire: list[dict[str, Any]] = []
        pending_ids: list[str] = []
        for i, m in enumerate(messages):
            if m.role == "assistant" and m.tool_calls:
                entry = self._wire(i, messages, m)
                pending_ids = [tc["id"] for tc in entry["tool_calls"]]
                wire.append(entry)
            elif m.role == "tool":
                cid = pending_ids.pop(0) if pending_ids else f"call_{i}"
                wire.append({"role": "tool", "content": m.content, "tool_call_id": cid})
            else:
                wire.append(self._wire(i, messages, m))
        payload = {
            "model": self.model,
            "messages": wire,
            "tools": [t.json_schema() for t in tools],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": seed % (2**31),
            "stream": False,
        }
        t0 = time.perf_counter()
        data = _post(f"{self.base_url}/chat/completions", payload, self.timeout, self._headers)
        wall = time.perf_counter() - t0
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"unexpected response shape: {str(data)[:200]}") from exc
        calls = tuple(
            ToolInvocation(str(tc["function"]["name"]), _args(tc["function"].get("arguments")))
            for tc in msg.get("tool_calls") or []
        )
        usage = data.get("usage") or {}
        return AgentAction(
            content=str(msg.get("content") or ""),
            tool_calls=calls,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_s=wall,
            meta={"finish_reason": data["choices"][0].get("finish_reason")},
        )
