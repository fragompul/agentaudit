"""Uniform text-based tool-calling protocol (ReAct-style) over any chat endpoint.

The benchmark, not the model's chat template, defines how tools are described and how calls
are parsed. This gives every model the *same* interface, and avoids silent tool-call loss in
server-side parsers (observed with Ollama 0.40.8 + qwen2.5:3b: a well-formed ``<tool_call>``
generation was dropped and an empty message returned when many tools were declared).

Protocol: tools are listed as JSON in the system prompt; the model calls a tool by emitting
``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` (several blocks allowed); tool
results are fed back as user messages wrapped in ``<tool_response>``.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from agentaudit.agents.base import Agent, AgentAction, Message
from agentaudit.agents.http import BackendError, _post
from agentaudit.bench.base import ToolInvocation, ToolSpec


@dataclass(frozen=True)
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


ChatFn = Callable[[list[dict[str, str]], int], ChatResult]


def tools_prompt(tools: Sequence[ToolSpec]) -> str:
    listing = "\n".join(json.dumps(t.json_schema()["function"]) for t in tools)
    return (
        "\n\n# Tools\n\nYou can call these tools (JSON signatures):\n<tools>\n"
        f"{listing}\n</tools>\n\n"
        "To call a tool, reply with exactly one block of this form and nothing else:\n"
        '<tool_call>\n{"name": "<tool name>", "arguments": {<arguments as JSON>}}\n</tool_call>\n'
        "You will receive the result inside <tool_response></tool_response>. "
        "Call one tool at a time."
    )


def parse_tool_calls(text: str) -> tuple[str, tuple[ToolInvocation, ...]]:
    """Extract tool calls from model text; returns ``(remaining_text, calls)``.

    A tool call is any JSON object with a string ``name`` and an ``arguments`` (or Llama-style
    ``parameters``) member, found anywhere in the text, with or without ``<tool_call>`` tags or
    code fences. Malformed JSON yields no call. Surrounding prose is returned as the remainder.
    """
    decoder = json.JSONDecoder()
    calls: list[ToolInvocation] = []
    keep: list[str] = []
    pos, last = 0, 0
    while True:
        start = text.find("{", pos)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pos = start + 1
            continue
        pos = end
        if not (isinstance(obj, dict) and isinstance(obj.get("name"), str)):
            continue
        if "arguments" not in obj and "parameters" not in obj:
            continue
        args = obj.get("arguments", obj.get("parameters"))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_unparsed": args}
        calls.append(ToolInvocation(obj["name"], args if isinstance(args, dict) else {}))
        keep.append(text[last:start])
        last = end
    keep.append(text[last:])
    rest = "".join(keep)
    rest = re.sub(r"</?tool_call>|```(?:json)?", "", rest).strip()
    return rest, tuple(calls)


def render(messages: Sequence[Message], tools: Sequence[ToolSpec]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        if m.role == "system":
            out.append({"role": "system", "content": m.content + tools_prompt(tools)})
        elif m.role == "assistant":
            body = m.content
            for c in m.tool_calls:
                body += (
                    "\n<tool_call>\n"
                    + json.dumps({"name": c.name, "arguments": c.arguments})
                    + "\n</tool_call>"
                )
            out.append({"role": "assistant", "content": body.strip()})
        elif m.role == "tool":
            out.append(
                {"role": "user", "content": f"<tool_response>\n{m.content}\n</tool_response>"}
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out


class TextProtocolAgent(Agent):
    """Agent that speaks the text tool protocol through an arbitrary ``ChatFn`` backend."""

    def __init__(self, name: str, chat: ChatFn, config: dict[str, object] | None = None) -> None:
        self.name = name
        self._chat = chat
        self._config = config or {}

    def identity(self) -> dict[str, object]:
        return {"protocol": "text-v1", **self._config}

    def act(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, seed: int
    ) -> AgentAction:
        res = self._chat(render(messages, tools), seed)
        rest, calls = parse_tool_calls(res.text)
        # one call per turn: the prompt says so, and later calls in the same message would be
        # issued before the model has seen any result (they are routinely hallucinated)
        return AgentAction(
            rest,
            calls[:1],
            res.prompt_tokens,
            res.completion_tokens,
            res.latency_s,
            {"raw": res.text, "parsed_calls": len(calls)},
        )


def ollama_chat(
    model: str,
    *,
    host: str = "http://127.0.0.1:11434",
    temperature: float = 0.7,
    num_ctx: int = 8192,
    num_predict: int = 256,
    timeout: float = 300.0,
    keep_alive: str = "30m",
) -> ChatFn:
    url = host.rstrip("/") + "/api/chat"

    def chat(messages: list[dict[str, str]], seed: int) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {
                "temperature": temperature,
                "seed": seed % (2**31),
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                # one call per turn: stop right after it (and before any invented tool result)
                "stop": ["</tool_call>", "<tool_response>"],
            },
        }
        t0 = time.perf_counter()
        data = _post(url, payload, timeout, {})
        wall = time.perf_counter() - t0
        msg = data.get("message")
        if not isinstance(msg, dict):
            raise BackendError(f"unexpected response: {str(data)[:200]}")
        return ChatResult(
            str(msg.get("content") or ""),
            int(data.get("prompt_eval_count") or 0),
            int(data.get("eval_count") or 0),
            wall,
        )

    return chat


def openai_chat(
    model: str,
    *,
    base_url: str = "http://127.0.0.1:8080/v1",
    temperature: float = 0.7,
    max_tokens: int = 400,
    timeout: float = 300.0,
) -> ChatFn:
    """Chat function for local OpenAI-compatible servers (llama.cpp, mlx_lm.server, LM Studio)."""
    url = base_url.rstrip("/") + "/chat/completions"

    def chat(messages: list[dict[str, str]], seed: int) -> ChatResult:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed % (2**31),
            "stream": False,
        }
        t0 = time.perf_counter()
        data = _post(url, payload, timeout, {})
        wall = time.perf_counter() - t0
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"unexpected response shape: {str(data)[:200]}") from exc
        usage = data.get("usage") or {}
        return ChatResult(
            str(text),
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            wall,
        )

    return chat


__all__ = [
    "ChatResult",
    "TextProtocolAgent",
    "ollama_chat",
    "openai_chat",
    "parse_tool_calls",
    "render",
    "tools_prompt",
]
