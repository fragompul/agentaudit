import json

import pytest

from agentaudit.agents.base import Message
from agentaudit.agents.text_protocol import (
    ChatResult,
    TextProtocolAgent,
    parse_tool_calls,
    render,
    tools_prompt,
)
from agentaudit.bench.base import Param, ToolInvocation, ToolSpec

TOOLS = [ToolSpec("ls", "list a dir", {"path": Param("string", "p")}), ToolSpec("finish", "done")]


@pytest.mark.parametrize(
    ("text", "name", "args"),
    [
        (
            '<tool_call>\n{"name": "ls", "arguments": {"path": "/"}}\n</tool_call>',
            "ls",
            {"path": "/"},
        ),
        ('{"name": "ls", "arguments": {"path": "/a"}}', "ls", {"path": "/a"}),
        ('```json\n{"name": "ls", "parameters": {"path": "/b"}}\n```', "ls", {"path": "/b"}),
        ('Let me look.\n<tool_call>{"name":"ls","arguments":{"path":"/c"}}', "ls", {"path": "/c"}),
        ('{"name": "ls", "arguments": "{\\"path\\": \\"/d\\"}"}', "ls", {"path": "/d"}),
        ('{"name": "finish", "arguments": {}}', "finish", {}),
    ],
)
def test_parse_accepts_common_formats(text: str, name: str, args: dict[str, str]) -> None:
    _, calls = parse_tool_calls(text)
    assert calls == (ToolInvocation(name, args),)


def test_parse_rejects_junk_and_keeps_prose() -> None:
    assert parse_tool_calls("I will do it.")[1] == ()
    assert parse_tool_calls('{"name": "ls"}')[1] == ()  # no arguments member
    assert parse_tool_calls('{"foo": 1}')[1] == ()
    assert parse_tool_calls('{"name": "ls", "arguments": {broken')[1] == ()
    assert parse_tool_calls('{"name": "ls", "arguments": "not json"}')[1][0].arguments == {
        "_unparsed": "not json"
    }
    assert parse_tool_calls('{"name": "ls", "arguments": [1]}')[1][0].arguments == {}
    rest, calls = parse_tool_calls('First.\n{"name": "ls", "arguments": {}}\nThen done.')
    assert len(calls) == 1 and "First." in rest and "Then done." in rest
    _, many = parse_tool_calls(
        '{"name": "ls", "arguments": {}} {"name": "finish", "arguments": {}}'
    )
    assert [c.name for c in many] == ["ls", "finish"]


def test_render_wraps_tool_results_and_injects_tools() -> None:
    msgs = [
        Message("system", "S"),
        Message("user", "U"),
        Message("assistant", "thinking", (ToolInvocation("ls", {"path": "/"}),)),
        Message("tool", "a.txt", tool_name="ls"),
    ]
    out = render(msgs, TOOLS)
    assert out[0]["content"].startswith("S") and "<tools>" in out[0]["content"]
    assert out[2]["role"] == "assistant" and "<tool_call>" in out[2]["content"]
    assert (
        json.loads(out[2]["content"].split("<tool_call>")[1].split("</tool_call>")[0])["name"]
        == "ls"
    )
    assert out[3] == {"role": "user", "content": "<tool_response>\na.txt\n</tool_response>"}
    assert "list a dir" in tools_prompt(TOOLS)


def test_agent_executes_only_first_call_and_records_raw() -> None:
    def chat(messages: list[dict[str, str]], seed: int) -> ChatResult:
        assert messages[0]["role"] == "system"
        return ChatResult(
            '{"name": "ls", "arguments": {"path": "/"}}\n{"name": "finish", "arguments": {}}',
            10,
            5,
            0.5,
        )

    agent = TextProtocolAgent("m", chat, {"model": "x"})
    act = agent.act([Message("system", "S"), Message("user", "U")], TOOLS, seed=1)
    assert [c.name for c in act.tool_calls] == ["ls"]
    assert act.meta["parsed_calls"] == 2 and (act.prompt_tokens, act.latency_s) == (10, 0.5)
    assert agent.identity() == {"protocol": "text-v1", "model": "x"}
