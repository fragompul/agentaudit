import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from agentaudit.agents.base import Message
from agentaudit.agents.http import BackendError, OllamaAgent, OpenAICompatAgent
from agentaudit.bench.base import Param, ToolInvocation, ToolSpec

TOOLS = [ToolSpec("ls", "list", {"path": Param("string", "p")})]
SEEN: list[dict[str, Any]] = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
        if self.path == "/api/chat":
            out: dict[str, Any] = {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "ls", "arguments": {"path": "/"}}}],
                },
                "prompt_eval_count": 11,
                "eval_count": 7,
                "done_reason": "stop",
            }
        elif self.path == "/v1/chat/completions":
            out = {
                "choices": [
                    {
                        "message": {
                            "content": "hi",
                            "tool_calls": [
                                {"function": {"name": "ls", "arguments": '{"path": "/x"}'}}
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }
        elif self.path == "/bad/chat/completions":
            out = {"nope": 1}
        else:
            self.send_response(500)
            self.end_headers()
            return
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


HISTORY = [
    Message("system", "s"),
    Message("user", "u"),
    Message("assistant", "", (ToolInvocation("ls", {"path": "/"}),)),
    Message("tool", "a.txt", tool_name="ls"),
]


def test_ollama_roundtrip(server: str) -> None:
    SEEN.clear()
    agent = OllamaAgent("m", host=server, temperature=0.2)
    act = agent.act(HISTORY, TOOLS, seed=5)
    assert act.tool_calls == (ToolInvocation("ls", {"path": "/"}),)
    assert (act.prompt_tokens, act.completion_tokens) == (11, 7) and act.latency_s > 0
    body = SEEN[-1]["body"]
    assert body["options"]["seed"] == 5 and body["options"]["temperature"] == 0.2
    assert body["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"path": "/"}
    assert body["messages"][3] == {"role": "tool", "content": "a.txt", "tool_name": "ls"}
    assert body["tools"][0]["function"]["name"] == "ls" and body["stream"] is False
    assert agent.identity()["model"] == "m" and agent.name == "ollama/m"


def test_openai_compat_roundtrip_and_ids(server: str) -> None:
    SEEN.clear()
    agent = OpenAICompatAgent("m", base_url=server + "/v1", api_key="k")
    act = agent.act(HISTORY, TOOLS, seed=1)
    assert act.tool_calls == (ToolInvocation("ls", {"path": "/x"}),) and act.content == "hi"
    assert (act.prompt_tokens, act.completion_tokens) == (5, 3)
    wire = SEEN[-1]["body"]["messages"]
    assert wire[2]["tool_calls"][0]["id"] == wire[3]["tool_call_id"]  # ids pair up
    assert SEEN[-1]["auth"] == "Bearer k"
    assert agent.identity()["backend"] == "openai-compat"


def test_errors_are_backend_errors(server: str) -> None:
    with pytest.raises(BackendError):
        OpenAICompatAgent("m", base_url=server + "/bad").act(HISTORY, TOOLS, seed=0)
    with pytest.raises(BackendError):
        OllamaAgent("m", host=server + "/nowhere").act(HISTORY, TOOLS, seed=0)
    with pytest.raises(BackendError):
        OllamaAgent("m", host="http://127.0.0.1:9", timeout=1).act(HISTORY, TOOLS, seed=0)
