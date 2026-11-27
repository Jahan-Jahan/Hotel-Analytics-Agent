"""Runs the REAL `openai` SDK against a local OpenAI-compatible mock server.

This verifies what the fake clients cannot: base_url handling, request serialisation (tools,
messages incl. tool results) and parsing of tool_calls/usage from a real HTTP response.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest
from openai import OpenAI

from hotel_agent.agent import Agent
from tests.fakes import StubToolBox


class MockLLM(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MockLLM.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        last = body["messages"][-1]
        if last["role"] == "tool":
            message = {"role": "assistant", "content": f"Tool said: {last['content']}"}
        else:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "run_sql", "arguments": json.dumps({"sql": "SELECT 1"})}}]}
        payload = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                   "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                   "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def server():
    MockLLM.requests.clear()
    httpd = HTTPServer(("127.0.0.1", 0), MockLLM)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def test_real_sdk_tool_calling_roundtrip(server):
    from hotel_agent.tools import TOOL_SCHEMAS

    toolbox = StubToolBox({"run_sql": [{"ok": True, "rows": [[1]]}]})
    toolbox.schemas = TOOL_SCHEMAS
    client = OpenAI(api_key="sk-test", base_url=server, max_retries=0,
                    http_client=httpx.Client(trust_env=False))   # a system proxy must not hijack localhost
    result = Agent(client, toolbox, "gpt-4o-mini").ask("hello")

    assert result.completed and result.answer.startswith("Tool said:")
    assert (result.prompt_tokens, result.completion_tokens) == (14, 6)
    first, second = MockLLM.requests
    assert first["path"] == "/v1/chat/completions" and first["auth"] == "Bearer sk-test"
    assert first["body"]["model"] == "gpt-4o-mini"
    assert [t["function"]["name"] for t in first["body"]["tools"]] == \
        ["run_sql", "predict_cancellation", "make_chart"]
    # the tool result is sent back linked to the tool call id
    assert second["body"]["messages"][-1]["tool_call_id"] == "call_1"
    assert second["body"]["messages"][-2]["tool_calls"][0]["function"]["name"] == "run_sql"
