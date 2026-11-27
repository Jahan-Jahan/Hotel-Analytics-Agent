"""Fake LLM clients so the agent loop and the eval harness can be tested without an API key."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS
from typing import Any


def make_response(content: str | None = None, tool_calls: list[tuple[str, Any]] | None = None,
                  usage: tuple[int, int] | None = (10, 5)):
    calls = None
    if tool_calls:
        calls = [NS(id=f"call_{i}", type="function",
                    function=NS(name=name, arguments=args if isinstance(args, str) else json.dumps(args)))
                 for i, (name, args) in enumerate(tool_calls)]
    message = NS(content=content, tool_calls=calls)
    return NS(choices=[NS(message=message)],
              usage=NS(prompt_tokens=usage[0], completion_tokens=usage[1]) if usage else None)


class ScriptedClient:
    """Returns the queued responses in order; records every request."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class StubToolBox:
    """No database: returns canned results by tool name."""

    def __init__(self, results: dict[str, list[dict]] | None = None):
        self.schemas = []
        self.results = {k: list(v) for k, v in (results or {}).items()}
        self.calls: list[tuple[str, str]] = []

    def call(self, name: str, raw_arguments: str) -> dict:
        self.calls.append((name, raw_arguments))
        queue = self.results.get(name)
        return queue.pop(0) if queue else {"ok": True}


class OracleClient:
    """Behaves like a perfect agent: uses the reference SQL / expected payload from the eval set.

    Used to prove that the eval harness scores a correct agent at 100%.
    """

    def __init__(self, cases: list[dict], runner):
        self.by_question = {c["question"]: c for c in cases}
        self.runner = runner
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, *, messages, **_):
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        case = self.by_question[messages[last_user]["content"]]
        tool_msgs = [m for m in messages[last_user:] if m["role"] == "tool"]
        cat = case["category"]

        if cat == "refusal":
            return make_response("Sorry, I can only answer questions about hotel bookings and I can't do that.")
        if not tool_msgs:
            if cat == "sql":
                return make_response(tool_calls=[("run_sql", {"sql": case["reference_sql"]})])
            if cat == "chart":
                return make_response(tool_calls=[("make_chart", case["chart"])])
            return make_response(tool_calls=[("predict_cancellation", case["expected_payload"])])

        result = json.loads(tool_msgs[-1]["content"])
        if cat == "sql":
            ref = self.runner.run(case["reference_sql"], max_rows=1000)
            return make_response(" | ".join(str(c) for row in ref["rows"] for c in row))
        if cat == "predict":
            return make_response(f"Cancellation probability: {result['cancel_probability']:.1%}")
        return make_response("Here is the chart.")
