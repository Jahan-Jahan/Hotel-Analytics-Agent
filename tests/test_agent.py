import json

import httpx
import openai

from hotel_agent.agent import MAX_HISTORY_TURNS, Agent
from tests.fakes import ScriptedClient, StubToolBox, make_response


def _agent(responses, toolbox=None, max_steps=5):
    client = ScriptedClient(responses)
    return Agent(client, toolbox or StubToolBox(), "test-model", max_steps), client


def test_tool_then_final_answer_and_accounting():
    toolbox = StubToolBox({"run_sql": [{"ok": True, "columns": ["n"], "rows": [[42]]}]})
    agent, client = _agent([
        make_response(tool_calls=[("run_sql", {"sql": "SELECT 42"})], usage=(100, 10)),
        make_response("The answer is 42.", usage=(150, 20)),
    ], toolbox)
    result = agent.ask("How many?")

    assert result.completed and result.answer == "The answer is 42."
    assert result.steps == 2 and len(result.tool_calls) == 1
    assert (result.prompt_tokens, result.completion_tokens) == (250, 30)
    # second request must contain the assistant tool call and the tool result
    roles = [m["role"] for m in client.requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert client.requests[0]["temperature"] == 0


def test_agent_sees_tool_errors_and_can_retry():
    toolbox = StubToolBox({"run_sql": [{"ok": False, "error": "Database error: boom"},
                                       {"ok": True, "rows": [[1]]}]})
    agent, client = _agent([
        make_response(tool_calls=[("run_sql", {"sql": "bad"})]),
        make_response(tool_calls=[("run_sql", {"sql": "good"})]),
        make_response("Fixed it."),
    ], toolbox)
    result = agent.ask("q")
    assert result.completed and len(result.tool_calls) == 2
    tool_msg = [m for m in client.requests[1]["messages"] if m["role"] == "tool"][0]
    assert "boom" in tool_msg["content"]


def test_max_steps_stops_a_looping_model():
    agent, _ = _agent([make_response(tool_calls=[("run_sql", {"sql": "x"})]) for _ in range(3)], max_steps=3)
    result = agent.ask("q")
    assert not result.completed and result.error == "max_steps_reached" and result.steps == 3


def test_llm_failure_is_reported_not_raised():
    error = openai.APIConnectionError(request=httpx.Request("POST", "http://llm.invalid"))
    agent, _ = _agent([error])
    result = agent.ask("q")
    assert not result.completed and "LLM request failed" in result.error


def test_missing_usage_from_provider_is_tolerated():
    agent, _ = _agent([make_response("hi", usage=None)])
    result = agent.ask("q")
    assert result.completed and result.prompt_tokens == 0


def test_malformed_tool_arguments_do_not_crash():
    toolbox = StubToolBox({"run_sql": [{"ok": False, "error": "Arguments are not valid JSON"}]})
    agent, _ = _agent([make_response(tool_calls=[("run_sql", "{not json")]), make_response("sorry")], toolbox)
    result = agent.ask("q")
    assert result.completed and result.tool_calls[0].arguments == {"_raw": "{not json"}


def test_history_is_kept_between_questions_and_trimmed():
    n = MAX_HISTORY_TURNS + 3
    agent, client = _agent([make_response(f"a{i}") for i in range(n)])
    for i in range(n):
        agent.ask(f"q{i}")
    last = client.requests[-1]["messages"]
    users = [m["content"] for m in last if m["role"] == "user"]
    assert users == [f"q{i}" for i in range(n - MAX_HISTORY_TURNS, n)]
    agent.reset()
    assert agent.turns == []


# ------------------------------------------------ automatic retry after a fixable tool error
FIXABLE = {"ok": False, "error": "Missing required fields: hotel.", "retryable": True}


def test_model_that_gives_up_after_a_fixable_error_is_nudged_once_and_recovers():
    toolbox = StubToolBox({"predict_cancellation": [FIXABLE, {"ok": True, "cancel_probability": 0.6}]})
    agent, client = _agent([
        make_response(tool_calls=[("predict_cancellation", {"adults": 2})]),
        make_response("Please confirm the hotel is Resort Hotel?"),          # gives up
        make_response(tool_calls=[("predict_cancellation", {"hotel": "Resort Hotel", "adults": 2})]),
        make_response("Probability: 60%."),
    ], toolbox)
    result = agent.ask("Resort Hotel booking for 2 adults ...")

    assert result.completed and result.answer == "Probability: 60%."
    assert result.retry_nudges == 1 and result.steps == 4 and len(result.tool_calls) == 2
    nudge = client.requests[2]["messages"][-1]
    assert nudge["role"] == "user" and nudge["content"].startswith("Automatic check:")
    assert "Missing required fields: hotel" in nudge["content"]


def test_nudge_happens_only_once_per_question():
    toolbox = StubToolBox({"predict_cancellation": [FIXABLE]})
    agent, _ = _agent([
        make_response(tool_calls=[("predict_cancellation", {})]),
        make_response("Which hotel?"),
        make_response("Sorry, still need the hotel."),          # gives up again: accepted, no loop
    ], toolbox)
    result = agent.ask("q")
    assert result.completed and result.retry_nudges == 1 and result.answer == "Sorry, still need the hotel."


def test_no_nudge_for_errors_that_are_not_fixable_by_retrying():
    toolbox = StubToolBox({"run_sql": [{"ok": False, "error": "Database unavailable"}]})
    agent, client = _agent([make_response(tool_calls=[("run_sql", {"sql": "SELECT 1"})]),
                            make_response("The database is down.")], toolbox)
    result = agent.ask("q")
    assert result.completed and result.retry_nudges == 0 and len(client.requests) == 2


def test_no_nudge_when_the_model_simply_answers():
    agent, _ = _agent([make_response("hello")])
    assert agent.ask("hi").retry_nudges == 0
