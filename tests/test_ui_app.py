"""Runs the real Streamlit script headlessly (AppTest) with a scripted fake agent."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from hotel_agent.agent import Agent
from hotel_agent.config import ConfigError
from tests.fakes import ScriptedClient, StubToolBox, make_response

APP = str(Path(__file__).resolve().parents[1] / "app.py")

SQL_OK = {"ok": True, "sql": "SELECT hotel, 30.1 AS rate FROM bookings", "columns": ["hotel", "rate"],
          "rows": [["City Hotel", 30.1]], "row_count": 1, "truncated": False}


def _fake_factory(rounds=3):
    def build(settings=None, toolbox=None):
        script = []
        for _ in range(rounds):
            script += [make_response(tool_calls=[("run_sql", {"sql": "SELECT 1"})]),
                       make_response("City Hotel has the higher rate: 30.1%.")]
        return Agent(ScriptedClient(script), StubToolBox({"run_sql": [SQL_OK] * rounds}), "fake-model")
    return build


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setattr("hotel_agent.agent.build_agent", _fake_factory())
    monkeypatch.setattr("hotel_agent.ui_helpers.check_services", lambda s, t: {
        "Database": (True, "87,229 bookings"), "Prediction API": (False, "unreachable"),
        "LLM": (True, "fake-model")})
    at = AppTest.from_file(APP, default_timeout=30)
    return at.run()


def test_initial_page(app):
    assert not app.exception
    assert app.title[0].value == "Ask your hotel data"
    assert len(app.chat_message) == 0
    sidebar_text = " ".join(m.value for m in app.sidebar.markdown)
    assert "🟢" in sidebar_text and "🔴" in sidebar_text and "87,229 bookings" in sidebar_text


def test_asking_a_question_shows_answer_sql_and_stats(app):
    app.chat_input[0].set_value("Which hotel cancels more?").run()
    assert not app.exception
    assert [m.name for m in app.chat_message] == ["user", "assistant"]
    assistant = app.chat_message[1]
    assert "30.1%" in " ".join(m.value for m in assistant.markdown)
    assert any("How I got this" in e.label for e in app.expander)
    assert any("SELECT hotel" in c.value for c in app.code)
    assert any("tokens" in c.value for c in app.caption)


def test_example_button_asks_the_question(app):
    next(b for b in app.sidebar.button if "higher cancellation rate" in b.label).click().run()
    assert not app.exception
    assert [m.name for m in app.chat_message] == ["user", "assistant"]


def test_clear_conversation(app):
    app.chat_input[0].set_value("hi").run()
    assert len(app.chat_message) == 2
    next(b for b in app.sidebar.button if b.label == "Clear conversation").click().run()
    assert len(app.chat_message) == 0 and not app.exception


def test_missing_api_key_shows_a_helpful_error(monkeypatch):
    def failing(settings=None, toolbox=None):
        raise ConfigError("LLM_API_KEY is not set. Copy .env.example to .env and fill it in.")
    monkeypatch.setattr("hotel_agent.agent.build_agent", failing)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and "LLM_API_KEY" in at.error[0].value
