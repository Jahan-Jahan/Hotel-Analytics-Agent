"""Needs PostgreSQL with the `bookings` view (Hotel-Booking-Demand: make db-up && make sql),
the read-only role (make db-role) and the prediction API (make serve there)."""
import httpx
import pytest
from sqlalchemy import create_engine, text

from hotel_agent.agent import Agent
from hotel_agent.config import Settings
from hotel_agent.evaluation import load_cases, run_eval, summarize
from hotel_agent.tools import ToolBox
from hotel_agent.tools.predict_tool import PredictClient
from hotel_agent.tools.sql_tool import SqlRunner
from tests.fakes import OracleClient

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def settings():
    s = Settings.from_env()
    try:
        with create_engine(s.database_url).connect() as con:
            con.execute(text("SELECT COUNT(*) FROM bookings"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL/bookings view/agent_ro role not available: {type(exc).__name__}")
    return s


@pytest.fixture(scope="module")
def runner(settings):
    return SqlRunner(settings)


def test_query_and_row_cap(runner):
    ok = runner.run("SELECT hotel, COUNT(*) AS n FROM bookings GROUP BY hotel ORDER BY hotel")
    assert ok["ok"] and ok["columns"] == ["hotel", "n"] and ok["row_count"] == 2
    capped = runner.run("SELECT * FROM bookings", max_rows=10)
    assert capped["row_count"] == 10 and capped["truncated"] is True


def test_guardrail_and_database_errors_are_returned_not_raised(runner):
    assert "Blocked by guardrails" in runner.run("DROP TABLE bookings")["error"]
    err = runner.run("SELECT ROUND(AVG(adr), 2) FROM bookings")["error"]
    assert "Hint: cast to numeric" in err


def test_stacked_statement_cannot_slip_through_even_if_validation_were_bypassed(settings):
    """Layer 2: the wrapper makes '; DROP' a syntax error in Postgres itself."""
    from hotel_agent.guardrails import wrap_with_limit
    engine = create_engine(settings.database_url)
    with engine.connect() as con, pytest.raises(Exception):
        con.exec_driver_sql(wrap_with_limit("SELECT 1; DROP TABLE bookings", 10))


def test_database_role_is_read_only_even_when_guardrails_are_bypassed(settings):
    """Layer 3: talk to Postgres directly, skipping every Python check."""
    engine = create_engine(settings.database_url)
    for statement in ("DELETE FROM bookings", "DROP MATERIALIZED VIEW bookings",
                      "SELECT * FROM hotel_bookings LIMIT 1", "CREATE TABLE evil (x int)"):
        with engine.connect() as con, pytest.raises(Exception):
            con.exec_driver_sql(statement)
            con.commit()


def test_timeout_stops_runaway_queries(settings):
    fast = SqlRunner(Settings(**{**settings.__dict__, "sql_timeout_ms": 300}))
    out = fast.run("SELECT COUNT(*) FROM bookings a CROSS JOIN bookings b CROSS JOIN bookings c")
    assert not out["ok"] and "timeout" in out["error"].lower()


def test_eval_harness_scores_a_perfect_agent_at_100_percent(settings, runner):
    """The oracle answers every case correctly, so any failure here is a harness bug."""
    try:
        httpx.Client(trust_env=False, timeout=3).get(f"{settings.predict_api_url}/health").raise_for_status()
    except Exception:
        pytest.skip("prediction API not running")
    cases = load_cases()
    toolbox = ToolBox(settings, runner=runner, predictor=PredictClient(settings.predict_api_url))
    agent = Agent(OracleClient(cases, runner), toolbox, "oracle", max_steps=4)
    results = run_eval(agent, cases, runner, toolbox.predictor)
    failures = [(r.id, r.detail) for r in results if not r.passed]
    assert not failures, failures
    assert summarize(results)["pass_rate"] == 1.0


def test_eval_harness_detects_a_wrong_agent(settings, runner):
    """An agent that returns wrong numbers must FAIL, otherwise the eval would be meaningless."""
    from hotel_agent.evaluation import evaluate_case

    case = next(c for c in load_cases() if c["id"] == "s01_overall_rate")
    wrong_case = {**case, "reference_sql": "SELECT 12.3"}      # the fake agent runs a wrong query

    class WrongAgentClient(OracleClient):
        def __init__(self):
            super().__init__([wrong_case], runner)

    agent = Agent(WrongAgentClient(), ToolBox(settings, runner=runner), "wrong", max_steps=4)
    result = evaluate_case(agent, case, runner, None)
    assert result.passed is False and result.sql_match is False


def test_ui_renders_real_results_table_chart_and_prediction(settings, runner, monkeypatch, tmp_path):
    """UI against the real database and prediction API (only the LLM is scripted)."""
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    APP = str(Path(__file__).resolve().parents[1] / "app.py")

    try:
        httpx.Client(trust_env=False, timeout=3).get(f"{settings.predict_api_url}/health").raise_for_status()
    except Exception:
        pytest.skip("prediction API not running")
    cases = load_cases()
    toolbox = ToolBox(Settings(**{**settings.__dict__, "chart_dir": tmp_path}), runner=runner,
                      predictor=PredictClient(settings.predict_api_url))
    monkeypatch.setattr("hotel_agent.agent.build_agent",
                        lambda s=None, t=None: Agent(OracleClient(cases, runner), toolbox, "oracle", 4))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    for case_id in ("s03_bookings_per_hotel", "c01_monthly_chart", "p01_risky_booking"):
        question = next(c["question"] for c in cases if c["id"] == case_id)
        at.chat_input[0].set_value(question).run()
        assert not at.exception, case_id
    assert len(at.dataframe) >= 1                       # SQL result table
    assert len(at.metric) == 1 and "%" in at.metric[0].value   # prediction probability
    assert len(at.get("image")) == 1                    # chart image
