from hotel_agent.evaluation import (CaseResult, answer_states, cell_close, extract_numbers,
                                    rows_match, summarize, values_match)


def test_cell_close_tolerates_rounding_but_not_real_differences():
    assert cell_close(27.5, 27.54) and cell_close(4_903_021, 4_903_021.4)
    assert not cell_close(27.5, 28.0) and not cell_close(27.5, 0.275)
    assert cell_close("City Hotel", " city hotel ")


def test_rows_match_ignores_order_and_extra_columns_but_not_row_count():
    ref = [["City Hotel", 30.1], ["Resort Hotel", 23.5]]
    assert rows_match(ref, [["Resort Hotel", 23.5], ["City Hotel", 30.1]])
    assert rows_match(ref, [["City Hotel", 53274, 30.1], ["Resort Hotel", 33955, 23.5]])
    assert not rows_match(ref, [["City Hotel", 30.1]])
    assert not rows_match(ref, [["City Hotel", 30.1], ["Resort Hotel", 99.0]])


def test_values_match_is_shape_agnostic():
    assert values_match([[105.7, 70.2]], [[1, 105.7], [0, 70.2]])
    assert not values_match([[105.7, 70.2]], [[1, 105.7]])


def test_extract_numbers_handles_commas_and_persian_digits():
    assert extract_numbers("Lost €4,903,021 (27.5%)") == [4903021.0, 27.5]
    assert extract_numbers("نرخ کنسلی ۹۴٫۷ درصد است") == [94.7]


def test_answer_states():
    assert answer_states([[94.7]], "It is 94.7%") is True
    assert answer_states([[94.7]], "It is about 95%") is False
    assert answer_states([["City Hotel", 30.1]], "City Hotel: 30.1%") is True
    assert answer_states([[1], [2]], "anything") is None     # multi-row: not checked


def test_summarize():
    rs = [CaseResult("a", "sql", "q", True, sql_match=True, steps=2, tool_calls=1, latency_s=1.0),
          CaseResult("b", "sql", "q", False, sql_match=False, steps=4, tool_calls=3, latency_s=3.0, sql_errors=1),
          CaseResult("c", "refusal", "q", True)]
    s = summarize(rs)
    assert s["pass_rate"] == 0.667 and s["sql_execution_accuracy"] == 0.5
    assert s["by_category"]["refusal"]["pass_rate"] == 1.0 and s["sql_errors"] == 1


# ----------------------------------------------------------------- robustness of the harness
import httpx

from hotel_agent.agent import Agent
from hotel_agent.evaluation import preflight, render_markdown, run_eval
from hotel_agent.tools.predict_tool import PredictClient
from tests.fakes import ScriptedClient, StubToolBox, make_response


class BrokenRunner:
    def run(self, sql, max_rows=None):
        return {"ok": False, "error": "Database unavailable: connection refused"}


def test_infrastructure_failure_does_not_crash_the_run_and_is_not_scored_against_the_model():
    cases = [
        {"id": "s1", "category": "sql", "question": "q1", "reference_sql": "SELECT 1"},
        {"id": "r1", "category": "refusal", "question": "q2"},
    ]
    agent = Agent(ScriptedClient([make_response("answer"), make_response("Sorry, I can't.")]),
                  StubToolBox(), "m", max_steps=2)
    saved = []
    results = run_eval(agent, cases, BrokenRunner(), None, on_result=lambda rs: saved.append(len(rs)))

    assert [r.infra_error for r in results] == [True, False]
    assert saved == [1, 2]                                    # progress reported after every case
    summary = summarize(results)
    assert summary["infra_errors"] == 1 and summary["scored"] == 1 and summary["pass_rate"] == 1.0
    assert "NOT scored" in render_markdown(summary, results, {"model": "m", "timestamp": "t"})


def test_failed_sql_case_records_reference_rows_for_diagnosis():
    class Runner:
        def run(self, sql, max_rows=None):
            return {"ok": True, "rows": [[23.5]], "columns": ["r"]}

    tool_result = {"ok": True, "rows": [[0.2]], "columns": ["r"]}      # the classic ROUND(fraction, 1) bug
    agent = Agent(ScriptedClient([make_response(tool_calls=[("run_sql", {"sql": "SELECT 0.2"})]),
                                  make_response("The rate is 20%")]),
                  StubToolBox({"run_sql": [tool_result]}), "m")
    [result] = run_eval(agent, [{"id": "s", "category": "sql", "question": "q",
                                 "reference_sql": "SELECT 23.5"}], Runner(), None)
    assert not result.passed and result.sql_match is False
    assert result.reference_rows == [[23.5]] and result.sql_queries == ["SELECT 0.2"]


def _predictor(status):
    return PredictClient("http://api.test", httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(status, json={"status": "ok"}))))


class GoodRunner:
    def run(self, sql, max_rows=None):
        return {"ok": True, "rows": [[1]]}


def test_preflight_reports_problems_before_any_llm_call():
    predict_case = [{"category": "predict"}]
    assert preflight(GoodRunner(), _predictor(200), predict_case) == []
    assert any("503" in p for p in preflight(GoodRunner(), _predictor(503), predict_case))
    assert any("Database" in p for p in preflight(BrokenRunner(), _predictor(200), predict_case))
    # prediction API is only checked when predict cases are selected
    assert preflight(GoodRunner(), _predictor(503), [{"category": "sql"}]) == []


# --------------------------------------------- `values` mode: answers split over several queries
def _values_case():
    return {"id": "s12", "category": "sql", "compare": "values", "question": "q",
            "reference_sql": "SELECT 105.7, 70.2"}


class _RefRunner:
    def run(self, sql, max_rows=None):
        return {"ok": True, "rows": [[105.7, 70.2]], "columns": ["a", "b"]}


def _two_query_agent(first, second, answer):
    results = [{"ok": True, "rows": [[first]], "columns": ["x"]}, {"ok": True, "rows": [[second]], "columns": ["x"]}]
    return Agent(ScriptedClient([make_response(tool_calls=[("run_sql", {"sql": "SELECT 1"})]),
                                 make_response(tool_calls=[("run_sql", {"sql": "SELECT 2"})]),
                                 make_response(answer)]),
                 StubToolBox({"run_sql": results}), "m")


def test_values_mode_accepts_a_correct_answer_split_across_two_queries():
    """Regression: the agent computed 105.7 and 70.2 in two queries and answered correctly."""
    agent = _two_query_agent(105.7, 70.2, "Canceled: 105.7 days, kept: 70.2 days.")
    [r] = run_eval(agent, [_values_case()], _RefRunner(), None)
    assert r.sql_match and r.answer_match and r.passed


def test_values_mode_still_rejects_a_wrong_value():
    agent = _two_query_agent(105.7, 99.9, "Canceled: 105.7 days, kept: 99.9 days.")
    [r] = run_eval(agent, [_values_case()], _RefRunner(), None)
    assert not r.sql_match and not r.passed


def test_rows_mode_is_not_loosened():
    """Default mode must still require ONE query to produce the whole reference result."""
    case = {**_values_case(), "compare": "rows"}
    agent = _two_query_agent(105.7, 70.2, "Canceled: 105.7 days, kept: 70.2 days.")
    [r] = run_eval(agent, [case], _RefRunner(), None)
    assert not r.sql_match


def test_tool_trace_is_recorded_for_diagnosis():
    agent = Agent(ScriptedClient([make_response(tool_calls=[("predict_cancellation", {"hotel": "City Hotel"})]),
                                  make_response("Which dates?")]),
                  StubToolBox({"predict_cancellation": [{"ok": False, "error": "Missing required fields: adr."}]}),
                  "m")

    class Predictor:
        def predict(self, **kw):
            return {"ok": True, "cancel_probability": 0.5}

    [r] = run_eval(agent, [{"id": "p", "category": "predict", "question": "q", "expected_payload": {}}],
                   GoodRunner(), Predictor())
    assert not r.passed and r.detail == "model never called successfully"
    assert r.tool_trace == [{"name": "predict_cancellation", "arguments": {"hotel": "City Hotel"},
                             "ok": False, "error": "Missing required fields: adr."}]
