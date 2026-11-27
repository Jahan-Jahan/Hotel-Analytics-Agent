import httpx

from hotel_agent.agent import AgentResult, ToolCallRecord
from hotel_agent.config import Settings
from hotel_agent.tools.predict_tool import PredictClient
from hotel_agent.ui_helpers import check_services, describe_call, make_payload


def test_describe_sql_call_success_and_failure():
    ok = describe_call(ToolCallRecord("run_sql", {"sql": "SELECT 1"}, {
        "ok": True, "sql": "SELECT 1", "columns": ["x"], "rows": [[1]], "truncated": False}, 0.123))
    assert ok["ok"] and ok["sql"] == "SELECT 1" and ok["rows"] == [[1]] and ok["seconds"] == 0.12
    bad = describe_call(ToolCallRecord("run_sql", {"sql": "DROP x"}, {"ok": False, "error": "Blocked"}, 0.0))
    assert not bad["ok"] and bad["error"] == "Blocked" and bad["sql"] == "DROP x" and "rows" not in bad


def test_describe_predict_and_chart():
    pred = describe_call(ToolCallRecord("predict_cancellation", {"hotel": "City Hotel"}, {
        "ok": True, "cancel_probability": 0.77, "will_cancel": True, "threshold_used": 0.21}, 0.5))
    assert pred["probability"] == 0.77 and pred["inputs"] == {"hotel": "City Hotel"} and pred["will_cancel"]
    chart = describe_call(ToolCallRecord("make_chart", {"sql": "SELECT 1"}, {"ok": True, "path": "c.png"}, 0.1))
    assert chart["path"] == "c.png" and chart["sql"] == "SELECT 1"


def test_make_payload_collects_stats_and_charts():
    result = AgentResult(answer="Done", completed=True, steps=2, prompt_tokens=100, completion_tokens=20,
                         latency_s=3.14159, tool_calls=[
                             ToolCallRecord("make_chart", {"sql": "s"}, {"ok": True, "path": "a.png"}, 0.1)])
    payload = make_payload(result)
    assert payload["content"] == "Done" and payload["charts"] == ["a.png"]
    assert payload["stats"] == {"steps": 2, "tool_calls": 1, "tokens": 120, "seconds": 3.1}


class _Runner:
    def __init__(self, ok):
        self.ok = ok

    def run(self, sql, max_rows=None):
        return {"ok": True, "rows": [[87229]]} if self.ok else {"ok": False, "error": "down"}


class _Box:
    def __init__(self, db_ok, api_status):
        self.runner = _Runner(db_ok)
        handler = (lambda r: httpx.Response(api_status)) if api_status else (
            lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")))
        self.predictor = PredictClient("http://api.test", httpx.Client(transport=httpx.MockTransport(handler)))


def test_check_services():
    settings = Settings.from_env()
    healthy = check_services(settings, _Box(True, 200))
    assert healthy["Database"] == (True, "87,229 bookings") and healthy["Prediction API"][0]
    broken = check_services(settings, _Box(False, None))
    assert broken["Database"][0] is False and broken["Prediction API"] == (False, "unreachable")
    assert check_services(settings, _Box(True, 503))["Prediction API"] == (False, "HTTP 503")
