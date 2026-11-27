import json

from hotel_agent.config import Settings
from hotel_agent.tools import TOOL_SCHEMAS, ToolBox


class FakeRunner:
    def run(self, sql, max_rows=None):
        return {"ok": True, "rows": [[1]], "sql": sql}


def _box():
    return ToolBox(Settings.from_env(), runner=FakeRunner(), predictor=object())


def test_dispatch_and_error_handling():
    box = _box()
    assert box.call("run_sql", json.dumps({"sql": "SELECT 1"}))["ok"]
    assert "Unknown tool" in box.call("drop_database", "{}")["error"]
    assert "not valid JSON" in box.call("run_sql", "{oops")["error"]
    assert "JSON object" in box.call("run_sql", "[1, 2]")["error"]
    assert "Bad arguments" in box.call("run_sql", json.dumps({"wrong": 1}))["error"]


def test_schemas_are_valid_and_required_fields_exist():
    names = [t["function"]["name"] for t in TOOL_SCHEMAS]
    assert names == ["run_sql", "predict_cancellation", "make_chart"]
    predict = TOOL_SCHEMAS[1]["function"]["parameters"]
    assert set(predict["required"]) <= set(predict["properties"])
