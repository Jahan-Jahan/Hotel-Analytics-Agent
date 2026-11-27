"""Tool registry: JSON schemas for the LLM + a dispatcher that never raises."""
from __future__ import annotations

import json
from typing import Any, Callable

from hotel_agent.config import Settings
from hotel_agent.tools import chart_tool
from hotel_agent.tools.predict_tool import MONTHS, OPTIONAL, REQUIRED, PredictClient
from hotel_agent.tools.sql_tool import SqlRunner

_PREDICT_PROPS: dict[str, dict] = {
    "hotel": {"type": "string", "enum": ["City Hotel", "Resort Hotel"],
              "description": "Required. The hotel named by the user."},
    "lead_time": {"type": "integer", "description": "Days between booking and arrival."},
    "arrival_date_year": {"type": "integer"},
    "arrival_date_month": {"type": "string", "enum": MONTHS},
    "arrival_date_day_of_month": {"type": "integer"},
    "stays_in_weekend_nights": {"type": "integer"},
    "stays_in_week_nights": {"type": "integer"},
    "adults": {"type": "integer"},
    "adr": {"type": "number", "description": "Average daily rate in EUR."},
    "children": {"type": "number"},
    "babies": {"type": "integer"},
    "meal": {"type": "string", "enum": ["BB", "HB", "FB", "SC", "Undefined"]},
    "country": {"type": "string", "description": "ISO 3166 alpha-3 code, e.g. PRT, GBR."},
    "market_segment": {"type": "string"},
    "distribution_channel": {"type": "string"},
    "is_repeated_guest": {"type": "integer", "enum": [0, 1]},
    "previous_cancellations": {"type": "integer"},
    "previous_bookings_not_canceled": {"type": "integer"},
    "reserved_room_type": {"type": "string"},
    "booking_changes": {"type": "integer"},
    "deposit_type": {"type": "string", "enum": ["No Deposit", "Non Refund", "Refundable"]},
    "agent": {"type": "number"},
    "company": {"type": "number"},
    "customer_type": {"type": "string", "enum": ["Transient", "Contract", "Transient-Party", "Group"]},
    "required_car_parking_spaces": {"type": "integer"},
    "total_of_special_requests": {"type": "integer"},
}

TOOL_SCHEMAS: list[dict] = [
    {"type": "function", "function": {
        "name": "run_sql",
        "description": ("Run ONE read-only PostgreSQL SELECT on the `bookings` view and get rows back. "
                        "Results are capped; aggregate in SQL instead of fetching raw rows."),
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
    }},
    {"type": "function", "function": {
        "name": "predict_cancellation",
        "description": ("Predict the cancellation probability of ONE specific, hypothetical booking with the "
                        "trained ML model. Use only when the user describes a concrete booking. Map EVERY detail "
                        "the user gives to a field, including optional ones (children, deposit_type, "
                        "market_segment, previous_cancellations, is_repeated_guest, special requests...): a "
                        "dropped detail silently changes the prediction. Use the exact field names."),
        "parameters": {"type": "object", "properties": _PREDICT_PROPS, "required": REQUIRED},
    }},
    {"type": "function", "function": {
        "name": "make_chart",
        "description": ("Plot the result of a SELECT query as a PNG chart. Give the SQL and the column names; "
                        "do not pass data values. Returns the image path."),
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "chart_type": {"type": "string", "enum": list(chart_tool.CHART_TYPES)},
            "sql": {"type": "string"},
            "x_column": {"type": "string"},
            "y_column": {"type": "string"},
            "series_column": {"type": "string", "description": "Optional: one line/bar group per value."},
        }, "required": ["title", "chart_type", "sql", "x_column", "y_column"]},
    }},
]


class ToolBox:
    def __init__(self, settings: Settings, runner: SqlRunner | None = None,
                 predictor: PredictClient | None = None):
        self.settings = settings
        self.runner = runner or SqlRunner(settings)
        self.predictor = predictor or PredictClient(settings.predict_api_url)
        self.schemas = TOOL_SCHEMAS
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "run_sql": lambda sql: self.runner.run(sql),
            "predict_cancellation": lambda **kw: self.predictor.predict(**kw),
            "make_chart": lambda **kw: chart_tool.make_chart(self.runner, settings.chart_dir, **kw),
        }

    def call(self, name: str, raw_arguments: str | None) -> dict[str, Any]:
        """Execute a tool call. Always returns a dict; errors become {"ok": False, ...}."""
        handler = self._handlers.get(name)
        if handler is None:
            return {"ok": False, "error": f"Unknown tool '{name}'."}
        try:
            args = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"Arguments are not valid JSON: {exc}", "retryable": True}
        if not isinstance(args, dict):
            return {"ok": False, "error": "Arguments must be a JSON object."}
        try:
            return handler(**args)
        except TypeError as exc:
            return {"ok": False, "error": f"Bad arguments for {name}: {exc}", "retryable": True}
        except Exception as exc:  # a tool bug must not crash the agent loop
            return {"ok": False, "error": f"Tool '{name}' failed: {type(exc).__name__}: {exc}"}
