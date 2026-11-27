"""UI logic that does not need Streamlit (so it can be unit-tested)."""
from __future__ import annotations

from typing import Any

import httpx

from hotel_agent.agent import AgentResult, ToolCallRecord
from hotel_agent.config import Settings

EXAMPLE_QUESTIONS = [
    "Which hotel has the higher cancellation rate?",
    "Chart the monthly cancellation rate over time",
    "Which market segment cancels the most (min 1000 bookings)?",
    "How much revenue did cancellations cost us in 2016?",
    "How likely is a City Hotel guest to cancel if they booked 200 days ahead for 10 August 2017, "
    "2 weekend + 3 week nights, 2 adults, via Online TA, ADR 120, no special requests?",
    "نرخ کنسلی رزروهایی که ودیعه Non Refund دارن چند درصده؟",
]


def check_services(settings: Settings, toolbox: Any) -> dict[str, tuple[bool, str]]:
    """Status of each dependency as {name: (ok, short message)}."""
    status: dict[str, tuple[bool, str]] = {}
    db = toolbox.runner.run("SELECT COUNT(*) AS n FROM bookings")
    status["Database"] = (True, f"{db['rows'][0][0]:,} bookings") if db["ok"] else (False, "unavailable")
    try:
        response = toolbox.predictor.client.get(f"{toolbox.predictor.base_url}/health")
        status["Prediction API"] = (response.status_code == 200, "ready" if response.status_code == 200
                                    else f"HTTP {response.status_code}")
    except httpx.HTTPError:
        status["Prediction API"] = (False, "unreachable")
    status["LLM"] = (bool(settings.llm_api_key), settings.llm_model if settings.llm_api_key else "no API key")
    return status


def describe_call(call: ToolCallRecord) -> dict[str, Any]:
    """Turn a tool call into plain data the UI can render."""
    result = call.result
    info: dict[str, Any] = {"tool": call.name, "ok": bool(result.get("ok")),
                            "error": result.get("error"), "seconds": round(call.duration_s, 2)}
    if call.name in ("run_sql", "make_chart"):
        info["sql"] = result.get("sql") or call.arguments.get("sql", "")
    if call.name == "run_sql" and info["ok"]:
        info.update(columns=result["columns"], rows=result["rows"], truncated=result["truncated"])
    elif call.name == "predict_cancellation":
        info["inputs"] = call.arguments
        if info["ok"]:
            info.update(probability=result["cancel_probability"], will_cancel=result["will_cancel"],
                        threshold=result["threshold_used"])
    elif call.name == "make_chart" and info["ok"]:
        info["path"] = result["path"]
    return info


def make_payload(result: AgentResult) -> dict[str, Any]:
    """Everything the UI needs to (re)draw one assistant message."""
    return {
        "role": "assistant",
        "content": result.answer,
        "calls": [describe_call(c) for c in result.tool_calls],
        "charts": result.charts,
        "stats": {"steps": result.steps, "tool_calls": len(result.tool_calls),
                  "tokens": result.prompt_tokens + result.completion_tokens,
                  "seconds": round(result.latency_s, 1)},
        "error": result.error,
    }
