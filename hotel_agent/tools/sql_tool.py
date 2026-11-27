"""run_sql tool: execute a validated read-only query and return JSON-safe rows."""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from hotel_agent.config import Settings
from hotel_agent.guardrails import GuardrailError, validate_sql, wrap_with_limit


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value


class SqlRunner:
    def __init__(self, settings: Settings, engine: Engine | None = None):
        self.settings = settings
        self.engine = engine or create_engine(
            settings.database_url,
            future=True,
            pool_pre_ping=True,
            connect_args={
                "options": f"-c statement_timeout={settings.sql_timeout_ms} "
                           f"-c default_transaction_read_only=on"
            },
        )

    def run(self, sql: str, max_rows: int | None = None) -> dict[str, Any]:
        """Validate, execute and return {"ok", "columns", "rows", ...} or {"ok": False, "error"}."""
        limit = max_rows or self.settings.max_rows
        try:
            clean = validate_sql(sql)
        except GuardrailError as exc:
            return {"ok": False, "error": f"Blocked by guardrails: {exc}"}

        try:
            with self.engine.connect() as con:
                result = con.exec_driver_sql(wrap_with_limit(clean, limit))
                columns = list(result.keys())
                fetched = result.fetchall()
        except DBAPIError as exc:  # SQL errors (bad column, timeout, permission, ...)
            message = str(getattr(exc, "orig", exc)).strip().splitlines()[0]
            hint = ""
            if "round(double precision" in message.lower():
                hint = " Hint: cast to numeric first, e.g. ROUND(AVG(adr)::numeric, 2)."
            return {"ok": False, "error": f"Database error: {message[:300]}.{hint}"}
        except SQLAlchemyError as exc:
            return {"ok": False, "error": f"Database unavailable: {str(exc)[:200]}"}

        truncated = len(fetched) > limit
        rows = [[_jsonable(c) for c in row] for row in fetched[:limit]]
        return {
            "ok": True,
            "sql": clean,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
        }
