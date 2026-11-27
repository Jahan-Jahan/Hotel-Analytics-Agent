"""make_chart tool: run a query and plot two columns. Data never passes through the LLM."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from hotel_agent.tools.sql_tool import SqlRunner

CHART_TYPES = ("bar", "barh", "line")
MAX_CHART_ROWS = 200


def make_chart(runner: SqlRunner, chart_dir: Path, *, title: str, chart_type: str, sql: str,
               x_column: str, y_column: str, series_column: str | None = None) -> dict[str, Any]:
    if chart_type not in CHART_TYPES:
        return {"ok": False, "error": f"chart_type must be one of {CHART_TYPES}."}

    result = runner.run(sql, max_rows=MAX_CHART_ROWS)
    if not result["ok"]:
        return result
    frame = pd.DataFrame(result["rows"], columns=result["columns"])
    needed = [c for c in (x_column, y_column, series_column) if c]
    absent = [c for c in needed if c not in frame.columns]
    if absent:
        return {"ok": False, "error": f"Columns {absent} not in query result {list(frame.columns)}."}
    if frame.empty:
        return {"ok": False, "error": "The query returned no rows, nothing to plot."}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        if series_column:
            data = frame.pivot(index=x_column, columns=series_column, values=y_column)
        else:
            data = frame.set_index(x_column)[[y_column]]
        kind = "bar" if chart_type == "bar" else chart_type
        data.plot(kind=kind, ax=ax, marker="o" if chart_type == "line" else None)
    except (ValueError, TypeError) as exc:
        plt.close(fig)
        return {"ok": False, "error": f"Could not plot these columns: {exc}"}
    ax.set_title(title)
    ax.set_xlabel(x_column if chart_type != "barh" else y_column)
    ax.set_ylabel(y_column if chart_type != "barh" else x_column)
    if not series_column:
        ax.get_legend() and ax.get_legend().remove()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()

    chart_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40] or "chart"
    path = chart_dir / f"{slug}_{time.time_ns() // 1_000_000}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"ok": True, "path": str(path), "points": int(len(frame))}
