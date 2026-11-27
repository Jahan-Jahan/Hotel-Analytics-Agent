"""Evaluation harness.   python -m hotel_agent.evaluation [--category sql] [--ids s01,s02] [--limit N]

Measures *behaviour*, not vibes: reference answers are computed from the database at run time,
predictions are checked against a direct call to the prediction API, and refusals are checked
from the tool-call trace.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

from hotel_agent.agent import Agent, AgentResult, build_agent
from hotel_agent.config import ROOT, ConfigError, Settings
from hotel_agent.tools.predict_tool import PredictClient
from hotel_agent.tools.sql_tool import SqlRunner

QUESTIONS_PATH = ROOT / "evals" / "questions.yaml"
RESULTS_DIR = ROOT / "evals" / "results"
ABS_TOL = 0.051          # a result rounded to 1-2 decimals still matches
REL_TOL = 1e-3
PROB_TOL = 0.01          # predict: |agent - direct API call|

_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩٫٬", "01234567890123456789.,")


# ----------------------------------------------------------------------------- comparison helpers
def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def cell_close(ref: Any, got: Any) -> bool:
    if _is_num(ref) and _is_num(got):
        return abs(ref - got) <= max(ABS_TOL, REL_TOL * abs(ref))
    return str(ref).strip().casefold() == str(got).strip().casefold()


def _row_covers(ref_row: list, got_row: list) -> bool:
    """Every reference cell is matched by a distinct cell of the candidate row."""
    pool = list(got_row)
    for cell in ref_row:
        for i, other in enumerate(pool):
            if cell_close(cell, other):
                pool.pop(i)
                break
        else:
            return False
    return True


def rows_match(ref_rows: list[list], got_rows: list[list]) -> bool:
    """Same number of rows; column order, extra columns and row order are ignored."""
    if len(ref_rows) != len(got_rows):
        return False
    unused = list(range(len(got_rows)))
    for ref in ref_rows:
        for i in unused:
            if _row_covers(ref, got_rows[i]):
                unused.remove(i)
                break
        else:
            return False
    return True


def values_match(ref_rows: list[list], got_rows: list[list]) -> bool:
    """Looser: all reference values appear somewhere in the candidate result."""
    flat = [c for row in got_rows for c in row]
    return _row_covers([c for row in ref_rows for c in row], flat)


def extract_numbers(text: str) -> list[float]:
    text = text.translate(_DIGIT_MAP)
    found = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return [float(f.replace(",", "")) for f in found]


def answer_states(ref_rows: list[list], answer: str) -> bool | None:
    """For single-row references: does the answer text state every reference value?"""
    if len(ref_rows) != 1:
        return None
    numbers, lowered = extract_numbers(answer), answer.translate(_DIGIT_MAP).casefold()
    for cell in ref_rows[0]:
        if _is_num(cell):
            if not any(cell_close(cell, n) for n in numbers):
                return False
        elif str(cell).strip().casefold() not in lowered:
            return False
    return True


class InfrastructureError(RuntimeError):
    """The database or prediction API failed: not the model's fault, so never scored as a failure."""


# ----------------------------------------------------------------------------- per-case evaluation
@dataclass
class CaseResult:
    id: str
    category: str
    question: str
    passed: bool
    detail: str = ""
    sql_match: bool | None = None
    answer_match: bool | None = None
    steps: int = 0
    tool_calls: int = 0
    sql_errors: int = 0
    blocked: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    answer: str = ""
    error: str | None = None
    sql_queries: list[str] = field(default_factory=list)
    reference_rows: list | None = None      # first reference rows (sql cases), for diagnosing failures
    tool_trace: list[dict] = field(default_factory=list)   # every tool call: name, arguments, ok, error
    infra_error: bool = False


def _ok_calls(result: AgentResult, name: str) -> list:
    return [c for c in result.tool_calls if c.name == name and c.result.get("ok")]


def evaluate_case(agent: Agent, case: dict, runner: SqlRunner, predictor: PredictClient) -> CaseResult:
    agent.reset()
    result = agent.ask(case["question"])
    category = case["category"]
    out = CaseResult(
        id=case["id"], category=category, question=case["question"], passed=False,
        steps=result.steps, tool_calls=len(result.tool_calls),
        prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
        latency_s=round(result.latency_s, 2), answer=result.answer, error=result.error,
        sql_queries=[c.arguments.get("sql", "") for c in result.tool_calls if c.arguments.get("sql")],
        tool_trace=[{"name": c.name, "arguments": c.arguments, "ok": bool(c.result.get("ok")),
                     "error": c.result.get("error")} for c in result.tool_calls],
    )
    for call in result.tool_calls:
        if not call.result.get("ok"):
            if str(call.result.get("error", "")).startswith("Blocked by guardrails"):
                out.blocked += 1
            elif call.name == "run_sql":
                out.sql_errors += 1

    if not result.completed:
        out.detail = f"agent did not finish ({result.error})"
        return out

    if category == "sql":
        ref = runner.run(case["reference_sql"], max_rows=1000)
        if not ref["ok"]:
            raise InfrastructureError(f"Reference SQL for {case['id']} failed: {ref['error']}")
        out.reference_rows = ref["rows"][:5]
        result_sets = [c.result["rows"] for c in _ok_calls(result, "run_sql")]
        if case.get("compare") == "values":
            # Shape-agnostic: the reference values may be spread over several queries
            # (e.g. one query per group), so pool the rows of all successful queries.
            out.sql_match = values_match(ref["rows"], [row for rows in result_sets for row in rows])
        else:
            out.sql_match = any(rows_match(ref["rows"], rows) for rows in result_sets)
        out.answer_match = answer_states(ref["rows"], result.answer)
        out.passed = bool(out.sql_match) and out.answer_match is not False
        out.detail = ("ok" if out.passed else
                      "no query matched the reference" if not out.sql_match else
                      "answer text does not state the reference value(s)")

    elif category == "chart":
        charts = _ok_calls(result, "make_chart")
        out.passed = bool(charts) and Path(charts[-1].result["path"]).exists() \
            and charts[-1].result["points"] >= 12
        out.detail = "ok" if out.passed else "no valid chart produced"

    elif category == "predict":
        direct = predictor.predict(**case["expected_payload"])
        calls = _ok_calls(result, "predict_cancellation")
        if not direct["ok"]:
            raise InfrastructureError(f"Direct prediction for {case['id']} failed: {direct['error']}")
        out.passed = any(abs(c.result["cancel_probability"] - direct["cancel_probability"]) <= PROB_TOL
                         for c in calls)
        out.detail = ("ok" if out.passed else
                      "model never called successfully" if not calls else
                      "inputs differ from the described booking "
                      f"(expected p={direct['cancel_probability']:.3f}, "
                      f"got {[c.result['cancel_probability'] for c in calls]})")

    elif category == "refusal":
        succeeded = [c.name for c in result.tool_calls if c.result.get("ok")]
        out.passed = not succeeded and bool(result.answer)
        out.detail = "ok" if out.passed else f"used tools successfully: {succeeded}"

    else:
        raise ValueError(f"Unknown category {category!r}")
    return out


# ----------------------------------------------------------------------------- aggregation / reporting
def summarize(results: list[CaseResult]) -> dict[str, Any]:
    scored = [r for r in results if not r.infra_error]
    by_cat: dict[str, list[CaseResult]] = defaultdict(list)
    for r in scored:
        by_cat[r.category].append(r)
    sql_cases = by_cat.get("sql", [])
    n = max(len(scored), 1)
    return {
        "cases": len(results),
        "scored": len(scored),
        "infra_errors": len(results) - len(scored),
        "passed": sum(r.passed for r in scored),
        "pass_rate": round(sum(r.passed for r in scored) / n, 3),
        "by_category": {c: {"cases": len(rs), "passed": sum(r.passed for r in rs),
                            "pass_rate": round(sum(r.passed for r in rs) / len(rs), 3)}
                        for c, rs in by_cat.items()},
        "sql_execution_accuracy": round(sum(bool(r.sql_match) for r in sql_cases) / len(sql_cases), 3)
        if sql_cases else None,
        "avg_steps": round(sum(r.steps for r in scored) / n, 2),
        "avg_tool_calls": round(sum(r.tool_calls for r in scored) / n, 2),
        "avg_latency_s": round(sum(r.latency_s for r in scored) / n, 2),
        "total_tokens": sum(r.prompt_tokens + r.completion_tokens for r in scored),
        "sql_errors": sum(r.sql_errors for r in scored),
        "guardrail_blocks": sum(r.blocked for r in scored),
    }


def render_markdown(summary: dict, results: list[CaseResult], meta: dict) -> str:
    lines = [f"# Evaluation run: {meta['model']}", "",
             f"- Date (UTC): {meta['timestamp']}", f"- Cases scored: {summary['scored']} of {summary['cases']}",
             f"- **Overall pass rate: {summary['pass_rate']:.0%}** ({summary['passed']}/{summary['scored']})",
             f"- SQL execution accuracy: {summary['sql_execution_accuracy']}",
             f"- Avg steps / tool calls / latency: {summary['avg_steps']} / {summary['avg_tool_calls']} / "
             f"{summary['avg_latency_s']}s", f"- Total tokens: {summary['total_tokens']}",
             f"- SQL errors (self-corrected or not): {summary['sql_errors']}, guardrail blocks: "
             f"{summary['guardrail_blocks']}", "", "| Category | Passed | Rate |", "|---|---|---|"]
    for cat, s in summary["by_category"].items():
        lines.append(f"| {cat} | {s['passed']}/{s['cases']} | {s['pass_rate']:.0%} |")
    if summary["infra_errors"]:
        lines += ["", f"**Warning:** {summary['infra_errors']} case(s) were NOT scored because the database "
                      f"or prediction API failed (infrastructure, not the model)."]
    lines += ["", "## Failures", ""]
    failures = [r for r in results if not r.passed and not r.infra_error]
    lines += [f"- `{r.id}` ({r.category}): {r.detail}" for r in failures] or ["None."]
    return "\n".join(lines) + "\n"


def _print_failure(r: CaseResult) -> None:
    for sql in r.sql_queries[:2]:
        print("         agent SQL :", " ".join(sql.split())[:220])
    if r.category in ("predict", "chart"):
        for call in r.tool_trace:
            args = json.dumps(call["arguments"], ensure_ascii=False)
            print(f"         tool call : {call['name']}({args[:260]}) -> "
                  f"{'ok' if call['ok'] else 'ERROR: ' + str(call['error'])[:200]}")
    if r.reference_rows is not None:
        print("         reference :", r.reference_rows)
    print("         answer    :", " ".join(r.answer.split())[:220])


def preflight(runner: SqlRunner, predictor: PredictClient, cases: list[dict]) -> list[str]:
    """Check the infrastructure BEFORE spending LLM calls. Returns a list of problems."""
    problems = []
    check = runner.run("SELECT 1 FROM bookings LIMIT 1")
    if not check["ok"]:
        problems.append(f"Database check failed: {check['error']}  (did you run `make db-role`?)")
    if any(c["category"] == "predict" for c in cases):
        try:
            response = predictor.client.get(f"{predictor.base_url}/health")
            if response.status_code != 200:
                problems.append(f"Prediction API {predictor.base_url}/health returned HTTP "
                                f"{response.status_code} (is `make serve` running in Hotel-Booking-Demand?)")
        except httpx.HTTPError as exc:
            problems.append(f"Prediction API unreachable at {predictor.base_url}: {type(exc).__name__}")
    return problems


def run_eval(agent: Agent, cases: list[dict], runner: SqlRunner, predictor: PredictClient,
             verbose: bool = False, on_result=None) -> list[CaseResult]:
    results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        try:
            r = evaluate_case(agent, case, runner, predictor)
        except InfrastructureError as exc:
            r = CaseResult(id=case["id"], category=case["category"], question=case["question"],
                           passed=False, detail=f"INFRASTRUCTURE ERROR: {exc}", infra_error=True)
        results.append(r)
        if on_result:
            on_result(results)               # lets the caller save progress after every case
        if verbose:
            status = "SKIP" if r.infra_error else "PASS" if r.passed else "FAIL"
            print(f"[{i:>2}/{len(cases)}] {status} {r.id:<28} {r.steps} steps, {r.latency_s:>5.1f}s  "
                  f"{'' if r.passed else r.detail}")
            if not r.passed and not r.infra_error:
                _print_failure(r)
    return results


def load_cases(path: Path = QUESTIONS_PATH) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the hotel analytics agent")
    parser.add_argument("--category", choices=["sql", "chart", "predict", "refusal"])
    parser.add_argument("--ids", help="comma-separated case ids")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    settings = Settings.from_env()
    try:
        agent = build_agent(settings)
    except ConfigError as exc:
        raise SystemExit(str(exc))

    cases = load_cases()
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.ids:
        wanted = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit:
        cases = cases[: args.limit]

    runner, predictor = agent.toolbox.runner, agent.toolbox.predictor
    problems = preflight(runner, predictor, cases)
    if problems:
        raise SystemExit("Cannot start the evaluation:\n  - " + "\n  - ".join(problems))

    meta = {"model": settings.llm_model, "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", settings.llm_model)
    json_path = RESULTS_DIR / f"{stamp}_{safe_model}.json"

    def save(partial: list[CaseResult]) -> None:      # called after every case: a crash loses nothing
        json_path.write_text(json.dumps(
            {"meta": meta, "summary": summarize(partial), "cases": [asdict(r) for r in partial]},
            indent=2, ensure_ascii=False), encoding="utf-8")

    results = run_eval(agent, cases, runner, predictor, verbose=True, on_result=save)
    summary = summarize(results)
    markdown = render_markdown(summary, results, meta)
    (RESULTS_DIR / "latest.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"Details: {json_path}")


if __name__ == "__main__":
    main()
