"""SQL guardrails: the first of several defence layers for LLM-written SQL.

Layers (outermost first):
  1. This module: parse the query and allow ONE read-only SELECT over the
     whitelisted view, with no dangerous functions.
  2. Execution wrapper: the validated query is run as a sub-select with an outer
     LIMIT, so a stacked statement ("...; DROP ...") is a syntax error in Postgres.
  3. Database: a role that can only SELECT from `bookings`, in a read-only
     transaction with a statement timeout (db/readonly_role.sql).
The LLM is never trusted: layers 2 and 3 still hold if this parser is fooled.
"""
from __future__ import annotations

import sqlglot
from sqlglot import exp

ALLOWED_TABLES = frozenset({"bookings"})

_SET_OP = getattr(exp, "SetOperation", exp.Union)

_FORBIDDEN_NODES = tuple(
    getattr(exp, name)
    for name in ("Insert", "Update", "Delete", "Create", "Drop", "Alter", "Merge",
                 "Command", "Into", "Lock", "Copy", "TruncateTable", "Grant", "Set")
    if hasattr(exp, name)
)

_FORBIDDEN_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink")
_FORBIDDEN_FUNCTIONS = frozenset({
    "set_config", "current_setting", "txid_current", "query_to_xml",
    "table_to_xml", "database_to_xml", "xmlparse", "nextval", "setval",
})


class GuardrailError(ValueError):
    """The query was rejected. The message is shown to the LLM so it can fix it."""


def validate_sql(sql: str, allowed_tables: frozenset[str] = ALLOWED_TABLES) -> str:
    """Return the cleaned query (no trailing ';') or raise GuardrailError."""
    if not sql or not sql.strip():
        raise GuardrailError("Empty query.")

    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except sqlglot.errors.SqlglotError as exc:
        raise GuardrailError(f"Could not parse the SQL: {str(exc)[:200]}") from exc

    if len(statements) != 1:
        raise GuardrailError("Exactly one SQL statement is allowed.")
    tree = statements[0]

    if not isinstance(tree, (exp.Select, _SET_OP)):
        raise GuardrailError("Only read-only SELECT queries are allowed.")

    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise GuardrailError(f"Forbidden SQL construct: {type(node).__name__}.")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if table.db or table.catalog:
            raise GuardrailError("Schema-qualified tables are not allowed; use `bookings` only.")
        if not isinstance(table.this, exp.Identifier):
            raise GuardrailError("Table-valued functions are not allowed in FROM.")
        name = table.name.lower()
        if name not in allowed_tables and name not in cte_names:
            raise GuardrailError(
                f"Table '{table.name}' is not available. You can only query: "
                f"{', '.join(sorted(allowed_tables))}."
            )

    for func in tree.find_all(exp.Func):
        name = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()
        if name.startswith(_FORBIDDEN_FUNCTION_PREFIXES) or name in _FORBIDDEN_FUNCTIONS:
            raise GuardrailError(f"Function '{name}' is not allowed.")

    return sql.strip().rstrip(";").strip()


def wrap_with_limit(clean_sql: str, max_rows: int) -> str:
    """Run the query as a sub-select with an outer LIMIT.

    Fetching max_rows + 1 lets the caller detect truncation. The newline before
    the closing parenthesis keeps a trailing `-- comment` from swallowing it.
    """
    return f"SELECT * FROM (\n{clean_sql}\n) AS _q LIMIT {int(max_rows) + 1}"
