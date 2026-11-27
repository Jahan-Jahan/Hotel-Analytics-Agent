import pytest

from hotel_agent.guardrails import GuardrailError, validate_sql, wrap_with_limit

ALLOWED = [
    "SELECT hotel, AVG(is_canceled) FROM bookings GROUP BY hotel;",
    "WITH m AS (SELECT year_month, COUNT(*) n FROM bookings GROUP BY 1) "
    "SELECT *, LAG(n) OVER (ORDER BY year_month) FROM m",
    "SELECT 1 UNION ALL SELECT 2",
    "select round(avg(adr)::numeric, 2) from bookings where is_canceled = 0 -- trailing comment",
    "SELECT COUNT(*) FILTER (WHERE is_canceled = 1) FROM bookings",
]

BLOCKED = [
    ("DROP TABLE bookings", "read-only"),
    ("DELETE FROM bookings", "read-only"),
    ("UPDATE bookings SET adr = 0", "read-only"),
    ("INSERT INTO bookings SELECT * FROM bookings", "read-only"),
    ("SELECT 1; DROP TABLE bookings", "one SQL statement"),
    ("SELECT * FROM hotel_bookings", "not available"),
    ("SELECT * FROM pg_catalog.pg_user", "Schema-qualified"),
    ("SELECT * FROM information_schema.tables", "Schema-qualified"),
    ("SELECT * INTO leaked FROM bookings", "Into"),
    ("SELECT pg_sleep(30)", "not allowed"),
    ("SELECT set_config('x', 'y', false)", "not allowed"),
    ("SELECT * FROM generate_series(1, 10)", "Table-valued"),
    ("SELECT * FROM bookings FOR UPDATE", "Lock"),
    ("COPY bookings TO '/tmp/out.csv'", "read-only"),
    ("", "Empty"),
    ("hello world", "read-only"),
    ("SELECT * FROM (SELECT * FROM pg_shadow) s", "not available"),
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allowed_queries_pass(sql):
    cleaned = validate_sql(sql)
    assert cleaned and not cleaned.endswith(";")


@pytest.mark.parametrize("sql,reason", BLOCKED)
def test_dangerous_queries_are_blocked(sql, reason):
    with pytest.raises(GuardrailError, match=reason):
        validate_sql(sql)


def test_cte_name_is_not_mistaken_for_a_forbidden_table():
    assert validate_sql("WITH hotel_bookings AS (SELECT * FROM bookings) SELECT * FROM hotel_bookings")


def test_wrapping_neutralises_stacked_statements_and_trailing_comments():
    wrapped = wrap_with_limit("SELECT 1 -- note", 50)
    assert wrapped.startswith("SELECT * FROM (")
    assert wrapped.endswith("LIMIT 51")
    assert "-- note\n)" in wrapped            # the closing paren is on its own line
