"""Create/refresh the read-only agent role.  Usage: python -m hotel_agent.setup_db"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine

from hotel_agent.config import ROOT, Settings

ROLE_SQL = ROOT / "db" / "readonly_role.sql"


def apply_readonly_role(admin_url: str, sql_path: Path = ROLE_SQL) -> None:
    engine = create_engine(admin_url, future=True)
    with engine.begin() as con:
        con.exec_driver_sql(sql_path.read_text())
    engine.dispose()


def main() -> None:
    apply_readonly_role(Settings.from_env().admin_database_url)
    print("Read-only role 'agent_ro' is ready (SELECT on bookings only).")


if __name__ == "__main__":
    main()
