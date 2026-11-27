"""Settings, read from environment variables (and a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class ConfigError(RuntimeError):
    """Raised when a required setting is missing."""


@dataclass(frozen=True)
class Settings:
    # LLM: any OpenAI-compatible endpoint (GapGPT, Groq, OpenRouter, Ollama, ...)
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    # Data: a READ-ONLY database user (see db/readonly_role.sql)
    database_url: str
    admin_database_url: str          # only used by `make db-role`
    # Prediction service from the Hotel-Booking-Demand project
    predict_api_url: str
    # Safety / cost limits
    sql_timeout_ms: int
    max_rows: int
    max_steps: int
    chart_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.getenv
        return cls(
            llm_base_url=env("LLM_BASE_URL", "https://api.gapgpt.app/v1"),
            llm_api_key=env("LLM_API_KEY", ""),
            llm_model=env("LLM_MODEL", "gpt-4o-mini"),
            database_url=env("DATABASE_URL", "postgresql+psycopg://agent_ro:agent_ro@localhost:5433/hotel"),
            admin_database_url=env("ADMIN_DATABASE_URL", "postgresql+psycopg://hotel:hotel@localhost:5433/hotel"),
            predict_api_url=env("PREDICT_API_URL", "http://localhost:8000"),
            sql_timeout_ms=int(env("SQL_TIMEOUT_MS", "5000")),
            max_rows=int(env("MAX_ROWS", "50")),
            max_steps=int(env("MAX_STEPS", "8")),
            chart_dir=Path(env("CHART_DIR", str(ROOT / "outputs" / "charts"))),
        )

    def require_llm(self) -> None:
        if not self.llm_api_key:
            raise ConfigError("LLM_API_KEY is not set. Copy .env.example to .env and fill it in.")
