"""Configuration shared by the Airflow DAG and orchestration tests."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

DAG_ID = "mlops_customer_churn_pipeline"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
# Keep the virtual-environment entry point intact. Resolving this symlink can
# produce the base interpreter path, which does not inherit the venv packages.
PYTHON_EXECUTABLE = Path(sys.executable)
DBT_EXECUTABLE = PYTHON_EXECUTABLE.parent / "dbt"


def environment_flag(name: str, *, default: bool = False) -> bool:
    """Parse a conventional boolean environment setting."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


def configured_schedule() -> str | None:
    """Return an optional schedule; empty means manual execution."""
    value = os.getenv("CHURN_DAG_SCHEDULE", "").strip()
    return value or None


def prepare_raw_data_command(use_synthetic: bool | None = None) -> str:
    """Select real ingestion or deterministic offline RAW generation."""
    synthetic = (
        environment_flag("CHURN_PIPELINE_USE_SYNTHETIC_DATA")
        if use_synthetic is None
        else use_synthetic
    )
    python = shlex.quote(str(PYTHON_EXECUTABLE))
    if synthetic:
        raw_path = os.getenv("TELCO_RAW_PATH", "data/raw/telco_customer_churn.csv")
        return (
            f"{python} -m churn_platform.testing.synthetic_telco "
            f"--output {shlex.quote(raw_path)}"
        )
    return f"{python} -m churn_platform.ingestion.telco"


def module_command(module: str) -> str:
    """Build a command using the interpreter that loaded the DAG."""
    return f"{shlex.quote(str(PYTHON_EXECUTABLE))} -m {shlex.quote(module)}"


def dbt_build_command() -> str:
    """Build the dbt mart with repository-local project and profiles."""
    database_path = Path(
        os.getenv("DBT_DUCKDB_PATH", "data/features/churn_analytics.duckdb")
    )
    return (
        f"mkdir -p {shlex.quote(str(database_path.parent))} && "
        f"{shlex.quote(str(DBT_EXECUTABLE))} build "
        "--project-dir dbt_project --profiles-dir dbt_project"
    )
