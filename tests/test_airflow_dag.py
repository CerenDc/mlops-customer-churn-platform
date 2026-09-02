"""Structural tests for the Airflow 3 customer churn DAG."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from airflow.sdk import TriggerRule

from churn_platform.orchestration.config import (
    PYTHON_EXECUTABLE,
    configured_repository_root,
    dbt_build_command,
    prepare_raw_data_command,
)

DAG_PATH = Path("orchestration/dags/churn_mlops_pipeline.py").resolve()
EXPECTED_CHAIN = (
    "prepare_raw_data",
    "spark_processing",
    "dbt_build",
    "train_models",
    "registry_lifecycle",
    "verify_champion",
)


@pytest.fixture(scope="module")
def churn_dag(tmp_path_factory: pytest.TempPathFactory):
    directory = tmp_path_factory.mktemp("airflow-dag-import")
    spec = importlib.util.spec_from_file_location("tested_churn_dag", DAG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("AIRFLOW_HOME", str(directory))
        monkeypatch.delenv("CHURN_DAG_SCHEDULE", raising=False)
        monkeypatch.delenv("CHURN_PIPELINE_USE_SYNTHETIC_DATA", raising=False)
        spec.loader.exec_module(module)
    return module.dag


def test_dag_imports_with_expected_identity(churn_dag) -> None:
    assert churn_dag.dag_id == "mlops_customer_churn_pipeline"
    assert set(churn_dag.task_ids) == set(EXPECTED_CHAIN)


def test_dag_manual_schedule_and_concurrency(churn_dag) -> None:
    assert churn_dag.schedule is None
    assert churn_dag.catchup is False
    assert churn_dag.max_active_runs == 1


def test_dag_has_strict_dependency_chain(churn_dag) -> None:
    for upstream, downstream in zip(EXPECTED_CHAIN, EXPECTED_CHAIN[1:], strict=False):
        assert churn_dag.get_task(downstream).upstream_task_ids == {upstream}
        assert churn_dag.get_task(upstream).downstream_task_ids == {downstream}


def test_failure_propagates_through_all_success_trigger_rules(churn_dag) -> None:
    for task_id in EXPECTED_CHAIN[1:]:
        assert churn_dag.get_task(task_id).trigger_rule == TriggerRule.ALL_SUCCESS


def test_tasks_disable_xcom_payloads(churn_dag) -> None:
    assert all(task.do_xcom_push is False for task in churn_dag.tasks)


def test_prepare_data_production_mode_uses_ingestion() -> None:
    assert "churn_platform.ingestion.telco" in prepare_raw_data_command(False)


def test_commands_preserve_virtual_environment_interpreter() -> None:
    assert Path(sys.executable) == PYTHON_EXECUTABLE
    assert str(PYTHON_EXECUTABLE) in prepare_raw_data_command(False)


def test_repository_root_can_be_configured_for_container(monkeypatch) -> None:
    monkeypatch.setenv("CHURN_REPOSITORY_ROOT", "/opt/mlops")
    assert Path("/opt/mlops") == configured_repository_root()


def test_prepare_data_synthetic_mode_uses_raw_fixture(monkeypatch) -> None:
    monkeypatch.setenv("TELCO_RAW_PATH", "/tmp/synthetic telco.csv")
    command = prepare_raw_data_command(True)
    assert "churn_platform.testing.synthetic_telco" in command
    assert "'/tmp/synthetic telco.csv'" in command


def test_dbt_command_creates_external_database_directory(monkeypatch) -> None:
    monkeypatch.setenv("DBT_DUCKDB_PATH", "/tmp/features dir/churn.duckdb")
    command = dbt_build_command()
    assert command.startswith("mkdir -p '/tmp/features dir' && ")
    assert "dbt build" in command


def test_task_retry_policy(churn_dag) -> None:
    assert {task.task_id: task.retries for task in churn_dag.tasks} == {
        "prepare_raw_data": 2,
        "spark_processing": 1,
        "dbt_build": 1,
        "train_models": 0,
        "registry_lifecycle": 1,
        "verify_champion": 1,
    }
