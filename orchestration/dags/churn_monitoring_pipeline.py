"""Independent monitoring workflow for the currently deployed champion."""

from __future__ import annotations

from datetime import timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

from churn_platform.orchestration.config import REPOSITORY_ROOT, module_command

with DAG(
    dag_id="churn_model_monitoring",
    description="Evaluate drift and publish operational model metrics",
    schedule=None,
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=15),
    default_args={"owner": "mlops-platform"},
    tags=["mlops", "churn", "monitoring", "drift"],
) as dag:
    evaluate_drift = BashOperator(
        task_id="evaluate_drift",
        bash_command=module_command("churn_platform.monitoring.run"),
        cwd=str(REPOSITORY_ROOT),
        append_env=True,
        do_xcom_push=False,
        retries=1,
        retry_delay=timedelta(minutes=1),
        execution_timeout=timedelta(minutes=10),
    )
