"""Airflow DAG orchestrating the complete customer churn MLOps workflow."""

from __future__ import annotations

from datetime import timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

from churn_platform.orchestration.config import (
    DAG_ID,
    REPOSITORY_ROOT,
    configured_schedule,
    dbt_build_command,
    module_command,
    prepare_raw_data_command,
)

COMMON_ENV = {
    "AIRFLOW_PIPELINE_RUN_ID": "{{ run_id }}",
    "AIRFLOW_PIPELINE_DAG_ID": "{{ dag.dag_id }}",
}
COMMON_TASK_ARGS = {
    "cwd": str(REPOSITORY_ROOT),
    "append_env": True,
    "do_xcom_push": False,
    "env": COMMON_ENV,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id=DAG_ID,
    description="Orchestrate churn data, training, registry, and validation",
    schedule=configured_schedule(),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=45),
    default_args={"owner": "mlops-platform"},
    tags=["mlops", "churn", "spark", "dbt", "mlflow"],
) as dag:
    prepare_raw_data = BashOperator(
        task_id="prepare_raw_data",
        bash_command=prepare_raw_data_command(),
        retries=2,
        execution_timeout=timedelta(minutes=10),
        **COMMON_TASK_ARGS,
    )
    spark_processing = BashOperator(
        task_id="spark_processing",
        bash_command=module_command("churn_platform.spark.transform_telco"),
        retries=1,
        execution_timeout=timedelta(minutes=15),
        **COMMON_TASK_ARGS,
    )
    dbt_build = BashOperator(
        task_id="dbt_build",
        bash_command=dbt_build_command(),
        retries=1,
        execution_timeout=timedelta(minutes=10),
        **COMMON_TASK_ARGS,
    )
    train_models = BashOperator(
        task_id="train_models",
        bash_command=module_command("churn_platform.ml.train"),
        retries=0,
        execution_timeout=timedelta(minutes=10),
        **COMMON_TASK_ARGS,
    )
    registry_lifecycle = BashOperator(
        task_id="registry_lifecycle",
        bash_command=module_command("churn_platform.ml.registry"),
        retries=1,
        execution_timeout=timedelta(minutes=5),
        **COMMON_TASK_ARGS,
    )
    verify_champion = BashOperator(
        task_id="verify_champion",
        bash_command=module_command("churn_platform.orchestration.verify"),
        retries=1,
        execution_timeout=timedelta(minutes=5),
        **COMMON_TASK_ARGS,
    )

    (
        prepare_raw_data
        >> spark_processing
        >> dbt_build
        >> train_models
        >> registry_lifecycle
        >> verify_champion
    )
