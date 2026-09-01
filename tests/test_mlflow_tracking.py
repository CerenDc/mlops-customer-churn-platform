"""Isolated MLflow integration tests for the complete training workflow."""

from __future__ import annotations

import duckdb
import mlflow.sklearn
import pandas as pd
import pytest
from mlflow import MlflowClient

from churn_platform.ml.data import EXCLUDED_COLUMNS
from churn_platform.ml.train import TrainingResult, train_models


@pytest.fixture(scope="module")
def tracked_training(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[TrainingResult, pd.DataFrame]:
    directory = tmp_path_factory.mktemp("mlflow-training")
    database_path = directory / "features.duckdb"
    rows = []
    for index in range(60):
        target = 1 if index % 4 == 0 else 0
        rows.append(
            {
                "customer_id": f"TRACK-{index:04d}",
                "churn": "Yes" if target else "No",
                "churn_flag": target,
                "tenure": index % 50,
                "monthly_charges": 30.0 + index,
                "total_charges": None if index % 13 == 0 else 100.0 + index * 10,
                "service_count": index % 8,
                "contract": "Month-to-month" if index % 3 else "One year",
                "internet_service": "DSL" if index % 2 else "Fiber optic",
            }
        )
    dataset = pd.DataFrame(rows)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("create schema main_marts")
        connection.register("training_fixture", dataset)
        connection.execute(
            "create table main_marts.fct_customer_churn_features "
            "as select * from training_fixture"
        )

    tracking_uri = f"sqlite:///{directory / 'mlflow.db'}"
    result = train_models(
        database_path=database_path,
        tracking_uri=tracking_uri,
        experiment_name="isolated-training-test",
    )
    return result, dataset


def test_mlflow_experiment_and_candidate_runs_exist(
    tracked_training: tuple[TrainingResult, pd.DataFrame],
) -> None:
    result, _ = tracked_training
    client = MlflowClient(tracking_uri=result.tracking_uri)
    runs = client.search_runs([result.experiment_id])
    assert len(runs) == 2
    assert {run.data.params["model_type"] for run in runs} == {
        "logistic_regression",
        "xgboost",
    }
    assert all(run.inputs.dataset_inputs for run in runs)


def test_only_selected_run_has_test_metrics(
    tracked_training: tuple[TrainingResult, pd.DataFrame],
) -> None:
    result, _ = tracked_training
    client = MlflowClient(tracking_uri=result.tracking_uri)
    for model_name, run_id in result.run_ids.items():
        run = client.get_run(run_id)
        if model_name == result.selected_model:
            assert "test_roc_auc" in run.data.metrics
            assert run.data.tags["candidate_status"] == "selected"
        else:
            assert "test_roc_auc" not in run.data.metrics
            assert run.data.tags["candidate_status"] == "rejected"


def test_logged_model_loads_and_predicts(
    tracked_training: tuple[TrainingResult, pd.DataFrame],
) -> None:
    result, dataset = tracked_training
    mlflow.set_tracking_uri(result.tracking_uri)
    model = mlflow.sklearn.load_model(result.selected_model_uri)
    sample = dataset.drop(columns=list(EXCLUDED_COLUMNS)).head(3)
    predictions = model.predict(sample)
    assert len(predictions) == 3
    assert set(predictions) <= {0, 1}


def test_training_result_records_reload_and_validation(
    tracked_training: tuple[TrainingResult, pd.DataFrame],
) -> None:
    result, _ = tracked_training
    assert result.reload_prediction_count == 5
    assert set(result.validation_results) == {"logistic_regression", "xgboost"}
    assert result.selected_run_id in result.run_ids.values()
