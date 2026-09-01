"""Isolated integration tests for MLflow registry aliases and idempotency."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import mlflow.sklearn
import pandas as pd
import pytest
from mlflow import MlflowClient

from churn_platform.ml.data import EXCLUDED_COLUMNS
from churn_platform.ml.promotion import PromotionPolicy
from churn_platform.ml.registry import RegistryResult, run_registry_lifecycle
from churn_platform.ml.train import train_models


@dataclass(frozen=True)
class RegistryScenario:
    tracking_uri: str
    database_path: Path
    dataset: pd.DataFrame
    first: RegistryResult
    repeated: RegistryResult
    rejected: RegistryResult
    replacement: RegistryResult


@pytest.fixture(scope="module")
def registry_scenario(tmp_path_factory: pytest.TempPathFactory) -> RegistryScenario:
    directory = tmp_path_factory.mktemp("registry")
    database_path = directory / "features.duckdb"
    rows = []
    for index in range(60):
        target = int(index % 4 == 0)
        rows.append(
            {
                "customer_id": f"REGISTRY-{index:04d}",
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
        connection.register("fixture", dataset)
        connection.execute(
            "create table main_marts.fct_customer_churn_features "
            "as select * from fixture"
        )

    tracking_uri = f"sqlite:///{directory / 'mlflow.db'}"
    experiment = "registry-integration-test"
    name = "isolated-churn-model"
    permissive = PromotionPolicy(0, 0, 0, 0, 1, 1)
    train_models(
        database_path=database_path,
        tracking_uri=tracking_uri,
        experiment_name=experiment,
    )
    first = run_registry_lifecycle(
        tracking_uri=tracking_uri,
        experiment_name=experiment,
        registered_model_name=name,
        policy=permissive,
        database_path=database_path,
    )
    repeated = run_registry_lifecycle(
        tracking_uri=tracking_uri,
        experiment_name=experiment,
        registered_model_name=name,
        policy=permissive,
        database_path=database_path,
    )

    train_models(
        database_path=database_path,
        tracking_uri=tracking_uri,
        experiment_name=experiment,
    )
    rejected = run_registry_lifecycle(
        tracking_uri=tracking_uri,
        experiment_name=experiment,
        registered_model_name=name,
        policy=PromotionPolicy(0, 0, 0, 0.001, 1, 1),
        database_path=database_path,
    )
    train_models(
        database_path=database_path,
        tracking_uri=tracking_uri,
        experiment_name=experiment,
    )
    replacement = run_registry_lifecycle(
        tracking_uri=tracking_uri,
        experiment_name=experiment,
        registered_model_name=name,
        policy=permissive,
        database_path=database_path,
    )
    return RegistryScenario(
        tracking_uri,
        database_path,
        dataset,
        first,
        repeated,
        rejected,
        replacement,
    )


def test_registered_model_and_version_are_created(
    registry_scenario: RegistryScenario,
) -> None:
    client = MlflowClient(tracking_uri=registry_scenario.tracking_uri)
    model = client.get_registered_model("isolated-churn-model")
    versions = client.search_model_versions("name = 'isolated-churn-model'")
    assert model.name == "isolated-churn-model"
    assert {version.version for version in versions} == {1, 2, 3}


def test_registered_model_metadata_is_preserved(
    registry_scenario: RegistryScenario,
) -> None:
    client = MlflowClient(tracking_uri=registry_scenario.tracking_uri)
    model = client.get_registered_model("isolated-churn-model")
    version = client.get_model_version("isolated-churn-model", "1")
    assert model.tags["target"] == "churn_flag"
    assert version.tags["source_run_id"] == registry_scenario.first.source_run_id
    assert version.tags["source_logged_model_id"]
    assert version.tags["test_roc_auc"]


def test_champion_alias_can_be_resolved(
    registry_scenario: RegistryScenario,
) -> None:
    client = MlflowClient(tracking_uri=registry_scenario.tracking_uri)
    champion = client.get_model_version_by_alias("isolated-churn-model", "champion")
    assert champion.version == 3
    assert registry_scenario.rejected.challenger_version == "2"


def test_exact_logged_model_is_not_registered_twice(
    registry_scenario: RegistryScenario,
) -> None:
    assert registry_scenario.first.version_count_before == 0
    assert registry_scenario.first.version_count_after == 1
    assert registry_scenario.repeated.reused_version
    assert registry_scenario.repeated.version_count_after == 1


def test_rejected_challenger_does_not_move_champion(
    registry_scenario: RegistryScenario,
) -> None:
    assert not registry_scenario.rejected.decision.promoted
    assert registry_scenario.rejected.champion_version == "1"
    assert registry_scenario.rejected.challenger_version == "2"
    assert "roc_auc_improvement" in registry_scenario.rejected.decision.reasons


def test_old_champion_remains_registered_after_rejection(
    registry_scenario: RegistryScenario,
) -> None:
    client = MlflowClient(tracking_uri=registry_scenario.tracking_uri)
    first = client.get_model_version("isolated-churn-model", "1")
    second = client.get_model_version("isolated-churn-model", "2")
    assert first.tags["promotion_status"] == "superseded"
    assert second.tags["promotion_status"] == "rejected"


def test_better_challenger_replaces_and_preserves_old_champion(
    registry_scenario: RegistryScenario,
) -> None:
    client = MlflowClient(tracking_uri=registry_scenario.tracking_uri)
    versions = client.search_model_versions("name = 'isolated-churn-model'")
    assert registry_scenario.replacement.decision.promoted
    assert registry_scenario.replacement.decision.previous_champion_version == "1"
    assert registry_scenario.replacement.champion_version == "3"
    assert {version.version for version in versions} == {1, 2, 3}


def test_champion_uri_loads_and_predicts(
    registry_scenario: RegistryScenario,
) -> None:
    mlflow.set_tracking_uri(registry_scenario.tracking_uri)
    model = mlflow.sklearn.load_model("models:/isolated-churn-model@champion")
    sample = registry_scenario.dataset.drop(columns=list(EXCLUDED_COLUMNS)).head(3)
    predictions = model.predict(sample)
    assert len(predictions) == 3
    assert set(predictions) <= {0, 1}


def test_lifecycle_validates_champion_predictions(
    registry_scenario: RegistryScenario,
) -> None:
    assert registry_scenario.first.champion_prediction_count == 5
    assert set(registry_scenario.first.champion_predictions) <= {0, 1}


def test_lifecycle_honors_configured_experiment_environment(
    registry_scenario: RegistryScenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for CI where the experiment name differs from the default."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", registry_scenario.tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "registry-integration-test")
    monkeypatch.setenv("MLFLOW_REGISTERED_MODEL_NAME", "environment-configured-model")

    result = run_registry_lifecycle(
        policy=PromotionPolicy(0, 0, 0, 0, 1, 1),
        database_path=registry_scenario.database_path,
    )

    assert result.registered_model_name == "environment-configured-model"
    assert result.source_run_id == registry_scenario.replacement.source_run_id
    assert result.champion_version == "1"
