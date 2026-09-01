"""Isolated tests for champion verification outside Airflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.models import infer_signature

from churn_platform.ml.data import separate_features_and_target
from churn_platform.ml.preprocessing import build_logistic_pipeline
from churn_platform.ml.tracking import configure_mlflow
from churn_platform.orchestration.verify import verify_champion


@dataclass(frozen=True)
class ChampionFixture:
    tracking_uri: str
    database_path: Path
    model_name: str


@pytest.fixture(scope="module")
def champion_fixture(tmp_path_factory: pytest.TempPathFactory) -> ChampionFixture:
    directory = tmp_path_factory.mktemp("champion-verification")
    database_path = directory / "features.duckdb"
    rows = []
    for index in range(40):
        target = int(index % 4 == 0)
        rows.append(
            {
                "customer_id": f"VERIFY-{index:04d}",
                "churn": "Yes" if target else "No",
                "churn_flag": target,
                "tenure": index,
                "monthly_charges": 30.0 + index,
                "contract": "Month-to-month" if index % 2 else "One year",
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

    features, target, metadata = separate_features_and_target(dataset)
    pipeline = build_logistic_pipeline(metadata)
    pipeline.fit(features, target)
    tracking_uri = f"sqlite:///{directory / 'mlflow.db'}"
    _, _, experiment_id = configure_mlflow(tracking_uri, "verification-test")
    with mlflow.start_run(experiment_id=experiment_id) as run:
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=infer_signature(features, pipeline.predict(features)),
            input_example=features.head(3),
        )
        run_id = run.info.run_id
    name = "verification-model"
    mlflow.register_model(model_info.model_uri, name)
    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name = '{name}'")
    version = next(version for version in versions if version.run_id == run_id)
    client.set_registered_model_alias(name, "champion", version.version)
    client.create_registered_model("aliasless-model")
    return ChampionFixture(tracking_uri, database_path, name)


def test_verification_fails_when_registered_model_missing(
    champion_fixture: ChampionFixture,
) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        verify_champion(
            tracking_uri=champion_fixture.tracking_uri,
            registered_model_name="missing-model",
            database_path=champion_fixture.database_path,
        )


def test_verification_fails_when_champion_alias_missing(
    champion_fixture: ChampionFixture,
) -> None:
    with pytest.raises(RuntimeError, match="no champion alias"):
        verify_champion(
            tracking_uri=champion_fixture.tracking_uri,
            registered_model_name="aliasless-model",
            database_path=champion_fixture.database_path,
        )


def test_valid_champion_loads_and_predicts(
    champion_fixture: ChampionFixture,
) -> None:
    result = verify_champion(
        tracking_uri=champion_fixture.tracking_uri,
        registered_model_name=champion_fixture.model_name,
        database_path=champion_fixture.database_path,
    )
    assert result.champion_version == "1"
    assert result.prediction_count == 5
    assert set(result.predictions) <= {0, 1}
