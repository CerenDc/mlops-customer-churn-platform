"""Verify that the registered champion serves valid churn predictions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from churn_platform.ml.data import load_feature_mart, separate_features_and_target
from churn_platform.ml.registry import DEFAULT_REGISTERED_MODEL_NAME
from churn_platform.ml.tracking import default_tracking_uri


@dataclass(frozen=True, slots=True)
class ChampionVerificationResult:
    """Auditable result from champion resolution and inference."""

    registered_model_name: str
    champion_version: str
    champion_uri: str
    prediction_count: int
    predictions: tuple[int, ...]


def verify_champion(
    *,
    tracking_uri: str | None = None,
    registered_model_name: str | None = None,
    database_path: Path | None = None,
    sample_size: int = 5,
) -> ChampionVerificationResult:
    """Resolve, load, and predict with the champion or fail clearly."""
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or default_tracking_uri()
    name = registered_model_name or os.getenv(
        "MLFLOW_REGISTERED_MODEL_NAME", DEFAULT_REGISTERED_MODEL_NAME
    )
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    try:
        client.get_registered_model(name)
    except MlflowException as error:
        raise RuntimeError(f"Registered model {name!r} does not exist") from error
    try:
        champion = client.get_model_version_by_alias(name, "champion")
    except MlflowException as error:
        raise RuntimeError(
            f"Registered model {name!r} has no champion alias"
        ) from error

    champion_uri = f"models:/{name}@champion"
    model = mlflow.sklearn.load_model(champion_uri)
    dataset = load_feature_mart(database_path)
    features, _, _ = separate_features_and_target(dataset)
    predictions = tuple(
        int(value) for value in model.predict(features.head(sample_size))
    )
    if len(predictions) != min(sample_size, len(features)):
        raise RuntimeError("Champion returned an unexpected prediction count")
    if not set(predictions) <= {0, 1}:
        raise RuntimeError("Champion returned invalid churn classes")
    return ChampionVerificationResult(
        registered_model_name=name,
        champion_version=str(champion.version),
        champion_uri=champion_uri,
        prediction_count=len(predictions),
        predictions=predictions,
    )


def main() -> None:
    """Run champion verification independently from Airflow."""
    result = verify_champion()
    print("Champion verification completed")
    print(f"Registered model: {result.registered_model_name}")
    print(f"Champion version: {result.champion_version}")
    print(f"Champion URI: {result.champion_uri}")
    print(f"Prediction count: {result.prediction_count}")
    print(f"Predictions: {list(result.predictions)}")


if __name__ == "__main__":
    main()
