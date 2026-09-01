"""Local MLflow experiment configuration and candidate run logging."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from churn_platform.ml.data import (
    EXCLUDED_COLUMNS,
    FEATURE_MART,
    TARGET_COLUMN,
    FeatureMetadata,
)
from churn_platform.ml.evaluate import EvaluationResult

DEFAULT_MLFLOW_ROOT = Path("data/mlflow")
DEFAULT_EXPERIMENT_NAME = "telco-customer-churn"


@dataclass(frozen=True, slots=True)
class CandidateRun:
    """A fitted candidate and its tracked validation result."""

    model_name: str
    run_id: str
    model_uri: str
    pipeline: Pipeline
    validation: EvaluationResult


def default_tracking_uri(root: Path = DEFAULT_MLFLOW_ROOT) -> str:
    """Return an absolute local SQLite tracking URI."""
    database_path = Path(root).resolve() / "mlflow.db"
    return f"sqlite:///{database_path}"


def configure_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> tuple[str, str, str]:
    """Configure and reuse a local experiment without any registry operations."""
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or default_tracking_uri()
    name = (
        experiment_name
        or os.getenv("MLFLOW_EXPERIMENT_NAME")
        or DEFAULT_EXPERIMENT_NAME
    )
    if uri.startswith("sqlite:///"):
        database_path = Path(uri.removeprefix("sqlite:///"))
        database_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_root = database_path.parent / "artifacts"
    elif uri.startswith("file://"):
        artifact_root = Path(uri.removeprefix("file://")) / "artifacts"
    else:
        artifact_root = DEFAULT_MLFLOW_ROOT.resolve() / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name, artifact_location=artifact_root.resolve().as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    return uri, name, experiment_id


def log_candidate_run(
    *,
    experiment_id: str,
    model_name: str,
    model_family: str,
    pipeline: Pipeline,
    validation: EvaluationResult,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    metadata: FeatureMetadata,
    database_path: Path,
    split_sizes: dict[str, int],
    model_parameters: dict[str, object],
) -> CandidateRun:
    """Log one fitted candidate, its lineage, model, and interpretation."""
    with mlflow.start_run(experiment_id=experiment_id, run_name=model_name) as run:
        mlflow.set_tags(
            {
                "project": "mlops-customer-churn-platform",
                "stage": "training",
                "data_layer": "dbt_feature_mart",
                "model_family": model_family,
                "candidate_status": "pending",
            }
        )
        parameters = {
            "model_type": model_name,
            "random_state": 42,
            "train_size": split_sizes["train"],
            "validation_size": split_sizes["validation"],
            "test_size": split_sizes["test"],
            "number_of_features": metadata.feature_count,
            "number_of_numeric_features": len(metadata.numeric_features),
            "number_of_categorical_features": len(metadata.categorical_features),
            **model_parameters,
        }
        mlflow.log_params(parameters)
        mlflow.log_metrics(
            {f"val_{name}": value for name, value in validation.metrics.items()}
        )
        mlflow.log_dict(
            validation.classification_report,
            "validation/classification_report.json",
        )
        mlflow.log_dict(
            {"labels": [0, 1], "matrix": validation.confusion_matrix},
            "validation/confusion_matrix.json",
        )
        mlflow.log_dict(
            {
                "target": TARGET_COLUMN,
                "excluded_columns": list(EXCLUDED_COLUMNS),
                "numeric_features": list(metadata.numeric_features),
                "categorical_features": list(metadata.categorical_features),
                "total_feature_count": metadata.feature_count,
            },
            "metadata/feature_schema.json",
        )

        lineage_frame = x_train.copy()
        lineage_frame[TARGET_COLUMN] = y_train
        training_dataset = mlflow.data.from_pandas(
            lineage_frame,
            source=str(database_path.resolve()),
            name=FEATURE_MART,
            targets=TARGET_COLUMN,
        )
        mlflow.log_input(training_dataset, context="training")

        input_example = x_train.head(5)
        signature = infer_signature(input_example, pipeline.predict(input_example))
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=input_example,
        )
        _log_interpretation_artifact(pipeline, model_name)
        return CandidateRun(
            model_name=model_name,
            run_id=run.info.run_id,
            model_uri=model_info.model_uri,
            pipeline=pipeline,
            validation=validation,
        )


def _log_interpretation_artifact(pipeline: Pipeline, model_name: str) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    if model_name == "logistic_regression":
        interpretation = pd.DataFrame(
            {"feature": feature_names, "coefficient": classifier.coef_[0]}
        ).sort_values("coefficient", key=abs, ascending=False)
        filename = "logistic_coefficients.csv"
    else:
        interpretation = pd.DataFrame(
            {"feature": feature_names, "importance": classifier.feature_importances_}
        ).sort_values("importance", ascending=False)
        filename = "feature_importance.csv"

    with tempfile.TemporaryDirectory(prefix="churn-interpretation-") as directory:
        path = Path(directory) / filename
        interpretation.to_csv(path, index=False)
        mlflow.log_artifact(str(path), artifact_path="interpretation")


def log_selected_test_result(run_id: str, result: EvaluationResult) -> None:
    """Attach final test metrics only to the validation-selected candidate."""
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {f"test_{name}": value for name, value in result.metrics.items()}
        )
        mlflow.log_dict(
            result.classification_report,
            "test/classification_report.json",
        )
        mlflow.log_dict(
            {"labels": [0, 1], "matrix": result.confusion_matrix},
            "test/confusion_matrix.json",
        )
