"""Train two churn candidates and track the experiment lifecycle in MLflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow.sklearn
from mlflow import MlflowClient

from churn_platform.ml.data import (
    RANDOM_STATE,
    DataSplits,
    FeatureMetadata,
    configured_duckdb_path,
    load_feature_mart,
    separate_features_and_target,
    split_dataset,
)
from churn_platform.ml.evaluate import evaluate_classifier
from churn_platform.ml.preprocessing import (
    build_logistic_pipeline,
    build_xgboost_pipeline,
)
from churn_platform.ml.tracking import (
    CandidateRun,
    configure_mlflow,
    log_candidate_run,
    log_selected_test_result,
)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Summary of a completed, reload-verified training experiment."""

    experiment_name: str
    experiment_id: str
    selected_model: str
    selected_run_id: str
    selected_model_uri: str
    validation_results: dict[str, dict[str, float]]
    test_metrics: dict[str, float]
    tracking_uri: str
    row_count: int
    feature_metadata: FeatureMetadata
    split_sizes: dict[str, int]
    class_distribution: dict[int, int]
    reload_prediction_count: int
    run_ids: dict[str, str]


def select_winner(candidates: list[CandidateRun]) -> CandidateRun:
    """Select exclusively by validation ROC AUC with stable name tie-breaking."""
    if not candidates:
        raise ValueError("At least one candidate is required")
    return max(
        candidates,
        key=lambda candidate: (
            candidate.validation.metrics["roc_auc"],
            candidate.model_name,
        ),
    )


def train_models(
    *,
    database_path: Path | None = None,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> TrainingResult:
    """Train, compare on validation, test once, track, and reload the winner."""
    feature_database = Path(database_path or configured_duckdb_path())
    dataset = load_feature_mart(feature_database)
    features, target, metadata = separate_features_and_target(dataset)
    splits = split_dataset(features, target, random_state=RANDOM_STATE)
    split_sizes = _split_sizes(splits)

    uri, resolved_experiment, experiment_id = configure_mlflow(
        tracking_uri, experiment_name
    )
    positive_count = int(splits.y_train.sum())
    negative_count = len(splits.y_train) - positive_count
    scale_pos_weight = negative_count / positive_count

    candidates = []
    definitions = (
        (
            "logistic_regression",
            "linear",
            build_logistic_pipeline(metadata),
            {
                "class_weight": "balanced",
                "max_iter": 1_000,
                "solver": "lbfgs",
            },
        ),
        (
            "xgboost",
            "boosted_tree",
            build_xgboost_pipeline(metadata, scale_pos_weight=scale_pos_weight),
            {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": scale_pos_weight,
                "tree_method": "hist",
                "n_jobs": 2,
            },
        ),
    )
    for model_name, family, pipeline, parameters in definitions:
        pipeline.fit(splits.x_train, splits.y_train)
        validation = evaluate_classifier(
            pipeline, splits.x_validation, splits.y_validation
        )
        candidates.append(
            log_candidate_run(
                experiment_id=experiment_id,
                model_name=model_name,
                model_family=family,
                pipeline=pipeline,
                validation=validation,
                x_train=splits.x_train,
                y_train=splits.y_train,
                metadata=metadata,
                database_path=feature_database,
                split_sizes=split_sizes,
                model_parameters=parameters,
            )
        )

    selected = select_winner(candidates)
    test_result = evaluate_classifier(selected.pipeline, splits.x_test, splits.y_test)
    client = MlflowClient(tracking_uri=uri)
    for candidate in candidates:
        status = "selected" if candidate.run_id == selected.run_id else "rejected"
        client.set_tag(candidate.run_id, "candidate_status", status)
    log_selected_test_result(selected.run_id, test_result)

    reloaded_model = mlflow.sklearn.load_model(selected.model_uri)
    reload_sample = splits.x_test.head(5)
    reloaded_predictions = reloaded_model.predict(reload_sample)
    if len(reloaded_predictions) != len(reload_sample):
        raise RuntimeError("Reloaded model returned an unexpected prediction count")
    if not set(reloaded_predictions) <= {0, 1}:
        raise RuntimeError("Reloaded model returned invalid churn classes")

    return TrainingResult(
        experiment_name=resolved_experiment,
        experiment_id=experiment_id,
        selected_model=selected.model_name,
        selected_run_id=selected.run_id,
        selected_model_uri=selected.model_uri,
        validation_results={
            candidate.model_name: candidate.validation.metrics
            for candidate in candidates
        },
        test_metrics=test_result.metrics,
        tracking_uri=uri,
        row_count=len(dataset),
        feature_metadata=metadata,
        split_sizes=split_sizes,
        class_distribution={
            int(label): int(count)
            for label, count in target.value_counts().sort_index().items()
        },
        reload_prediction_count=len(reloaded_predictions),
        run_ids={candidate.model_name: candidate.run_id for candidate in candidates},
    )


def _split_sizes(splits: DataSplits) -> dict[str, int]:
    return {
        "train": len(splits.x_train),
        "validation": len(splits.x_validation),
        "test": len(splits.x_test),
    }


def _format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())


def main() -> None:
    """Run complete local training and print a concise summary."""
    result = train_models()
    print("ML training completed")
    print(f"Experiment: {result.experiment_name} ({result.experiment_id})")
    print(f"Customers: {result.row_count}")
    print(
        "Features: "
        f"{result.feature_metadata.feature_count} "
        f"({len(result.feature_metadata.numeric_features)} numeric, "
        f"{len(result.feature_metadata.categorical_features)} categorical)"
    )
    print(f"Splits: {result.split_sizes}")
    for model_name, metrics in result.validation_results.items():
        print(f"{model_name} validation: {_format_metrics(metrics)}")
    print(f"Selected model: {result.selected_model}")
    print(f"Selected run: {result.selected_run_id}")
    print(f"Test: {_format_metrics(result.test_metrics)}")
    print(f"Reload predictions: {result.reload_prediction_count}")
    print(f"MLflow tracking: {result.tracking_uri}")


if __name__ == "__main__":
    main()
