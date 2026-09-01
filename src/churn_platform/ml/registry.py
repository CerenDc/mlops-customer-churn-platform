"""Register the latest selected experiment model and manage lifecycle aliases."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.exceptions import MlflowException

from churn_platform.ml.data import load_feature_mart, separate_features_and_target
from churn_platform.ml.promotion import (
    PromotionDecision,
    PromotionPolicy,
    evaluate_promotion,
)
from churn_platform.ml.tracking import (
    DEFAULT_EXPERIMENT_NAME,
    configure_mlflow,
)

DEFAULT_REGISTERED_MODEL_NAME = "telco-churn-classifier"
REQUIRED_TEST_METRICS = (
    "test_roc_auc",
    "test_pr_auc",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_accuracy",
    "test_log_loss",
)


@dataclass(frozen=True, slots=True)
class RegistryResult:
    """Summary of one complete registry lifecycle execution."""

    registered_model_name: str
    candidate_version: str
    source_run_id: str
    source_logged_model_id: str
    source_logged_model_uri: str
    reused_version: bool
    version_count_before: int
    version_count_after: int
    decision: PromotionDecision
    champion_version: str | None
    challenger_version: str | None
    champion_prediction_count: int
    champion_predictions: tuple[int, ...]
    tracking_uri: str


def find_latest_selected_run(client: MlflowClient, experiment_id: str) -> Run:
    """Return the newest selected run containing every final test metric."""
    runs = client.search_runs(
        [experiment_id],
        filter_string="tags.candidate_status = 'selected'",
        order_by=["start_time DESC"],
    )
    for run in runs:
        if all(metric in run.data.metrics for metric in REQUIRED_TEST_METRICS):
            return run
    raise RuntimeError(
        "No selected MLflow run with complete test metrics exists. Run: "
        "python -m churn_platform.ml.train"
    )


def resolve_logged_model(client: MlflowClient, experiment_id: str, run_id: str):
    """Resolve and validate the intended READY pipeline logged by a run."""
    candidates = [
        model
        for model in client.search_logged_models(
            experiment_ids=[experiment_id], max_results=1000
        )
        if model.source_run_id == run_id
        and model.name == "model"
        and getattr(model.status, "value", model.status) == "READY"
    ]
    if not candidates:
        raise RuntimeError(
            f"Selected run {run_id} has no READY logged model named model"
        )
    model = max(candidates, key=lambda item: item.creation_timestamp)
    model_info = mlflow.models.get_model_info(model.model_uri)
    if model_info.signature is None:
        raise RuntimeError(f"Logged model {model.model_id} has no usable signature")
    mlflow.sklearn.load_model(model.model_uri)
    return model


def _model_versions(client: MlflowClient, name: str):
    return list(client.search_model_versions(f"name = '{name}'"))


def _alias(client: MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def _metrics_from_version(version) -> dict[str, float]:
    missing = [metric for metric in REQUIRED_TEST_METRICS if metric not in version.tags]
    if missing:
        raise RuntimeError(
            f"Registered model version {version.version} lacks metrics: "
            f"{', '.join(missing)}"
        )
    return {metric: float(version.tags[metric]) for metric in REQUIRED_TEST_METRICS}


def _register_or_reuse(
    client: MlflowClient,
    *,
    name: str,
    run: Run,
    logged_model,
):
    versions = _model_versions(client, name)
    for version in versions:
        if version.tags.get("source_logged_model_id") == logged_model.model_id:
            return version, True, len(versions)

    version_tags = {
        "source_run_id": run.info.run_id,
        "source_logged_model_id": logged_model.model_id,
        "model_type": run.data.params["model_type"],
        "validation_roc_auc": str(run.data.metrics["val_roc_auc"]),
        "data_layer": "dbt_feature_mart",
        "promotion_status": "challenger",
        **{metric: str(run.data.metrics[metric]) for metric in REQUIRED_TEST_METRICS},
    }
    version = mlflow.register_model(
        model_uri=logged_model.model_uri,
        name=name,
        tags=version_tags,
    )
    return client.get_model_version(name, version.version), False, len(versions)


def run_registry_lifecycle(
    *,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    registered_model_name: str | None = None,
    policy: PromotionPolicy | None = None,
    database_path: Path | None = None,
) -> RegistryResult:
    """Register/reuse the latest selected run and apply alias-based promotion."""
    uri, _, experiment_id = configure_mlflow(
        tracking_uri, experiment_name or DEFAULT_EXPERIMENT_NAME
    )
    name = registered_model_name or os.getenv(
        "MLFLOW_REGISTERED_MODEL_NAME", DEFAULT_REGISTERED_MODEL_NAME
    )
    active_policy = policy or PromotionPolicy.from_environment()
    client = MlflowClient(tracking_uri=uri)
    run = find_latest_selected_run(client, experiment_id)
    logged_model = resolve_logged_model(client, experiment_id, run.info.run_id)

    version, reused, count_before = _register_or_reuse(
        client, name=name, run=run, logged_model=logged_model
    )
    for key, value in {
        "project": "mlops-customer-churn-platform",
        "problem_type": "binary_classification",
        "target": "churn_flag",
        "framework": "sklearn_pipeline",
    }.items():
        client.set_registered_model_tag(name, key, value)

    champion = _alias(client, name, "champion")
    candidate_metrics = {
        metric: run.data.metrics[metric] for metric in REQUIRED_TEST_METRICS
    }
    if champion is not None and champion.version == version.version:
        decision = PromotionDecision(
            promoted=True,
            candidate_version=str(version.version),
            previous_champion_version=str(champion.version),
            new_champion_version=str(champion.version),
            reasons=("exact candidate is already champion",),
            candidate_metrics=candidate_metrics,
            champion_metrics=_metrics_from_version(champion),
            gate_results={"already_champion": True},
        )
    else:
        client.set_registered_model_alias(name, "challenger", version.version)
        resolved = client.get_model_version_by_alias(name, "challenger")
        if resolved.version != version.version:
            raise RuntimeError("Challenger alias did not resolve to the candidate")
        champion_metrics = _metrics_from_version(champion) if champion else None
        decision = evaluate_promotion(
            candidate_metrics,
            active_policy,
            champion_metrics,
            candidate_version=str(version.version),
            champion_version=str(champion.version) if champion else None,
        )
        if decision.promoted:
            client.set_registered_model_alias(name, "champion", version.version)
            client.set_model_version_tag(
                name, version.version, "promotion_status", "champion"
            )
            if champion is not None:
                client.set_model_version_tag(
                    name, champion.version, "promotion_status", "superseded"
                )
            client.delete_registered_model_alias(name, "challenger")
        else:
            reason = ", ".join(decision.reasons)
            client.set_model_version_tag(
                name, version.version, "promotion_status", "rejected"
            )
            client.set_model_version_tag(
                name, version.version, "promotion_reason", reason
            )

    champion = _alias(client, name, "champion")
    challenger = _alias(client, name, "challenger")
    predictions: tuple[int, ...] = ()
    if champion is not None:
        champion_uri = f"models:/{name}@champion"
        model = mlflow.sklearn.load_model(champion_uri)
        dataset = load_feature_mart(database_path)
        features, _, _ = separate_features_and_target(dataset)
        values = model.predict(features.head(5))
        predictions = tuple(int(value) for value in values)
        if len(predictions) != 5 or not set(predictions) <= {0, 1}:
            raise RuntimeError(
                "Champion reload validation returned invalid predictions"
            )
    elif challenger is not None:
        mlflow.sklearn.load_model(f"models:/{name}@challenger")

    return RegistryResult(
        registered_model_name=name,
        candidate_version=str(version.version),
        source_run_id=run.info.run_id,
        source_logged_model_id=logged_model.model_id,
        source_logged_model_uri=logged_model.model_uri,
        reused_version=reused,
        version_count_before=count_before,
        version_count_after=len(_model_versions(client, name)),
        decision=decision,
        champion_version=str(champion.version) if champion else None,
        challenger_version=str(challenger.version) if challenger else None,
        champion_prediction_count=len(predictions),
        champion_predictions=predictions,
        tracking_uri=uri,
    )


def main() -> None:
    """Execute and report the local registry lifecycle."""
    result = run_registry_lifecycle()
    decision = "PROMOTED" if result.decision.promoted else "REJECTED"
    print("MLflow registry lifecycle completed")
    print(f"Registered model: {result.registered_model_name}")
    print(f"Candidate version: {result.candidate_version}")
    print(f"Source run: {result.source_run_id}")
    print(f"Logged model: {result.source_logged_model_id}")
    print(f"Version reused: {result.reused_version}")
    for metric in ("test_roc_auc", "test_f1", "test_recall"):
        print(f"{metric}: {result.decision.candidate_metrics[metric]:.4f}")
    for gate, passed in result.decision.gate_results.items():
        print(f"Gate {gate}: {'PASS' if passed else 'FAIL'}")
    print(f"Promotion decision: {decision}")
    if result.decision.reasons:
        print(f"Reason: {', '.join(result.decision.reasons)}")
    print(f"Champion version: {result.champion_version or 'none'}")
    print(f"Challenger version: {result.challenger_version or 'none'}")
    if result.champion_version:
        print(f"Champion URI: models:/{result.registered_model_name}@champion")
        print(f"Champion predictions: {list(result.champion_predictions)}")


if __name__ == "__main__":
    main()
