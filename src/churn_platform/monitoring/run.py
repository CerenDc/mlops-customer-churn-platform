"""Run reproducible drift and optional champion performance monitoring."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from churn_platform.ml.data import (
    TARGET_COLUMN,
    load_feature_mart,
    separate_features_and_target,
)
from churn_platform.ml.evaluate import evaluate_classifier
from churn_platform.ml.registry import DEFAULT_REGISTERED_MODEL_NAME
from churn_platform.ml.tracking import default_tracking_uri
from churn_platform.monitoring.drift import analyze_drift, generate_drifted_dataset
from churn_platform.monitoring.state import (
    configured_monitoring_root,
    write_metrics_state,
)

LOGGER = logging.getLogger(__name__)
MAX_ROC_AUC_DEGRADATION = 0.05


def load_monitoring_dataset(path: Path | None) -> pd.DataFrame:
    """Load an explicit CSV or reuse the existing dbt feature mart."""
    return pd.read_csv(path) if path else load_feature_mart()


def run_monitoring(
    *,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path,
    metrics_state_path: Path | None = None,
    include_model: bool = True,
) -> dict[str, object]:
    """Evaluate drift, optional champion performance, and publish one snapshot."""
    drift = analyze_drift(reference, current, output_dir)
    state: dict[str, object] = {
        "churn_monitoring_last_run_timestamp_seconds": time.time(),
        "churn_monitoring_status": 1 if drift.detected else 0,
        "churn_data_feature_rows": len(current),
        "churn_data_drift_detected": drift.detected,
        "churn_data_drift_share": drift.drift_share,
        "churn_data_drifted_features": drift.drifted_features,
        "churn_data_total_features": drift.total_features,
        "features": [
            {
                "feature": item.feature,
                "score": item.score,
                "method": item.method,
                "threshold": item.threshold,
                "detected": item.detected,
            }
            for item in drift.features
        ],
        "recommendation": "INVESTIGATE" if drift.detected else "NO_ACTION",
        "report_html": str(drift.html_path),
        "report_json": str(drift.json_path),
        **_dbt_test_metrics(),
        **_airflow_pipeline_metrics(),
    }
    if include_model:
        model_state = _champion_performance(reference, current)
        state.update(model_state)
        delta = model_state.get("churn_model_roc_auc_delta")
        if delta is not None and float(delta) < -MAX_ROC_AUC_DEGRADATION:
            state["churn_monitoring_status"] = 2
            state["recommendation"] = "RETRAIN_RECOMMENDED"

    state_path = write_metrics_state(state, metrics_state_path)
    LOGGER.info(
        "event=drift_analysis_completed drift_detected=%s drift_share=%.4f "
        "drifted_features=%d total_features=%d status=%s recommendation=%s",
        drift.detected,
        drift.drift_share,
        drift.drifted_features,
        drift.total_features,
        state["churn_monitoring_status"],
        state["recommendation"],
    )
    LOGGER.info("event=monitoring_state_published path=%s", state_path)
    return state


def _champion_performance(
    reference: pd.DataFrame, current: pd.DataFrame
) -> dict[str, object]:
    if TARGET_COLUMN not in reference or TARGET_COLUMN not in current:
        return {}
    uri = os.getenv("MLFLOW_TRACKING_URI") or default_tracking_uri()
    name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", DEFAULT_REGISTERED_MODEL_NAME)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    try:
        champion = client.get_model_version_by_alias(name, "champion")
    except MlflowException:
        LOGGER.info("event=champion_monitoring_skipped reason=champion_not_available")
        return {}

    model = mlflow.sklearn.load_model(f"models:/{name}@champion")
    metrics = evaluate_model_performance(model, reference, current)
    metrics["champion"] = {
        "model": champion.tags.get("model_type", "unknown"),
        "version": str(champion.version),
    }
    return metrics


def evaluate_model_performance(
    model, reference: pd.DataFrame, current: pd.DataFrame
) -> dict[str, object]:
    """Evaluate one fitted champion where delayed ground truth is available."""
    reference_x, reference_y, _ = separate_features_and_target(reference)
    current_x, current_y, _ = separate_features_and_target(current)
    reference_result = evaluate_classifier(model, reference_x, reference_y)
    current_result = evaluate_classifier(model, current_x, current_y)
    metrics: dict[str, object] = {}
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        metrics[f"churn_model_reference_{metric}"] = reference_result.metrics[metric]
        metrics[f"churn_model_current_{metric}"] = current_result.metrics[metric]
    metrics["churn_model_roc_auc_delta"] = (
        current_result.metrics["roc_auc"] - reference_result.metrics["roc_auc"]
    )
    return metrics


def _dbt_test_metrics() -> dict[str, int]:
    path = Path("dbt_project/target/run_results.json")
    if not path.is_file():
        return {}
    results = json.loads(path.read_text()).get("results", [])
    tests = [item for item in results if item.get("unique_id", "").startswith("test.")]
    passed = sum(item.get("status") == "pass" for item in tests)
    return {
        "churn_dbt_tests_total": len(tests),
        "churn_dbt_tests_passed": passed,
        "churn_dbt_tests_failed": len(tests) - passed,
    }


def _airflow_pipeline_metrics() -> dict[str, object]:
    """Read bounded aggregate metrics from Airflow's metadata database."""
    try:
        from airflow.models import DagRun, TaskInstance
        from airflow.utils.session import create_session
    except ImportError:
        return {}

    try:
        with create_session() as session:
            runs = (
                session.query(DagRun)
                .filter(DagRun.dag_id == "mlops_customer_churn_pipeline")
                .order_by(DagRun.logical_date.desc())
                .all()
            )
            completed = [run for run in runs if run.end_date and run.start_date]
            successful = [run for run in runs if run.state == "success"]
            failed = [run for run in runs if run.state == "failed"]
            metrics: dict[str, object] = {
                "churn_pipeline_runs_total": len(runs),
                "churn_pipeline_success_total": len(successful),
                "churn_pipeline_failures_total": len(failed),
            }
            if successful:
                latest_success = successful[0]
                metrics["churn_pipeline_last_success_timestamp_seconds"] = (
                    latest_success.end_date.timestamp()
                )
            if completed:
                latest = completed[0]
                metrics["churn_pipeline_duration_seconds"] = (
                    latest.end_date - latest.start_date
                ).total_seconds()
                tasks = (
                    session.query(TaskInstance)
                    .filter(
                        TaskInstance.dag_id == latest.dag_id,
                        TaskInstance.run_id == latest.run_id,
                    )
                    .all()
                )
                metrics["stages"] = [
                    {
                        "stage": task.task_id,
                        "duration": (task.end_date - task.start_date).total_seconds(),
                    }
                    for task in tasks
                    if task.start_date and task.end_date
                ]
            return metrics
    except Exception as error:  # Airflow availability must not block drift analysis.
        LOGGER.warning("event=airflow_metrics_unavailable error=%s", error)
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--scenario", choices=("normal", "drifted"), default="normal")
    parser.add_argument("--output-dir", type=Path, default=configured_monitoring_root())
    parser.add_argument("--metrics-state", type=Path)
    parser.add_argument("--skip-model", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    reference = load_monitoring_dataset(arguments.reference)
    if arguments.current:
        current = load_monitoring_dataset(arguments.current)
    elif arguments.scenario == "drifted":
        current = generate_drifted_dataset(reference)
    else:
        current = reference.copy()
    state = run_monitoring(
        reference=reference,
        current=current,
        output_dir=arguments.output_dir / arguments.scenario,
        metrics_state_path=arguments.metrics_state,
        include_model=not arguments.skip_model,
    )
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
