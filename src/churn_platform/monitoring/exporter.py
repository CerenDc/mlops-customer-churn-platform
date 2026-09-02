"""Small HTTP exporter for the latest persistent monitoring snapshot."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from prometheus_client import CollectorRegistry, Gauge, generate_latest

from churn_platform.monitoring.state import read_metrics_state


def build_registry(state_path: Path | None = None) -> CollectorRegistry:
    """Build an isolated registry from the latest atomic JSON snapshot."""
    registry = CollectorRegistry()
    state = read_metrics_state(state_path)
    available = _gauge(
        registry, "churn_monitoring_snapshot_available", "Latest snapshot exists"
    )
    available.set(bool(state))
    if not state:
        return registry

    scalar_metrics = {
        "churn_monitoring_last_run_timestamp_seconds": "Last monitoring run time",
        "churn_monitoring_status": "Status: 0 healthy, 1 warning, 2 critical",
        "churn_data_feature_rows": "Rows in the monitored current feature dataset",
        "churn_data_drift_detected": "Whether dataset-level drift was detected",
        "churn_data_drift_share": "Share of monitored features that drifted",
        "churn_data_drifted_features": "Number of drifted features",
        "churn_data_total_features": "Number of monitored features",
        "churn_dbt_tests_total": "dbt tests in the latest structured result",
        "churn_dbt_tests_passed": "Successful dbt tests",
        "churn_dbt_tests_failed": "Failed dbt tests",
        "churn_pipeline_runs_total": "Observed Airflow training DAG runs",
        "churn_pipeline_success_total": "Successful Airflow training DAG runs",
        "churn_pipeline_failures_total": "Failed Airflow training DAG runs",
        "churn_pipeline_last_success_timestamp_seconds": "Last successful run time",
        "churn_pipeline_duration_seconds": "Duration of the latest completed run",
    }
    for name, documentation in scalar_metrics.items():
        if name in state:
            _gauge(registry, name, documentation).set(float(state[name]))

    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        for scope in ("reference", "current"):
            key = f"churn_model_{scope}_{metric}"
            if key in state:
                _gauge(registry, key, f"Champion {scope} {metric}").set(state[key])
    if "churn_model_roc_auc_delta" in state:
        _gauge(
            registry,
            "churn_model_roc_auc_delta",
            "Current minus reference champion ROC-AUC",
        ).set(state["churn_model_roc_auc_delta"])

    feature_score = Gauge(
        "churn_feature_drift_score",
        "Evidently drift score; see method label for semantics",
        ("feature", "method"),
        registry=registry,
    )
    feature_detected = Gauge(
        "churn_feature_drift_detected",
        "Whether Evidently detected drift for a bounded feature",
        ("feature",),
        registry=registry,
    )
    for feature in state.get("features", []):
        feature_score.labels(feature["feature"], feature["method"]).set(
            feature["score"]
        )
        feature_detected.labels(feature["feature"]).set(feature["detected"])

    stage_duration = Gauge(
        "churn_pipeline_stage_duration_seconds",
        "Latest completed Airflow task duration",
        ("stage",),
        registry=registry,
    )
    for stage in state.get("stages", []):
        stage_duration.labels(stage["stage"]).set(stage["duration"])

    champion = state.get("champion")
    if champion:
        Gauge(
            "churn_model_champion_info",
            "Current champion identity (bounded local registry cardinality)",
            ("model", "version"),
            registry=registry,
        ).labels(champion["model"], champion["version"]).set(1)
    return registry


def _gauge(registry: CollectorRegistry, name: str, documentation: str) -> Gauge:
    return Gauge(name, documentation, registry=registry)


def handler_for(state_path: Path | None = None):
    """Create an HTTP handler bound to a configured state file."""

    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._respond(
                    HTTPStatus.OK,
                    b'{"status":"available"}\n',
                    "application/json",
                )
                return
            if self.path == "/metrics":
                self._respond(
                    HTTPStatus.OK,
                    generate_latest(build_registry(state_path)),
                    "text/plain; version=0.0.4; charset=utf-8",
                )
                return
            self._respond(HTTPStatus.NOT_FOUND, b"not found\n", "text/plain")

        def _respond(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return MetricsHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--state", type=Path)
    arguments = parser.parse_args()
    server = ThreadingHTTPServer(
        (arguments.host, arguments.port), handler_for(arguments.state)
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
