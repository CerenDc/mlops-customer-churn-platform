from __future__ import annotations

import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from churn_platform.ml.data import separate_features_and_target
from churn_platform.ml.preprocessing import build_logistic_pipeline
from churn_platform.monitoring.drift import (
    analyze_drift,
    generate_drifted_dataset,
)
from churn_platform.monitoring.exporter import handler_for
from churn_platform.monitoring.run import evaluate_model_performance, run_monitoring


def monitoring_frame(rows: int = 240) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"customer-{index}" for index in range(rows)],
            "churn": ["Yes" if index % 4 == 0 else "No" for index in range(rows)],
            "churn_flag": [1 if index % 4 == 0 else 0 for index in range(rows)],
            "tenure": [index % 72 for index in range(rows)],
            "monthly_charges": [30.0 + index % 50 for index in range(rows)],
            "contract": [
                ["Month-to-month", "One year", "Two year"][index % 3]
                for index in range(rows)
            ],
            "payment_method": [
                ["Electronic check", "Credit card"][index % 2] for index in range(rows)
            ],
        }
    )


def test_identical_data_has_no_drift(tmp_path: Path) -> None:
    reference = monitoring_frame()
    result = analyze_drift(reference, reference.copy(), tmp_path)

    assert result.detected is False
    assert result.drifted_features == 0
    assert result.drift_share == 0
    assert result.html_path.is_file()
    assert result.json_path.is_file()


def test_deterministic_shift_is_detected(tmp_path: Path) -> None:
    reference = monitoring_frame()
    current = generate_drifted_dataset(reference)
    result = analyze_drift(reference, current, tmp_path)

    assert result.detected is True
    assert result.drifted_features >= 2
    assert result.drift_share >= 0.5


def test_monitoring_state_is_exposed_over_http(tmp_path: Path) -> None:
    reference = monitoring_frame()
    state_path = tmp_path / "metrics.json"
    state = run_monitoring(
        reference=reference,
        current=reference.copy(),
        output_dir=tmp_path / "report",
        metrics_state_path=state_path,
        include_model=False,
    )
    assert state["recommendation"] == "NO_ACTION"

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(state_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        root = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(f"{root}/health") as response:
            assert response.status == 200
        with urllib.request.urlopen(f"{root}/metrics") as response:
            metrics = response.read().decode()
        assert "churn_monitoring_snapshot_available 1.0" in metrics
        assert "churn_data_drift_detected 0.0" in metrics
        assert "churn_feature_drift_score" in metrics
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_ground_truth_model_monitoring_is_deterministic() -> None:
    reference = monitoring_frame()
    features, target, metadata = separate_features_and_target(reference)
    model = build_logistic_pipeline(metadata).fit(features, target)

    metrics = evaluate_model_performance(model, reference, reference.copy())

    assert metrics["churn_model_roc_auc_delta"] == 0
    assert 0 <= metrics["churn_model_current_f1"] <= 1
