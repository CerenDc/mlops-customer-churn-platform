"""HTTP and loader regression tests for champion model serving."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churn_platform.serving.app import create_app
from churn_platform.serving.model_loader import (
    ChampionModelLoader,
    ModelUnavailableError,
    Prediction,
)
from churn_platform.serving.schemas import ChurnFeatures


def valid_payload() -> dict[str, object]:
    """Return one feature row matching the dbt mart contract."""
    return {
        "tenure": 5,
        "monthly_charges": 75.5,
        "total_charges": 377.5,
        "senior_citizen": 0,
        "gender": "Female",
        "partner": "No",
        "dependents": "No",
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "has_internet": 1,
        "has_phone": 1,
        "service_count": 5,
        "is_month_to_month": 1,
        "has_long_term_contract": 0,
        "monthly_to_total_charge_ratio": 0.2,
        "tenure_group": "0-12 months",
    }


@dataclass
class ReadyLoader:
    """Deterministic local test double at the HTTP dependency boundary."""

    def info(self) -> dict[str, str | None]:
        return {
            "model_name": "telco-churn-classifier",
            "model_version": "7",
            "model_alias": "champion",
            "loading_status": "ready",
            "error": None,
        }

    def load(self) -> None:
        return None

    def predict(self, features: ChurnFeatures) -> Prediction:
        assert features.tenure == 5
        return Prediction(
            value=1,
            probability=0.82,
            model_name="telco-churn-classifier",
            model_version="7",
            model_alias="champion",
        )


class UnavailableLoader(ReadyLoader):
    """Test double representing an unavailable registry backend."""

    def predict(self, features: ChurnFeatures) -> Prediction:
        raise ModelUnavailableError("champion alias is unavailable")

    def load(self) -> None:
        raise ModelUnavailableError("champion alias is unavailable")


def test_health_does_not_require_a_loaded_model() -> None:
    response = TestClient(create_app(loader=UnavailableLoader())).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_reports_resolved_champion() -> None:
    response = TestClient(create_app(loader=ReadyLoader())).get("/model/info")
    assert response.status_code == 200
    assert response.json() == {
        "model_name": "telco-churn-classifier",
        "model_version": "7",
        "model_alias": "champion",
        "loading_status": "ready",
        "error": None,
    }


def test_readiness_requires_champion() -> None:
    ready = TestClient(create_app(loader=ReadyLoader())).get("/ready")
    unavailable = TestClient(create_app(loader=UnavailableLoader())).get("/ready")
    assert ready.status_code == 200
    assert unavailable.status_code == 503


def test_metrics_endpoint_exposes_serving_counters() -> None:
    response = TestClient(create_app(loader=ReadyLoader())).get("/metrics")
    assert response.status_code == 200
    assert "churn_serving_inference_requests_total" in response.text


def test_successful_prediction_returns_probability_and_identity() -> None:
    response = TestClient(create_app(loader=ReadyLoader())).post(
        "/predict", json=valid_payload()
    )
    assert response.status_code == 200
    assert response.json() == {
        "prediction": 1,
        "churn": True,
        "churn_probability": 0.82,
        "model_name": "telco-churn-classifier",
        "model_version": "7",
        "model_alias": "champion",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [("tenure", -1), ("gender", "unknown"), ("unexpected", "value")],
)
def test_invalid_request_is_rejected(field: str, value: object) -> None:
    payload = valid_payload()
    payload[field] = value
    response = TestClient(create_app(loader=ReadyLoader())).post(
        "/predict", json=payload
    )
    assert response.status_code == 422


def test_malformed_json_is_rejected() -> None:
    response = TestClient(create_app(loader=ReadyLoader())).post(
        "/predict",
        content="not-json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_unavailable_model_returns_service_unavailable() -> None:
    response = TestClient(create_app(loader=UnavailableLoader())).post(
        "/predict", json=valid_payload()
    )
    assert response.status_code == 503
    assert "champion alias is unavailable" in response.json()["detail"]


def test_loader_resolves_alias_and_uses_complete_pipeline(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class Version:
        version = "12"

    class Client:
        def __init__(self, *, tracking_uri: str) -> None:
            observed["tracking_uri"] = tracking_uri

        def get_model_version_by_alias(self, name: str, alias: str) -> Version:
            observed["alias"] = (name, alias)
            return Version()

    class Pipeline:
        def predict(self, frame: pd.DataFrame) -> list[int]:
            observed["columns"] = tuple(frame.columns)
            return [0]

        def predict_proba(self, frame: pd.DataFrame) -> list[list[float]]:
            return [[0.73, 0.27]]

    monkeypatch.setattr("churn_platform.serving.model_loader.MlflowClient", Client)
    monkeypatch.setattr(
        "churn_platform.serving.model_loader.mlflow.sklearn.load_model",
        lambda uri: observed.setdefault("model_uri", uri) and Pipeline(),
    )
    loader = ChampionModelLoader(
        tracking_uri="sqlite:////tmp/test.db",
        model_name="telco",
        model_alias="champion",
    )
    result = loader.predict(ChurnFeatures.model_validate(valid_payload()))

    assert observed["alias"] == ("telco", "champion")
    assert observed["model_uri"] == "models:/telco@champion"
    assert len(observed["columns"]) == 26
    assert result.value == 0
    assert result.probability == 0.27
    assert result.model_version == "12"


def test_loader_failure_is_explicit(monkeypatch) -> None:
    class BrokenClient:
        def __init__(self, *, tracking_uri: str) -> None:
            pass

        def get_model_version_by_alias(self, name: str, alias: str):
            raise RuntimeError("registry offline")

    monkeypatch.setattr(
        "churn_platform.serving.model_loader.MlflowClient", BrokenClient
    )
    loader = ChampionModelLoader(
        tracking_uri="sqlite:////tmp/missing.db", model_name="telco"
    )

    with pytest.raises(
        ModelUnavailableError, match="Unable to load configured champion"
    ):
        loader.load()
    assert loader.loading_status == "error"
    assert loader.model_version is None
    assert loader.error == "RuntimeError: configured champion could not be loaded"
