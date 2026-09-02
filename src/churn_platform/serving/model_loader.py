"""Thread-safe lazy loading and inference for the MLflow champion model."""

from __future__ import annotations

import os
from contextlib import suppress
from dataclasses import dataclass
from threading import Lock
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from churn_platform.ml.registry import DEFAULT_REGISTERED_MODEL_NAME
from churn_platform.ml.tracking import default_tracking_uri
from churn_platform.serving.schemas import ChurnFeatures

DEFAULT_MODEL_ALIAS = "champion"


class ModelUnavailableError(RuntimeError):
    """Raised when the configured registry model cannot serve predictions."""


@dataclass(frozen=True, slots=True)
class Prediction:
    """Internal prediction with resolved model identity."""

    value: int
    probability: float
    model_name: str
    model_version: str
    model_alias: str


class ChampionModelLoader:
    """Load one immutable alias resolution and reuse its sklearn pipeline."""

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        model_name: str | None = None,
        model_alias: str | None = None,
    ) -> None:
        self.tracking_uri = (
            tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or default_tracking_uri()
        )
        self.model_name = model_name or os.getenv(
            "MLFLOW_REGISTERED_MODEL_NAME", DEFAULT_REGISTERED_MODEL_NAME
        )
        self.model_alias = model_alias or os.getenv(
            "MLFLOW_MODEL_ALIAS", DEFAULT_MODEL_ALIAS
        )
        self.model_version: str | None = None
        self.loading_status = "not_loaded"
        self.error: str | None = None
        self._model: Any | None = None
        self._lock = Lock()

    @property
    def model_uri(self) -> str:
        """Return the alias URI used for a genuine registry load."""
        return f"models:/{self.model_name}@{self.model_alias}"

    def load(self) -> None:
        """Resolve the alias and load the complete preprocessing pipeline once."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                client = MlflowClient(tracking_uri=self.tracking_uri)
                version = client.get_model_version_by_alias(
                    self.model_name, self.model_alias
                )
                self._model = mlflow.sklearn.load_model(self.model_uri)
                self.model_version = str(version.version)
                self.loading_status = "ready"
                self.error = None
            except Exception as error:
                self.loading_status = "error"
                self.error = (
                    f"{type(error).__name__}: configured champion could not be loaded"
                )
                raise ModelUnavailableError(
                    f"Unable to load configured champion {self.model_uri}"
                ) from error

    def info(self) -> dict[str, str | None]:
        """Return non-secret loader metadata, attempting a lazy load first."""
        if self.loading_status == "not_loaded":
            with suppress(ModelUnavailableError):
                self.load()
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_alias": self.model_alias,
            "loading_status": self.loading_status,
            "error": self.error,
        }

    def predict(self, features: ChurnFeatures) -> Prediction:
        """Predict with the exact registered preprocessing-and-model pipeline."""
        self.load()
        frame = pd.DataFrame([features.model_dump()])
        value = int(self._model.predict(frame)[0])
        if value not in {0, 1}:
            raise RuntimeError(f"Champion returned invalid class {value}")
        if not hasattr(self._model, "predict_proba"):
            raise RuntimeError("Champion model does not expose predict_proba")
        probability = float(self._model.predict_proba(frame)[0][1])
        if not 0 <= probability <= 1:
            raise RuntimeError("Champion returned an invalid churn probability")
        if self.model_version is None:
            raise RuntimeError("Champion version was not resolved")
        return Prediction(
            value=value,
            probability=probability,
            model_name=self.model_name,
            model_version=self.model_version,
            model_alias=self.model_alias,
        )
