"""Evidently-backed data drift analysis for bounded churn features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from churn_platform.ml.data import EXCLUDED_COLUMNS

DATASET_DRIFT_SHARE_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    """One feature's Evidently score and configured decision threshold."""

    feature: str
    score: float
    method: str
    threshold: float
    detected: bool


@dataclass(frozen=True, slots=True)
class DriftResult:
    """Aggregated Evidently drift result."""

    detected: bool
    drifted_features: int
    total_features: int
    drift_share: float
    features: tuple[FeatureDrift, ...]
    html_path: Path
    json_path: Path


def monitoring_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return bounded model inputs, excluding identifiers and targets."""
    return dataset.drop(columns=list(EXCLUDED_COLUMNS), errors="ignore")


def generate_drifted_dataset(reference: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic, schema-preserving distribution shift."""
    current = reference.copy()
    if "tenure" in current:
        current["tenure"] = (current["tenure"].astype(float) + 48).clip(upper=72)
    if "monthly_charges" in current:
        current["monthly_charges"] = current["monthly_charges"].astype(float) * 1.8
    if "contract" in current:
        current["contract"] = "Month-to-month"
    if "payment_method" in current:
        current["payment_method"] = "Electronic check"
    return current


def analyze_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path,
) -> DriftResult:
    """Run Evidently and persist human- and machine-readable reports."""
    reference_features = monitoring_features(reference)
    current_features = monitoring_features(current)
    if tuple(reference_features.columns) != tuple(current_features.columns):
        raise ValueError("Reference and current monitoring schemas must match")
    if reference_features.empty or current_features.empty:
        raise ValueError("Monitoring datasets must contain rows and features")

    snapshot = Report([DataDriftPreset(drift_share=DATASET_DRIFT_SHARE_THRESHOLD)]).run(
        current_data=current_features, reference_data=reference_features
    )
    payload = snapshot.dict()
    feature_results = tuple(_feature_results(payload))
    count_metric = next(
        metric
        for metric in payload["metrics"]
        if metric["config"]["type"].endswith("DriftedColumnsCount")
    )
    drifted = int(count_metric["value"]["count"])
    share = float(count_metric["value"]["share"])

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "data_drift_report.html"
    json_path = output_dir / "data_drift_report.json"
    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))
    return DriftResult(
        detected=share >= DATASET_DRIFT_SHARE_THRESHOLD,
        drifted_features=drifted,
        total_features=len(feature_results),
        drift_share=share,
        features=feature_results,
        html_path=html_path,
        json_path=json_path,
    )


def _feature_results(payload: dict[str, object]):
    for metric in payload["metrics"]:
        config = metric["config"]
        if not config["type"].endswith("ValueDrift"):
            continue
        score = float(metric["value"])
        threshold = float(config["threshold"])
        method = str(config["method"])
        # Evidently defaults in this report are p-value based; smaller than the
        # configured threshold means the distributions differ significantly.
        detected = score < threshold if "p_value" in method else score >= threshold
        yield FeatureDrift(
            feature=str(config["column"]),
            score=score,
            method=method,
            threshold=threshold,
            detected=detected,
        )
