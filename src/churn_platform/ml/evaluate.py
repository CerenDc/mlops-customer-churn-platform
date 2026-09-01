"""Classification metrics and structured evaluation artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Metrics and JSON-compatible evaluation details."""

    metrics: dict[str, float]
    classification_report: dict[str, object]
    confusion_matrix: list[list[int]]


def evaluate_classifier(
    model: Pipeline, features: pd.DataFrame, target: pd.Series
) -> EvaluationResult:
    """Evaluate a fitted binary classifier using probabilities and classes."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "pr_auc": float(average_precision_score(target, probabilities)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(target, predictions)),
        "log_loss": float(log_loss(target, probabilities, labels=[0, 1])),
    }
    report = classification_report(
        target, predictions, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(target, predictions, labels=[0, 1]).tolist()
    return EvaluationResult(metrics, report, matrix)
