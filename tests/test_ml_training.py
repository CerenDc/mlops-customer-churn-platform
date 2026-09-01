"""Tests for preprocessing, candidate models, evaluation, and selection."""

from __future__ import annotations

import pandas as pd

from churn_platform.ml.data import separate_features_and_target, split_dataset
from churn_platform.ml.evaluate import EvaluationResult, evaluate_classifier
from churn_platform.ml.preprocessing import (
    build_logistic_pipeline,
    build_xgboost_pipeline,
)
from churn_platform.ml.tracking import CandidateRun
from churn_platform.ml.train import select_winner


def _prepared_data(dataset: pd.DataFrame):
    features, target, metadata = separate_features_and_target(dataset)
    return split_dataset(features, target), metadata


def test_logistic_regression_pipeline_trains(
    synthetic_feature_mart: pd.DataFrame,
) -> None:
    splits, metadata = _prepared_data(synthetic_feature_mart)
    pipeline = build_logistic_pipeline(metadata)
    pipeline.fit(splits.x_train, splits.y_train)
    assert len(pipeline.predict(splits.x_validation)) == len(splits.x_validation)


def test_xgboost_pipeline_trains(synthetic_feature_mart: pd.DataFrame) -> None:
    splits, metadata = _prepared_data(synthetic_feature_mart)
    pipeline = build_xgboost_pipeline(metadata, scale_pos_weight=3.0)
    pipeline.fit(splits.x_train, splits.y_train)
    assert len(pipeline.predict(splits.x_validation)) == len(splits.x_validation)


def test_pipeline_handles_unseen_category(
    synthetic_feature_mart: pd.DataFrame,
) -> None:
    splits, metadata = _prepared_data(synthetic_feature_mart)
    pipeline = build_logistic_pipeline(metadata)
    pipeline.fit(splits.x_train, splits.y_train)
    unseen = splits.x_validation.head(1).copy()
    unseen["contract"] = "Previously unseen contract"
    assert pipeline.predict(unseen)[0] in {0, 1}


def test_evaluation_returns_all_required_metrics(
    synthetic_feature_mart: pd.DataFrame,
) -> None:
    splits, metadata = _prepared_data(synthetic_feature_mart)
    pipeline = build_logistic_pipeline(metadata)
    pipeline.fit(splits.x_train, splits.y_train)
    result = evaluate_classifier(pipeline, splits.x_validation, splits.y_validation)
    assert set(result.metrics) == {
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "log_loss",
    }
    assert len(result.confusion_matrix) == 2


def test_winner_selection_uses_validation_roc_auc() -> None:
    lower = EvaluationResult({"roc_auc": 0.70}, {}, [[0, 0], [0, 0]])
    higher = EvaluationResult({"roc_auc": 0.80}, {}, [[0, 0], [0, 0]])
    candidates = [
        CandidateRun("logistic_regression", "a", "uri-a", None, lower),  # type: ignore[arg-type]
        CandidateRun("xgboost", "b", "uri-b", None, higher),  # type: ignore[arg-type]
    ]
    assert select_winner(candidates).model_name == "xgboost"
