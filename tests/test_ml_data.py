"""Tests for feature-mart validation and deterministic splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from churn_platform.ml.data import (
    EXCLUDED_COLUMNS,
    FeatureMartValidationError,
    separate_features_and_target,
    split_dataset,
    validate_feature_mart,
)


def test_valid_feature_mart_passes(synthetic_feature_mart: pd.DataFrame) -> None:
    validated = validate_feature_mart(synthetic_feature_mart)
    assert len(validated) == 60


def test_missing_target_fails(synthetic_feature_mart: pd.DataFrame) -> None:
    with pytest.raises(FeatureMartValidationError, match="churn_flag"):
        validate_feature_mart(synthetic_feature_mart.drop(columns="churn_flag"))


def test_duplicate_customer_id_fails(synthetic_feature_mart: pd.DataFrame) -> None:
    synthetic_feature_mart.loc[1, "customer_id"] = synthetic_feature_mart.loc[
        0, "customer_id"
    ]
    with pytest.raises(FeatureMartValidationError, match="duplicate"):
        validate_feature_mart(synthetic_feature_mart)


def test_invalid_target_fails(synthetic_feature_mart: pd.DataFrame) -> None:
    synthetic_feature_mart.loc[0, "churn_flag"] = 2
    with pytest.raises(FeatureMartValidationError, match="invalid"):
        validate_feature_mart(synthetic_feature_mart)


def test_leakage_columns_are_excluded(synthetic_feature_mart: pd.DataFrame) -> None:
    features, _, _ = separate_features_and_target(synthetic_feature_mart)
    assert not set(EXCLUDED_COLUMNS) & set(features.columns)


def test_numeric_and_categorical_features_are_identified(
    synthetic_feature_mart: pd.DataFrame,
) -> None:
    _, _, metadata = separate_features_and_target(synthetic_feature_mart)
    assert "tenure" in metadata.numeric_features
    assert "contract" in metadata.categorical_features
    assert metadata.feature_count == 6


def test_split_preserves_total_row_count(
    synthetic_feature_mart: pd.DataFrame,
) -> None:
    features, target, _ = separate_features_and_target(synthetic_feature_mart)
    splits = split_dataset(features, target)
    assert len(splits.x_train) + len(splits.x_validation) + len(splits.x_test) == 60


def test_split_is_deterministic(synthetic_feature_mart: pd.DataFrame) -> None:
    features, target, _ = separate_features_and_target(synthetic_feature_mart)
    first = split_dataset(features, target)
    second = split_dataset(features, target)
    assert first.x_train.index.tolist() == second.x_train.index.tolist()
    assert first.x_validation.index.tolist() == second.x_validation.index.tolist()
    assert first.x_test.index.tolist() == second.x_test.index.tolist()


def test_both_classes_exist_in_every_split(
    synthetic_feature_mart: pd.DataFrame,
) -> None:
    features, target, _ = separate_features_and_target(synthetic_feature_mart)
    splits = split_dataset(features, target)
    assert set(splits.y_train) == {0, 1}
    assert set(splits.y_validation) == {0, 1}
    assert set(splits.y_test) == {0, 1}
