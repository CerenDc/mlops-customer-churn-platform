"""Load, validate, and split the dbt customer churn feature mart."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DUCKDB_PATH = Path("data/features/churn_analytics.duckdb")
FEATURE_MART = "main_marts.fct_customer_churn_features"
TARGET_COLUMN = "churn_flag"
EXCLUDED_COLUMNS = ("customer_id", "churn", TARGET_COLUMN)
RANDOM_STATE = 42


class FeatureMartValidationError(ValueError):
    """Raised when the analytical feature contract is invalid for training."""


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    """Feature groups derived from the feature mart schema."""

    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]

    @property
    def feature_count(self) -> int:
        """Return the total number of predictive input fields."""
        return len(self.numeric_features) + len(self.categorical_features)


@dataclass(frozen=True, slots=True)
class DataSplits:
    """Reproducible train, validation, and untouched test partitions."""

    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def configured_duckdb_path() -> Path:
    """Return the feature database path configured for this process."""
    return Path(os.getenv("DBT_DUCKDB_PATH", str(DEFAULT_DUCKDB_PATH)))


def load_feature_mart(database_path: Path | None = None) -> pd.DataFrame:
    """Load only the dbt feature mart used as the ML training contract."""
    path = Path(database_path or configured_duckdb_path())
    if not path.is_file():
        raise FileNotFoundError(_missing_mart_message(path))

    try:
        with duckdb.connect(str(path), read_only=True) as connection:
            dataset = connection.execute(f"select * from {FEATURE_MART}").fetchdf()
    except duckdb.CatalogException as error:
        raise FileNotFoundError(_missing_mart_message(path)) from error
    return validate_feature_mart(dataset)


def _missing_mart_message(path: Path) -> str:
    return (
        f"Feature mart not found in {path}. Run the data pipeline first:\n"
        "python -m churn_platform.ingestion.telco\n"
        "python -m churn_platform.spark.transform_telco\n"
        "dbt build --project-dir dbt_project --profiles-dir dbt_project"
    )


def validate_feature_mart(dataset: pd.DataFrame) -> pd.DataFrame:
    """Validate the customer-grain ML data contract and return it unchanged."""
    if dataset.empty:
        raise FeatureMartValidationError("Feature mart contains no rows")

    required = set(EXCLUDED_COLUMNS)
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise FeatureMartValidationError(
            f"Feature mart is missing required columns: {', '.join(missing)}"
        )
    if dataset["customer_id"].isna().any():
        raise FeatureMartValidationError("customer_id contains null values")
    if not dataset["customer_id"].is_unique:
        raise FeatureMartValidationError("customer_id contains duplicate values")
    if dataset[TARGET_COLUMN].isna().any():
        raise FeatureMartValidationError("churn_flag contains null values")

    target_values = set(dataset[TARGET_COLUMN].unique())
    if not target_values <= {0, 1}:
        raise FeatureMartValidationError(
            f"churn_flag contains invalid values: {sorted(target_values - {0, 1})}"
        )
    if target_values != {0, 1}:
        raise FeatureMartValidationError("Both churn classes 0 and 1 must be present")
    return dataset


def separate_features_and_target(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, FeatureMetadata]:
    """Exclude identifiers and both target representations from model inputs."""
    validated = validate_feature_mart(dataset)
    features = validated.drop(columns=list(EXCLUDED_COLUMNS))
    leakage = set(features.columns) & set(EXCLUDED_COLUMNS)
    if leakage:
        raise FeatureMartValidationError(
            f"Target leakage columns remain in features: {sorted(leakage)}"
        )

    numeric = tuple(features.select_dtypes(include="number").columns)
    categorical = tuple(column for column in features if column not in numeric)
    if not numeric or not categorical:
        raise FeatureMartValidationError(
            "Training requires both numeric and categorical features"
        )
    metadata = FeatureMetadata(numeric, categorical)
    return features, validated[TARGET_COLUMN].astype(int), metadata


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    random_state: int = RANDOM_STATE,
) -> DataSplits:
    """Create deterministic 70/15/15 stratified partitions."""
    x_train, x_remaining, y_train, y_remaining = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=random_state,
        stratify=target,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_remaining,
        y_remaining,
        test_size=0.50,
        random_state=random_state,
        stratify=y_remaining,
    )
    return DataSplits(
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )
