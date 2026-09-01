"""Leakage-safe preprocessing and candidate model pipelines."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from churn_platform.ml.data import RANDOM_STATE, FeatureMetadata


def build_preprocessor(
    metadata: FeatureMetadata, *, scale_numeric: bool
) -> ColumnTransformer:
    """Build preprocessing fitted only when its enclosing pipeline is trained."""
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(metadata.numeric_features)),
            (
                "categorical",
                categorical_pipeline,
                list(metadata.categorical_features),
            ),
        ],
        verbose_feature_names_out=True,
    )


def build_logistic_pipeline(metadata: FeatureMetadata) -> Pipeline:
    """Create the interpretable class-balanced baseline pipeline."""
    estimator = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(metadata, scale_numeric=True)),
            ("classifier", estimator),
        ]
    )


def build_xgboost_pipeline(
    metadata: FeatureMetadata, *, scale_pos_weight: float
) -> Pipeline:
    """Create a lightweight deterministic CPU boosted-tree pipeline."""
    estimator = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(metadata, scale_numeric=False)),
            ("classifier", estimator),
        ]
    )
