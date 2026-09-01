"""Tests for deterministic champion/challenger promotion policy."""

from __future__ import annotations

import pytest

from churn_platform.ml.promotion import PromotionPolicy, evaluate_promotion

PASSING = {
    "test_roc_auc": 0.85,
    "test_f1": 0.65,
    "test_recall": 0.78,
}


def test_promotion_configuration_parses_environment(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_MIN_TEST_ROC_AUC", "0.81")
    monkeypatch.setenv("MODEL_MAX_F1_REGRESSION", "0.03")
    policy = PromotionPolicy.from_environment()
    assert policy.min_test_roc_auc == 0.81
    assert policy.max_f1_regression == 0.03


def test_invalid_promotion_configuration_fails(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_MIN_TEST_F1", "not-a-number")
    with pytest.raises(ValueError, match="must be a number"):
        PromotionPolicy.from_environment()


def test_bootstrap_candidate_passing_all_gates() -> None:
    decision = evaluate_promotion(PASSING, PromotionPolicy())
    assert decision.promoted


@pytest.mark.parametrize(
    ("metric", "value", "failed_gate"),
    [
        ("test_roc_auc", 0.799, "absolute_test_roc_auc"),
        ("test_f1", 0.599, "absolute_test_f1"),
        ("test_recall", 0.699, "absolute_test_recall"),
    ],
)
def test_bootstrap_candidate_failing_floor(
    metric: str, value: float, failed_gate: str
) -> None:
    candidate = {**PASSING, metric: value}
    decision = evaluate_promotion(candidate, PromotionPolicy())
    assert not decision.promoted
    assert failed_gate in decision.reasons


def test_existing_champion_with_sufficient_improvement_passes() -> None:
    champion = {"test_roc_auc": 0.84, "test_f1": 0.66, "test_recall": 0.79}
    decision = evaluate_promotion(PASSING, PromotionPolicy(), champion)
    assert decision.promoted


def test_insufficient_roc_improvement_fails() -> None:
    champion = {"test_roc_auc": 0.85, "test_f1": 0.65, "test_recall": 0.78}
    decision = evaluate_promotion(PASSING, PromotionPolicy(), champion)
    assert not decision.promoted
    assert "roc_auc_improvement" in decision.reasons


def test_excessive_f1_regression_fails() -> None:
    champion = {"test_roc_auc": 0.84, "test_f1": 0.68, "test_recall": 0.78}
    decision = evaluate_promotion(PASSING, PromotionPolicy(), champion)
    assert not decision.promoted
    assert "f1_guardrail" in decision.reasons


def test_excessive_recall_regression_fails() -> None:
    champion = {"test_roc_auc": 0.84, "test_f1": 0.65, "test_recall": 0.81}
    decision = evaluate_promotion(PASSING, PromotionPolicy(), champion)
    assert not decision.promoted
    assert "recall_guardrail" in decision.reasons


def test_exact_boundaries_pass() -> None:
    policy = PromotionPolicy()
    champion = {"test_roc_auc": 0.84, "test_f1": 0.65, "test_recall": 0.78}
    candidate = {
        "test_roc_auc": 0.841,
        "test_f1": 0.63,
        "test_recall": 0.76,
    }
    assert evaluate_promotion(candidate, policy, champion).promoted
