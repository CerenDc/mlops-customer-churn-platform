"""Pure champion/challenger promotion policy."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    """Configurable absolute floors and incumbent comparison guardrails."""

    min_test_roc_auc: float = 0.80
    min_test_f1: float = 0.60
    min_test_recall: float = 0.70
    min_roc_auc_improvement: float = 0.001
    max_f1_regression: float = 0.02
    max_recall_regression: float = 0.02

    @classmethod
    def from_environment(cls) -> PromotionPolicy:
        """Build policy from finite, non-negative environment values."""
        defaults = cls()
        fields = {
            "min_test_roc_auc": "MODEL_MIN_TEST_ROC_AUC",
            "min_test_f1": "MODEL_MIN_TEST_F1",
            "min_test_recall": "MODEL_MIN_TEST_RECALL",
            "min_roc_auc_improvement": "MODEL_MIN_ROC_AUC_IMPROVEMENT",
            "max_f1_regression": "MODEL_MAX_F1_REGRESSION",
            "max_recall_regression": "MODEL_MAX_RECALL_REGRESSION",
        }
        values = {}
        for field, variable in fields.items():
            raw = os.getenv(variable, str(getattr(defaults, field)))
            try:
                value = float(raw)
            except ValueError as error:
                raise ValueError(f"{variable} must be a number, got {raw!r}") from error
            if not 0 <= value <= 1:
                raise ValueError(f"{variable} must be between 0 and 1, got {value}")
            values[field] = value
        return cls(**values)


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Explain a deterministic promotion decision."""

    promoted: bool
    candidate_version: str | None
    previous_champion_version: str | None
    new_champion_version: str | None
    reasons: tuple[str, ...]
    candidate_metrics: dict[str, float]
    champion_metrics: dict[str, float] | None
    gate_results: dict[str, bool]


def evaluate_promotion(
    candidate_metrics: dict[str, float],
    policy: PromotionPolicy,
    champion_metrics: dict[str, float] | None = None,
    *,
    candidate_version: str | None = None,
    champion_version: str | None = None,
) -> PromotionDecision:
    """Apply bootstrap floors or incumbent-relative gates to a candidate."""
    required = {"test_roc_auc", "test_f1", "test_recall"}
    missing = sorted(required - candidate_metrics.keys())
    if missing:
        raise ValueError(f"Candidate metrics missing: {', '.join(missing)}")

    gates = {
        "absolute_test_roc_auc": (
            candidate_metrics["test_roc_auc"] >= policy.min_test_roc_auc
        ),
        "absolute_test_f1": candidate_metrics["test_f1"] >= policy.min_test_f1,
        "absolute_test_recall": (
            candidate_metrics["test_recall"] >= policy.min_test_recall
        ),
    }
    if champion_metrics is not None:
        missing = sorted(required - champion_metrics.keys())
        if missing:
            raise ValueError(f"Champion metrics missing: {', '.join(missing)}")
        gates.update(
            {
                "roc_auc_improvement": candidate_metrics["test_roc_auc"]
                >= champion_metrics["test_roc_auc"] + policy.min_roc_auc_improvement,
                "f1_guardrail": candidate_metrics["test_f1"]
                >= champion_metrics["test_f1"] - policy.max_f1_regression,
                "recall_guardrail": candidate_metrics["test_recall"]
                >= champion_metrics["test_recall"] - policy.max_recall_regression,
            }
        )

    reasons = tuple(name for name, passed in gates.items() if not passed)
    promoted = not reasons
    return PromotionDecision(
        promoted=promoted,
        candidate_version=candidate_version,
        previous_champion_version=champion_version,
        new_champion_version=candidate_version if promoted else champion_version,
        reasons=reasons,
        candidate_metrics=dict(candidate_metrics),
        champion_metrics=dict(champion_metrics) if champion_metrics else None,
        gate_results=gates,
    )
