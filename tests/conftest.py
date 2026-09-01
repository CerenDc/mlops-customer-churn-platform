"""Shared deterministic test data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_feature_mart() -> pd.DataFrame:
    """Return a customer-grain mart with enough rows for stratified splits."""
    rows = []
    for index in range(60):
        churn_flag = 1 if index % 4 == 0 else 0
        rows.append(
            {
                "customer_id": f"CUSTOMER-{index:04d}",
                "churn": "Yes" if churn_flag else "No",
                "churn_flag": churn_flag,
                "tenure": index % 48,
                "monthly_charges": np.nan if index == 1 else 30.0 + index,
                "total_charges": np.nan if index % 13 == 0 else 100.0 + index * 20,
                "service_count": index % 8,
                "contract": np.nan
                if index == 2
                else ("Month-to-month" if index % 3 else "One year"),
                "internet_service": "Fiber optic" if index % 2 else "DSL",
            }
        )
    return pd.DataFrame(rows)
