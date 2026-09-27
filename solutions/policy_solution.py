"""Reference solution for the decision-policy task."""

from __future__ import annotations

import numpy as np
import pandas as pd

from solutions.causal_solution import t_learner_uplift
from src.policy import select_under_budget
from src.predictive import fit_purchase_baseline
from src.workshop_data import ARTIFACT_COLUMNS, GROWTH_BUDGET, split_train_test


def expected_incremental_profit(
    uplift: np.ndarray,
    expected_order_margin: np.ndarray,
    expected_coupon_cost: np.ndarray,
) -> np.ndarray:
    """Money added per targeted customer after expected coupon cost."""
    return uplift * expected_order_margin - expected_coupon_cost


def reference_uplift(observed: pd.DataFrame) -> np.ndarray:
    """The pilot's winner: a logistic T-Learner fit on the training split, scored on everyone."""
    train, _ = split_train_test(observed)
    return t_learner_uplift(train, observed)[2]


def build_customer_scores(
    observed: pd.DataFrame, uplift: np.ndarray | None = None
) -> pd.DataFrame:
    """Produce the artifact contract dataframe (ARTIFACT_COLUMNS)."""
    if uplift is None:
        uplift = reference_uplift(observed)
    margin = observed["expected_order_margin"].to_numpy()
    cost = observed["expected_coupon_cost"].to_numpy()
    profit = expected_incremental_profit(uplift, margin, cost)
    return pd.DataFrame(
        {
            "customer_id": observed["customer_id"].to_numpy(),
            "purchase_probability": fit_purchase_baseline(observed),
            "estimated_uplift": uplift,
            "expected_incremental_profit": profit,
            "recommended": select_under_budget(profit, cost, GROWTH_BUDGET),
        }
    )[ARTIFACT_COLUMNS]
