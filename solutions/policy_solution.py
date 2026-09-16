"""Reference solution for the decision-policy task."""

from __future__ import annotations

import numpy as np
import pandas as pd

from solutions.causal_solution import t_learner_uplift
from src.predictive import fit_purchase_baseline
from src.workshop_data import ARTIFACT_COLUMNS, GROWTH_BUDGET


def expected_incremental_profit(
    uplift: np.ndarray,
    expected_order_margin: np.ndarray,
    expected_coupon_cost: np.ndarray,
) -> np.ndarray:
    """Money added per targeted customer after expected coupon cost."""
    return uplift * expected_order_margin - expected_coupon_cost


def select_under_budget(
    profit: np.ndarray, cost: np.ndarray, budget: float = GROWTH_BUDGET
) -> np.ndarray:
    """Greedy: highest profit first, positive-value only, until budget is spent."""
    recommended = np.zeros(len(profit), dtype=np.int64)
    spent = 0.0
    for index in np.argsort(-profit):
        if profit[index] <= 0:
            break
        if spent + cost[index] > budget:
            continue
        recommended[index] = 1
        spent += cost[index]
    return recommended


def build_customer_scores(observed: pd.DataFrame) -> pd.DataFrame:
    """Produce the artifact contract dataframe (ARTIFACT_COLUMNS)."""
    purchase_probability = fit_purchase_baseline(observed)
    _, _, uplift = t_learner_uplift(observed)
    margin = observed["expected_order_margin"].to_numpy()
    cost = observed["expected_coupon_cost"].to_numpy()
    profit = expected_incremental_profit(uplift, margin, cost)
    recommended = select_under_budget(profit, cost)
    return pd.DataFrame(
        {
            "customer_id": observed["customer_id"].to_numpy(),
            "purchase_probability": purchase_probability,
            "estimated_uplift": uplift,
            "expected_incremental_profit": profit,
            "recommended": recommended,
        }
    )[ARTIFACT_COLUMNS]
