"""Budget-constrained selection, provided so participants can focus on the model."""

from __future__ import annotations

import numpy as np


def select_under_budget(
    profit: np.ndarray, cost: np.ndarray, budget: float
) -> np.ndarray:
    """Highest expected profit first, positive values only, until the budget is spent.

    Returns a 0/1 array: 1 means "send this customer a coupon".
    """
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
