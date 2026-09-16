"""Reference solutions for the causal-estimation tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.workshop_data import FEATURES


def naive_observed_difference(observed: pd.DataFrame) -> float:
    """Observed treated-minus-control purchase rate. NOT an ATE."""
    treated = observed.loc[observed["coupon_sent"] == 1, "purchased_7d"].mean()
    control = observed.loc[observed["coupon_sent"] == 0, "purchased_7d"].mean()
    return float(treated - control)


def t_learner_uplift(
    observed: pd.DataFrame, features: list[str] = FEATURES
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-model T-Learner. Returns (mu0, mu1, uplift) for every customer.

    Interpreting uplift as CATE assumes the measured pre-treatment features are
    sufficient for exchangeability, plus overlap and consistency.
    """
    treated = observed[observed["coupon_sent"] == 1]
    control = observed[observed["coupon_sent"] == 0]
    model_treated = LogisticRegression(max_iter=2000).fit(
        treated[features], treated["purchased_7d"]
    )
    model_control = LogisticRegression(max_iter=2000).fit(
        control[features], control["purchased_7d"]
    )
    mu1 = model_treated.predict_proba(observed[features])[:, 1]
    mu0 = model_control.predict_proba(observed[features])[:, 1]
    return mu0, mu1, mu1 - mu0
