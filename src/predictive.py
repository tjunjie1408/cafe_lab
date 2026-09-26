"""Facilitator-provided predictive purchase baseline (not a causal estimate)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.workshop_data import FEATURES


def fit_purchase_baseline(observed: pd.DataFrame) -> np.ndarray:
    """Pooled logistic regression predicting purchase from behavioral features.

    This is an ordinary predictive score: who is likely to purchase. It is NOT
    a treatment effect and must not be interpreted as one.
    """
    model = LogisticRegression(max_iter=2000)
    model.fit(observed[FEATURES], observed["purchased_7d"])
    return model.predict_proba(observed[FEATURES])[:, 1]
