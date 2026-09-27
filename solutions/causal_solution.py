"""Reference solutions for the causal-estimation and evaluation tasks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.workshop_data import FEATURES


def make_logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=2000)


def make_boosting() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=0)


def naive_observed_difference(observed: pd.DataFrame) -> float:
    """Observed treated-minus-control purchase rate. NOT an ATE."""
    treated = observed.loc[observed["coupon_sent"] == 1, "purchased_7d"].mean()
    control = observed.loc[observed["coupon_sent"] == 0, "purchased_7d"].mean()
    return float(treated - control)


def t_learner_uplift(
    train: pd.DataFrame,
    score: pd.DataFrame,
    features: list[str] = FEATURES,
    make_model: Callable = make_logistic,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one outcome model per arm on ``train``; return (mu0, mu1, uplift) for ``score``.

    Interpreting uplift as CATE assumes the measured pre-treatment features are
    sufficient for exchangeability, plus overlap and consistency.
    """
    treated = train[train["coupon_sent"] == 1]
    control = train[train["coupon_sent"] == 0]
    model_treated = make_model().fit(treated[features], treated["purchased_7d"])
    model_control = make_model().fit(control[features], control["purchased_7d"])
    mu1 = model_treated.predict_proba(score[features])[:, 1]
    mu0 = model_control.predict_proba(score[features])[:, 1]
    return mu0, mu1, mu1 - mu0


def qini_curve(pilot: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Cumulative incremental purchases when targeting randomized customers by ``score``.

    At each cut-off: treated buyers minus control buyers scaled to the treated
    count, so both arms are compared at the same size.
    """
    order = np.argsort(-np.asarray(score))
    treated = pilot["coupon_sent"].to_numpy()[order]
    bought = pilot["purchased_7d"].to_numpy()[order]
    n_treated = np.cumsum(treated)
    n_control = np.cumsum(1 - treated)
    buyers_treated = np.cumsum(bought * treated)
    buyers_control = np.cumsum(bought * (1 - treated))
    scale = np.divide(n_treated, n_control, out=np.zeros(len(order)), where=n_control > 0)
    gain = buyers_treated - buyers_control * scale
    share = np.arange(1, len(order) + 1) / len(order)
    return pd.DataFrame({"share": share, "gain": gain})
