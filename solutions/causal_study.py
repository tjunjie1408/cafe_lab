"""Fit candidates and score fixed, independent facilitator evaluation populations."""

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from src.causal_models import fit_s_learner
from src.evaluation import qini_coefficient
from src.policy import select_under_budget
from solutions.causal_solution import qini_curve

METHODS = ("S-logistic", "T-logistic", "T-boosting", "CausalForestDML")
FOREST_SETTINGS = dict(n_estimators=120, max_depth=12, min_samples_leaf=40,
                       cv=3, honest=True, inference=False, n_jobs=2)


def fit_candidate(method, train, features, seed):
    """Return a fitted bundle and measured fit wall time, excluding import/setup."""
    if method == "CausalForestDML":
        from econml.dml import CausalForestDML
        model = CausalForestDML(
            model_y=HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15, random_state=seed),
            model_t=HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, random_state=seed),
            discrete_treatment=True, random_state=seed, **FOREST_SETTINGS,
        )
        start = time.perf_counter()
        model.fit(train["purchased_7d"].to_numpy(), train["coupon_sent"].to_numpy(),
                  X=train[features].to_numpy())
        bundle = {"kind": "causal_forest", "features": list(features), "model": model}
    elif method == "S-logistic":
        start = time.perf_counter()
        bundle = fit_s_learner(train, features)
    elif method in {"T-logistic", "T-boosting"}:
        factory = (lambda: LogisticRegression(max_iter=2_000)) if method == "T-logistic" else (
            lambda: HistGradientBoostingClassifier(random_state=seed)
        )
        arm0, arm1 = factory(), factory()
        control, treated = train.loc[train.coupon_sent == 0], train.loc[train.coupon_sent == 1]
        start = time.perf_counter()
        arm0.fit(control[features], control["purchased_7d"])
        arm1.fit(treated[features], treated["purchased_7d"])
        bundle = {"kind": "t", "features": list(features), "model_control": arm0, "model_treated": arm1}
    else:
        raise ValueError(f"Unknown method: {method}")
    return bundle, time.perf_counter() - start


def score_predictions(evaluation, truth, pilot, uplift, pilot_uplift, budget=2_500):
    """Evaluate estimates only; no fitted estimator or training data is accepted."""
    if not evaluation.customer_id.equals(truth.customer_id):
        raise ValueError("Evaluation rows and truth must be aligned")
    uplift, pilot_uplift = np.asarray(uplift), np.asarray(pilot_uplift)
    if uplift.shape != (len(evaluation),) or pilot_uplift.shape != (len(pilot),):
        raise ValueError("Prediction lengths do not match evaluation populations")
    if not np.isfinite(uplift).all() or not np.isfinite(pilot_uplift).all():
        raise ValueError("Predictions must be finite")
    tau = truth.true_cate.to_numpy()
    margin = evaluation.expected_order_margin.to_numpy()
    cost = evaluation.expected_coupon_cost.to_numpy()
    expected_profit = uplift * margin - cost
    selected = select_under_budget(expected_profit, cost, budget).astype(bool)
    reference = select_under_budget(tau * margin - cost, cost, budget).astype(bool)
    metrics = {
        "cate_rmse": float(np.sqrt(np.mean((uplift - tau) ** 2))),
        "cate_mae": float(np.mean(np.abs(uplift - tau))),
        "ate_bias": float(np.mean(uplift) - np.mean(tau)),
        "ate_abs_error": float(abs(np.mean(uplift) - np.mean(tau))),
        "pilot_qini": qini_coefficient(qini_curve(pilot, pilot_uplift)),
        "true_policy_profit": float(np.sum((tau * margin - cost)[selected])),
        "estimated_policy_profit": float(expected_profit[selected].sum()),
        "truth_ranked_reference_profit": float(np.sum((tau * margin - cost)[reference])),
        "spend": float(cost[selected].sum()), "recommended": int(selected.sum()),
        "true_ate": float(tau.mean()), "estimated_ate": float(uplift.mean()),
    }
    return metrics, selected
