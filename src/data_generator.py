"""Pedagogical observational-data simulator for Causal Cafe Lab."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Evaluate the logistic function without overflowing at extreme values."""
    result = np.empty_like(values, dtype=float)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def generate_cafe_data(
    n_customers: int = 10_000,
    seed: int = 42,
    randomized_coupon_rate: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return participant observations and matched facilitator-only truth.

    By default coupon assignment favors customers with high old-model purchase
    scores but remains stochastic. All common causes of assignment and purchase
    are included in the participant-visible covariates.

    With ``randomized_coupon_rate`` every customer gets a coupon with that
    probability instead (a randomized pilot). The same random draws are used in
    the same order, so the default output is unchanged.
    """
    if n_customers < 100:
        raise ValueError("n_customers must be at least 100")
    if randomized_coupon_rate is not None and not 0 < randomized_coupon_rate < 1:
        raise ValueError("randomized_coupon_rate must be between 0 and 1")

    rng = np.random.default_rng(seed)
    n = n_customers

    loyalty_score = np.round(rng.beta(2.2, 2.0, n), 4)
    # Price sensitivity and offer affinity are primarily independent response
    # traits, but a small loyalty component removes the artificial perfect
    # independence of Narrative v1 without turning them into confounders: they
    # still do not enter the baseline outcome, so identification is unchanged.
    price_raw = rng.beta(2.0, 2.0, n)
    offer_raw = rng.beta(2.0, 2.0, n)
    price_sensitivity = np.round(
        np.clip(0.85 * price_raw + 0.15 * (1.0 - loyalty_score), 0.0, 1.0), 4
    )
    offer_affinity = np.round(
        np.clip(0.85 * offer_raw + 0.15 * loyalty_score, 0.0, 1.0), 4
    )
    app_sessions_30d = rng.poisson(1.0 + 7.0 * loyalty_score)
    orders_30d = rng.poisson(0.3 + 4.0 * loyalty_score)
    days_since_last_purchase = np.round(
        np.clip(
            rng.gamma(
                shape=2.0,
                scale=18.0 * (1.15 - 0.75 * loyalty_score),
            ),
            0.0,
            120.0,
        ),
        1,
    )
    average_order_value = np.round(
        np.clip(
            8.0 + 13.0 * loyalty_score + rng.normal(0.0, 2.5, n),
            5.0,
            35.0,
        ),
        2,
    )

    baseline_logit = (
        -2.4
        + 1.15 * loyalty_score
        + 0.20 * orders_30d
        + 0.055 * app_sessions_30d
        - 0.018 * days_since_last_purchase
    )
    true_mu0 = _sigmoid(baseline_logit)

    old_score_logit = (
        -2.25
        + 1.05 * loyalty_score
        + 0.18 * orders_30d
        + 0.050 * app_sessions_30d
        - 0.016 * days_since_last_purchase
    )
    old_predictive_score = _sigmoid(old_score_logit)
    treatment_propensity = np.clip(
        _sigmoid(-1.5 + 2.0 * old_predictive_score + 0.4 * loyalty_score),
        0.10,
        0.90,
    )
    if randomized_coupon_rate is not None:
        treatment_propensity = np.full(n, randomized_coupon_rate)
    coupon_sent = rng.binomial(1, treatment_propensity, n).astype(np.int64)

    moderate_inactivity = np.exp(
        -((days_since_last_purchase - 35.0) / 24.0) ** 2
    )
    lost_customer = np.clip(
        (days_since_last_purchase - 75.0) / 45.0,
        0.0,
        1.0,
    )
    treatment_logit_shift = (
        -0.35
        - 0.45 * loyalty_score
        + 1.10 * price_sensitivity
        + 0.90 * offer_affinity
        + 0.40 * moderate_inactivity
        - 0.85 * lost_customer
    )
    true_mu1 = _sigmoid(baseline_logit + treatment_logit_shift)
    true_cate = true_mu1 - true_mu0

    observed_purchase_probability = np.where(
        coupon_sent == 1,
        true_mu1,
        true_mu0,
    )
    purchased_7d = rng.binomial(
        1, observed_purchase_probability, n
    ).astype(np.int64)

    customer_id = pd.Series(
        [f"CAFE-{index:06d}" for index in range(1, n + 1)],
        dtype="string",
    )
    observed = pd.DataFrame(
        {
            "customer_id": customer_id,
            "days_since_last_purchase": days_since_last_purchase,
            "orders_30d": orders_30d.astype(np.int64),
            "app_sessions_30d": app_sessions_30d.astype(np.int64),
            "average_order_value": average_order_value,
            "loyalty_score": loyalty_score,
            "price_sensitivity": price_sensitivity,
            "offer_affinity": offer_affinity,
            "expected_order_margin": np.round(0.72 * average_order_value, 2),
            # Expected cost per targeted customer is the coupon face value times
            # the probability that the customer actually redeems it. Redemption
            # rises with observed baseline engagement (loyal, recent customers
            # redeem more) and stays in a realistic 0.20-0.80 band. This makes
            # high-baseline customers cost more, reinforcing that couponing a
            # customer who would have bought anyway is wasted spend.
            "expected_coupon_cost": np.round(
                np.clip(
                    0.32 + 0.50 * loyalty_score - 0.0018 * days_since_last_purchase,
                    0.20,
                    0.80,
                )
                * (0.20 * average_order_value),
                2,
            ),
            "coupon_sent": coupon_sent,
            "purchased_7d": purchased_7d,
        }
    )

    rounded_mu0 = np.round(true_mu0, 6)
    rounded_mu1 = np.round(true_mu1, 6)
    truth = pd.DataFrame(
        {
            "customer_id": customer_id.copy(),
            "old_predictive_score": np.round(old_predictive_score, 6),
            "treatment_propensity": np.round(treatment_propensity, 6),
            "true_mu0": rounded_mu0,
            "true_mu1": rounded_mu1,
            "true_cate": np.round(rounded_mu1 - rounded_mu0, 6),
        }
    )

    return observed, truth
