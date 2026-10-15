"""Facilitator-only paired simulation scenarios; the classroom data is unchanged."""

import numpy as np
from scipy.special import expit, logit

from src.data_generator import generate_cafe_data

SCENARIOS = ("smooth", "nonlinear", "hidden_confounder")


def make_population(n, seed, scenario, namespace, *, randomized=False):
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    observed, truth = generate_cafe_data(
        n_customers=n, seed=seed, randomized_coupon_rate=0.5 if randomized else None,
    )
    ids = namespace + ":" + observed["customer_id"]
    observed["customer_id"] = ids
    truth["customer_id"] = ids
    if scenario == "smooth":
        return observed, truth

    rng = np.random.default_rng(seed + 700_000)
    mu0 = truth["true_mu0"].to_numpy()
    if scenario == "nonlinear":
        shift = (
            -0.7
            + 2.8 * ((observed["price_sensitivity"] > 0.60) & (observed["offer_affinity"] > 0.50))
            + 1.2 * observed["days_since_last_purchase"].between(20, 55)
            - 1.4 * (observed["loyalty_score"] > 0.65)
        )
        mu1 = expit(logit(mu0) + shift.to_numpy())
        propensity = truth["treatment_propensity"].to_numpy()
    else:
        u = rng.binomial(1, 0.5, n)
        # U changes baseline outcome and assignment, while the risk-difference
        # effect depends only on visible X. Thus missing-U RMSE is not caused by
        # an extra unobserved effect modifier.
        mu0 = 0.05 + 0.25 * mu0 + 0.28 * u
        tau = (
            0.03 + 0.13 * observed["price_sensitivity"]
            + 0.10 * observed["offer_affinity"] - 0.08 * observed["loyalty_score"]
        ).to_numpy()
        mu1 = mu0 + tau
        propensity = expit(logit(truth["treatment_propensity"].to_numpy()) + 2.0 * (2 * u - 1))
        observed["latent_confounder"] = u  # instructor audit only; excluded from normal model features
        truth["latent_confounder"] = u
    if randomized:
        propensity = np.full(n, 0.5)
    treatment = rng.binomial(1, propensity, n)
    observed["coupon_sent"] = treatment
    observed["purchased_7d"] = rng.binomial(1, np.where(treatment == 1, mu1, mu0), n)
    truth["true_mu0"] = mu0
    truth["true_mu1"] = mu1
    truth["true_cate"] = mu1 - mu0
    truth["treatment_propensity"] = propensity
    return observed, truth
