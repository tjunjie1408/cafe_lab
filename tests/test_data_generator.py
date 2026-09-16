import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_cafe_data


OBSERVED_COLUMNS = [
    "customer_id",
    "days_since_last_purchase",
    "orders_30d",
    "app_sessions_30d",
    "average_order_value",
    "loyalty_score",
    "price_sensitivity",
    "offer_affinity",
    "expected_order_margin",
    "expected_coupon_cost",
    "coupon_sent",
    "purchased_7d",
]

TRUTH_COLUMNS = [
    "customer_id",
    "old_predictive_score",
    "treatment_propensity",
    "true_mu0",
    "true_mu1",
    "true_cate",
]


@pytest.fixture(scope="module")
def generated_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate_cafe_data(n_customers=10_000, seed=42)


def test_rejects_too_few_customers() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        generate_cafe_data(n_customers=99, seed=42)


def test_schema_and_truth_are_isolated(
    generated_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    observed, truth = generated_data

    assert observed.columns.tolist() == OBSERVED_COLUMNS
    assert truth.columns.tolist() == TRUTH_COLUMNS
    assert len(observed) == len(truth) == 10_000
    assert observed["customer_id"].is_unique
    pd.testing.assert_series_equal(
        observed["customer_id"], truth["customer_id"], check_names=False
    )
    assert set(TRUTH_COLUMNS[1:]).isdisjoint(observed.columns)
    assert set(observed["coupon_sent"].unique()) <= {0, 1}
    assert set(observed["purchased_7d"].unique()) <= {0, 1}
    assert observed[["coupon_sent", "purchased_7d"]].dtypes.eq("int64").all()
    assert observed[
        ["loyalty_score", "price_sensitivity", "offer_affinity"]
    ].apply(lambda column: column.between(0.0, 1.0).all()).all()
    assert observed[
        ["average_order_value", "expected_order_margin", "expected_coupon_cost"]
    ].gt(0.0).all().all()
    assert truth[
        ["old_predictive_score", "treatment_propensity", "true_mu0", "true_mu1"]
    ].apply(lambda column: column.between(0.0, 1.0).all()).all()


def test_truth_is_reconstructable_from_participant_visible_confounders(
    generated_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    observed, truth = generated_data
    loyalty = observed["loyalty_score"].to_numpy()
    price = observed["price_sensitivity"].to_numpy()
    affinity = observed["offer_affinity"].to_numpy()
    orders = observed["orders_30d"].to_numpy()
    sessions = observed["app_sessions_30d"].to_numpy()
    recency = observed["days_since_last_purchase"].to_numpy()

    baseline_logit = (
        -2.4 + 1.15 * loyalty + 0.20 * orders + 0.055 * sessions - 0.018 * recency
    )
    score_logit = (
        -2.25
        + 1.05 * loyalty
        + 0.18 * orders
        + 0.050 * sessions
        - 0.016 * recency
    )
    old_score = 1.0 / (1.0 + np.exp(-score_logit))
    propensity = np.clip(
        1.0 / (1.0 + np.exp(-(-1.5 + 2.0 * old_score + 0.4 * loyalty))),
        0.10,
        0.90,
    )
    mu0 = 1.0 / (1.0 + np.exp(-baseline_logit))
    moderate_inactivity = np.exp(-((recency - 35.0) / 24.0) ** 2)
    lost_customer = np.clip((recency - 75.0) / 45.0, 0.0, 1.0)
    treatment_shift = (
        -0.35
        - 0.45 * loyalty
        + 1.10 * price
        + 0.90 * affinity
        + 0.40 * moderate_inactivity
        - 0.85 * lost_customer
    )
    mu1 = 1.0 / (1.0 + np.exp(-(baseline_logit + treatment_shift)))

    np.testing.assert_allclose(truth["old_predictive_score"], old_score, atol=5e-7)
    np.testing.assert_allclose(truth["treatment_propensity"], propensity, atol=5e-7)
    np.testing.assert_allclose(truth["true_mu0"], mu0, atol=5e-7)
    np.testing.assert_allclose(truth["true_mu1"], mu1, atol=5e-7)


def test_generation_is_reproducible() -> None:
    observed_a, truth_a = generate_cafe_data(n_customers=1_000, seed=7)
    observed_b, truth_b = generate_cafe_data(n_customers=1_000, seed=7)
    observed_c, truth_c = generate_cafe_data(n_customers=1_000, seed=8)

    pd.testing.assert_frame_equal(observed_a, observed_b)
    pd.testing.assert_frame_equal(truth_a, truth_b)
    assert not observed_a.equals(observed_c)
    assert not truth_a.equals(truth_c)


def test_assignment_is_stochastic_biased_and_has_overlap(
    generated_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    observed, truth = generated_data
    combined = observed[["coupon_sent"]].join(truth.drop(columns="customer_id"))

    assert truth["treatment_propensity"].between(0.10, 0.90).all()

    score_decile = pd.qcut(truth["old_predictive_score"], 10, duplicates="raise")
    counts = combined.groupby(score_decile, observed=False)["coupon_sent"].agg(
        treated="sum", total="count"
    )
    counts["untreated"] = counts["total"] - counts["treated"]
    assert counts[["treated", "untreated"]].ge(100).all().all()

    calibration = combined.assign(propensity_decile=pd.qcut(
        truth["treatment_propensity"], 10, duplicates="raise"
    )).groupby("propensity_decile", observed=False).agg(
        realized_rate=("coupon_sent", "mean"),
        declared_rate=("treatment_propensity", "mean"),
    )
    assert (
        calibration["realized_rate"] - calibration["declared_rate"]
    ).abs().max() <= 0.05
    assert calibration["realized_rate"].iloc[-1] > calibration["realized_rate"].iloc[0]

    lower_cut = truth["old_predictive_score"].quantile(0.25)
    upper_cut = truth["old_predictive_score"].quantile(0.75)
    bottom = truth.loc[
        truth["old_predictive_score"] <= lower_cut, "treatment_propensity"
    ].mean()
    top = truth.loc[
        truth["old_predictive_score"] >= upper_cut, "treatment_propensity"
    ].mean()
    assert top - bottom >= 0.10


def test_naive_observed_difference_overstates_true_ate(
    generated_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    observed, truth = generated_data
    treated_rate = observed.loc[observed["coupon_sent"] == 1, "purchased_7d"].mean()
    untreated_rate = observed.loc[
        observed["coupon_sent"] == 0, "purchased_7d"
    ].mean()
    naive_difference = treated_rate - untreated_rate
    true_ate = truth["true_cate"].mean()

    assert treated_rate > untreated_rate
    assert naive_difference - true_ate >= 0.02


def test_effects_are_heterogeneous_without_an_inverse_baseline_shortcut(
    generated_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    observed, truth = generated_data
    combined = observed.join(truth.drop(columns="customer_id"))

    assert (truth["true_cate"] < 0.0).mean() >= 0.02
    assert (truth["true_cate"] > 0.05).mean() >= 0.20
    assert abs(truth[["true_mu0", "true_cate"]].corr().iloc[0, 1]) < 0.80

    low_price, high_price = observed["price_sensitivity"].quantile([0.25, 0.75])
    low_offer, high_offer = observed["offer_affinity"].quantile([0.25, 0.75])
    assert combined.loc[
        combined["price_sensitivity"] >= high_price, "true_cate"
    ].mean() > combined.loc[
        combined["price_sensitivity"] <= low_price, "true_cate"
    ].mean()
    assert combined.loc[
        combined["offer_affinity"] >= high_offer, "true_cate"
    ].mean() > combined.loc[
        combined["offer_affinity"] <= low_offer, "true_cate"
    ].mean()


def test_true_cate_is_the_difference_between_conditional_outcomes(
    generated_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    _, truth = generated_data
    np.testing.assert_allclose(
        truth["true_cate"], truth["true_mu1"] - truth["true_mu0"], atol=1e-6
    )


def test_expected_coupon_cost_scales_with_redemption_likelihood(
    generated_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Cost per targeted customer must vary with baseline redemption, not be flat.

    Higher-baseline customers redeem more, so their expected coupon cost is
    larger. This is what makes couponing a would-buy-anyway customer a double
    waste: low uplift and high redemption cost.
    """
    observed, truth = generated_data
    cost = observed["expected_coupon_cost"]

    assert cost.between(0.0, observed["average_order_value"] * 0.20 + 1e-9).all()
    # Not a near-constant column: it genuinely spreads across customers.
    assert cost.std() >= 0.30
    assert cost.max() - cost.min() >= 1.0
    # Cost rises with baseline purchase likelihood.
    assert cost.corr(truth["old_predictive_score"]) >= 0.40


def test_causal_value_targeting_beats_predictive_targeting(
    generated_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The oracle value policy must capture more true EV than predictive ranking.

    This is the workshop's central claim. Using synthetic truth, ranking by
    expected incremental value must beat ranking by predicted purchase
    probability at a fixed budget, and the value policy must be net positive.
    """
    observed, truth = generated_data
    combined = observed.join(truth.drop(columns="customer_id"))
    expected_value = (
        combined["true_cate"] * combined["expected_order_margin"]
        - combined["expected_coupon_cost"]
    )
    combined = combined.assign(expected_value=expected_value)

    budget = int(len(combined) * 0.20)
    value_selected = combined.nlargest(budget, "expected_value")["expected_value"].sum()
    predictive_selected = combined.nlargest(
        budget, "old_predictive_score"
    )["expected_value"].sum()

    assert value_selected > 0.0
    assert predictive_selected < 0.0
    assert value_selected - predictive_selected >= 500.0
    # A meaningful, but not universal, minority is profitably targetable.
    assert 0.15 <= (expected_value > 0.0).mean() <= 0.45


@pytest.mark.parametrize("seed", [40, 41, 42, 43, 44])
def test_pedagogical_signal_survives_neighboring_seeds(seed: int) -> None:
    observed, truth = generate_cafe_data(n_customers=10_000, seed=seed)
    treated_rate = observed.loc[
        observed["coupon_sent"] == 1, "purchased_7d"
    ].mean()
    untreated_rate = observed.loc[
        observed["coupon_sent"] == 0, "purchased_7d"
    ].mean()

    assert treated_rate - untreated_rate - truth["true_cate"].mean() >= 0.02
    assert (truth["true_cate"] < 0.0).mean() >= 0.02
    assert (truth["true_cate"] > 0.05).mean() >= 0.20
