from pathlib import Path

import numpy as np
import pandas as pd

from src.workshop_data import (
    ARTIFACT_COLUMNS,
    FEATURES,
    GROWTH_BUDGET,
    load_observed_data,
    load_facilitator_truth,
)

ROOT = Path(__file__).resolve().parents[1]


def test_constants_are_stable() -> None:
    assert FEATURES == [
        "days_since_last_purchase",
        "orders_30d",
        "app_sessions_30d",
        "average_order_value",
        "loyalty_score",
        "price_sensitivity",
        "offer_affinity",
    ]
    assert ARTIFACT_COLUMNS == [
        "customer_id",
        "purchase_probability",
        "estimated_uplift",
        "expected_incremental_profit",
        "recommended",
    ]
    assert GROWTH_BUDGET == 2500.0


def test_loaders_expose_expected_columns() -> None:
    observed = load_observed_data()
    truth = load_facilitator_truth()
    assert set(FEATURES).issubset(observed.columns)
    assert {"coupon_sent", "purchased_7d"}.issubset(observed.columns)
    # Truth fields must never leak into the participant frame.
    assert {"true_mu0", "true_mu1", "true_cate"}.isdisjoint(observed.columns)
    assert {"true_mu0", "true_mu1", "true_cate"}.issubset(truth.columns)
    assert len(observed) == len(truth) == 10_000


def test_purchase_baseline_is_a_ranked_probability() -> None:
    from src.predictive import fit_purchase_baseline

    observed = load_observed_data()
    probability = fit_purchase_baseline(observed)
    assert probability.shape == (len(observed),)
    assert ((probability >= 0.0) & (probability <= 1.0)).all()
    # A real signal: buyers score higher on average than non-buyers.
    buyers = probability[observed["purchased_7d"].to_numpy() == 1]
    non_buyers = probability[observed["purchased_7d"].to_numpy() == 0]
    assert buyers.mean() > non_buyers.mean() + 0.05


def test_naive_difference_overstates_true_ate() -> None:
    from solutions.causal_solution import naive_observed_difference

    observed = load_observed_data()
    truth = load_facilitator_truth()
    naive = naive_observed_difference(observed)
    treated = observed.loc[observed["coupon_sent"] == 1, "purchased_7d"].mean()
    control = observed.loc[observed["coupon_sent"] == 0, "purchased_7d"].mean()
    assert naive == treated - control
    assert naive > truth["true_cate"].mean()  # it overstates


def test_t_learner_recovers_cate() -> None:
    from solutions.causal_solution import t_learner_uplift

    observed = load_observed_data()
    truth = load_facilitator_truth()
    mu0, mu1, uplift = t_learner_uplift(observed)
    assert mu0.shape == mu1.shape == uplift.shape == (len(observed),)
    np.testing.assert_allclose(uplift, mu1 - mu0)
    corr = np.corrcoef(uplift, truth["true_cate"].to_numpy())[0, 1]
    mae = np.abs(uplift - truth["true_cate"].to_numpy()).mean()
    assert corr >= 0.85  # reference achieves ~0.915
    assert mae <= 0.04  # reference achieves ~0.024


def test_expected_incremental_profit_formula() -> None:
    from solutions.policy_solution import expected_incremental_profit

    uplift = np.array([0.10, 0.02, -0.01])
    margin = np.array([10.0, 10.0, 10.0])
    cost = np.array([1.5, 1.5, 1.5])
    got = expected_incremental_profit(uplift, margin, cost)
    np.testing.assert_allclose(got, uplift * margin - cost)


def test_policy_respects_budget_and_positivity() -> None:
    from solutions.policy_solution import build_customer_scores

    observed = load_observed_data()
    scores = build_customer_scores(observed)
    assert scores.columns.tolist() == ARTIFACT_COLUMNS
    assert set(scores["recommended"].unique()) <= {0, 1}
    assert (
        (scores["purchase_probability"] >= 0) & (scores["purchase_probability"] <= 1)
    ).all()

    chosen = scores["recommended"] == 1
    cost = observed.loc[chosen, "expected_coupon_cost"].sum()
    assert cost <= GROWTH_BUDGET + 1e-6
    # Only positive-value customers are ever recommended.
    assert (scores.loc[chosen, "expected_incremental_profit"] > 0).all()
    # Budget is binding, so not everyone profitable is selected.
    assert 1000 <= chosen.sum() <= 2000


def test_causal_targeting_beats_predictive_on_true_value() -> None:
    from solutions.policy_solution import build_customer_scores
    from src.predictive import fit_purchase_baseline

    observed = load_observed_data()
    truth = load_facilitator_truth()
    scores = build_customer_scores(observed)
    true_ev = (
        truth["true_cate"].to_numpy() * observed["expected_order_margin"].to_numpy()
        - observed["expected_coupon_cost"].to_numpy()
    )
    causal_value = true_ev[(scores["recommended"] == 1).to_numpy()].sum()
    # Spend the same budget on the highest predicted-purchase customers instead.
    order = np.argsort(-fit_purchase_baseline(observed))
    spent, predictive_mask = 0.0, np.zeros(len(observed), dtype=bool)
    costs = observed["expected_coupon_cost"].to_numpy()
    for i in order:
        if spent + costs[i] > 2500.0:
            continue
        predictive_mask[i] = True
        spent += costs[i]
    predictive_value = true_ev[predictive_mask].sum()
    assert causal_value > 0 > predictive_value


def test_backup_artifact_exists_and_matches_schema() -> None:
    path = ROOT / "artifacts" / "customer_scores.csv"
    assert path.exists(), "run scripts/build_solution_artifact.py"
    backup = pd.read_csv(path, dtype={"customer_id": "string"})
    assert backup.columns.tolist() == ARTIFACT_COLUMNS
    assert len(backup) == 10_000


def test_plots_return_figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from src.visualisation import plot_uplift_distribution, plot_policy_comparison

    fig1 = plot_uplift_distribution(np.linspace(-0.05, 0.3, 100))
    fig2 = plot_policy_comparison(["random", "causal"], [-1000.0, 900.0])
    assert fig1.axes and fig2.axes


def test_notebook_is_participant_safe() -> None:
    source = (ROOT / "workshop.py").read_text(encoding="utf-8")
    assert "import solutions" not in source and "from solutions" not in source
    assert source.count("raise NotImplementedError") == 3  # Tasks 2, 3, 4
    assert 'Path("artifacts").mkdir(parents=True, exist_ok=True)' in source
    assert (ROOT / "workshop.ipynb").exists()
