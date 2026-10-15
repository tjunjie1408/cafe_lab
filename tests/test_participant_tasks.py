from pathlib import Path

import numpy as np
import pytest
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
    from src.workshop_data import split_train_test

    observed = load_observed_data()
    truth = load_facilitator_truth()
    train, _ = split_train_test(observed)
    mu0, mu1, uplift = t_learner_uplift(train, observed)
    assert mu0.shape == mu1.shape == uplift.shape == (len(observed),)
    np.testing.assert_allclose(uplift, mu1 - mu0)
    corr = np.corrcoef(uplift, truth["true_cate"].to_numpy())[0, 1]
    mae = np.abs(uplift - truth["true_cate"].to_numpy()).mean()
    assert corr >= 0.85  # reference achieves ~0.893
    assert mae <= 0.04  # reference achieves ~0.026


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
    # Figures render themselves inline even in a kernel that never imported pyplot.
    assert fig1._repr_png_().startswith(b"\x89PNG")


def test_results_board_renders() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from src.visualisation import (
        plot_budget_curve,
        plot_effect_summary,
        plot_model_check,
        plot_targeting_map,
    )

    rng = np.random.default_rng(0)
    uplift = rng.normal(0.1, 0.05, 200)
    probability = rng.uniform(0, 1, 200)
    cost = rng.uniform(1, 3, 200)
    curve = pd.DataFrame({"share": np.linspace(0.005, 1, 200), "gain": np.linspace(0, 20, 200)})
    figures = [
        plot_effect_summary(0.14, uplift),
        plot_model_check({"logistic": (0.69, 0.68), "boosting": (0.65, 0.64)}, {"logistic": curve}),
        plot_budget_curve(uplift * 30 - cost, cost, probability, 100.0),
        plot_targeting_map(probability, uplift, (uplift > 0.12).astype(int)),
    ]
    for fig in figures:
        assert fig._repr_png_().startswith(b"\x89PNG")


def test_confounding_overlap_plot_highlights_imbalance() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from src.visualisation import ORANGE, plot_confounding_overlap
    from matplotlib.colors import to_hex

    smd = pd.Series({"balanced": 0.02, "imbalanced": 0.3, "negative": -0.2})
    rng = np.random.default_rng(0)
    fig = plot_confounding_overlap(smd, rng.uniform(0.1, 0.6, 300), rng.integers(0, 2, 300))
    assert fig._repr_png_().startswith(b"\x89PNG")
    bars = fig.axes[0].patches
    highlighted = {bar.get_width() for bar in bars if to_hex(bar.get_facecolor()) == ORANGE.lower()}
    assert highlighted == {0.3, -0.2}


def test_answer_notebook_is_in_sync_with_workshop() -> None:
    from scripts.build_answer_notebook import build_answers

    expected = build_answers((ROOT / "workshop.py").read_text(encoding="utf-8"))
    committed = (ROOT / "solutions" / "workshop_answers.py").read_text(encoding="utf-8")
    assert committed == expected, "run scripts/build_answer_notebook.py"


def test_answer_notebook_reproduces_reference_numbers(tmp_path: Path, monkeypatch) -> None:
    """Run the filled-in participant notebook in a participant-like folder."""
    import runpy
    import shutil

    import matplotlib

    matplotlib.use("Agg")
    shutil.copytree(ROOT / "src", tmp_path / "src")
    (tmp_path / "data").mkdir()
    for name in ("observed_data.csv", "pilot_rct.csv"):
        shutil.copy(ROOT / "data" / name, tmp_path / "data" / name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    result = runpy.run_path(str(ROOT / "solutions" / "workshop_answers.py"))

    from src.evaluation import qini_coefficient

    np.testing.assert_allclose(result["naive"], 0.1382, atol=5e-5)
    scores = pd.read_csv(tmp_path / "artifacts" / "customer_scores.csv")
    assert scores.columns.tolist() == ARTIFACT_COLUMNS
    chosen = scores["recommended"] == 1
    assert chosen.sum() == 1_691
    np.testing.assert_allclose(scores.loc[chosen, "expected_incremental_profit"].sum(), 927.19, atol=0.01)
    qini = {name: qini_coefficient(curve) for name, curve in result["curves"].items()}
    assert qini["T-Learner, logistic"] > qini["purchase probability (old AI)"] > qini["T-Learner, boosting"]

    # Data overview: the budget covers only a small share of a full campaign.
    overview = result["overview"]
    assert overview["customers"] == 10_000
    assert overview["cost to coupon everyone"] > 6 * overview["Growth Budget"]

    # Balance and overlap: the five history features are confounded, the two
    # response traits are not, and no customer lacks a comparable counterpart.
    smd = result["balance"]["SMD"]
    history = {"days_since_last_purchase", "orders_30d", "app_sessions_30d",
               "average_order_value", "loyalty_score"}
    assert set(smd.index[smd.abs() > 0.1]) == history
    np.testing.assert_allclose(smd["loyalty_score"], 0.261, atol=5e-3)
    propensity = result["propensity"]
    assert 0.05 < propensity.min() and propensity.max() < 0.95

    # List profile: the old AI picks likelier buyers yet loses money.
    profile = result["list_profile"]
    causal, old_ai = profile["causal list"], profile["old AI list"]
    assert causal["customers"] == 1_691
    assert old_ai["spend"] <= GROWTH_BUDGET + 1e-6
    assert old_ai["mean purchase probability"] > causal["mean purchase probability"]
    np.testing.assert_allclose(causal["estimated incremental profit"], 927.19, atol=0.01)
    assert old_ai["estimated incremental profit"] < 0


def test_concept_diagrams_render() -> None:
    import io

    import matplotlib

    matplotlib.use("Agg")
    from src.visualisation import (
        plot_causal_dag,
        plot_t_learner_schematic,
        plot_uplift_archetypes,
        plot_workshop_roadmap,
    )

    for build in (
        plot_workshop_roadmap,
        plot_uplift_archetypes,
        plot_causal_dag,
        plot_t_learner_schematic,
    ):
        buffer = io.BytesIO()
        build().savefig(buffer, format="png")
        assert buffer.tell() > 10_000, build.__name__


def test_complete_analysis_matches_reference(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    from solutions.policy_solution import build_customer_scores

    env = {
        **os.environ,
        "CAUSAL_CAFE_OUTPUT_DIR": str(tmp_path),
        "MPLBACKEND": "Agg",
        "PYTHONIOENCODING": "utf-8",
    }
    subprocess.run(
        [sys.executable, str(ROOT / "solutions" / "causal_cafe_complete.py")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
    )
    for name in [
        "policy_comparison.csv",
        "archetype_summary.csv",
        "analysis_summary.csv",
        "decision_memo.md",
        "03_causal_dag.png",
        "12_archetype_reveal.png",
    ]:
        assert (tmp_path / name).exists(), name

    observed = load_observed_data()
    reference = build_customer_scores(observed)
    chosen = reference["recommended"] == 1
    summary = pd.read_csv(tmp_path / "analysis_summary.csv").set_index("metric")["value"]
    assert summary["customers recommended"] == chosen.sum()
    np.testing.assert_allclose(
        summary["budget spent"], observed.loc[chosen, "expected_coupon_cost"].sum()
    )
    np.testing.assert_allclose(
        summary["estimated incremental profit"],
        reference.loc[chosen, "expected_incremental_profit"].sum(),
    )

    policies = pd.read_csv(tmp_path / "policy_comparison.csv").set_index("policy")
    value = policies["true_incremental_profit"]
    assert value["Oracle (true CATE)"] >= value["Causal value ranking"] > 0
    assert value["Predictive ranking"] < 0


def test_notebook_is_participant_safe() -> None:
    source = (ROOT / "workshop.py").read_text(encoding="utf-8")
    assert "import solutions" not in source and "from solutions" not in source
    assert source.count("raise NotImplementedError") == 4  # Tasks 2, 3, 4, 5
    # Simulator truth stays on the projector, never in the participant notebook.
    assert "true_cate" not in source and "data/facilitator" not in source
    assert 'Path("artifacts").mkdir(parents=True, exist_ok=True)' in source
    assert (ROOT / "workshop.ipynb").exists()


def test_train_test_split_is_stable_and_stratified() -> None:
    from src.workshop_data import split_train_test

    observed = load_observed_data()
    train, test = split_train_test(observed)
    again, _ = split_train_test(observed)
    assert len(train) == 7_000 and len(test) == 3_000
    assert train["customer_id"].tolist() == again["customer_id"].tolist()
    assert abs(train["coupon_sent"].mean() - test["coupon_sent"].mean()) < 0.005


def test_pilot_is_randomized_and_disjoint() -> None:
    from src.workshop_data import FEATURES, load_pilot_data

    pilot = load_pilot_data()
    observed = load_observed_data()
    assert len(pilot) == 5_000
    assert set(pilot.columns) == set(observed.columns)
    assert set(pilot["customer_id"]).isdisjoint(observed["customer_id"])
    assert 0.45 < pilot["coupon_sent"].mean() < 0.55
    treated = pilot[pilot["coupon_sent"] == 1]
    control = pilot[pilot["coupon_sent"] == 0]
    smd = (treated[FEATURES].mean() - control[FEATURES].mean()) / np.sqrt(
        (treated[FEATURES].var() + control[FEATURES].var()) / 2
    )
    assert smd.abs().max() < 0.1  # a coin flip balances the groups


def test_qini_curve_counts_incremental_purchases() -> None:
    from solutions.causal_solution import qini_curve
    from src.evaluation import qini_coefficient

    pilot = pd.DataFrame({"coupon_sent": [1, 0, 1, 0], "purchased_7d": [1, 0, 0, 1]})
    curve = qini_curve(pilot, np.array([0.9, 0.8, 0.7, 0.6]))
    np.testing.assert_allclose(curve["share"], [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(curve["gain"], [1.0, 1.0, 1.0, 0.0])
    assert qini_coefficient(curve) == pytest.approx(0.75)


def test_pilot_prefers_logistic_over_boosting_and_the_old_ai() -> None:
    from sklearn.linear_model import LogisticRegression

    from solutions.causal_solution import make_boosting, qini_curve, t_learner_uplift
    from src.evaluation import qini_coefficient
    from src.workshop_data import FEATURES, load_pilot_data, split_train_test

    observed = load_observed_data()
    pilot = load_pilot_data()
    train, _ = split_train_test(observed)
    logistic = qini_coefficient(qini_curve(pilot, t_learner_uplift(train, pilot)[2]))
    boosting = qini_coefficient(
        qini_curve(pilot, t_learner_uplift(train, pilot, make_model=make_boosting)[2])
    )
    baseline = LogisticRegression(max_iter=2000).fit(train[FEATURES], train["purchased_7d"])
    old_ai = qini_coefficient(qini_curve(pilot, baseline.predict_proba(pilot[FEATURES])[:, 1]))
    assert logistic > old_ai > boosting > 0  # reference: 29.2 > 17.6 > 6.6


def test_budget_helper_stops_at_budget_and_positive_value() -> None:
    from src.policy import select_under_budget

    profit = np.array([5.0, 4.0, -1.0, 3.0])
    cost = np.array([2.0, 2.0, 1.0, 2.0])
    np.testing.assert_array_equal(select_under_budget(profit, cost, 4.0), [1, 1, 0, 0])
    np.testing.assert_array_equal(select_under_budget(profit, cost, 100.0), [1, 1, 0, 1])
