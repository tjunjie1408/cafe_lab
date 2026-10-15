# %% [markdown]
# # Causal Cafe: End-to-End Data Analysis and Decision-Making (Facilitator Edition)
#
# This file integrates data ingestion, confounding and overlap diagnostics, predictive modeling,
# T-Learner causal uplift estimation, ranking quality evaluation, budget-constrained targeting policy,
# ground-truth reveal, and a decision memo into a single executable workflow.
#
# **After-workshop reading.** This is the full analysis behind the reveal. Unlike your
# workshop notebook, it reads the simulator's ground truth (`data/facilitator/`):
# every customer's true coupon effect, which no real company ever has. It also
# contains reference answers to all five tasks, so it is shared only after the session.
# Every number checked against truth here is a teaching check on synthetic data,
# not evidence about a real campaign.
#
# Key distinctions:
#
# - `purchase_probability` answers: "Who is more likely to buy?";
# - `estimated_uplift` answers: "Who will change their purchasing behavior because of the coupon?";
# - `expected_incremental_profit` converts uplift into business value;
# - `recommended` provides the final targeting list under a fixed budget constraint.
#
# How to run (from repository root): `python solutions/causal_cafe_complete.py`.
# All charts and tables are written to `artifacts/complete_analysis/`,
# which can be overridden via the `CAUSAL_CAFE_OUTPUT_DIR` environment variable.

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def find_project_root() -> Path:
    """Walk up from this file (script) or the working directory (notebook)."""
    starts = [Path.cwd().resolve()]
    if "__file__" in globals():
        starts.insert(0, Path(__file__).resolve().parent)
    for start in starts:
        for candidate in [start, *start.parents]:
            if (candidate / "data" / "observed_data.csv").exists():
                return candidate
    raise FileNotFoundError(
        "Cannot find data/observed_data.csv. Run this file inside the repository."
    )


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualisation import (  # noqa: E402
    BLUE,
    GREEN,
    GREY,
    ORANGE,
    PURPLE,
    RED,
    plot_causal_dag,
    plot_t_learner_schematic,
    plot_uplift_archetypes,
    plot_workshop_roadmap,
)
from src.workshop_data import (  # noqa: E402
    ARTIFACT_COLUMNS,
    FEATURES,
    GROWTH_BUDGET,
    load_facilitator_truth,
    load_observed_data,
    load_pilot_data,
    split_train_test,
)
from src.evaluation import qini_coefficient  # noqa: E402
from src.policy import select_under_budget as _select_positive  # noqa: E402
from src.causal_models import (  # noqa: E402
    fit_s_learner, predict_outcomes, predict_uplift, save_bundle, load_bundle,
)

OUTPUT_DIR = Path(
    os.environ.get("CAUSAL_CAFE_OUTPUT_DIR", PROJECT_ROOT / "artifacts" / "complete_analysis")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IN_NOTEBOOK = "ipykernel" in sys.modules

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def display_path(path: Path) -> str:
    """Print paths relative to the repository so no machine-specific folder is shown."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save every chart; display it inline when running inside a notebook."""
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    if IN_NOTEBOOK:
        from IPython.display import display

        display(fig)
    plt.close(fig)

# %% [markdown]
# ## 0. Workshop Roadmap and Customer Archetypes
#
# Align terminology using two conceptual diagrams first: Prediction -> Causality -> Policy -> Communication;
# and the fundamental distinction that "Will they buy?" and "Did the coupon change their purchase decision?" are two distinct questions.

# %%
save_figure(plot_workshop_roadmap(), "01_workshop_roadmap.png")
save_figure(plot_uplift_archetypes(), "02_uplift_archetypes_concept.png")

# %% [markdown]
# ## 1. Load and Inspect Data
#
# `observed` contains customer data available during analysis; `truth` stores true potential outcomes from the simulator,
# used solely for pedagogical validation. Real-world business datasets typically do not provide this ground truth.

# %%
observed = load_observed_data()
truth = load_facilitator_truth()

required_observed = {
    "customer_id",
    "coupon_sent",
    "purchased_7d",
    "expected_order_margin",
    "expected_coupon_cost",
    *FEATURES,
}
missing = required_observed.difference(observed.columns)
if missing:
    raise ValueError(f"Observed data is missing columns: {sorted(missing)}")
if not observed["customer_id"].equals(truth["customer_id"]):
    raise ValueError("Observed data and facilitator truth are not row-aligned")

treatment = observed["coupon_sent"].to_numpy()
outcome = observed["purchased_7d"].to_numpy()
margin = observed["expected_order_margin"].to_numpy()
coupon_cost = observed["expected_coupon_cost"].to_numpy()
true_cate = truth["true_cate"].to_numpy()

data_summary = pd.DataFrame(
    {
        "metric": [
            "customers",
            "coupon assignment rate",
            "overall purchase rate",
            "average coupon cost",
            "cost to coupon everyone",
            "growth budget",
        ],
        "value": [
            len(observed),
            treatment.mean(),
            outcome.mean(),
            coupon_cost.mean(),
            coupon_cost.sum(),
            GROWTH_BUDGET,
        ],
    }
)
data_summary

# %% [markdown]
# ## 2. Causal Question and DAG (Task 1 Reference Solution)
#
# - Treatment: `coupon_sent` (20% discount coupon)
# - Outcome: `purchased_7d` (Purchased within 7 days)
# - Confounders: loyalty, recent orders, app sessions, recency (days since last purchase), average order value
# - Do not control: post-treatment variables occurring after coupon delivery (e.g., actual coupon redemption, post-coupon browsing)
# - Budget rule: Paid from headquarters' fixed Growth Budget via `expected_coupon_cost`

# %%
save_figure(plot_causal_dag(), "03_causal_dag.png")

# %% [markdown]
# ## 3. Observational Correlation: Treated vs. Control Group (Task 2)
#
# The between-group difference below is an observational result, not the Average Treatment Effect (ATE). Historical coupon assignment was non-random, meaning the two groups were likely systematically different to begin with.

# %%
arm_summary = (
    observed.groupby("coupon_sent", as_index=False)
    .agg(customers=("customer_id", "size"), purchase_rate=("purchased_7d", "mean"))
    .assign(group=lambda frame: frame["coupon_sent"].map({0: "No coupon", 1: "Coupon"}))
)

naive_difference = float(
    outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
)
true_ate = float(true_cate.mean())

fig, ax = plt.subplots(figsize=(7.5, 4.8))
bars = ax.bar(
    arm_summary["group"],
    arm_summary["purchase_rate"],
    color=[GREY, BLUE],
    width=0.62,
)
ax.bar_label(bars, labels=[f"{value:.1%}" for value in arm_summary["purchase_rate"]], padding=4)
ax.set_ylim(0, arm_summary["purchase_rate"].max() * 1.25)
ax.set_ylabel("Observed purchase rate")
ax.set_xlabel("")
ax.set_title("Coupon recipients purchased more, but assignment was not random")
ax.text(
    0.01,
    -0.12,
    f"Observed difference: {naive_difference:.1%} | Simulator true ATE: {true_ate:.1%}",
    transform=ax.transAxes,
    fontsize=10,
)
save_figure(fig, "04_observed_purchase_rates.png")

# %% [markdown]
# ### Why Observational Differences Are Biased: Confounding and Overlap Diagnostics
#
# Left: Standardized Mean Difference (SMD). Prior to coupon delivery, the two groups already differed systematically in loyalty, order counts, etc.,
# representing the backdoor path `T <- X -> Y` in the DAG. |SMD| > 0.1 is conventionally considered imbalanced.
#
# Right: Distribution of propensity scores `P(T=1 | X)`. T-Learner relies on positivity (overlap):
# customers with similar characteristics must appear in both groups; otherwise, `mu0` or `mu1` relies solely on extrapolation.

# %%
treated_rows = observed.loc[treatment == 1, FEATURES]
control_rows = observed.loc[treatment == 0, FEATURES]
standardized_mean_difference = (
    (treated_rows.mean() - control_rows.mean())
    / np.sqrt((treated_rows.var() + control_rows.var()) / 2)
).sort_values()

propensity_model = LogisticRegression(max_iter=2_000)
propensity_model.fit(observed[FEATURES], treatment)
propensity = propensity_model.predict_proba(observed[FEATURES])[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9))
axes[0].barh(
    standardized_mean_difference.index,
    standardized_mean_difference.to_numpy(),
    color=[ORANGE if abs(value) > 0.1 else GREY for value in standardized_mean_difference],
)
for threshold in (-0.1, 0.1):
    axes[0].axvline(threshold, color="black", linestyle="--", linewidth=1)
axes[0].axvline(0, color="black", linewidth=1)
axes[0].set_title(
    f"Pre-coupon balance: {(standardized_mean_difference.abs() > 0.1).sum()} of "
    f"{len(FEATURES)} features have |SMD| > 0.1"
)
axes[0].set_xlabel("Standardized mean difference (coupon − no coupon)")

bins = np.linspace(0, 1, 41)
axes[1].hist(propensity[treatment == 0], bins=bins, alpha=0.6, color=GREY, label="No coupon", density=True)
axes[1].hist(propensity[treatment == 1], bins=bins, alpha=0.6, color=BLUE, label="Coupon", density=True)
axes[1].set_title(
    f"Propensity range {propensity.min():.2f}–{propensity.max():.2f}; "
    f"{((propensity < 0.05) | (propensity > 0.95)).mean():.1%} outside [0.05, 0.95]"
)
axes[1].set_xlabel("Estimated propensity P(coupon_sent = 1 | X)")
axes[1].set_ylabel("Density")
axes[1].legend()
save_figure(fig, "05_confounding_and_overlap.png")

overlap_summary = pd.DataFrame(
    {
        "metric": [
            "max |SMD|",
            "propensity min",
            "propensity max",
            "share outside [0.05, 0.95]",
        ],
        "value": [
            float(standardized_mean_difference.abs().max()),
            float(propensity.min()),
            float(propensity.max()),
            float(((propensity < 0.05) | (propensity > 0.95)).mean()),
        ],
    }
)
overlap_summary

# %% [markdown]
# ## 4. Predictive Model: Who Is Naturally Likely to Buy

# %%
predictive_model = LogisticRegression(max_iter=2_000)
predictive_model.fit(observed[FEATURES], outcome)
purchase_probability = predictive_model.predict_proba(observed[FEATURES])[:, 1]

# %% [markdown]
# ## 5. T-Learner: Estimating Individual Coupon Uplift (Task 3)
#
# Fit separate purchase models on the treated and control groups, then predict both counterfactual scenarios for each customer:
# `mu1 = P(Y=1 | do(coupon=1), X)` and `mu0 = P(Y=1 | do(coupon=0), X)`.
# Under the assumptions of exchangeability, positivity (overlap), and consistency, `mu1 - mu0` can be interpreted as the Conditional Average Treatment Effect (CATE).

# %%
save_figure(plot_t_learner_schematic(), "06_t_learner_schematic.png")

# %% [markdown]
# Train on 70% of the customers and keep 30% aside, so every accuracy check below
# uses customers the models never saw. The uplift is then scored for everyone.

# %%
train, test = split_train_test(observed)
treated = train.loc[train["coupon_sent"] == 1]
control = train.loc[train["coupon_sent"] == 0]

treated_model = LogisticRegression(max_iter=2_000)
control_model = LogisticRegression(max_iter=2_000)
treated_model.fit(treated[FEATURES], treated["purchased_7d"])
control_model.fit(control[FEATURES], control["purchased_7d"])

model_path = save_bundle(
    {"kind": "t", "features": list(FEATURES),
     "model_control": control_model, "model_treated": treated_model},
    OUTPUT_DIR / "models" / "t_learner.joblib", training_rows=len(train), split_seed=42,
)
restored = load_bundle(model_path)
control_model, treated_model = restored["model_control"], restored["model_treated"]
mu1 = treated_model.predict_proba(observed[FEATURES])[:, 1]
mu0 = control_model.predict_proba(observed[FEATURES])[:, 1]
estimated_uplift = mu1 - mu0
estimated_ate = float(estimated_uplift.mean())

uplift_diagnostics = pd.DataFrame(
    {
        "metric": ["mean estimated uplift", "mean true CATE", "correlation", "MAE"],
        "value": [
            estimated_ate,
            true_ate,
            np.corrcoef(estimated_uplift, true_cate)[0, 1],
            np.abs(estimated_uplift - true_cate).mean(),
        ],
    }
)
uplift_diagnostics

# %% [markdown]
# ### Analyze the saved model
# The models above were saved and reloaded before computing any uplift.
# Diagnostics, policy scores and pilot evaluation all reuse these loaded objects.

# %%
restored_mu0 = restored["model_control"].predict_proba(observed[restored["features"]])[:, 1]
restored_mu1 = restored["model_treated"].predict_proba(observed[restored["features"]])[:, 1]
np.testing.assert_allclose(restored_mu0, mu0, rtol=1e-12)
np.testing.assert_allclose(restored_mu1, mu1, rtol=1e-12)
print(f"Saved and reloaded: {display_path(model_path)} ({model_path.stat().st_size:,} bytes)")

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [0.8, 1, 1]})

ate_labels = ["Naive observed\ndifference", "T-Learner\n(adjusted)", "Simulator\ntrue ATE"]
ate_values = [naive_difference, estimated_ate, true_ate]
ate_bars = axes[0].bar(ate_labels, ate_values, color=[GREY, BLUE, GREEN], width=0.62)
axes[0].bar_label(ate_bars, labels=[f"{value:.1%}" for value in ate_values], padding=4)
axes[0].set_ylim(0, max(ate_values) * 1.25)
axes[0].set_title(
    f"Bias vs. truth: naive {naive_difference - true_ate:+.1%}, adjusted {estimated_ate - true_ate:+.1%}"
)
axes[0].set_ylabel("Average coupon effect on purchase")

axes[1].hist(estimated_uplift, bins=42, color=BLUE, alpha=0.88, edgecolor="white")
axes[1].axvline(0, color="black", linewidth=1.2, linestyle="--")
axes[1].axvline(estimated_ate, color=ORANGE, linewidth=2, label=f"Mean: {estimated_ate:.1%}")
axes[1].set_title("Estimated coupon uplift varies across customers")
axes[1].set_xlabel("Estimated uplift (mu1 - mu0)")
axes[1].set_ylabel("Customers")
axes[1].legend()

plot_sample = np.random.default_rng(42).choice(len(observed), size=2_500, replace=False)
axes[2].scatter(
    true_cate[plot_sample],
    estimated_uplift[plot_sample],
    s=13,
    alpha=0.35,
    color=BLUE,
    edgecolors="none",
)
limits = [
    min(true_cate.min(), estimated_uplift.min()),
    max(true_cate.max(), estimated_uplift.max()),
]
axes[2].plot(limits, limits, color=ORANGE, linestyle="--", linewidth=1.6, label="Perfect fit")
axes[2].set_xlim(limits)
axes[2].set_ylim(limits)
axes[2].set_title(
    f"Estimated vs. true CATE: r = {np.corrcoef(estimated_uplift, true_cate)[0, 1]:.2f}"
)
axes[2].set_xlabel("True CATE (teaching data only)")
axes[2].set_ylabel("Estimated uplift")
axes[2].legend()

save_figure(fig, "07_uplift_diagnostics.png")

# %% [markdown]
# ### Ranking Quality: Does Uplift Ranking Prioritize Those Truly Influenced by Coupons?
#
# Left: Customers partitioned into 10 deciles by estimated uplift, comparing average estimated uplift against ground truth (calibration).
#
# Right: Cumulative gain curve. Targeting customers in descending order of different scores, with the y-axis showing cumulative true incremental purchases.
# Curves closer to the top-left indicate superior ranking; random targeting forms the diagonal, and Oracle (ranking by true CATE) serves as the theoretical upper bound.

# %%
decile = pd.qcut(pd.Series(estimated_uplift).rank(method="first"), 10, labels=False) + 1
calibration = (
    pd.DataFrame({"decile": decile, "estimated": estimated_uplift, "true": true_cate})
    .groupby("decile")
    .mean()
)

targeted_share = np.arange(1, len(observed) + 1) / len(observed)


def cumulative_true_gain(rank_score: np.ndarray) -> np.ndarray:
    return np.cumsum(true_cate[np.argsort(-rank_score)])


fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9))
positions = np.arange(1, 11)
axes[0].bar(positions - 0.2, calibration["estimated"], width=0.4, color=BLUE, label="Estimated uplift")
axes[0].bar(positions + 0.2, calibration["true"], width=0.4, color=GREEN, label="True CATE")
axes[0].axhline(0, color="black", linewidth=1)
axes[0].set_xticks(positions)
axes[0].set_title(
    "Decile calibration: max gap "
    f"{(calibration['estimated'] - calibration['true']).abs().max():.3f}"
)
axes[0].set_xlabel("Estimated-uplift decile (1 = lowest, 10 = highest)")
axes[0].set_ylabel("Mean effect on purchase probability")
axes[0].legend()

axes[1].plot(targeted_share, cumulative_true_gain(true_cate), color=GREEN, linestyle=":", linewidth=2, label="Oracle (true CATE)")
axes[1].plot(targeted_share, cumulative_true_gain(estimated_uplift), color=ORANGE, linewidth=2.4, label="T-Learner uplift")
axes[1].plot(targeted_share, cumulative_true_gain(purchase_probability), color=GREY, linewidth=2.2, label="Purchase probability")
axes[1].plot([0, 1], [0, true_cate.sum()], color="black", linestyle="--", linewidth=1, label="Random")
axes[1].set_title("Cumulative true incremental purchases by ranking")
axes[1].set_xlabel("Share of customers targeted (sorted by score)")
axes[1].set_ylabel("Cumulative true incremental purchases")
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
axes[1].legend()
save_figure(fig, "08_uplift_ranking_quality.png")

# %% [markdown]
# ### Model Check: Held-Out AUC vs. the Randomized Pilot (Tasks 3–4)
#
# AUC measures how well each arm's model predicts *purchase* on held-out customers.
# It cannot measure *uplift*, because no customer is seen in both arms. Last
# month's pilot assigned coupons by coin flip, so there the two arms are
# comparable and a Qini curve scores the uplift ranking directly: at each cut-off,
# treated buyers minus control buyers scaled to the treated count.
#
# Compare training and held-out AUC as well as pilot Qini. A large generalization
# gap is evidence of overfitting; model complexity alone does not establish it.

# %%
def make_logistic():
    return LogisticRegression(max_iter=2_000)


def make_boosting():
    return HistGradientBoostingClassifier(random_state=0)


def fit_arm_models(make_model):
    return (
        make_model().fit(control[FEATURES], control["purchased_7d"]),
        make_model().fit(treated[FEATURES], treated["purchased_7d"]),
    )


def qini_curve(frame: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-np.asarray(score))
    arm = frame["coupon_sent"].to_numpy()[order]
    bought = frame["purchased_7d"].to_numpy()[order]
    n_treated, n_control = np.cumsum(arm), np.cumsum(1 - arm)
    scale = np.divide(n_treated, n_control, out=np.zeros(len(order)), where=n_control > 0)
    gain = np.cumsum(bought * arm) - np.cumsum(bought * (1 - arm)) * scale
    return pd.DataFrame({"share": np.arange(1, len(order) + 1) / len(order), "gain": gain})


pilot = load_pilot_data()
pilot_truth = pd.read_csv(PROJECT_ROOT / "data" / "facilitator" / "pilot_truth.csv")
test_treated = test["coupon_sent"].to_numpy() == 1
test_outcome = test["purchased_7d"].to_numpy()
baseline_on_train = LogisticRegression(max_iter=2_000).fit(train[FEATURES], train["purchased_7d"])

model_rows, pilot_curves = [], {}
gb0, gb1 = fit_arm_models(make_boosting)
gb_path = save_bundle(
    {"kind": "t", "features": list(FEATURES), "model_control": gb0, "model_treated": gb1},
    OUTPUT_DIR / "models" / "t_learner_boosting.joblib", training_rows=len(train), split_seed=42,
)
s_path = save_bundle(
    fit_s_learner(train, FEATURES), OUTPUT_DIR / "models" / "s_learner.joblib",
    training_rows=len(train), split_seed=42,
)
candidate_bundles = {
    "T-Learner, logistic": restored,
    "T-Learner, boosting": load_bundle(gb_path),
    "S-Learner, logistic + interactions": load_bundle(s_path),
}
for name, bundle in candidate_bundles.items():
    test_mu0, test_mu1 = predict_outcomes(bundle, test)
    train_mu0, train_mu1 = predict_outcomes(bundle, train)
    train_treated = train["coupon_sent"].to_numpy() == 1
    train_outcome = train["purchased_7d"].to_numpy()
    pilot_uplift = predict_uplift(bundle, pilot)
    pilot_curves[name] = qini_curve(pilot, pilot_uplift)
    model_rows.append(
        {
            "model": name,
            "training AUC mu0": roc_auc_score(train_outcome[~train_treated], train_mu0[~train_treated]),
            "training AUC mu1": roc_auc_score(train_outcome[train_treated], train_mu1[train_treated]),
            "held-out AUC mu0": roc_auc_score(
                test_outcome[~test_treated], test_mu0[~test_treated]
            ),
            "held-out AUC mu1": roc_auc_score(
                test_outcome[test_treated], test_mu1[test_treated]
            ),
            "pilot Qini": qini_coefficient(pilot_curves[name]),
        }
    )
pilot_curves["purchase probability (old AI)"] = qini_curve(
    pilot, baseline_on_train.predict_proba(pilot[FEATURES])[:, 1]
)
pilot_curves["perfect knowledge (simulator)"] = qini_curve(pilot, pilot_truth["true_cate"].to_numpy())
for name in ["purchase probability (old AI)", "perfect knowledge (simulator)"]:
    model_rows.append({"model": name, "pilot Qini": qini_coefficient(pilot_curves[name])})
model_check = pd.DataFrame(model_rows)
model_check.to_csv(OUTPUT_DIR / "model_check.csv", index=False)
model_check

# %% [markdown]
# ### Are the T-Learner and S-Learner really tied? Check against the truth
#
# The pilot Qini can only rank the candidates on 5,000 randomized customers. The simulator
# lets us go one step further: compare each candidate's uplift with every customer's true
# effect, and spend the same budget with each candidate's ranking. This is possible only
# on synthetic data.

# %%
candidate_truth_rows = []
for name, bundle in candidate_bundles.items():
    candidate_uplift = predict_uplift(bundle, observed)
    candidate_pick = _select_positive(candidate_uplift * margin - coupon_cost, coupon_cost, GROWTH_BUDGET) == 1
    candidate_truth_rows.append(
        {
            "model": name,
            "mean estimated uplift": candidate_uplift.mean(),
            "correlation with true CATE": np.corrcoef(candidate_uplift, true_cate)[0, 1],
            "MAE vs true CATE": np.abs(candidate_uplift - true_cate).mean(),
            "customers picked": int(candidate_pick.sum()),
            "true incremental profit": float((true_cate * margin - coupon_cost)[candidate_pick].sum()),
        }
    )
candidate_truth = pd.DataFrame(candidate_truth_rows).set_index("model")
candidate_truth.to_csv(OUTPUT_DIR / "candidate_truth_check.csv")
candidate_truth.round(3)

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [0.8, 1.2]})
auc_table = model_check.dropna(subset=["held-out AUC mu0"]).set_index("model")
positions = np.arange(len(auc_table))
axes[0].bar(positions - 0.18, auc_table["held-out AUC mu0"], width=0.36, color=GREY, label="mu0 (no coupon)")
axes[0].bar(positions + 0.18, auc_table["held-out AUC mu1"], width=0.36, color=BLUE, label="mu1 (coupon)")
axes[0].set_xticks(positions, ["T-logistic", "T-boosting", "S-logistic"], fontsize=9)
axes[0].set_ylim(0.5, 0.75)
axes[0].set_ylabel("AUC on the 30% held-out customers")
axes[0].set_title("Held-out purchase prediction")
axes[0].legend()

styles = ["-", "--", "-.", "-", ":"]
colors = [BLUE, ORANGE, PURPLE, GREY, GREEN]
for (name, curve), color, style in zip(pilot_curves.items(), colors, styles):
    axes[1].plot(curve["share"], curve["gain"], color=color, linestyle=style, linewidth=2.2,
                 label=f"{name} (Qini {qini_coefficient(curve):.1f})")
axes[1].plot([0, 1], [0, pilot_curves["T-Learner, logistic"]["gain"].iloc[-1]],
             color="black", linestyle="--", linewidth=1, label="random line")
axes[1].set_title("Uplift check on the randomized pilot")
axes[1].set_xlabel("Share of pilot customers targeted (highest score first)")
axes[1].set_ylabel("Incremental purchases")
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
axes[1].legend(fontsize=9)
save_figure(fig, "08b_holdout_auc_vs_pilot_qini.png")

# %% [markdown]
# ## 6. From Uplift to Profit: Budget-Constrained Customer Targeting (Task 5)
#
# `expected_incremental_profit = uplift × expected_order_margin - expected_coupon_cost`
#
# Dimensional check: (Probability change) × (Gross margin per incremental purchase) - (Expected coupon cost per targeted customer)
# = Expected incremental profit per targeted customer.
#
# The policy only considers customers with positive expected incremental profit, prioritizing them in descending order of profit until the budget is exhausted.

# %%
expected_incremental_profit = estimated_uplift * margin - coupon_cost
true_incremental_profit = true_cate * margin - coupon_cost


def select_under_budget(rank_score: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
    """The same `src.policy.select_under_budget` your workshop notebook used, as a boolean mask.

    It walks customers from the highest score down, skips anyone who no longer fits
    the budget, and stops at the first non-positive score. Purchase probabilities and
    random draws are always positive, so the same helper also ranks the old AI and the
    random policy below.
    """
    return _select_positive(np.asarray(rank_score), cost, budget).astype(bool)


selected = select_under_budget(expected_incremental_profit, coupon_cost, GROWTH_BUDGET)

scores = pd.DataFrame(
    {
        "customer_id": observed["customer_id"],
        "purchase_probability": purchase_probability,
        "estimated_uplift": estimated_uplift,
        "expected_incremental_profit": expected_incremental_profit,
        "recommended": selected.astype(np.int64),
    }
)[ARTIFACT_COLUMNS]
# The artifact itself is written by scripts/build_solution_artifact.py
# (artifacts/customer_scores.csv); this analysis only reports on it.

budget_spent = float(coupon_cost[selected].sum())
estimated_policy_profit = float(expected_incremental_profit[selected].sum())
policy_summary = pd.DataFrame(
    {
        "metric": [
            "customers recommended",
            "budget spent",
            "budget remaining",
            "estimated incremental profit",
            "average estimated uplift among recommended",
            "customers with positive estimated profit",
        ],
        "value": [
            int(selected.sum()),
            budget_spent,
            GROWTH_BUDGET - budget_spent,
            estimated_policy_profit,
            estimated_uplift[selected].mean(),
            int((expected_incremental_profit > 0).sum()),
        ],
    }
)
policy_summary

# %% [markdown]
# ### Budget Curve: Why Target Only Positive-Profit Customers and Treat Budget as a Hard Constraint
#
# Accumulate customers in descending order of expected incremental profit. Estimated cumulative profit initially rises, peaking where incremental profit turns negative;
# the portion within the budget limit constitutes the final targeting list. The grey curve illustrates targeting by purchase probability under the exact same spend.
#
# **Comparing with your workshop notebook:** there, both curves were *estimated* profit, so the
# old AI's line ended near −1,009. Here the grey and green curves use the simulator's *true*
# profit, so the old AI ends near −946 and the causal list near +845 instead of the estimated +927.
# The orange curve is the same estimate you saw in the workshop.

# %%
causal_order = np.argsort(-expected_incremental_profit)
predictive_order = np.argsort(-purchase_probability)

fig, ax = plt.subplots(figsize=(9.5, 5.4))
ax.plot(
    np.cumsum(coupon_cost[causal_order]),
    np.cumsum(expected_incremental_profit[causal_order]),
    color=ORANGE,
    linewidth=2.4,
    label="Causal ranking: estimated profit",
)
ax.plot(
    np.cumsum(coupon_cost[causal_order]),
    np.cumsum(true_incremental_profit[causal_order]),
    color=GREEN,
    linewidth=2,
    linestyle="--",
    label="Causal ranking: true profit (simulator)",
)
ax.plot(
    np.cumsum(coupon_cost[predictive_order]),
    np.cumsum(true_incremental_profit[predictive_order]),
    color=GREY,
    linewidth=2,
    label="Predictive ranking: true profit (simulator)",
)
ax.axvline(GROWTH_BUDGET, color="black", linestyle=":", linewidth=1.5)
ax.text(GROWTH_BUDGET, 0.96, f"  Growth Budget = {GROWTH_BUDGET:,.0f}", fontsize=10,
        va="top", transform=ax.get_xaxis_transform())
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Profit peaks, then falls: stop at positive value and the budget")
ax.set_xlabel("Cumulative expected coupon spend")
ax.set_ylabel("Cumulative incremental profit")
ax.legend(loc="lower left")
save_figure(fig, "09_budget_curve.png")

# %% [markdown]
# ## 7. Visualizing the Final Targeting Policy

# %%
fig, ax = plt.subplots(figsize=(8.4, 6.1))
ax.scatter(
    purchase_probability[~selected],
    estimated_uplift[~selected],
    s=9,
    alpha=0.18,
    color=GREY,
    label="Not recommended",
    edgecolors="none",
)
ax.scatter(
    purchase_probability[selected],
    estimated_uplift[selected],
    s=18,
    alpha=0.72,
    color=ORANGE,
    label="Recommended",
    edgecolors="none",
)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.set_title("High purchase probability is not the same as high uplift")
ax.set_xlabel("Predicted purchase probability")
ax.set_ylabel("Estimated coupon uplift")
ax.legend(frameon=True)
save_figure(fig, "10_targeting_map.png")

# %% [markdown]
# ## 8. Reveal: Comparing Four Policies Under the Same Budget (Facilitator Only)
#
# - Random: Randomly allocate coupons until budget is exhausted (average of 100 simulations);
# - Predictive ranking: Legacy system, ranking customers by purchase probability;
# - Causal value ranking: The causal policy proposed in this analysis;
# - Oracle: Ranking by simulator ground-truth CATE (unobservable in reality; theoretical upper bound).

# %%
predictive_selected = select_under_budget(purchase_probability, coupon_cost, GROWTH_BUDGET)
oracle_selected = select_under_budget(true_incremental_profit, coupon_cost, GROWTH_BUDGET)
random_rng = np.random.default_rng(2026)
random_values = [
    true_incremental_profit[
        select_under_budget(random_rng.random(len(observed)), coupon_cost, GROWTH_BUDGET)
    ].sum()
    for _ in range(100)
]

policy_masks = {
    "Predictive ranking": predictive_selected,
    "Causal value ranking": selected,
    "Oracle (true CATE)": oracle_selected,
}
policy_comparison = pd.DataFrame(
    [
        {
            "policy": "Random (mean of 100)",
            "customers": np.nan,
            "spend": np.nan,
            "true_incremental_profit": float(np.mean(random_values)),
        },
        *[
            {
                "policy": name,
                "customers": int(mask.sum()),
                "spend": float(coupon_cost[mask].sum()),
                "true_incremental_profit": float(true_incremental_profit[mask].sum()),
            }
            for name, mask in policy_masks.items()
        ],
    ]
)
policy_comparison.to_csv(OUTPUT_DIR / "policy_comparison.csv", index=False)
policy_comparison

# %%
fig, ax = plt.subplots(figsize=(9.2, 5.2))
values = policy_comparison["true_incremental_profit"]
bars = ax.bar(
    policy_comparison["policy"].str.replace(" (", "\n(", regex=False),
    values,
    color=[GREY, RED, ORANGE, GREEN],
    width=0.62,
)
ax.axhline(0, color="black", linewidth=1)
ax.bar_label(bars, labels=[f"{'-' if value < 0 else ''}${abs(value):,.0f}" for value in values], padding=4)
ax.margins(y=0.15)
ax.set_title(f"True incremental profit under the same {GROWTH_BUDGET:,.0f} budget")
ax.set_ylabel("True incremental profit (simulator evaluation)")
save_figure(fig, "11_policy_comparison.png")

# %% [markdown]
# ### Archetype Reveal: Who Did the Two Policies Spend the Budget On?
#
# Classify customers into archetypes using simulator ground truth, checking the rules in this order:
#
# - Negative responders: `true_cate < 0`
# - Persuadables: `true_cate >= 0.10`
# - Likely purchasers: `true_mu0 >= 0.35` and `true_cate < 0.05`
# - Unlikely purchasers: `true_mu1 < 0.15` and `true_cate < 0.10`
# - Remaining: Moderate responders
#
# The classes are mutually exclusive: a high-baseline customer with CATE < 0 counts as a Negative
# responder, not a Likely purchaser, which leaves 163 Likely purchasers (1.6%).

# %%
ARCHETYPE_ORDER = [
    "Persuadables",
    "Moderate responders",
    "Likely purchasers",
    "Unlikely purchasers",
    "Negative responders",
]
ARCHETYPE_COLORS = [GREEN, BLUE, GREY, "#D9D9D9", RED]
true_mu0 = truth["true_mu0"].to_numpy()
true_mu1 = truth["true_mu1"].to_numpy()
archetype = np.select(
    [
        true_cate < 0,
        true_cate >= 0.10,
        (true_mu0 >= 0.35) & (true_cate < 0.05),
        (true_mu1 < 0.15) & (true_cate < 0.10),
    ],
    ["Negative responders", "Persuadables", "Likely purchasers", "Unlikely purchasers"],
    default="Moderate responders",
)

archetype_summary = (
    pd.DataFrame(
        {
            "archetype": archetype,
            "true_mu0": true_mu0,
            "true_mu1": true_mu1,
            "true_cate": true_cate,
            "coupon_cost": coupon_cost,
            "predictive_pick": predictive_selected,
            "causal_pick": selected,
        }
    )
    .groupby("archetype")
    .agg(
        customers=("true_cate", "size"),
        mean_true_mu0=("true_mu0", "mean"),
        mean_true_mu1=("true_mu1", "mean"),
        mean_true_cate=("true_cate", "mean"),
        mean_coupon_cost=("coupon_cost", "mean"),
        predictive_picks=("predictive_pick", "sum"),
        causal_picks=("causal_pick", "sum"),
    )
    .reindex(ARCHETYPE_ORDER)
)
archetype_summary.insert(1, "share", archetype_summary["customers"] / len(observed))
archetype_summary.to_csv(OUTPUT_DIR / "archetype_summary.csv")
archetype_summary

# %%
composition = pd.DataFrame(
    {
        "All customers": archetype_summary["customers"],
        "Predictive picks": archetype_summary["predictive_picks"],
        "Causal picks": archetype_summary["causal_picks"],
    }
)
composition = composition / composition.sum()

fig, axes = plt.subplots(
    1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.25, 1]}, layout="constrained"
)
left = np.zeros(len(composition.columns))
for name, color in zip(ARCHETYPE_ORDER, ARCHETYPE_COLORS):
    shares = composition.loc[name].to_numpy()
    axes[0].barh(composition.columns, shares, left=left, color=color, label=name)
    for y, (start, share) in enumerate(zip(left, shares)):
        if share >= 0.06:
            axes[0].text(start + share / 2, y, f"{share:.0%}", ha="center", va="center",
                         fontsize=9, color="white" if color not in (GREY, "#D9D9D9") else "black")
    left += shares
axes[0].invert_yaxis()
axes[0].set_xlim(0, 1)
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
axes[0].set_title("Who each policy spends the budget on")
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

y_positions = np.arange(len(ARCHETYPE_ORDER))
axes[1].hlines(y_positions, archetype_summary["mean_true_mu0"], archetype_summary["mean_true_mu1"],
               color=GREY, linewidth=3)
axes[1].scatter(archetype_summary["mean_true_mu0"], y_positions, color=GREY, s=70, zorder=3, label="Without coupon (mu0)")
axes[1].scatter(archetype_summary["mean_true_mu1"], y_positions, color=ORANGE, s=70, zorder=3, label="With coupon (mu1)")
axes[1].set_yticks(y_positions, ARCHETYPE_ORDER)
axes[1].invert_yaxis()
axes[1].set_title("True purchase probability with and without a coupon")
axes[1].set_xlabel("Mean true purchase probability")
axes[1].legend(loc="lower right")
save_figure(fig, "12_archetype_reveal.png")

# %% [markdown]
# ## 9. Save Summary Results
#
# These CSV and PNG files jointly provide auditable results: charts convey intuition, while tables preserve exact values.

# %%
analysis_summary = pd.concat(
    [
        pd.DataFrame(
            {
                "metric": ["naive observed difference", "true ATE"],
                "value": [naive_difference, true_ate],
            }
        ),
        overlap_summary,
        uplift_diagnostics,
        pd.DataFrame(
            {
                "metric": [f"pilot Qini: {name}" for name in model_check["model"]],
                "value": model_check["pilot Qini"],
            }
        ),
        policy_summary,
    ],
    ignore_index=True,
)
analysis_summary.to_csv(OUTPUT_DIR / "analysis_summary.csv", index=False)

print(f"Input rows: {len(observed):,}")
print(f"Recommended customers: {selected.sum():,}")
print(f"Budget spent: ${budget_spent:,.2f} / ${GROWTH_BUDGET:,.2f}")
print(f"Estimated incremental profit: ${estimated_policy_profit:,.2f}")
print(f"Outputs written to: {display_path(OUTPUT_DIR)}")

# %% [markdown]
# ## 10. Decision Memo (Reference Solution, Auto-Populated)
#
# The five fixed headings align with the facilitator rubric. Simulator ground truth appears only in the final section,
# explicitly designated as pedagogical validation rather than production evidence.

# %%
causal_true_value = float(true_incremental_profit[selected].sum())
predictive_true_value = float(true_incremental_profit[predictive_selected].sum())
decision_memo = f"""# Decision memo — Causal Cafe coupon campaign

## 1. Estimated effect
The naive treated-minus-control gap is {naive_difference:.1%}, but coupons were
sent preferentially to likely buyers, so this is not an ATE. After adjusting for
the seven pre-treatment features with a T-Learner, the estimated average effect
is {estimated_ate:+.1%} on 7-day purchase probability, ranging from
{np.percentile(estimated_uplift, 5):+.1%} to {np.percentile(estimated_uplift, 95):+.1%}
(5th–95th percentile) across customers.

## 2. Identification assumptions
- Conditional exchangeability: the DAG assumes loyalty, orders, app sessions,
  recency and order value capture all common causes of coupon and purchase.
- Overlap: estimated propensities range {propensity.min():.2f}–{propensity.max():.2f};
  {((propensity < 0.05) | (propensity > 0.95)).mean():.1%} of customers fall outside [0.05, 0.95].
- Consistency: one coupon type (20%), one outcome window (7 days).
- No post-treatment variables (redemption, post-coupon browsing) are adjusted for.

## 3. Budget impact
{int(selected.sum()):,} customers recommended; expected coupon spend
${budget_spent:,.2f} of the ${GROWTH_BUDGET:,.0f} Growth Budget; estimated
incremental profit ${estimated_policy_profit:,.2f}.
{int((expected_incremental_profit > 0).sum()):,} customers have positive expected value, so the budget binds.

## 4. Recommended action
Send coupons only to the recommended list: positive expected incremental profit,
ranked from highest, until the budget is spent. Stop ranking by purchase
probability; it spends the budget on customers who would buy anyway.

## 5. What still needs validation
- The ranking was checked on last month's randomized pilot: the logistic T-Learner
  scored a Qini of {model_check.set_index("model").loc["T-Learner, logistic", "pilot Qini"]:.1f},
  against {model_check.set_index("model").loc["T-Learner, boosting", "pilot Qini"]:.1f} for gradient boosting and
  {model_check.set_index("model").loc["purchase probability (old AI)", "pilot Qini"]:.1f} for purchase probability.
  The S-Learner (logistic with coupon interactions) scored
  {model_check.set_index("model").loc["S-Learner, logistic + interactions", "pilot Qini"]:.1f}, practically tied;
  the gap is within what a different pilot sample could produce, and choosing a model
  on this pilot would use up the pilot as an independent check.
- Run a randomized holdout inside the recommended list before scaling.
- Check calibration of the profit estimate: in the synthetic check the realised
  value was ${causal_true_value:,.2f} against ${estimated_policy_profit:,.2f}
  estimated, so treat the estimate as optimistic.
- Monitor redemption cost and re-fit when customer mix changes.
- Synthetic check only (not production evidence): the predictive ranking would
  have produced {'-' if predictive_true_value < 0 else ''}${abs(predictive_true_value):,.2f} under the same budget.
"""
(OUTPUT_DIR / "decision_memo.md").write_text(decision_memo, encoding="utf-8")
print(decision_memo)

# %% [markdown]
# ## Decision Conclusions
#
# 1. The coupon group exhibits a higher purchase rate, but due to selection bias in historical coupon assignment, this observational difference cannot be treated directly as the ATE.
# 2. After adjusting for pre-treatment features, the T-Learner average effect closely tracks ground truth; customers exhibit significant heterogeneity in coupon response, and high baseline purchase probability does not imply high uplift.
# 3. Final targeting should be driven by expected incremental profit, strictly bounded by the headquarters' fixed budget constraint.
# 4. The legacy predictive policy expends most of the budget on customers who would have bought anyway, yielding negative true incremental profit under the same budget.
# 5. Simulator ground truth is used solely for educational validation; prior to production deployment, a randomized holdout experiment is still required to verify the true incremental business value.
