# %% [markdown]
# # Causal Cafe Lab — Growth Intelligence Team
# Facilitator builds the interface; you build the causal decision logic.
#
# > Prediction estimates what will happen. Causality estimates what an action
# > will change. Policy decides what we should do.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.workshop_data import (
    FEATURES, ARTIFACT_COLUMNS, GROWTH_BUDGET,
    load_observed_data, load_facilitator_truth,
)
from src.predictive import fit_purchase_baseline
from src.visualisation import plot_uplift_distribution, plot_policy_comparison

observed = load_observed_data()
observed.head()

# %% [markdown]
# The old AI ranks customers by predicted purchase probability. High reported
# ROI made the campaign look successful — until the audit.

# %%
observed["purchase_probability"] = fit_purchase_baseline(observed)
observed[["customer_id", "purchase_probability", "coupon_sent", "purchased_7d"]].head()

# %% [markdown]
# ## Task 1 — Specify the question (worksheet)
# In the cell below, write your answers as plain text:
#
# - treatment
# - outcome
# - three plausible confounders
# - one variable you must NOT control for because it happens after treatment
# - a one-line DAG sketch
# - the supplied funding rule (headquarters Growth Budget)

# %% [markdown]
# _Your answer here._

# %% [markdown]
# ## Task 2 — Naive observed difference
# Compute E[Y|T=1] - E[Y|T=0]. This is what we observed, not what we want.
# It must NOT be labeled ATE.

# %%
def naive_observed_difference(df: pd.DataFrame) -> float:
    # TODO: return treated purchase rate minus control purchase rate.
    raise NotImplementedError


naive = naive_observed_difference(observed)
print("naive observed difference:", naive)

# %% [markdown]
# ## Task 3 — T-Learner
# Split by coupon_sent, fit one outcome model per arm on FEATURES, then predict
# mu0 and mu1 for every customer and take uplift = mu1 - mu0.

# %%
def t_learner_uplift(df: pd.DataFrame, features=FEATURES):
    # TODO: fit two LogisticRegression models and return (mu0, mu1, uplift).
    raise NotImplementedError


mu0, mu1, uplift = t_learner_uplift(observed)
plot_uplift_distribution(uplift)

# %% [markdown]
# ## Task 4 — Campaign policy and decision memo
# expected_incremental_profit = uplift * expected_order_margin - expected_coupon_cost
#
# Recommend positive-value customers under the fixed GROWTH_BUDGET, then write
# artifacts/customer_scores.csv with columns ARTIFACT_COLUMNS.

# %%
def build_customer_scores(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: assemble purchase_probability, estimated_uplift,
    # expected_incremental_profit, recommended; enforce the budget.
    raise NotImplementedError


scores = build_customer_scores(observed)
Path("artifacts").mkdir(parents=True, exist_ok=True)
scores.to_csv("artifacts/customer_scores.csv", index=False)
scores.head()

# %% [markdown]
# ## Reveal (facilitator truth)
# Compare your uplift with the simulator's known CATE. Real observational data
# never contains both potential outcomes; this is a synthetic teaching check.

# %%
truth = load_facilitator_truth().set_index("customer_id").loc[observed["customer_id"]]
print("your uplift vs true CATE corr:",
      np.corrcoef(uplift, truth["true_cate"].to_numpy())[0, 1])

# %% [markdown]
# ### Decision memo (fill in)
# Estimated effect · identification assumptions · budget impact ·
# recommended action · what still needs validation (e.g. a holdout experiment).
