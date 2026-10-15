# %% [markdown]
# # Causal Cafe: effect heterogeneity beyond the T-Learner
# **After-workshop reading, part 2.** Does a heavier method or more data change the
# answer? This appendix reads a pre-run instructor study (240 fits on simulated
# data: three scenarios, three training sizes, five seeds, four methods) and fits
# one real CausalForestDML. To reproduce it, run `scripts/run_causal_study.py` and
# `scripts/render_causal_study.py` first. The fixed grid includes every training seed,
# sample size and scenario; simulator truth is used only for evaluation.
# This notebook adds a real CausalForestDML fit on one declared grid case and
# reads the complete pre-run results. It never selects a favorable seed.

# %%
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
from threadpoolctl import threadpool_limits

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/workshop_data.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.workshop_data import FEATURES
from src.causal_models import save_bundle, load_bundle, predict_uplift
from solutions.experiment_data import make_population
from solutions.causal_study import fit_candidate, score_predictions, FOREST_SETTINGS

STUDY = ROOT / "artifacts/causal_study_20261015"
manifest = json.loads((STUDY / "manifest.json").read_text(encoding="utf-8"))
assert manifest["status"] == "complete" and manifest["completed_runs"] == 240
metrics = pd.read_csv(STUDY / "metrics.csv")

# %% [markdown]
# ## What changes in a Causal Forest?
# Outcome and treatment models are cross-fitted before residualization. The
# forest targets heterogeneity in the treatment effect; honest splitting uses
# separate observations to choose tree splits and estimate leaves. This still
# requires observed confounders and overlap. Inference is disabled in this
# experiment, so seed bands below must not be interpreted as CATE intervals.

# %%
display(Image(filename=str(STUDY / "figures/01_effect_estimators.png")))
print("Fixed forest settings:", FOREST_SETTINGS)

# %% [markdown]
# ## Fit a real forest on the declared nonlinear case
# 10,000 actual training customers, seed 42, 5,000 independent evaluation
# customers and an independent randomized pilot. The training subset is the
# same nested subset used by the batch experiment.

# %%
pool, _ = make_population(50_000,42,"nonlinear","train-42")
pool = pool.iloc[np.random.default_rng(42+123).permutation(len(pool))].reset_index(drop=True)
train = pool.iloc[:10_000].copy()
evaluation, truth = make_population(5_000,91_001,"nonlinear","evaluation")
pilot, _ = make_population(5_000,91_002,"nonlinear","pilot",randomized=True)
with threadpool_limits(limits=1):
    forest_bundle, fit_seconds = fit_candidate("CausalForestDML",train,FEATURES,42)
    path = save_bundle(forest_bundle,STUDY/"appendix_demo/causal_forest.joblib",
                       training_rows=len(train),fit_seconds=fit_seconds,seed=42,scenario="nonlinear")
    loaded = load_bundle(path)
    uplift = predict_uplift(loaded,evaluation)
    pilot_uplift = predict_uplift(loaded,pilot)
result, selected = score_predictions(evaluation,truth,pilot,uplift,pilot_uplift)
display(pd.Series(result).to_frame("measured result"))
print(f"Actual forest fitting time in this notebook: {fit_seconds:.3f}s")

# %% [markdown]
# ## Does the forest recover the pattern, not just the average?
# Each point is a decile of predicted uplift. The diagonal is group-level
# calibration, not a guarantee that an individual's effect is observable.

# %%
check = pd.DataFrame({"estimate":uplift,"truth":truth.true_cate})
check["decile"] = pd.qcut(check.estimate.rank(method="first"),10,labels=False)
calibration = check.groupby("decile")[["estimate","truth"]].mean()
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(calibration.estimate,calibration.truth,"o-",label="Causal Forest DML")
limits = [float(calibration.min().min())-.02,float(calibration.max().max())+.02]
ax.plot(limits,limits,"k--",label="Calibrated group means")
ax.set(xlabel="Mean estimated effect",ylabel="Mean true CATE",title="Nonlinear case: effect calibration")
ax.legend(); fig.tight_layout(); plt.show()

# %% [markdown]
# ## Read all sample sizes and training seeds
# The three main panels retain every candidate and every seed. No hypothesis
# test or confidence interval is inferred from five repetitions.

# %%
display(Image(filename=str(STUDY / "figures/02_learning_curves.png")))
display(Image(filename=str(STUDY / "figures/03_policy_learning_curves.png")))
display(metrics.loc[~metrics.reveal_u].groupby(["scenario","training_rows","method"])[
    ["cate_rmse","ate_bias","true_policy_profit","fit_seconds"]].mean())

# %% [markdown]
# ## What happens when identification fails?
# Here U influences treatment assignment and baseline purchase, while CATE
# depends only on visible X. The paired audit includes U in adjustment on
# exactly the same rows and outcomes. Larger samples alone do not supply U.

# %%
display(Image(filename=str(STUDY / "figures/10_hidden_confounding_dag.png")))
display(Image(filename=str(STUDY / "figures/04_hidden_variable_audit.png")))

# %% [markdown]
# ## Why Qini, calibration and profit answer different questions
# Qini checks ranking on a randomized pilot; calibration checks effect scale;
# profit additionally incorporates margin, coupon cost and a fixed budget.
# These are synthetic pedagogical checks, not production business evidence.

# %%
display(Image(filename=str(STUDY / "figures/05_randomized_pilot_qini.png")))
display(Image(filename=str(STUDY / "figures/07_effect_calibration.png")))
display(Image(filename=str(STUDY / "figures/08_budget_sensitivity.png")))
