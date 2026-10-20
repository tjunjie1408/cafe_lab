# Causal Cafe Lab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tjunjie1408/cafe_lab/blob/main/workshop.ipynb)

A hands-on causal machine learning workshop. You join Causal Cafe's Growth
Intelligence Team after an audit finds that the coupon campaign confused
*attributed revenue* with *incremental profit*. Your job: estimate who the
coupon actually changes, and decide who should get one under a fixed budget.

> Prediction estimates what will happen.  
> Causality estimates what an action will change.  
> Policy decides what we should do.  
> Language explains why.

The workshop uses a **Colab notebook + Notion book**. The notebook contains
the coding tasks, explanations and result summaries. Companion lecture notes
are provided in an online Notion document.

## Before the session

- A laptop with Chrome, Edge or Firefox, and a Google account for Colab.
- Comfortable with Python and pandas; you have called scikit-learn's `fit()`
  and `predict_proba()` before.
- Nothing to install.

## During the session

1. Click **Open in Colab** above, then **File → Save a copy in Drive** so your
   work is kept.
2. Run the first code cell (setup). It downloads this repository into your
   Colab session.
3. Work through the five tasks:
   1. specify the causal question and DAG;
   2. calculate the naive observed difference;
   3. train a T-Learner on a train/test split, check it with held-out AUC, and
      compare gradient boosting and an S-Learner with explicit treatment interactions;
   4. score your uplift ranking with a Qini curve on a randomized pilot;
   5. build the budget-constrained targeting policy.
4. The *Check your policy* cell prints the numbers for your five-part
   decision memo.
5. Download your trained T-Learner as a ZIP containing fitted models and JSON
   metadata. Review the closing campaign results and selected-customer profiles
   to explain who should receive coupons and the estimated business value.
6. The reveal happens on the projector.

The notebook keeps four coding TODOs. Each candidate is trained once, saved,
and reloaded before evaluation. Held-out checks, pilot Qini curves, and the
campaign policy use predictions from those loaded models.

If Colab disconnects, reopen your Drive copy and choose
**Runtime → Run all**; the setup cell fetches the files again.

## Files

| Path | What it is |
| --- | --- |
| `workshop.ipynb` | The notebook you work in |
| `data/observed_data.csv` | 10,000 synthetic customers: features, coupon, purchase |
| `data/pilot_rct.csv` | 5,000 customers from last month's pilot, where a coin flip decided who got a coupon |
| `src/` | Helpers the notebook imports (data loading, baseline model, budget selection, Qini scoring, plots) |
| `src/causal_models.py` | S-Learner, model persistence, and prediction helpers |

The notebook creates `artifacts/customer_scores.csv` and three model bundles
under `artifacts/models/`: logistic T-Learner, boosting T-Learner, and S-Learner.
The model-download cell packages the logistic T-Learner bundle and its metadata.

All data is synthetic.
