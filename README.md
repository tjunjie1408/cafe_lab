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
3. Work through the four tasks:
   1. specify the causal question and DAG;
   2. calculate the naive observed difference;
   3. implement a T-Learner;
   4. build the budget-constrained targeting policy.
4. The *Check your policy* cell prints the numbers for your five-part
   decision memo.
5. The reveal happens on the projector.

If Colab disconnects, reopen your Drive copy and choose
**Runtime → Run all**; the setup cell fetches the files again.

## Files

| Path | What it is |
| --- | --- |
| `workshop.ipynb` | The notebook you work in |
| `data/observed_data.csv` | 10,000 synthetic customers: features, coupon, purchase |
| `src/` | Helpers the notebook imports (data loading, baseline model, plots) |

All data is synthetic.
