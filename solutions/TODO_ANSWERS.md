# TODO Answers — `workshop.ipynb` (facilitator only)

The four TODO cells in the participant notebook, the code that fills each one,
and what the cell prints once it works. The filled-in notebook is
`solutions/workshop_answers.ipynb`, built by `scripts/build_answer_notebook.py`
from `workshop.py`; run it top to bottom to see every output, including the
results board. Never put this file or the answer notebook on `main`.

Numbers below are from the pinned `requirements.txt`; in Colab the last decimals
can differ, but signs and customer counts should not.

## Task 2 — naive observed difference

```python
def naive_observed_difference(df: pd.DataFrame) -> float:
    treated = df.loc[df["coupon_sent"] == 1, "purchased_7d"].mean()
    control = df.loc[df["coupon_sent"] == 0, "purchased_7d"].mean()
    return float(treated - control)
```

Prints `naive observed difference: 0.1381972...`

Say: this is 13.8 points, but coupons went to likely buyers, so it is not the
ATE. The simulator's true ATE is 10.9 (keep that for the reveal).

## Task 3 — T-Learner

```python
def t_learner_uplift(train, score, features=FEATURES, make_model=make_logistic):
    treated = train[train["coupon_sent"] == 1]
    control = train[train["coupon_sent"] == 0]
    model_1 = make_model().fit(treated[features], treated["purchased_7d"])
    model_0 = make_model().fit(control[features], control["purchased_7d"])
    mu1 = model_1.predict_proba(score[features])[:, 1]
    mu0 = model_0.predict_proba(score[features])[:, 1]
    return mu0, mu1, mu1 - mu0
```

The cells after it print:

```text
logistic  AUC mu0 0.688 · mu1 0.684
boosting  AUC mu0 0.646 · mu1 0.636
```

Common mistakes:

- fitting on `score` instead of `train`, so the AUC is measured on data the
  model has already seen;
- calling `make_model` once and reusing it for both arms: the second `fit`
  overwrites the first, so `mu0 == mu1` and the uplift is all zeros;
- returning `mu0 - mu1`: the uplift histogram is centered on a negative value.

Say: boosting is the stronger model on paper and loses on held-out AUC here.
The training arms hold only 2,078 treated and 4,922 control customers, so it
overfits.

Stretch (break the DAG): mean uplift with all features is **0.098**, without
`loyalty_score` **0.099** (the other history features still carry its signal),
and without all five history features **0.131**, back near the naive 0.138.

```python
drop = {"days_since_last_purchase", "orders_30d", "app_sessions_30d",
        "average_order_value", "loyalty_score"}
t_learner_uplift(train, observed, features=[f for f in FEATURES if f != "loyalty_score"])[2].mean()
t_learner_uplift(train, observed, features=[f for f in FEATURES if f not in drop])[2].mean()
```

## Task 4 — Qini curve on the randomized pilot

```python
def qini_curve(pilot, score):
    order = np.argsort(-np.asarray(score))
    treated = pilot["coupon_sent"].to_numpy()[order]
    bought = pilot["purchased_7d"].to_numpy()[order]
    n_treated = np.cumsum(treated)
    n_control = np.cumsum(1 - treated)
    buyers_treated = np.cumsum(bought * treated)
    buyers_control = np.cumsum(bought * (1 - treated))
    scale = np.divide(n_treated, n_control, out=np.zeros(len(order)), where=n_control > 0)
    gain = buyers_treated - buyers_control * scale
    share = np.arange(1, len(order) + 1) / len(order)
    return pd.DataFrame({"share": share, "gain": gain})
```

The chart legend shows **logistic 29.2 · boosting 6.6 · old AI 17.6**.

Common mistakes:

- sorting ascending: every curve sinks below the random line;
- dividing by `n_control` without the `where=` guard: `inf`/`nan` at the top of
  the ranking, where there may be no control customers yet;
- forgetting to reorder `bought` with the same `order` as `treated`.

Say: AUC picked the right winner by accident; only randomized data can score
the uplift itself. Boosting barely beats random, and even the old AI's ranking
beats it, because likely buyers overlap with persuadables.

## Task 5 — policy under the Growth Budget

```python
def build_customer_scores(df, uplift):
    margin = df["expected_order_margin"].to_numpy()
    cost = df["expected_coupon_cost"].to_numpy()
    profit = uplift * margin - cost
    return pd.DataFrame(
        {
            "customer_id": df["customer_id"].to_numpy(),
            "purchase_probability": fit_purchase_baseline(df),
            "estimated_uplift": uplift,
            "expected_incremental_profit": profit,
            "recommended": select_under_budget(profit, cost, GROWTH_BUDGET),
        }
    )[ARTIFACT_COLUMNS]
```

*Check your policy* prints:

```text
customers recommended:        1,691
expected coupon spend:        2,499.84 / 2,500
expected incremental orders:  333.2
expected incremental profit:  927.19
mean estimated uplift (ATE):  +0.098
```

Common mistake: passing `uplift` from Task 3 (the 3,000 test customers)
instead of `uplift_all` (all 10,000). The notebook already passes
`uplift_all`; a participant who retypes the call gets a length error.

## Results board (no TODO)

Four charts built only from the participant's own output, mirroring the
complete analysis without the simulator truth:

| Results board | Complete analysis equivalent | What to read off |
| --- | --- | --- |
| `plot_effect_summary` | `07_uplift_diagnostics.png` (left two panels) | 13.8% naive vs 9.8% adjusted; uplift spread |
| `plot_model_check` | `08b_holdout_auc_vs_pilot_qini.png` | logistic wins on AUC *and* pilot Qini |
| `plot_budget_curve` | `09_budget_curve.png` | estimated profit at the budget: **+927** causal vs **−1,009** old AI |
| `plot_targeting_map` | `10_targeting_map.png` | the 1,691 recommended are high-uplift, not high-probability |

Then switch to the projector for the truth: `11_policy_comparison.png`
(causal **+$845**, old AI **−$946**, random −$715, oracle +$1,029) and
`12_archetype_reveal.png` (1,677 of the 1,691 are persuadables).
