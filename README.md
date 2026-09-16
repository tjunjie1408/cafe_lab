# Causal Cafe Lab

Causal Cafe Lab is a 2-3 hour causal machine learning workshop in which
participants join a beverage chain's Growth Intelligence Team and repair a
coupon campaign that confuses attributed revenue with incremental profit.

Under explicit identification assumptions, the common workshop path uses a
hand-built T-Learner to estimate customer-level uplift, converts probability
uplift into expected incremental profit, and allocates a fixed headquarters
Growth Budget. Participants produce a targeting artifact and an audit-ready
decision memo for a facilitator-provided Streamlit dashboard.

The conceptual framework is:

> Prediction estimates what will happen.  
> Causality estimates what an action will change.  
> Policy decides what we should do.  
> Language explains why.

## Setup

Requires **Python 3.12**. Direct dependencies are pinned in
[`requirements.txt`](requirements.txt), with a full transitive lock in
[`requirements.lock.txt`](requirements.lock.txt) for an identical, verified
facilitator environment.

Guaranteed path — standard library only, no extra tooling to install:

```bash
py -3.12 -m venv .venv                 # Windows;  python3.12 -m venv .venv on macOS/Linux
.venv\Scripts\activate                 # Windows;  source .venv/bin/activate on macOS/Linux
python -m pip install -r requirements.txt
python -m pytest -q --basetemp tmp/check  # optional: 30 checks should pass
```

If the Windows `py` launcher is unavailable, run the first command with an
installed Python 3.12 executable path, then continue with the activated
environment's `python`.

Optional fast path, only if you already have [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

uv is not required for the workshop. Use the `venv` path unless you already
have uv installed.

Regenerate the datasets from the simulator (already bundled under `data/`):

```bash
python scripts/generate_cafe_data.py
```

## Participant quick start

Open [`workshop.ipynb`](workshop.ipynb) and complete its four participant
tasks:

1. specify the causal question and DAG;
2. calculate the naive observed difference;
3. implement the T-Learner;
4. build the budget-constrained policy and decision memo.

The completed notebook writes `artifacts/customer_scores.csv` with the stable
five-column artifact contract.

## Facilitator quick start

Keep `data/facilitator/`, `solutions/`, and the backup artifact hidden until the
reveal or live recovery.

Regenerate verified assets when needed:

```powershell
python scripts/generate_cafe_data.py
python scripts/build_solution_artifact.py
python -m jupytext --to ipynb workshop.py
python -m jupytext --test workshop.py --to ipynb
python -m pytest -q --basetemp tmp/facilitator-check
```

Expected result: `30 passed`.


## Repository map

| Path | Responsibility |
| --- | --- |
| [`workshop.py`](workshop.py) / [`workshop.ipynb`](workshop.ipynb) | Participant notebook source and generated notebook |
| [`src/`](src/) | Facilitator-owned simulator, loading, predictive baseline, and plotting helpers |
| [`solutions/`](solutions/) | Facilitator-only causal and policy reference answers |
| [`scripts/`](scripts/) | Active dataset and backup-artifact generation commands |
| [`data/`](data/) | Participant observations and isolated facilitator truth |
| [`tests/`](tests/) | Data, export, estimator, policy, artifact, plot, and notebook safeguards |
