# Causal Cafe Lab

Causal Cafe Lab is a 2-3 hour causal machine learning workshop in which
participants join a beverage chain's Growth Intelligence Team and repair a
coupon campaign that confuses attributed revenue with incremental profit.

Under explicit identification assumptions, the common workshop path uses a
hand-built T-Learner to estimate customer-level uplift, converts probability
uplift into expected incremental profit, and allocates a fixed headquarters
Growth Budget. Participants produce a targeting artifact and an audit-ready
decision memo; the facilitator reveals the synthetic truth at the end.

The conceptual framework is:

> Prediction estimates what will happen.  
> Causality estimates what an action will change.  
> Policy decides what we should do.  
> Language explains why.

## Setup

The current teaching format is **Colab notebook + Notion book**; slides are
retained as historical assets. Start with [`workshop.ipynb`](workshop.ipynb) for
the live reading-and-coding sequence. Companion lecture notes are provided
in an online Notion document.

Requires **Python 3.12**. Direct dependencies are pinned in
[`requirements.txt`](requirements.txt), with a full transitive lock in
[`requirements.lock.txt`](requirements.lock.txt) for an identical, verified
base workshop environment. The optional instructor study adds the packages in
`requirements-instructor.txt`; its executed environment is frozen in
`artifacts/causal_study_20261015/environment.lock.txt`.

Guaranteed path — standard library only, no extra tooling to install:

```bash
py -3.12 -m venv .venv                 # Windows;  python3.12 -m venv .venv on macOS/Linux
.venv\Scripts\activate                 # Windows;  source .venv/bin/activate on macOS/Linux
python -m pip install -r requirements.txt
python -m pip install -r requirements-instructor.txt  # facilitator algorithms and execution tools
python -m pytest -q --basetemp tmp/check
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

Open [`workshop.ipynb`](workshop.ipynb) in Google Colab (the participant
`main` branch README carries the *Open in Colab* badge), run the setup cell,
and complete the five participant tasks:

1. specify the causal question and DAG;
2. calculate the naive observed difference;
3. train a T-Learner on a 70/30 split, check it with held-out AUC, and compare
   logistic regression with gradient boosting;
4. score each ranking with a Qini curve on the randomized pilot
   (`data/pilot_rct.csv`);
5. build the budget-constrained policy and decision memo.

The completed notebook writes `artifacts/customer_scores.csv` with the stable
five-column artifact contract and prints the numbers the decision memo needs.
At the reveal the facilitator projects the complete analysis notebook
(`solutions/causal_cafe_complete`). Colab setup and the day-of flow are in
[docs/guides/COLAB_GUIDE.md](docs/guides/COLAB_GUIDE.md).

## Facilitator quick start

Keep `data/facilitator/`, `solutions/`, and the backup artifact hidden until the
reveal or live recovery.

Regenerate verified assets when needed:

```powershell
python scripts/generate_cafe_data.py
python scripts/build_solution_artifact.py
python -m jupytext --to ipynb workshop.py
python -m jupytext --test workshop.py --to ipynb
python scripts/build_answer_notebook.py
python -m pytest -q --basetemp tmp/facilitator-check
```

The complete instructor environment currently passes `54` checks, including
reconciliation of all 240 saved study cases. The artifact audit is skipped
until the full study has been run.

Live-coding answers for the four TODOs, with the output each cell should print:
[solutions/workshop_answers.ipynb](solutions/workshop_answers.ipynb). The filled-in notebook
is `solutions/workshop_answers.ipynb`.

Full answer key with every chart (concept diagrams, confounding/overlap
check, uplift ranking quality, budget curve, policy and archetype reveal) and
an auto-filled decision memo, written to `artifacts/complete_analysis/`:

```powershell
python solutions/causal_cafe_complete.py
```

After-workshop reading for participants: one HTML page that combines the
complete analysis and the Causal Forest appendix, with a contents section. It
contains simulator truth and every task answer, so hand it out only after the
session. Rebuild it from the executed notebooks:

```powershell
python scripts/execute_answer_notebooks.py
python scripts/export_reading_html.py   # -> artifacts/reading/causal_cafe_reading.html
```

Both answer workflows now save their fitted T-Learner, without retraining for
export. `solutions/workshop_answers.ipynb` writes `artifacts/models/t_learner.joblib`;
the complete analysis writes `artifacts/complete_analysis/models/t_learner.joblib`
(or below `CAUSAL_CAFE_OUTPUT_DIR`). Each bundle contains the control and treated
models, ordered features, training row count, and package versions, with matching
readable `.json` metadata. A save/load prediction check runs before continuing.

The participant `workshop.ipynb` keeps its four TODOs. Once they are filled in,
it saves the same model bundle and its last cell downloads `t_learner.zip` in
Google Colab. This is a local notebook update; Colab opened from GitHub will use
the revised cells only after the participant notebook is published to `main`.

Reuse your own saved model with the recorded package versions:

```python
import joblib
from src.workshop_data import load_observed_data

bundle = joblib.load("artifacts/models/t_learner.joblib")
customers = load_observed_data()
features = customers[bundle["features"]]
mu0 = bundle["model_control"].predict_proba(features)[:, 1]
mu1 = bundle["model_treated"].predict_proba(features)[:, 1]
uplift = mu1 - mu0
```


## Repository map

### Instructor robustness study

Install `requirements-instructor.txt` into the project environment, then run
`python scripts/run_causal_study.py` and `python scripts/render_causal_study.py`.
The study reports every size × seed × scenario combination and an additional
paired hidden-variable audit. See `artifacts/causal_study_20261015/PLAN.md` for
the fixed protocol and `REVIEW.md` in that directory for the executed findings.
The participant notebook is not executed by the study runner.
Use `python scripts/sync_workshop_notebooks.py` to synchronize all notebook
sources and `python scripts/execute_answer_notebooks.py` to execute only the
three filled-in instructor notebooks. Executed copies retain their figures
under `artifacts/causal_study_20261015/executed/`.

| Path | Responsibility |
| --- | --- |
| [`workshop.py`](workshop.py) / [`workshop.ipynb`](workshop.ipynb) | Participant notebook source and generated notebook |
| [`src/`](src/) | Facilitator-owned simulator, loading, predictive baseline, plotting and concept diagrams |
| [`solutions/`](solutions/) | Facilitator-only causal and policy reference answers, plus the complete analysis notebook `causal_cafe_complete` |
| [`scripts/`](scripts/) | Active dataset and backup-artifact generation commands |
| [`data/`](data/) | Participant observations and isolated facilitator truth |
| [`tests/`](tests/) | Data, export, estimator, policy, artifact, plot, and notebook safeguards |
| [`docs/guides/`](docs/guides/) | Facilitator guide, code walkthrough, delivery checklist, Colab delivery |
