"""Audit the complete saved experiment against independently recomputed quantities."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/causal_study_20261015"


@pytest.mark.skipif(not (OUT/"metrics.csv").exists(),reason="full study not yet run")
def test_every_grid_case_has_consistent_data_and_independent_metric_reconciliation():
    manifest=json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"]=="complete" and manifest["completed_runs"]==240
    for name,digest in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    metrics=pd.read_csv(OUT/"metrics.csv")
    assert len(metrics)==240 and (~metrics.reveal_u).sum()==180
    assert not metrics.duplicated(["scenario","training_rows","seed","method","reveal_u"]).any()
    for scenario,part in metrics.groupby("scenario"):
        assert part.evaluation_sha256.nunique()==part.pilot_sha256.nunique()==1
        evaluation=pd.read_csv(OUT/"evaluation"/scenario/"customers.csv.gz")
        truth=pd.read_csv(OUT/"evaluation"/scenario/"truth.csv.gz")
        for row in part.itertuples():
            folder=(OUT/row.model_path).parent
            assert (folder/"model.joblib").is_file()
            metadata=json.loads((folder/"model.json").read_text(encoding="utf-8"))
            assert metadata["training_rows"]==row.training_rows
            assert ("latent_confounder" in metadata["features"])==row.reveal_u
            pred=pd.read_csv(folder/"predictions.csv.gz")
            assert pred.customer_id.equals(evaluation.customer_id)
            np.testing.assert_allclose(pred.true_cate,truth.true_cate,atol=1e-14)
            error=pred.estimated_uplift.to_numpy()-truth.true_cate.to_numpy()
            assert row.cate_rmse==pytest.approx(np.sqrt(np.mean(error**2)),abs=1e-12)
            assert row.ate_bias==pytest.approx(np.mean(error),abs=1e-12)
            chosen=pred.recommended.astype(bool)
            cost=evaluation.expected_coupon_cost
            margin=evaluation.expected_order_margin
            true_profit=truth.true_cate*margin-cost
            assert row.true_policy_profit==pytest.approx(true_profit[chosen].sum(),abs=1e-9)
            assert row.spend==pytest.approx(cost[chosen].sum(),abs=1e-9)
            assert row.spend<=2_500+1e-9
            assert row.recommended==chosen.sum()
            assert row.reload_max_abs_difference<1e-12


@pytest.mark.skipif(not (OUT/"notebook_execution.json").exists(),reason="instructor notebooks not yet executed")
def test_notebook_sources_match_and_executed_copies_have_real_outputs():
    import jupytext
    import nbformat
    for stem in ["workshop","solutions/workshop_answers","solutions/causal_cafe_complete","solutions/causal_forest_appendix"]:
        source=jupytext.read(ROOT/f"{stem}.py")
        notebook=nbformat.read(ROOT/f"{stem}.ipynb",as_version=4)
        nbformat.validate(notebook)
        assert [(c.cell_type,c.source) for c in source.cells]==[(c.cell_type,c.source) for c in notebook.cells]
    records=json.loads((OUT/"notebook_execution.json").read_text(encoding="utf-8"))
    assert len(records)==3 and all(r["status"]=="passed" for r in records)
    for record in records:
        name=Path(record["source"]).stem
        notebook=nbformat.read(OUT/"executed"/f"{name}.executed.ipynb",as_version=4)
        current=nbformat.read(ROOT/record["source"],as_version=4)
        assert [(c.cell_type,c.source) for c in notebook.cells]==[(c.cell_type,c.source) for c in current.cells]
        code=[c for c in notebook.cells if c.cell_type=="code" and c.source.strip()]
        assert all(c.execution_count is not None for c in code)
        outputs=[o for c in code for o in c.outputs]
        assert not any(o.output_type=="error" for o in outputs)
        assert sum("image/png" in o.get("data",{}) for o in outputs)>=4
