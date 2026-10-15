"""Independent checks of experiment controls, effect targets and score formulas."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from src.workshop_data import FEATURES
from src.causal_models import load_bundle, predict_uplift, save_bundle
from solutions.experiment_data import make_population
from solutions.causal_study import fit_candidate, score_predictions


def test_populations_are_disjoint_reproducible_and_pilot_is_randomized():
    for scenario in ["smooth", "nonlinear", "hidden_confounder"]:
        train, _ = make_population(1_000,40,scenario,"train")
        evaluation, _ = make_population(1_000,91_001,scenario,"evaluation")
        pilot, truth = make_population(1_000,91_002,scenario,"pilot",randomized=True)
        repeat, _ = make_population(1_000,40,scenario,"train")
        pd.testing.assert_frame_equal(train, repeat)
        assert not set(train.customer_id) & set(evaluation.customer_id)
        assert not set(train.customer_id) & set(pilot.customer_id)
        np.testing.assert_array_equal(truth.treatment_propensity,0.5)
        assert 0.4 < pilot.coupon_sent.mean() < 0.6


def test_hidden_variable_changes_assignment_and_baseline_but_not_cate_target():
    frame,truth=make_population(5_000,42,"hidden_confounder","test")
    expected=0.03+0.13*frame.price_sensitivity+0.10*frame.offer_affinity-0.08*frame.loyalty_score
    np.testing.assert_allclose(truth.true_cate,expected,atol=1e-15)
    assert "latent_confounder" not in FEATURES
    e=truth.groupby("latent_confounder").treatment_propensity.mean()
    assert e[1]-e[0]>0.5
    assert truth.true_mu0.between(0,1).all() and truth.true_mu1.between(0,1).all()


def test_nonlinear_scenario_has_specified_threshold_interactions():
    frame,truth=make_population(1_000,41,"nonlinear","test")
    shift=(-0.7+2.8*((frame.price_sensitivity>0.60)&(frame.offer_affinity>0.50))
           +1.2*frame.days_since_last_purchase.between(20,55)-1.4*(frame.loyalty_score>0.65))
    np.testing.assert_allclose(truth.true_mu1,expit(logit(truth.true_mu0)+shift))


def test_scoring_uses_true_incremental_value_and_respects_budget():
    evaluation=pd.DataFrame({"customer_id":["a","b","c"],"expected_order_margin":[10.,10.,10.],
                             "expected_coupon_cost":[2.,2.,2.]})
    truth=pd.DataFrame({"customer_id":["a","b","c"],"true_cate":[0.3,0.1,-0.2]})
    pilot=pd.DataFrame({"coupon_sent":[1,0,1,0],"purchased_7d":[1,0,0,0]})
    metrics,selected=score_predictions(evaluation,truth,pilot,np.array([0.3,0.1,-0.2]),np.array([4,3,2,1]),budget=4)
    np.testing.assert_array_equal(selected,[True,False,False])
    assert metrics["cate_rmse"]==0 and metrics["ate_bias"]==0
    assert metrics["spend"]==2 and metrics["true_policy_profit"]==pytest.approx(1)
    with pytest.raises(ValueError,match="aligned"):
        score_predictions(evaluation,truth.iloc[::-1],pilot,np.zeros(3),np.zeros(4))


def test_real_causal_forest_survives_reload_and_predicts_without_training(tmp_path,monkeypatch):
    from econml.dml import CausalForestDML
    train,_=make_population(1_000,40,"smooth","train")
    evaluation,_=make_population(250,91_001,"smooth","evaluation")
    bundle,seconds=fit_candidate("CausalForestDML",train,FEATURES,40)
    assert isinstance(bundle["model"],CausalForestDML) and seconds>0
    expected=predict_uplift(bundle,evaluation)
    restored=load_bundle(save_bundle(bundle,tmp_path/"forest.joblib",training_rows=len(train)))
    def forbidden(*args,**kwargs):
        raise AssertionError("analysis may not fit a forest")
    monkeypatch.setattr(CausalForestDML,"fit",forbidden)
    # Parallel forest reductions can differ in the last floating-point bits.
    np.testing.assert_allclose(predict_uplift(restored,evaluation),expected,rtol=1e-12,atol=1e-12)
    assert expected.shape==(250,) and np.isfinite(expected).all()
