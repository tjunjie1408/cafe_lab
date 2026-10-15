"""Regression guards for prediction-only evaluation and portable model bundles."""

from pathlib import Path
import hashlib
import runpy
import shutil

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script",["workshop_answers","causal_cafe_complete"])
def test_answer_does_not_refit_identical_outcome_models(tmp_path, monkeypatch, script):
    import matplotlib
    matplotlib.use("Agg")
    shutil.copytree(ROOT / "src", tmp_path / "src")
    shutil.copytree(ROOT / "data", tmp_path / "data")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAUSAL_CAFE_OUTPUT_DIR",str(tmp_path/"artifacts"))
    original = LogisticRegression.fit
    fitted = set()

    def fit_once(self, x, y, *args, **kwargs):
        digest = hashlib.sha256(np.asarray(x).tobytes() + np.asarray(y).tobytes()).hexdigest()
        key = (np.asarray(x).shape, digest, repr(self.get_params()))
        assert key not in fitted, "an identical logistic model was refitted during evaluation"
        fitted.add(key)
        return original(self, x, y, *args, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", fit_once)
    original_boosting = HistGradientBoostingClassifier.fit
    boosting_fitted = set()

    def fit_boosting_once(self,x,y,*args,**kwargs):
        key=hashlib.sha256(np.asarray(x).tobytes()+np.asarray(y).tobytes()).hexdigest()
        assert key not in boosting_fitted,"an identical boosting model was refitted during evaluation"
        boosting_fitted.add(key)
        return original_boosting(self,x,y,*args,**kwargs)

    monkeypatch.setattr(HistGradientBoostingClassifier,"fit",fit_boosting_once)
    result = runpy.run_path(str(ROOT / "solutions" / f"{script}.py"))
    assert "S-Learner, logistic + interactions" in result.get("curves",result.get("pilot_curves"))
    assert len(fitted) == 6  # pooled baseline, propensity, T0, T1, S, pilot baseline
    assert len(boosting_fitted)==2  # one control and one treated outcome model


def test_saved_s_learner_predicts_without_fit_and_obeys_feature_order(tmp_path, monkeypatch):
    from src.causal_models import fit_s_learner, predict_uplift, save_bundle, load_bundle
    from src.workshop_data import FEATURES, load_observed_data, split_train_test
    train, test = split_train_test(load_observed_data())
    bundle = fit_s_learner(train, FEATURES)
    expected = predict_uplift(bundle, test)
    path = save_bundle(bundle, tmp_path / "s.joblib", training_rows=len(train))
    restored = load_bundle(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("prediction must not train")

    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    np.testing.assert_array_equal(predict_uplift(restored, test), expected)
    np.testing.assert_array_equal(predict_uplift(restored, test[test.columns[::-1]]), expected)
    assert np.std(expected) > 0.01
    with pytest.raises(ValueError, match="Missing"):
        predict_uplift(restored, test.drop(columns=[FEATURES[0]]))


@pytest.mark.parametrize("script",["workshop_answers","causal_cafe_complete"])
def test_analysis_uses_reloaded_t_learner(tmp_path, monkeypatch, script):
    """Held-out predictions, pilot and policy must call the loaded objects."""
    import matplotlib
    matplotlib.use("Agg")
    shutil.copytree(ROOT / "src", tmp_path / "src")
    shutil.copytree(ROOT / "data", tmp_path / "data")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAUSAL_CAFE_OUTPUT_DIR",str(tmp_path/"artifacts"))
    original_load = joblib.load

    loaded_ids = set()
    prediction_calls = {}
    original_predict = LogisticRegression.predict_proba

    def observe_loaded_predictions(self, *args, **kwargs):
        if id(self) in loaded_ids:
            prediction_calls[id(self)] = prediction_calls.get(id(self), 0) + 1
        return original_predict(self, *args, **kwargs)

    def observe_loaded_models(path, *args, **kwargs):
        bundle = original_load(path, *args, **kwargs)
        if Path(path).name == "t_learner.joblib":
            loaded_ids.update([id(bundle["model_control"]), id(bundle["model_treated"])])
        return bundle

    monkeypatch.setattr(joblib, "load", observe_loaded_models)
    monkeypatch.setattr(LogisticRegression, "predict_proba", observe_loaded_predictions)
    result = runpy.run_path(str(ROOT / "solutions" / f"{script}.py"))
    x = result["observed"][result["FEATURES"]]
    saved = original_load(tmp_path / "artifacts/models/t_learner.joblib")
    original_uplift = saved["model_treated"].predict_proba(x)[:, 1] - saved["model_control"].predict_proba(x)[:, 1]
    np.testing.assert_allclose(result["scores"]["estimated_uplift"], original_uplift)
    assert len(loaded_ids) == 2
    assert all(prediction_calls.get(key, 0) >= 3 for key in loaded_ids)
