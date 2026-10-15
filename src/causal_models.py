"""Prediction and persistence shared by classroom and instructor analyses."""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression


def _features(bundle, frame):
    names = bundle["features"]
    missing = set(names).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    return frame[names]


def s_design(frame, features, treatment):
    """An explicit X, T, X*T design; treatment can be observed or counterfactual."""
    x = frame[features].to_numpy(dtype=float)
    t = np.broadcast_to(np.asarray(treatment, dtype=float), (len(frame),)).reshape(-1, 1)
    return np.column_stack([x, t, x * t])


def fit_s_learner(train, features):
    model = LogisticRegression(max_iter=4_000)
    model.fit(s_design(train, features, train["coupon_sent"]), train["purchased_7d"])
    return {"kind": "s", "features": list(features), "model": model}


def predict_outcomes(bundle, frame):
    """Predict both mean outcomes without fitting or changing any model."""
    x = _features(bundle, frame)
    if bundle.get("kind", "t") == "t":
        return (
            bundle["model_control"].predict_proba(x)[:, 1],
            bundle["model_treated"].predict_proba(x)[:, 1],
        )
    if bundle["kind"] == "s":
        return (
            bundle["model"].predict_proba(s_design(frame, bundle["features"], 0))[:, 1],
            bundle["model"].predict_proba(s_design(frame, bundle["features"], 1))[:, 1],
        )
    raise ValueError("Causal Forest estimates effects, not two outcome probabilities")


def predict_uplift(bundle, frame):
    if bundle.get("kind", "t") == "causal_forest":
        return np.asarray(bundle["model"].effect(_features(bundle, frame).to_numpy())).reshape(-1)
    mu0, mu1 = predict_outcomes(bundle, frame)
    return mu1 - mu0


def save_bundle(bundle, path, *, training_rows, **run_metadata):
    """Save fitted estimators and readable provenance; never train on export."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kind": bundle.get("kind", "t"),
        "features": list(bundle["features"]),
        "treatment": "coupon_sent", "outcome": "purchased_7d",
        "training_rows": int(training_rows),
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
        "joblib_version": joblib.__version__, "python_version": sys.version.split()[0],
        **run_metadata,
    }
    joblib.dump({**bundle, **metadata}, path, compress=3)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def load_bundle(path):
    """Load only trusted workshop artifacts, preserving their feature order."""
    bundle = joblib.load(path)
    features = bundle.get("features", [])
    if not features or len(features) != len(set(features)):
        raise ValueError("Model bundle requires nonempty, unique ordered features")
    kind = bundle.get("kind", "t")
    keys = ["model_control", "model_treated"] if kind == "t" else ["model"]
    if kind not in {"t", "s", "causal_forest"} or any(key not in bundle for key in keys):
        raise ValueError("Invalid model bundle")
    return bundle
