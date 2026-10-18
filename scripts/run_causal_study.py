"""Run the preregistered-size facilitator grid and retain every fitted model."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Keep nuisance and BLAS work bounded while the forest uses two CPU threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.causal_models import load_bundle, predict_uplift, save_bundle
from src.workshop_data import FEATURES
from solutions.experiment_data import SCENARIOS, make_population
from solutions.causal_study import METHODS, FOREST_SETTINGS, fit_candidate, score_predictions


def fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def run_case(output, train, evaluation, truth, pilot, *, scenario, n, seed, method, reveal_u, signature):
    key = f"{scenario}/n{n}/seed{seed}/{method}{'_observed_U' if reveal_u else ''}"
    folder = output / "runs" / key
    folder.mkdir(parents=True, exist_ok=True)
    features = [*FEATURES, "latent_confounder"] if reveal_u else list(FEATURES)
    bundle, fit_seconds = fit_candidate(method, train, features, seed)
    expected = predict_uplift(bundle, evaluation)
    model_path = save_bundle(
        bundle, folder / "model.joblib", training_rows=n, fit_seconds=fit_seconds,
        seed=seed, scenario=scenario, method=method, observed_hidden_variable=reveal_u,
        training_sha256=fingerprint(train), study_signature=signature,
    )
    loaded = load_bundle(model_path)
    start = time.perf_counter()
    uplift = predict_uplift(loaded, evaluation)
    pilot_uplift = predict_uplift(loaded, pilot)
    inference_seconds = time.perf_counter() - start
    np.testing.assert_allclose(uplift, expected, rtol=1e-12, atol=1e-12)
    metrics, selected = score_predictions(evaluation, truth, pilot, uplift, pilot_uplift)
    row = dict(scenario=scenario, training_rows=n, seed=seed, method=method,
               reveal_u=reveal_u, fit_seconds=fit_seconds, inference_seconds=inference_seconds,
               model_bytes=model_path.stat().st_size, reload_max_abs_difference=float(np.max(abs(uplift-expected))),
               training_sha256=fingerprint(train), evaluation_sha256=fingerprint(evaluation),
               pilot_sha256=fingerprint(pilot), model_path=model_path.relative_to(output).as_posix(), **metrics)
    predictions = pd.DataFrame({"customer_id": evaluation.customer_id, "true_cate": truth.true_cate,
                                "estimated_uplift": uplift, "recommended": selected.astype(int)})
    predictions.to_csv(folder / "predictions.csv.gz", index=False)
    pd.DataFrame({"customer_id": pilot.customer_id, "estimated_uplift": pilot_uplift}).to_csv(
        folder / "pilot_predictions.csv.gz", index=False)
    (folder / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"{key}: RMSE={row['cate_rmse']:.4f} bias={row['ate_bias']:+.4f} "
          f"profit={row['true_policy_profit']:+.2f} fit={fit_seconds:.2f}s", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/causal_study_20261015")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = dict(sizes=[2_000, 10_000, 50_000], seeds=list(range(40,45)), scenarios=list(SCENARIOS),
                  methods=list(METHODS), evaluation_rows=5_000, pilot_rows=5_000,
                  evaluation_seed=91_001, pilot_seed=91_002, budget=2_500,
                  forest=FOREST_SETTINGS, hidden_variable_audit=True,
                  selection="all fixed candidates reported; no tuning or winner selected on evaluation outcomes",
                  seed_spread="descriptive min/max across five training seeds, not confidence intervals")
    sources = ["src/data_generator.py", "src/causal_models.py", "src/policy.py", "src/evaluation.py",
               "solutions/experiment_data.py", "solutions/causal_study.py", "scripts/run_causal_study.py"]
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    signature = hashlib.sha256(json.dumps([config,hashes],sort_keys=True).encode()).hexdigest()
    manifest = {"config":config,"source_sha256":hashes,"signature":signature,
                "versions":{p:importlib.metadata.version(p) for p in ["numpy","pandas","scikit-learn","econml","joblib"]},
                "device":"CPU; forest n_jobs=2; BLAS and OpenMP limited to one thread", "status":"running"}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    started=time.perf_counter()
    rows=[]
    with threadpool_limits(limits=1):
        for scenario in SCENARIOS:
            evaluation, truth = make_population(5_000,91_001,scenario,"evaluation")
            pilot,pilot_truth = make_population(5_000,91_002,scenario,"pilot",randomized=True)
            data_dir=output/"evaluation"/scenario
            data_dir.mkdir(parents=True,exist_ok=True)
            for name,frame in [("customers",evaluation),("truth",truth),("pilot",pilot),("pilot_truth",pilot_truth)]:
                frame.to_csv(data_dir/f"{name}.csv.gz",index=False)
            for seed in config["seeds"]:
                pool,_=make_population(50_000,seed,scenario,f"train-{seed}")
                pool=pool.iloc[np.random.default_rng(seed+123).permutation(len(pool))].reset_index(drop=True)
                assert not set(pool.customer_id) & (set(evaluation.customer_id)|set(pilot.customer_id))
                for n in config["sizes"]:
                    train=pool.iloc[:n].copy()
                    for reveal_u in ([False,True] if scenario=="hidden_confounder" else [False]):
                        for method in METHODS:
                            row=run_case(output,train,evaluation,truth,pilot,scenario=scenario,n=n,seed=seed,
                                         method=method,reveal_u=reveal_u,signature=signature)
                            rows.append(row)
                            pd.DataFrame(rows).to_csv(output/"metrics.csv",index=False)
    manifest.update(status="complete", elapsed_seconds=time.perf_counter()-started, completed_runs=len(rows),
                    expected_runs=240, main_grid_runs=180, paired_hidden_variable_runs=60)
    if len(rows)!=240:
        raise AssertionError(f"Expected 240 runs, got {len(rows)}")
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Completed {len(rows)} runs in {manifest['elapsed_seconds']:.1f}s",flush=True)


if __name__ == "__main__":
    main()
