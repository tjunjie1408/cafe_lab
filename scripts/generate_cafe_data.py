"""Generate the stable Phase 1 Causal Cafe workshop datasets."""

from pathlib import Path
import os
import shutil
import sys
import tempfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generator import generate_cafe_data

PILOT_CUSTOMERS = 5_000
PILOT_SEED = 2026


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def _replace_file(source: Path, destination: Path) -> None:
    source.replace(destination)


def _write_dataset_pair(
    observed: pd.DataFrame,
    truth: pd.DataFrame,
    observed_path: Path,
    truth_path: Path,
) -> None:
    if len(observed) != len(truth):
        raise ValueError("observed and truth datasets must have matching row counts")
    if not observed["customer_id"].equals(truth["customer_id"]):
        raise ValueError("observed and truth customer_id values must match in order")

    observed_temp: Path | None = None
    truth_temp: Path | None = None
    backups: dict[Path, Path | None] = {observed_path: None, truth_path: None}
    installed: set[Path] = set()

    try:
        observed_temp = _temporary_path(observed_path)
        truth_temp = _temporary_path(truth_path)
        observed.to_csv(observed_temp, index=False)
        truth.to_csv(truth_temp, index=False)

        serialized_observed = pd.read_csv(
            observed_temp, dtype={"customer_id": "string"}
        )
        serialized_truth = pd.read_csv(
            truth_temp, dtype={"customer_id": "string"}
        )
        if serialized_observed.columns.tolist() != observed.columns.tolist():
            raise ValueError("serialized observed dataset has an invalid schema")
        if serialized_truth.columns.tolist() != truth.columns.tolist():
            raise ValueError("serialized truth dataset has an invalid schema")
        if len(serialized_observed) != len(observed) or len(serialized_truth) != len(
            truth
        ):
            raise ValueError("serialized datasets have invalid row counts")
        if not serialized_observed["customer_id"].equals(
            serialized_truth["customer_id"]
        ):
            raise ValueError("serialized dataset customer_id values do not match")

        for destination in backups:
            if destination.exists():
                backup = _temporary_path(destination)
                try:
                    shutil.copy2(destination, backup)
                except Exception:
                    backup.unlink(missing_ok=True)
                    raise
                backups[destination] = backup

        _replace_file(observed_temp, observed_path)
        installed.add(observed_path)
        _replace_file(truth_temp, truth_path)
        installed.add(truth_path)
    except Exception:
        for destination, backup in backups.items():
            if backup is not None and backup.exists():
                backup.replace(destination)
            elif destination in installed and destination.exists():
                destination.unlink()
        raise
    finally:
        for temporary in [observed_temp, truth_temp, *backups.values()]:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def main() -> None:
    observed, truth = generate_cafe_data(n_customers=10_000, seed=42)

    observed_path = PROJECT_ROOT / "data" / "observed_data.csv"
    truth_path = PROJECT_ROOT / "data" / "facilitator" / "simulator_truth.csv"
    _write_dataset_pair(observed, truth, observed_path, truth_path)

    print(f"Observed dataset: {len(observed):,} rows -> {observed_path}")
    print(f"Facilitator truth: {len(truth):,} rows -> {truth_path}")

    # Last month's randomized pilot: new customers, coupons by coin flip.
    pilot, pilot_truth = generate_cafe_data(
        n_customers=PILOT_CUSTOMERS, seed=PILOT_SEED, randomized_coupon_rate=0.5
    )
    for frame in (pilot, pilot_truth):
        frame["customer_id"] = frame["customer_id"].str.replace("CAFE-", "PILOT-", regex=False)
    pilot_path = PROJECT_ROOT / "data" / "pilot_rct.csv"
    pilot_truth_path = PROJECT_ROOT / "data" / "facilitator" / "pilot_truth.csv"
    _write_dataset_pair(pilot, pilot_truth, pilot_path, pilot_truth_path)

    print(f"Randomized pilot: {len(pilot):,} rows -> {pilot_path}")
    print(f"Pilot truth: {len(pilot_truth):,} rows -> {pilot_truth_path}")


if __name__ == "__main__":
    main()
