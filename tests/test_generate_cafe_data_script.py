from pathlib import Path

import pandas as pd
import pytest

from scripts import generate_cafe_data as export_script
from src.data_generator import generate_cafe_data


def test_dataset_pair_export_writes_matching_valid_csvs(tmp_path: Path) -> None:
    observed, truth = generate_cafe_data(n_customers=100, seed=42)
    observed_path = tmp_path / "data" / "observed_data.csv"
    truth_path = tmp_path / "data" / "facilitator" / "simulator_truth.csv"

    export_script._write_dataset_pair(observed, truth, observed_path, truth_path)

    actual_observed = pd.read_csv(observed_path, dtype={"customer_id": "string"})
    actual_truth = pd.read_csv(truth_path, dtype={"customer_id": "string"})
    pd.testing.assert_frame_equal(actual_observed, observed)
    pd.testing.assert_frame_equal(actual_truth, truth)


def test_dataset_pair_export_rolls_back_when_second_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed, truth = generate_cafe_data(n_customers=100, seed=42)
    observed_path = tmp_path / "data" / "observed_data.csv"
    truth_path = tmp_path / "data" / "facilitator" / "simulator_truth.csv"
    truth_path.parent.mkdir(parents=True)
    observed_path.write_bytes(b"old observed\n")
    truth_path.write_bytes(b"old truth\n")
    original_observed = observed_path.read_bytes()
    original_truth = truth_path.read_bytes()

    real_replace = export_script._replace_file
    replace_count = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("injected second replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(export_script, "_replace_file", fail_second_replace)

    with pytest.raises(OSError, match="injected second replacement failure"):
        export_script._write_dataset_pair(
            observed, truth, observed_path, truth_path
        )

    assert observed_path.read_bytes() == original_observed
    assert truth_path.read_bytes() == original_truth


def test_dataset_pair_export_cleans_first_temp_when_second_acquisition_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed, truth = generate_cafe_data(n_customers=100, seed=42)
    observed_path = tmp_path / "data" / "observed_data.csv"
    truth_path = tmp_path / "data" / "facilitator" / "simulator_truth.csv"
    real_temporary_path = export_script._temporary_path
    acquisition_count = 0

    def fail_second_acquisition(destination: Path) -> Path:
        nonlocal acquisition_count
        acquisition_count += 1
        if acquisition_count == 2:
            raise OSError("injected second temporary-file failure")
        return real_temporary_path(destination)

    monkeypatch.setattr(
        export_script, "_temporary_path", fail_second_acquisition
    )

    with pytest.raises(OSError, match="injected second temporary-file failure"):
        export_script._write_dataset_pair(
            observed, truth, observed_path, truth_path
        )

    assert not list(tmp_path.rglob("*.tmp"))
    assert not observed_path.exists()
    assert not truth_path.exists()


def test_dataset_pair_export_cleans_backup_temp_when_backup_copy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed, truth = generate_cafe_data(n_customers=100, seed=42)
    observed_path = tmp_path / "data" / "observed_data.csv"
    truth_path = tmp_path / "data" / "facilitator" / "simulator_truth.csv"
    truth_path.parent.mkdir(parents=True)
    observed_path.write_bytes(b"old observed\n")
    truth_path.write_bytes(b"old truth\n")
    original_observed = observed_path.read_bytes()
    original_truth = truth_path.read_bytes()

    def fail_backup_copy(source: Path, destination: Path) -> None:
        raise OSError("injected backup-copy failure")

    monkeypatch.setattr(export_script.shutil, "copy2", fail_backup_copy)

    with pytest.raises(OSError, match="injected backup-copy failure"):
        export_script._write_dataset_pair(
            observed, truth, observed_path, truth_path
        )

    assert not list(tmp_path.rglob("*.tmp"))
    assert observed_path.read_bytes() == original_observed
    assert truth_path.read_bytes() == original_truth
