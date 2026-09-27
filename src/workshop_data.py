"""Facilitator data loaders and shared constants for Causal Cafe Lab."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parents[1]

FEATURES = [
    "days_since_last_purchase",
    "orders_30d",
    "app_sessions_30d",
    "average_order_value",
    "loyalty_score",
    "price_sensitivity",
    "offer_affinity",
]
ARTIFACT_COLUMNS = [
    "customer_id",
    "purchase_probability",
    "estimated_uplift",
    "expected_incremental_profit",
    "recommended",
]
GROWTH_BUDGET = 2500.0
TEST_SIZE = 0.3
SPLIT_SEED = 42


def load_observed_data(path: str | Path | None = None) -> pd.DataFrame:
    """Participant-visible observational data."""
    csv = Path(path) if path is not None else _ROOT / "data" / "observed_data.csv"
    return pd.read_csv(csv, dtype={"customer_id": "string"})


def load_facilitator_truth(path: str | Path | None = None) -> pd.DataFrame:
    """Facilitator-only simulator truth. Do not show before the reveal."""
    csv = (
        Path(path)
        if path is not None
        else _ROOT / "data" / "facilitator" / "simulator_truth.csv"
    )
    return pd.read_csv(csv, dtype={"customer_id": "string"})


def load_pilot_data(path: str | Path | None = None) -> pd.DataFrame:
    """Last month's randomized pilot: coupons were sent by a coin flip."""
    csv = Path(path) if path is not None else _ROOT / "data" / "pilot_rct.csv"
    return pd.read_csv(csv, dtype={"customer_id": "string"})


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """70/30 split, stratified by coupon_sent so both arms appear in each part."""
    return train_test_split(
        df, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=df["coupon_sent"]
    )
