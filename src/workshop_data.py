"""Facilitator data loaders and shared constants for Causal Cafe Lab."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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
