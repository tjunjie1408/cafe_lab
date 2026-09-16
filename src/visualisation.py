"""Facilitator plotting helpers."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure


def plot_uplift_distribution(uplift: np.ndarray) -> Figure:
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    ax.hist(uplift, bins=40)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("estimated uplift (mu1 - mu0)")
    ax.set_ylabel("customers")
    ax.set_title("Estimated coupon uplift distribution")
    return fig


def plot_policy_comparison(labels: list[str], values: list[float]) -> Figure:
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    ax.bar(labels, values)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("true expected value captured")
    ax.set_title("Targeting policy comparison")
    return fig
