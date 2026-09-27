"""Scoring an uplift ranking on randomized data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def qini_coefficient(curve: pd.DataFrame) -> float:
    """Average height of a Qini curve above the straight random-targeting line.

    ``curve`` has columns ``share`` (fraction of customers targeted, ending at 1)
    and ``gain`` (cumulative incremental purchases). Higher is better; random
    targeting scores about 0.
    """
    share = curve["share"].to_numpy()
    gain = curve["gain"].to_numpy()
    return float(np.mean(gain - share * gain[-1]))
