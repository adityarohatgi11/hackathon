"""Risk modelling utilities used by bidding strategies.

This module purposefully keeps the implementation light-weight while
illustrating core quantitative concepts such as *Value-at-Risk* (VaR)
and *Conditional Value-at-Risk* (CVaR).  The functions are designed to be
side-effect-free (pure) and therefore trivial to unit-test.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

__all__ = [
    "historical_var",
    "historical_cvar",
    "risk_adjustment_factor",
]


def _validate_returns(returns: pd.Series) -> pd.Series:
    if returns.empty:
        raise ValueError("'returns' series must not be empty")
    return returns.dropna()


def historical_var(returns: pd.Series, alpha: float = 0.95) -> float:
    """Compute the historical VaR at confidence level *alpha*.

    Parameters
    ----------
    returns : pd.Series
        Percentage returns (e.g. ``price.pct_change()``) expressed as
        **decimal fractions** (i.e. 0.01 == 1 %).
    alpha : float, default 0.95
        Confidence level. 0.95 corresponds to the 5 % worst outcomes.
    """
    clean = _validate_returns(returns)
    quantile = np.quantile(clean, 1 - alpha)
    # VaR is conventionally reported as a *positive* number.
    return abs(quantile)


def historical_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    """Compute the historical CVaR (Expected Shortfall).

    CVaR is defined as the *expected* loss conditional on the loss being
    larger than the VaR.
    """
    clean = _validate_returns(returns)
    var_threshold = np.quantile(clean, 1 - alpha)
    tail_losses = clean[clean <= var_threshold]
    return abs(tail_losses.mean())


def risk_adjustment_factor(
    returns: pd.Series,
    target_risk: float = 0.05,
    alpha: float = 0.95,
) -> float:
    """Return a scaling factor ∈ (0, 1] to keep CVaR below *target_risk*.

    The function is intentionally simple – it scales down proportionally
    to the ratio of the desired risk over the current risk measure.
    """
    current_risk = historical_cvar(returns, alpha=alpha)
    if current_risk <= 0:
        return 1.0
    scaling = min(1.0, target_risk / current_risk)
    return scaling 