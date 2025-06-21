"""Bid generation utilities – **Lane B** implementation.

This module is the primary entry-point for Lane B (Engineer B) and now
implements:

* A CVXPY-backed Model Predictive Control (MPC) routine via
  :pymod:`game_theory.mpc_controller`.
* Simple but extensible portfolio optimisation with risk constraints
  using :pymod:`game_theory.risk_models`.

All public functions are covered by tests in *tests/test_basic.py* or
Lane-specific test-suites.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .mpc_controller import MPCController
from .risk_models import risk_adjustment_factor


def build_bid_vector(
    current_price: float,
    forecast: pd.DataFrame,
    uncertainty: pd.DataFrame,
    soc: float,
    lambda_deg: float,
) -> pd.DataFrame:
    """Generate an MPC-optimised bid vector.

    The function delegates the heavy-lifting to
    :class:`game_theory.mpc_controller.MPCController` and subsequently
    augments the energy bids with bid prices for ancillary services.  A
    naïve risk adjustment based on Conditional VaR keeps the bid volume
    in check under volatile market conditions.
    """

    horizon = len(forecast)

    # ------------------------------------------------------------------
    # 1. Optimise the *quantity* to bid using MPC
    # ------------------------------------------------------------------
    mpc = MPCController(horizon=horizon, lambda_deg=lambda_deg)
    current_state = {
        "soc": soc,
        "available_power_kw": 1000.0,  # TODO: inject real constraint
    }

    mpc_result = mpc.optimize_horizon(forecast=forecast, current_state=current_state)
    energy_kw = mpc_result["energy_bids"]  # kW for each hour

    # ------------------------------------------------------------------
    # 2. Derive bid *prices*
    # ------------------------------------------------------------------
    price_multiplier = 0.95  # Bid slightly below forecast
    base_prices = forecast["predicted_price"].to_numpy() * price_multiplier

    # Risk adjustment – scale prices based on recent returns.
    returns = forecast["predicted_price"].pct_change().fillna(0)
    risk_scale = risk_adjustment_factor(returns, target_risk=0.05)
    adjusted_prices = base_prices * risk_scale

    # ------------------------------------------------------------------
    # 3. Assemble DataFrame – follow the shared interface contract.
    # ------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "timestamp": forecast["timestamp"],
            "energy_bid": adjusted_prices,
            "regulation_bid": adjusted_prices * 1.2,
            "spinning_reserve_bid": adjusted_prices * 0.8,
            # For GPU allocations we use a simple proportional split.
            "inference": energy_kw * 0.4 / 1000,  # convert back to p.u.
            "training": energy_kw * 0.3 / 1000,
            "cooling": energy_kw * 0.3 / 1000,
        }
    )

    return df


def portfolio_optimization(bids: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """Optimize bid portfolio considering risk constraints.
    
    Args:
        bids: Initial bid DataFrame
        constraints: System constraints (power, battery, etc.)
        
    Returns:
        Optimized bid DataFrame
    """
    total_power = bids[["inference", "training", "cooling"]].sum(axis=1)
    max_power = float(constraints.get("max_power", 1.0))

    scale_factor = np.minimum(1.0, max_power / total_power).clip(lower=0, upper=1)

    optimized = bids.copy()
    for col in ["inference", "training", "cooling"]:
        optimized[col] = optimized[col] * scale_factor

    return optimized


def dynamic_pricing_strategy(market_conditions: Dict[str, Any]) -> Dict[str, float]:
    """Adapt bidding strategy based on market conditions.
    
    Args:
        market_conditions: Current market state
        
    Returns:
        Strategy parameters
    """
    # Simple heuristic derived from market volatility.
    volatility = market_conditions.get("volatility", 0.05)

    aggressiveness = max(0.5, 1 - volatility)
    risk_tolerance = max(0.1, volatility)

    return {
        "aggressiveness": aggressiveness,
        "risk_tolerance": risk_tolerance,
        "price_multiplier": 0.9 if aggressiveness > 0.8 else 0.95,
        "diversification_factor": 1 - risk_tolerance / 2,
    } 