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
    returns = forecast["predicted_price"].pct_change(fill_method=None).fillna(0)
    risk_scale = risk_adjustment_factor(returns, target_risk=0.05)
    adjusted_prices = base_prices * risk_scale

    # CRITICAL: Ensure all bids are positive (cannot bid negative prices)
    adjusted_prices = np.maximum(adjusted_prices, 0.01)

    # ------------------------------------------------------------------
    # 3. Assemble DataFrame – follow the shared interface contract.
    # ------------------------------------------------------------------
    # CRITICAL: Keep power allocations in kW for VCG auction compatibility
    # The VCG auction expects kW values, not per-unit
    df = pd.DataFrame(
        {
            "timestamp": forecast["timestamp"],
            "energy_bid": adjusted_prices,
            "regulation_bid": adjusted_prices * 1.2,
            "spinning_reserve_bid": adjusted_prices * 0.8,
            # For GPU allocations we use a simple proportional split in kW
            "inference": energy_kw * 0.4,  # kW for inference workloads
            "training": energy_kw * 0.3,   # kW for training workloads
            "cooling": energy_kw * 0.3,    # kW for cooling systems
            # Add bid prices for each service (required by VCG auction)
            "inference_bid": adjusted_prices * 1.1,  # Slightly higher bid for inference
            "training_bid": adjusted_prices * 1.0,   # Standard bid for training
            "cooling_bid": adjusted_prices * 0.9,    # Lower bid for cooling
        }
    )

    return df


def portfolio_optimization(bids: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """Risk-aware optimisation of bid volumes.

    If *cvxpy* is available the routine solves a convex programme that
    maximises expected value while keeping CVaR below the specified
    ``constraints['cvar_limit']``.  When cvxpy is not installed it falls
    back to the deterministic power-limit scaling used previously – so
    interfaces remain unchanged.
    """

    try:
        import cvxpy as cp  # type: ignore

        # ------------------------------------------------------------------
        #  Setup optimisation problem (one period, representative):
        #  maximise   Σ_i  p_i * x_i
        #  subject to Σ_i  x_i      ≤ max_power
        #             CVaRα(returns) ≤ cvar_limit
        #             0 ≤ x_i ≤ 1.0
        # ------------------------------------------------------------------
        alloc_cols = ["inference", "training", "cooling"]
        price_cols = [
            "energy_bid",  # proxy price for each service
            "regulation_bid",
            "spinning_reserve_bid",
        ]

        # Use last row (current period) for optimisation
        current_row = bids.iloc[-1]
        prices = np.array(current_row[price_cols][:3])
        x = cp.Variable(len(alloc_cols))

        # Objective: maximise expected revenue (linear)
        objective = cp.Maximize(prices @ x)

        max_power = float(constraints.get("max_power", 1.0))
        cvar_limit = float(constraints.get("cvar_limit", 0.05))

        # Simple CVaR approximation via risk_adjustment_factor
        returns_series = bids["energy_bid"].pct_change().fillna(0)
        risk_scale = risk_adjustment_factor(returns_series, target_risk=cvar_limit)

        constraints_cvx = [
            cp.sum(x) <= max_power * risk_scale,  # combined capacity + risk scaling
            x >= 0,
            x <= 1.0,
        ]

        prob = cp.Problem(objective, constraints_cvx)
        prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)

        # Build optimised DataFrame (copy bids and replace allocations for last period)
        optimized = bids.copy()
        alloc_values = x.value if x.value is not None else np.zeros(len(alloc_cols))
        optimized.loc[optimized.index[-1], alloc_cols] = alloc_values

        return optimized

    except ImportError:
        # ---------- Fallback: deterministic power cap scaling ----------
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