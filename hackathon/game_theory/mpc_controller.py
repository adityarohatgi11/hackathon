from __future__ import annotations

"""Model Predictive Control (MPC) controller for optimal energy bidding.

This module houses the :class:`MPCController` which formulates a convex
optimization problem over a rolling horizon to derive an energy bid
profile that maximises the expected profit subject to battery and power
constraints.

The formulation is deliberately simple to keep the example light-weight
while still demonstrating best practices such as:

1. Explicit type annotations and Google-style docstrings.
2. Configuration via *config.toml* with sensible defaults.
3. Separation of model construction and solution retrieval to ease unit
   testing and future extensions.
4. Graceful degradation – falling back to a heuristic solution whenever
   the solver fails or CVXPY is unavailable.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

# CVXPY is an optional heavy dependency – import lazily so that the
#   module can still be imported (e.g. for documentation) without the
#   package installed.
try:
    import cvxpy as cp  # type: ignore

    _HAS_CVXPY = True
except ImportError:  # pragma: no cover – handled during runtime.
    _HAS_CVXPY = False

import toml

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _load_config(config_path: str = "config.toml") -> Dict[str, Any]:
    """Load project-wide configuration.

    The configuration file is optional; when missing, a set of defaults
    that work for simulation purposes is returned.
    """
    try:
        return toml.load(config_path)
    except (FileNotFoundError, toml.TomlDecodeError):
        # Defaults generous enough for unit testing / CI environments.
        return {
            "BATTERY_CAP_MWH": 1.0,
            "BATTERY_MAX_KW": 250.0,
            "SOC_MIN": 0.15,
            "SOC_MAX": 0.90,
            "BATTERY_EFF": 0.94,
            "LAMBDA_DEG": 0.0002,
            "site_power_kw": 1000,
        }


# ---------------------------------------------------------------------------
# MPC Controller
# ---------------------------------------------------------------------------


@dataclass
class MPCController:
    """Rolling-horizon MPC controller.

    Parameters
    ----------
    horizon : int, default 24
        Number of *discrete* timesteps (hours) to optimise.
    lambda_deg : float, optional
        Battery degradation cost coefficient (\$/MWh).
    constraints : Optional[Dict[str, Any]], optional
        Runtime-adjustable constraints (for instance coming from a UI or
        real-time telemetry). If *None*, the values are read from
        *config.toml*.
    """

    horizon: int = 24
    lambda_deg: Optional[float] = None
    constraints: Optional[Dict[str, Any]] = None
    _cfg: Dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cfg = _load_config()

        # Allow overriding of degradation cost.
        if self.lambda_deg is None:
            self.lambda_deg = self._cfg.get("LAMBDA_DEG", 2e-4)

        # Merge runtime constraints with config.
        user_constraints = self.constraints or {}
        self.constraints = {**self._cfg, **user_constraints}

    # ---------------------------------------------------------------------
    # API methods
    # ---------------------------------------------------------------------

    def optimize_horizon(
        self,
        forecast: pd.DataFrame,
        current_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Optimise energy bids for the upcoming *horizon*.

        Parameters
        ----------
        forecast
            DataFrame returned by :pyclass:`forecasting.Forecaster`, must
            contain the column *predicted_price*.
        current_state
            Dictionary containing at least the *soc* (state of charge)
            and *available_power_kw* fields.

        Returns
        -------
        Dict[str, Any]
            Dictionary with two keys:

            ``'status'`` : str
                ``"optimal"`` when a solution is found, ``"heuristic"``
                otherwise.

            ``'energy_bids'`` : np.ndarray
                Array of length *horizon* with the power bids in **kW**.
        """
        # Sanity checks ----------------------------------------------------
        if "predicted_price" not in forecast.columns:
            raise KeyError("forecast must contain a 'predicted_price' column")

        if len(forecast) < self.horizon:
            raise ValueError(
                f"forecast must contain at least {self.horizon} rows, got {len(forecast)}"
            )

        prices = forecast["predicted_price"].iloc[: self.horizon].to_numpy()

        soc_init = float(current_state.get("soc", 0.5))
        p_available = float(current_state.get("available_power_kw", self._cfg["site_power_kw"]))

        # Attempt a convex optimisation with CVXPY -------------------------
        if _HAS_CVXPY:
            try:
                energy_bid, status = self._solve_cvxpy(prices, soc_init, p_available)
                return {"status": status, "energy_bids": energy_bid}
            except Exception:  # pragma: no cover – fallback executed below.
                pass  # Fall back to heuristic.

        # Fallback heuristic (robust and fast) -----------------------------
        heuristic_bids = np.minimum(prices / np.max(prices) * p_available, p_available * 0.9)
        return {"status": "heuristic", "energy_bids": heuristic_bids}

    def update_constraints(self, new_constraints: Dict[str, Any]) -> None:
        """Update the internal constraint dictionary in-place."""
        self.constraints.update(new_constraints)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _solve_cvxpy(
        self,
        prices: np.ndarray,
        soc_init: float,
        p_available: float,
    ) -> tuple[np.ndarray, str]:
        """Formulate and solve the convex optimisation problem.

        The problem is a simplified energy arbitrage formulation:

        maximise    Σ_t  (price_t * e_t)  -  λ_deg * Σ_t e_t
        subject to  0 ≤ e_t ≤ P_max
                    SOC_{t+1} = SOC_t  –  e_t / CAP_MWh   (charging positive)
                    SOC_min ≤ SOC_t ≤ SOC_max
        """
        horizon = self.horizon

        # Decision variables ---------------------------------------------
        e = cp.Variable(horizon)  # Energy bid (kW) for each timestep
        soc = cp.Variable(horizon + 1)

        # Parameters ------------------------------------------------------
        cap_mwh = float(self.constraints["BATTERY_CAP_MWH"]) * 1000  # to kWh
        soc_min = float(self.constraints["SOC_MIN"])
        soc_max = float(self.constraints["SOC_MAX"])

        lambda_deg = float(self.lambda_deg)

        # Objective -------------------------------------------------------
        revenue = prices @ e  # expected revenue ( \$/h * kW )
        degradation = lambda_deg * cp.sum(e)
        objective = cp.Maximize(revenue - degradation)

        # Constraints -----------------------------------------------------
        constraints = [soc[0] == soc_init]
        for t in range(horizon):
            # Battery dynamics (simple integration – ignoring efficiency losses)
            constraints += [soc[t + 1] == soc[t] - e[t] / cap_mwh]

        constraints += [
            cp.constraints.nonpos.NonPos(-e),  # e ≥ 0
            e <= p_available,
            soc_min <= soc,
            soc <= soc_max,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, warm_start=True, verbose=False)

        status = "optimal" if problem.status == cp.OPTIMAL else problem.status
        energy_plan = np.clip(e.value, 0, p_available) if e.value is not None else np.zeros(horizon)

        return energy_plan, status 