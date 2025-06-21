"""
Advanced Stochastic Models for GridPilot-GT
Implements cutting-edge mathematical models for energy price forecasting and optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    # Advanced numerical libraries
    from scipy import stats, optimize, integrate
    from scipy.special import gamma, beta
    from scipy.linalg import cholesky, solve_triangular
    import scipy.sparse as sp
    
    # Optional advanced libraries
    try:
        import cvxpy as cp
        HAS_CVXPY = True
    except ImportError:
        HAS_CVXPY = False
        
    try:
        from numba import jit, njit
        HAS_NUMBA = True
    except ImportError:
        HAS_NUMBA = False
        
    HAS_ADVANCED_LIBS = True
except ImportError:
    HAS_ADVANCED_LIBS = False
    HAS_CVXPY = False
    HAS_NUMBA = False

logger = logging.getLogger(__name__)

def jit_decorator(func):
    """Conditional JIT decorator."""
    if HAS_NUMBA:
        return jit(nopython=True)(func)
    return func

def njit_decorator(func):
    """Conditional NJIT decorator."""
    if HAS_NUMBA:
        return njit(func)
    return func


class StochasticDifferentialEquation:
    """
    Stochastic Differential Equation (SDE) models for energy price dynamics.
    
    Implements multiple SDE formulations:
    1. Geometric Brownian Motion (GBM)
    2. Mean-Reverting Ornstein-Uhlenbeck (OU) process
    3. Jump Diffusion (Merton model)
    4. Heston Stochastic Volatility
    """
    
    def __init__(self, model_type: str = "mean_reverting", **params):
        """
        Initialize SDE model.
        
        Args:
            model_type: "gbm", "mean_reverting", "jump_diffusion", "heston"
            **params: Model-specific parameters
        """
        self.model_type = model_type
        self.params = params
        self.fitted_params = {}
        self.dt = 1.0 / 24  # Hourly timestep
        
        # Default parameters
        self.default_params = {
            "mu": 0.0,      # Drift
            "sigma": 0.2,   # Volatility
            "theta": 0.1,   # Mean reversion speed
            "kappa": 2.0,   # Volatility mean reversion
            "xi": 0.3,      # Vol of vol
            "rho": -0.5,    # Correlation
            "lambda": 0.1,  # Jump intensity
            "mu_j": -0.1,   # Jump mean
            "sigma_j": 0.2  # Jump volatility
        }
        
        # Update with provided parameters
        for key, value in self.default_params.items():
            if key not in self.params:
                self.params[key] = value
                
        logger.info(f"Initialized SDE model: {model_type}")
    
    def fit(self, prices: pd.Series, method: str = "mle") -> Dict[str, float]:
        """
        Fit SDE model to historical price data.
        
        Args:
            prices: Historical price series
            method: "mle" (Maximum Likelihood) or "moments"
            
        Returns:
            Dictionary of fitted parameters
        """
        if len(prices) < 50:
            logger.warning("Insufficient data for SDE fitting")
            return self.params
        
        try:
            if self.model_type == "mean_reverting":
                return self._fit_ou_process(prices, method)
            elif self.model_type == "gbm":
                return self._fit_gbm(prices, method)
            elif self.model_type == "jump_diffusion":
                return self._fit_jump_diffusion(prices, method)
            elif self.model_type == "heston":
                return self._fit_heston(prices, method)
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return self.params
                
        except Exception as e:
            logger.error(f"SDE fitting failed: {e}")
            return self.params
    
    def _fit_ou_process(self, prices: pd.Series, method: str) -> Dict[str, float]:
        """Fit Ornstein-Uhlenbeck mean-reverting process."""
        log_prices = np.log(prices)
        returns = log_prices.diff().dropna()
        
        if method == "mle":
            # Maximum likelihood estimation
            def neg_log_likelihood(params):
                theta, mu, sigma = params
                if theta <= 0 or sigma <= 0:
                    return 1e10
                
                n = len(returns)
                dt = self.dt
                
                # OU process likelihood
                expected_returns = theta * (mu - log_prices[:-1]) * dt
                variance = sigma**2 * dt
                
                ll = -0.5 * n * np.log(2 * np.pi * variance)
                ll -= 0.5 * np.sum((returns - expected_returns)**2) / variance
                
                return -ll
            
            # Initial guess
            x0 = [0.1, log_prices.mean(), returns.std()]
            
            # Optimize
            result = optimize.minimize(
                neg_log_likelihood, x0, 
                method='L-BFGS-B',
                bounds=[(0.001, 10), (-10, 10), (0.001, 5)]
            )
            
            if result.success:
                theta, mu, sigma = result.x
                self.fitted_params = {
                    "theta": theta,
                    "mu": np.exp(mu),  # Convert back to price level
                    "sigma": sigma
                }
            else:
                # Fallback to method of moments
                return self._fit_ou_moments(log_prices, returns)
        else:
            return self._fit_ou_moments(log_prices, returns)
        
        logger.info(f"OU process fitted: {self.fitted_params}")
        return self.fitted_params
    
    def _fit_ou_moments(self, log_prices: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """Method of moments estimation for OU process."""
        # Simple moment matching
        mean_return = returns.mean()
        var_return = returns.var()
        
        # Estimate parameters
        theta = -np.log(1 + mean_return) / self.dt if mean_return > -1 else 0.1
        mu = log_prices.mean()
        sigma = np.sqrt(var_return / self.dt)
        
        self.fitted_params = {
            "theta": max(theta, 0.001),
            "mu": np.exp(mu),
            "sigma": max(sigma, 0.001)
        }
        
        return self.fitted_params
    
    def _fit_gbm(self, prices: pd.Series, method: str) -> Dict[str, float]:
        """Fit Geometric Brownian Motion."""
        log_prices = np.log(prices)
        returns = log_prices.diff().dropna()
        
        # GBM parameters
        mu = returns.mean() / self.dt
        sigma = returns.std() / np.sqrt(self.dt)
        
        self.fitted_params = {
            "mu": mu,
            "sigma": max(sigma, 0.001)
        }
        
        return self.fitted_params
    
    def _fit_jump_diffusion(self, prices: pd.Series, method: str) -> Dict[str, float]:
        """Fit Jump Diffusion (Merton) model."""
        log_prices = np.log(prices)
        returns = log_prices.diff().dropna()
        
        # Identify jumps using threshold method
        threshold = 3 * returns.std()
        jumps = returns[np.abs(returns) > threshold]
        
        # Estimate jump parameters
        lambda_jump = len(jumps) / len(returns) / self.dt
        mu_j = jumps.mean() if len(jumps) > 0 else 0.0
        sigma_j = jumps.std() if len(jumps) > 1 else 0.1
        
        # Estimate diffusion parameters (excluding jumps)
        normal_returns = returns[np.abs(returns) <= threshold]
        mu_diffusion = normal_returns.mean() / self.dt if len(normal_returns) > 0 else 0.0
        sigma_diffusion = normal_returns.std() / np.sqrt(self.dt) if len(normal_returns) > 1 else 0.1
        
        self.fitted_params = {
            "mu": mu_diffusion,
            "sigma": max(sigma_diffusion, 0.001),
            "lambda": max(lambda_jump, 0.001),
            "mu_j": mu_j,
            "sigma_j": max(sigma_j, 0.001)
        }
        
        return self.fitted_params
    
    def _fit_heston(self, prices: pd.Series, method: str) -> Dict[str, float]:
        """Fit Heston stochastic volatility model (simplified)."""
        # Simplified Heston fitting using realized volatility proxy
        log_prices = np.log(prices)
        returns = log_prices.diff().dropna()
        
        # Estimate volatility series using rolling window
        vol_window = min(24, len(returns) // 4)
        realized_vol = returns.rolling(window=vol_window).std()
        vol_returns = realized_vol.diff().dropna()
        
        # Estimate Heston parameters
        mu = returns.mean() / self.dt
        v0 = realized_vol.iloc[-1]**2 if len(realized_vol) > 0 else 0.04
        kappa = 2.0  # Default mean reversion speed
        theta = realized_vol.mean()**2 if len(realized_vol) > 0 else 0.04
        xi = vol_returns.std() if len(vol_returns) > 1 else 0.3
        rho = np.corrcoef(returns[:-1], vol_returns)[0, 1] if len(vol_returns) > 10 else -0.5
        
        self.fitted_params = {
            "mu": mu,
            "v0": max(v0, 0.001),
            "kappa": max(kappa, 0.1),
            "theta": max(theta, 0.001),
            "xi": max(xi, 0.001),
            "rho": np.clip(rho, -0.99, 0.99)
        }
        
        return self.fitted_params
    
    def simulate(self, n_steps: int, n_paths: int = 1000, initial_price: float = None) -> np.ndarray:
        """
        Simulate price paths using fitted SDE model.
        
        Args:
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            initial_price: Starting price (uses last fitted price if None)
            
        Returns:
            Array of shape (n_paths, n_steps) with simulated prices
        """
        if not self.fitted_params:
            logger.warning("Model not fitted, using default parameters")
            params = self.params
        else:
            params = self.fitted_params
        
        if initial_price is None:
            initial_price = 50.0  # Default
        
        if self.model_type == "mean_reverting":
            return self._simulate_ou(n_steps, n_paths, initial_price, params)
        elif self.model_type == "gbm":
            return self._simulate_gbm(n_steps, n_paths, initial_price, params)
        elif self.model_type == "jump_diffusion":
            return self._simulate_jump_diffusion(n_steps, n_paths, initial_price, params)
        elif self.model_type == "heston":
            return self._simulate_heston(n_steps, n_paths, initial_price, params)
        else:
            # Fallback to GBM
            return self._simulate_gbm(n_steps, n_paths, initial_price, params)
    
    @staticmethod
    @jit_decorator
    def _simulate_ou_core(n_steps: int, n_paths: int, initial_price: float, 
                         theta: float, mu: float, sigma: float, dt: float) -> np.ndarray:
        """Core OU simulation (JIT compiled if available)."""
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = np.log(initial_price)
        
        for t in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            paths[:, t] = (paths[:, t-1] + 
                          theta * (np.log(mu) - paths[:, t-1]) * dt + 
                          sigma * dW)
        
        return np.exp(paths)
    
    def _simulate_ou(self, n_steps: int, n_paths: int, initial_price: float, params: Dict) -> np.ndarray:
        """Simulate Ornstein-Uhlenbeck process."""
        return self._simulate_ou_core(
            n_steps, n_paths, initial_price,
            params["theta"], params["mu"], params["sigma"], self.dt
        )
    
    @staticmethod
    @jit_decorator  
    def _simulate_gbm_core(n_steps: int, n_paths: int, initial_price: float,
                          mu: float, sigma: float, dt: float) -> np.ndarray:
        """Core GBM simulation (JIT compiled if available)."""
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = np.log(initial_price)
        
        for t in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            paths[:, t] = (paths[:, t-1] + 
                          (mu - 0.5 * sigma**2) * dt + 
                          sigma * dW)
        
        return np.exp(paths)
    
    def _simulate_gbm(self, n_steps: int, n_paths: int, initial_price: float, params: Dict) -> np.ndarray:
        """Simulate Geometric Brownian Motion."""
        return self._simulate_gbm_core(
            n_steps, n_paths, initial_price,
            params["mu"], params["sigma"], self.dt
        )
    
    def _simulate_jump_diffusion(self, n_steps: int, n_paths: int, initial_price: float, params: Dict) -> np.ndarray:
        """Simulate Jump Diffusion (Merton) model."""
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = np.log(initial_price)
        
        for t in range(1, n_steps):
            # Diffusion component
            dW = np.random.normal(0, np.sqrt(self.dt), n_paths)
            diffusion = (params["mu"] - 0.5 * params["sigma"]**2) * self.dt + params["sigma"] * dW
            
            # Jump component
            jump_times = np.random.poisson(params["lambda"] * self.dt, n_paths)
            jumps = np.zeros(n_paths)
            
            for i in range(n_paths):
                if jump_times[i] > 0:
                    jump_sizes = np.random.normal(params["mu_j"], params["sigma_j"], jump_times[i])
                    jumps[i] = np.sum(jump_sizes)
            
            paths[:, t] = paths[:, t-1] + diffusion + jumps
        
        return np.exp(paths)
    
    def _simulate_heston(self, n_steps: int, n_paths: int, initial_price: float, params: Dict) -> np.ndarray:
        """Simulate Heston stochastic volatility model."""
        S = np.zeros((n_paths, n_steps))
        V = np.zeros((n_paths, n_steps))
        
        S[:, 0] = initial_price
        V[:, 0] = params["v0"]
        
        for t in range(1, n_steps):
            # Correlated random numbers
            Z1 = np.random.normal(0, 1, n_paths)
            Z2 = np.random.normal(0, 1, n_paths)
            W1 = Z1
            W2 = params["rho"] * Z1 + np.sqrt(1 - params["rho"]**2) * Z2
            
            # Variance process (with Feller condition)
            V[:, t] = np.maximum(
                V[:, t-1] + params["kappa"] * (params["theta"] - V[:, t-1]) * self.dt + 
                params["xi"] * np.sqrt(np.maximum(V[:, t-1], 0)) * np.sqrt(self.dt) * W2,
                0.001  # Floor for numerical stability
            )
            
            # Price process
            S[:, t] = S[:, t-1] * np.exp(
                (params["mu"] - 0.5 * V[:, t-1]) * self.dt + 
                np.sqrt(np.maximum(V[:, t-1], 0)) * np.sqrt(self.dt) * W1
            )
        
        return S


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for risk assessment and scenario analysis.
    """
    
    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        """
        Initialize Monte Carlo engine.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        logger.info(f"Monte Carlo engine initialized: {n_simulations} simulations")
    
    def value_at_risk(self, returns: np.ndarray, confidence_level: float = 0.05) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR).
        
        Args:
            returns: Array of portfolio returns
            confidence_level: VaR confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            Dictionary with VaR and CVaR metrics
        """
        if len(returns) == 0:
            return {"var": 0.0, "cvar": 0.0, "expected_return": 0.0, "volatility": 0.0}
        
        # Sort returns
        sorted_returns = np.sort(returns)
        
        # VaR calculation
        var_index = int(confidence_level * len(sorted_returns))
        var = sorted_returns[var_index] if var_index < len(sorted_returns) else sorted_returns[-1]
        
        # CVaR calculation (expected shortfall)
        cvar = np.mean(sorted_returns[:var_index]) if var_index > 0 else var
        
        # Additional statistics
        expected_return = np.mean(returns)
        volatility = np.std(returns)
        
        return {
            "var": float(var),
            "cvar": float(cvar),
            "expected_return": float(expected_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(expected_return / volatility) if volatility > 0 else 0.0
        }
    
    def scenario_analysis(self, price_model: StochasticDifferentialEquation, 
                         allocation_strategy: Callable, 
                         horizon: int = 24,
                         initial_conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive scenario analysis using Monte Carlo simulation.
        
        Args:
            price_model: Fitted SDE model for price simulation
            allocation_strategy: Function that takes prices and returns allocation
            horizon: Forecast horizon
            initial_conditions: Initial system state
            
        Returns:
            Dictionary with scenario analysis results
        """
        if initial_conditions is None:
            initial_conditions = {"price": 50.0, "soc": 0.5, "power_available": 1000.0}
        
        # Generate price scenarios
        price_paths = price_model.simulate(
            n_steps=horizon, 
            n_paths=self.n_simulations,
            initial_price=initial_conditions["price"]
        )
        
        # Simulate allocation decisions and outcomes
        revenues = []
        allocations = []
        risks = []
        
        for path_idx in range(self.n_simulations):
            try:
                prices = price_paths[path_idx, :]
                
                # Create price forecast DataFrame
                forecast_df = pd.DataFrame({
                    'timestamp': pd.date_range(start=datetime.now(), periods=horizon, freq='H'),
                    'predicted_price': prices,
                    'σ_energy': prices * 0.1,  # 10% uncertainty
                    'σ_hash': prices * 0.05,
                    'σ_token': prices * 0.03
                })
                
                # Get allocation decision
                allocation = allocation_strategy(forecast_df, initial_conditions)
                
                # Calculate revenue for this scenario
                total_allocation = sum(allocation.values()) if isinstance(allocation, dict) else allocation
                revenue = np.sum(prices * total_allocation * 0.001)  # Convert to revenue
                
                revenues.append(revenue)
                allocations.append(total_allocation)
                
                # Calculate risk metrics
                price_volatility = np.std(prices) / np.mean(prices)
                risks.append(price_volatility)
                
            except Exception as e:
                logger.warning(f"Simulation path {path_idx} failed: {e}")
                revenues.append(0.0)
                allocations.append(0.0)
                risks.append(0.0)
        
        # Aggregate results
        revenues = np.array(revenues)
        allocations = np.array(allocations)
        risks = np.array(risks)
        
        # Calculate statistics
        revenue_stats = self.value_at_risk(revenues)
        
        return {
            "revenue_statistics": revenue_stats,
            "allocation_statistics": {
                "mean": float(np.mean(allocations)),
                "std": float(np.std(allocations)),
                "min": float(np.min(allocations)),
                "max": float(np.max(allocations))
            },
            "risk_statistics": {
                "mean_volatility": float(np.mean(risks)),
                "max_volatility": float(np.max(risks)),
                "volatility_95th": float(np.percentile(risks, 95))
            },
            "scenario_outcomes": {
                "best_case_revenue": float(np.max(revenues)),
                "worst_case_revenue": float(np.min(revenues)),
                "median_revenue": float(np.median(revenues)),
                "probability_positive": float(np.mean(revenues > 0))
            },
            "n_simulations": self.n_simulations
        }


class ReinforcementLearningAgent:
    """
    Q-Learning agent for adaptive bidding strategy optimization.
    """
    
    def __init__(self, state_size: int = 64, action_size: int = 5, 
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 epsilon: float = 0.1):
        """
        Initialize RL agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for Q-table updates
            discount_factor: Future reward discount factor
            epsilon: Exploration rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size))
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
        logger.info(f"RL Agent initialized: {state_size} states, {action_size} actions")
    
    def discretize_state(self, continuous_state: Dict[str, float]) -> int:
        """
        Convert continuous state to discrete state index.
        
        Args:
            continuous_state: Dictionary with state variables
            
        Returns:
            Discrete state index
        """
        # Simple discretization scheme
        price = continuous_state.get("price", 50.0)
        soc = continuous_state.get("soc", 0.5)
        volatility = continuous_state.get("volatility", 0.1)
        demand = continuous_state.get("demand", 0.5)
        
        # Normalize and discretize
        price_bin = min(int(price / 20), 3)  # 0-3 based on price ranges
        soc_bin = min(int(soc * 4), 3)      # 0-3 based on SOC quartiles
        vol_bin = min(int(volatility * 10), 3)  # 0-3 based on volatility
        demand_bin = min(int(demand * 4), 3)  # 0-3 based on demand
        
        # Combine into single state index
        state_index = price_bin * 64 + soc_bin * 16 + vol_bin * 4 + demand_bin
        return min(state_index, self.state_size - 1)
    
    def choose_action(self, state_index: int, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state_index: Current state index
            training: Whether in training mode (exploration) or testing
            
        Returns:
            Action index
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.action_size)
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state_index, :])
    
    def update_q_table(self, state: int, action: int, reward: float, next_state: int):
        """
        Update Q-table using Q-learning rule.
        
        Args:
            state: Current state index
            action: Action taken
            reward: Reward received
            next_state: Next state index
        """
        # Q-learning update rule
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state, :])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state, action] = new_q
        
        # Store history
        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)
    
    def get_bidding_strategy(self, state: Dict[str, float]) -> Dict[str, float]:
        """
        Convert RL action to bidding strategy parameters.
        
        Args:
            state: Current market state
            
        Returns:
            Dictionary with bidding strategy parameters
        """
        state_index = self.discretize_state(state)
        action = self.choose_action(state_index, training=False)
        
        # Map action to strategy parameters
        strategies = [
            {"aggressiveness": 0.2, "risk_tolerance": 0.1},  # Conservative
            {"aggressiveness": 0.4, "risk_tolerance": 0.2},  # Moderate-Conservative
            {"aggressiveness": 0.6, "risk_tolerance": 0.3},  # Balanced
            {"aggressiveness": 0.8, "risk_tolerance": 0.4},  # Moderate-Aggressive
            {"aggressiveness": 1.0, "risk_tolerance": 0.5},  # Aggressive
        ]
        
        if action < len(strategies):
            return strategies[action]
        else:
            return strategies[2]  # Default to balanced
    
    def train_episode(self, market_data: pd.DataFrame, 
                     reward_function: Callable) -> float:
        """
        Train the agent on one episode of market data.
        
        Args:
            market_data: Historical market data
            reward_function: Function to calculate rewards
            
        Returns:
            Total episode reward
        """
        total_reward = 0.0
        
        for i in range(len(market_data) - 1):
            # Current state
            current_state = {
                "price": market_data.iloc[i]["price"],
                "soc": 0.5,  # Simplified
                "volatility": market_data.iloc[i].get("price_volatility_24h", 0.1),
                "demand": 0.5  # Simplified
            }
            
            state_index = self.discretize_state(current_state)
            action = self.choose_action(state_index, training=True)
            
            # Next state
            next_state = {
                "price": market_data.iloc[i + 1]["price"],
                "soc": 0.5,
                "volatility": market_data.iloc[i + 1].get("price_volatility_24h", 0.1),
                "demand": 0.5
            }
            
            next_state_index = self.discretize_state(next_state)
            
            # Calculate reward
            reward = reward_function(current_state, action, next_state)
            
            # Update Q-table
            self.update_q_table(state_index, action, reward, next_state_index)
            
            total_reward += reward
        
        return total_reward


class StochasticOptimalControl:
    """
    Stochastic optimal control for energy trading decisions.
    Implements Hamilton-Jacobi-Bellman (HJB) equation solutions.
    """
    
    def __init__(self, horizon: int = 24, n_states: int = 100):
        """
        Initialize stochastic control problem.
        
        Args:
            horizon: Time horizon
            n_states: Number of discretized states
        """
        self.horizon = horizon
        self.n_states = n_states
        self.value_function = np.zeros((n_states, horizon + 1))
        self.optimal_policy = np.zeros((n_states, horizon))
        
        logger.info(f"Stochastic control initialized: {horizon}h horizon, {n_states} states")
    
    def solve_hjb(self, price_dynamics: StochasticDifferentialEquation,
                  cost_function: Callable, 
                  terminal_condition: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve Hamilton-Jacobi-Bellman equation using dynamic programming.
        
        Args:
            price_dynamics: SDE model for price evolution
            cost_function: Running cost function
            terminal_condition: Terminal cost function
            
        Returns:
            Tuple of (value_function, optimal_policy)
        """
        # State space discretization
        state_grid = np.linspace(0.1, 1.0, self.n_states)  # SOC grid
        action_grid = np.linspace(0, 1000, 21)  # Power allocation grid
        
        # Terminal condition
        for i, soc in enumerate(state_grid):
            self.value_function[i, self.horizon] = terminal_condition(soc)
        
        # Backward induction
        for t in range(self.horizon - 1, -1, -1):
            for i, soc in enumerate(state_grid):
                min_cost = np.inf
                best_action = 0
                
                for action in action_grid:
                    # Expected cost-to-go
                    expected_cost = self._expected_cost_to_go(
                        soc, action, t, price_dynamics, cost_function
                    )
                    
                    if expected_cost < min_cost:
                        min_cost = expected_cost
                        best_action = action
                
                self.value_function[i, t] = min_cost
                self.optimal_policy[i, t] = best_action
        
        return self.value_function, self.optimal_policy
    
    def _expected_cost_to_go(self, soc: float, action: float, time: int,
                           price_dynamics: StochasticDifferentialEquation,
                           cost_function: Callable) -> float:
        """Calculate expected cost-to-go using Monte Carlo integration."""
        n_samples = 100
        costs = []
        
        # Sample future prices
        price_samples = price_dynamics.simulate(n_steps=1, n_paths=n_samples, initial_price=50.0)
        
        for price in price_samples[:, 0]:
            # State transition
            next_soc = max(0, min(1, soc - action / 1000))  # Simplified dynamics
            
            # Running cost
            running_cost = cost_function(soc, action, price)
            
            # Continuation value (interpolated)
            next_state_index = int(next_soc * (self.n_states - 1))
            next_state_index = max(0, min(next_state_index, self.n_states - 1))
            
            continuation_value = self.value_function[next_state_index, time + 1]
            
            total_cost = running_cost + continuation_value
            costs.append(total_cost)
        
        return np.mean(costs)
    
    def get_optimal_action(self, soc: float, time: int) -> float:
        """
        Get optimal action for given state and time.
        
        Args:
            soc: State of charge
            time: Current time step
            
        Returns:
            Optimal power allocation
        """
        if time >= self.horizon:
            return 0.0
        
        # Interpolate policy
        state_index = int(soc * (self.n_states - 1))
        state_index = max(0, min(state_index, self.n_states - 1))
        
        return self.optimal_policy[state_index, time]


# Utility functions for advanced stochastic methods
def create_stochastic_forecaster(model_type: str = "mean_reverting", **kwargs) -> StochasticDifferentialEquation:
    """Factory function to create stochastic forecasting models."""
    return StochasticDifferentialEquation(model_type=model_type, **kwargs)

def create_monte_carlo_engine(n_simulations: int = 10000) -> MonteCarloEngine:
    """Factory function to create Monte Carlo engine."""
    return MonteCarloEngine(n_simulations=n_simulations)

def create_rl_agent(**kwargs) -> ReinforcementLearningAgent:
    """Factory function to create RL agent."""
    return ReinforcementLearningAgent(**kwargs)

def create_stochastic_control(horizon: int = 24) -> StochasticOptimalControl:
    """Factory function to create stochastic control solver."""
    return StochasticOptimalControl(horizon=horizon) 