"""Advanced quantitative models for energy price forecasting and risk management."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import advanced quantitative libraries
try:
    # Advanced time series
    from arch import arch_model  # GARCH models
    from filterpy.kalman import KalmanFilter
    from scipy import signal
    from scipy.stats import norm, t
    import pywt  # Wavelets
    
    # Advanced ML
    import xgboost as xgb
    from sklearn.ensemble import IsolationForest
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
    
    # Portfolio optimization
    from scipy.optimize import minimize
    import cvxpy as cp
    
    HAS_ADVANCED_LIBS = True
except ImportError:
    HAS_ADVANCED_LIBS = False
    logging.warning("Advanced quantitative libraries not available")

logger = logging.getLogger(__name__)


class GARCHVolatilityModel:
    """GARCH model for advanced volatility forecasting."""
    
    def __init__(self, p: int = 1, q: int = 1):
        """Initialize GARCH model.
        
        Args:
            p: Order of GARCH terms
            q: Order of ARCH terms
        """
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None
        
    def fit(self, returns: pd.Series) -> None:
        """Fit GARCH model to return series."""
        if not HAS_ADVANCED_LIBS:
            logger.warning("GARCH model requires 'arch' library")
            return
            
        try:
            # Remove outliers for better fit
            returns_clean = returns.dropna()
            returns_clean = returns_clean[np.abs(returns_clean) < returns_clean.std() * 3]
            
            # Fit GARCH(p,q) model
            self.model = arch_model(
                returns_clean * 100,  # Scale for numerical stability
                vol='Garch', 
                p=self.p, 
                q=self.q,
                dist='t'  # Student-t distribution for fat tails
            )
            
            self.fitted_model = self.model.fit(disp='off')
            logger.info(f"GARCH({self.p},{self.q}) model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting GARCH model: {e}")
            self.fitted_model = None
    
    def forecast_volatility(self, horizon: int = 24) -> np.ndarray:
        """Forecast volatility for given horizon."""
        if self.fitted_model is None:
            return np.array([0.1] * horizon)  # Fallback
            
        try:
            forecast = self.fitted_model.forecast(horizon=horizon)
            vol_forecast = np.sqrt(forecast.variance.values[-1, :]) / 100  # Unscale
            return vol_forecast
        except Exception as e:
            logger.error(f"Error forecasting GARCH volatility: {e}")
            return np.array([0.1] * horizon)


class KalmanStateEstimator:
    """Kalman filter for state-space modeling of energy prices."""
    
    def __init__(self, dim_state: int = 3):
        """Initialize Kalman filter.
        
        Args:
            dim_state: Dimension of state vector (level, trend, seasonality)
        """
        self.dim_state = dim_state
        self.kf = None
        self.state_means = []
        self.state_covariances = []
        
    def setup_filter(self, initial_price: float) -> None:
        """Setup Kalman filter matrices."""
        if not HAS_ADVANCED_LIBS:
            logger.warning("Kalman filter requires 'filterpy' library")
            return
            
        # State: [level, trend, seasonal]
        self.kf = KalmanFilter(dim_x=self.dim_state, dim_z=1)
        
        # State transition matrix (level + trend model)
        self.kf.F = np.array([
            [1., 1., 1.],  # level(t) = level(t-1) + trend(t-1) + seasonal(t-1)
            [0., 1., 0.],  # trend(t) = trend(t-1)
            [0., 0., 0.9]  # seasonal(t) = 0.9 * seasonal(t-1) (decay)
        ])
        
        # Observation matrix
        self.kf.H = np.array([[1., 0., 0.]])  # Observe level only
        
        # Process noise
        self.kf.Q *= 0.1
        
        # Measurement noise
        self.kf.R *= 1.0
        
        # Initial state
        self.kf.x = np.array([initial_price, 0., 0.])
        self.kf.P *= 10.0
        
        logger.info("Kalman filter initialized")
    
    def update(self, price: float) -> Dict[str, float]:
        """Update filter with new price observation."""
        if self.kf is None:
            return {'level': price, 'trend': 0., 'seasonal': 0.}
            
        try:
            self.kf.predict()
            self.kf.update(price)
            
            # Store state estimates
            self.state_means.append(self.kf.x.copy())
            self.state_covariances.append(self.kf.P.copy())
            
            return {
                'level': float(self.kf.x[0]),
                'trend': float(self.kf.x[1]),
                'seasonal': float(self.kf.x[2]),
                'uncertainty': float(np.sqrt(self.kf.P[0, 0]))
            }
        except Exception as e:
            logger.error(f"Error updating Kalman filter: {e}")
            return {'level': price, 'trend': 0., 'seasonal': 0.}
    
    def forecast(self, steps: int = 24) -> Dict[str, np.ndarray]:
        """Multi-step ahead forecast."""
        if self.kf is None or len(self.state_means) == 0:
            return {'forecast': np.array([50.] * steps), 'uncertainty': np.array([5.] * steps)}
        
        # Start from last state
        x = self.state_means[-1].copy()
        P = self.state_covariances[-1].copy()
        
        forecasts = []
        uncertainties = []
        
        for step in range(steps):
            # Predict next state
            x = self.kf.F @ x
            P = self.kf.F @ P @ self.kf.F.T + self.kf.Q
            
            # Forecast observation
            forecast = self.kf.H @ x
            forecast_var = self.kf.H @ P @ self.kf.H.T
            
            forecasts.append(float(forecast[0]))
            uncertainties.append(float(np.sqrt(forecast_var[0, 0])))
        
        return {
            'forecast': np.array(forecasts),
            'uncertainty': np.array(uncertainties)
        }


class WaveletAnalyzer:
    """Wavelet analysis for multi-scale decomposition."""
    
    def __init__(self, wavelet: str = 'db4', levels: int = 4):
        """Initialize wavelet analyzer.
        
        Args:
            wavelet: Wavelet type
            levels: Decomposition levels
        """
        self.wavelet = wavelet
        self.levels = levels
        
    def decompose(self, prices: pd.Series) -> Dict[str, np.ndarray]:
        """Wavelet decomposition of price series."""
        if not HAS_ADVANCED_LIBS:
            logger.warning("Wavelet analysis requires 'pywt' library")
            return {'approximation': prices.values, 'details': []}
        
        try:
            # Wavelet decomposition
            coeffs = pywt.wavedec(prices.values, self.wavelet, level=self.levels)
            
            # Separate approximation and details
            approximation = coeffs[0]
            details = coeffs[1:]
            
            # Reconstruct components
            reconstructed = {}
            
            # Low-frequency component (trend)
            trend_coeffs = [approximation] + [np.zeros_like(d) for d in details]
            reconstructed['trend'] = pywt.waverec(trend_coeffs, self.wavelet)
            
            # High-frequency components (noise, cycles)
            for i, detail in enumerate(details):
                detail_coeffs = [np.zeros_like(approximation)] + [np.zeros_like(d) for d in details]
                detail_coeffs[i + 1] = detail
                reconstructed[f'detail_{i+1}'] = pywt.waverec(detail_coeffs, self.wavelet)
            
            logger.info(f"Wavelet decomposition complete: {self.levels} levels")
            return reconstructed
            
        except Exception as e:
            logger.error(f"Error in wavelet decomposition: {e}")
            return {'approximation': prices.values, 'details': []}
    
    def denoise(self, prices: pd.Series, threshold_mode: str = 'soft') -> np.ndarray:
        """Wavelet denoising."""
        if not HAS_ADVANCED_LIBS:
            return prices.values
            
        try:
            # Decompose
            coeffs = pywt.wavedec(prices.values, self.wavelet, level=self.levels)
            
            # Estimate noise level (sigma)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            
            # Threshold details
            threshold = sigma * np.sqrt(2 * np.log(len(prices)))
            coeffs_thresh = coeffs.copy()
            
            for i in range(1, len(coeffs)):
                coeffs_thresh[i] = pywt.threshold(coeffs[i], threshold, mode=threshold_mode)
            
            # Reconstruct
            denoised = pywt.waverec(coeffs_thresh, self.wavelet)
            
            # Ensure same length as input
            if len(denoised) != len(prices):
                denoised = denoised[:len(prices)]
                
            return denoised
            
        except Exception as e:
            logger.error(f"Error in wavelet denoising: {e}")
            return prices.values


class XGBoostForecaster:
    """XGBoost model for advanced gradient boosting forecasting."""
    
    def __init__(self, **xgb_params):
        """Initialize XGBoost forecaster."""
        self.default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        self.params = {**self.default_params, **xgb_params}
        self.model = None
        self.feature_importance_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit XGBoost model."""
        if not HAS_ADVANCED_LIBS:
            logger.warning("XGBoost forecaster requires 'xgboost' library")
            return
            
        try:
            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(X, y)
            self.feature_importance_ = dict(zip(X.columns, self.model.feature_importances_))
            logger.info("XGBoost model fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting XGBoost model: {e}")
            self.model = None
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            return np.full(len(X), 50.0)  # Fallback
        
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error in XGBoost prediction: {e}")
            return np.full(len(X), 50.0)
    
    def predict_with_uncertainty(self, X: pd.DataFrame, n_estimators: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction with uncertainty using bootstrap."""
        if self.model is None:
            pred = np.full(len(X), 50.0)
            uncertainty = np.full(len(X), 5.0)
            return pred, uncertainty
        
        try:
            # Use multiple random seeds for uncertainty estimation
            predictions = []
            for seed in range(n_estimators):
                temp_model = xgb.XGBRegressor(**{**self.params, 'random_state': seed})
                # Use bootstrap sample indices (simplified)
                indices = np.random.choice(len(X), size=len(X), replace=True)
                temp_pred = self.model.predict(X.iloc[indices])
                predictions.append(np.mean(temp_pred))
            
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)
            
            return np.full(len(X), pred_mean), np.full(len(X), pred_std)
            
        except Exception as e:
            logger.error(f"Error in XGBoost uncertainty prediction: {e}")
            pred = self.predict(X)
            uncertainty = np.abs(pred * 0.1)
            return pred, uncertainty


class GaussianProcessForecaster:
    """Gaussian Process Regression for uncertainty quantification."""
    
    def __init__(self):
        """Initialize Gaussian Process."""
        self.model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Gaussian Process."""
        if not HAS_ADVANCED_LIBS:
            logger.warning("Gaussian Process requires 'sklearn' library")
            return
            
        try:
            # Define kernel
            kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
            
            self.model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10,
                random_state=42
            )
            
            # Fit on subset for computational efficiency
            n_samples = min(500, len(X))
            indices = np.random.choice(len(X), size=n_samples, replace=False)
            
            self.model.fit(X.iloc[indices], y.iloc[indices])
            logger.info("Gaussian Process fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting Gaussian Process: {e}")
            self.model = None
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty."""
        if self.model is None:
            pred = np.full(len(X), 50.0)
            std = np.full(len(X), 5.0)
            return pred, std
        
        try:
            pred, std = self.model.predict(X, return_std=True)
            return pred, std
        except Exception as e:
            logger.error(f"Error in Gaussian Process prediction: {e}")
            pred = np.full(len(X), 50.0)
            std = np.full(len(X), 5.0)
            return pred, std


class PortfolioOptimizer:
    """Advanced portfolio optimization for energy trading."""
    
    def __init__(self):
        """Initialize portfolio optimizer."""
        self.results = {}
        
    def mean_variance_optimization(self, expected_returns: np.ndarray, 
                                 cov_matrix: np.ndarray, 
                                 risk_aversion: float = 1.0) -> Dict[str, Any]:
        """Mean-variance optimization."""
        try:
            n_assets = len(expected_returns)
            
            if HAS_ADVANCED_LIBS:
                # Use CVXPY for convex optimization
                w = cp.Variable(n_assets)
                portfolio_return = expected_returns.T @ w
                portfolio_risk = cp.quad_form(w, cov_matrix)
                
                # Objective: maximize return - risk_aversion * risk
                objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
                
                # Constraints
                constraints = [
                    cp.sum(w) == 1,  # Weights sum to 1
                    w >= 0,          # Long-only
                    w <= 0.4         # Max 40% in any asset
                ]
                
                # Solve
                problem = cp.Problem(objective, constraints)
                problem.solve()
                
                if w.value is not None:
                    return {
                        'weights': w.value,
                        'expected_return': float(portfolio_return.value),
                        'risk': float(np.sqrt(portfolio_risk.value)),
                        'sharpe_ratio': float(portfolio_return.value / np.sqrt(portfolio_risk.value))
                    }
            
            # Fallback: equal weights
            weights = np.ones(n_assets) / n_assets
            portfolio_ret = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            return {
                'weights': weights,
                'expected_return': float(portfolio_ret),
                'risk': float(portfolio_vol),
                'sharpe_ratio': float(portfolio_ret / portfolio_vol) if portfolio_vol > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {e}")
            n_assets = len(expected_returns)
            return {
                'weights': np.ones(n_assets) / n_assets,
                'expected_return': 0.0,
                'risk': 0.1,
                'sharpe_ratio': 0.0
            }
    
    def risk_parity_optimization(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Risk parity portfolio optimization."""
        try:
            n_assets = cov_matrix.shape[0]
            
            def risk_budget_objective(weights, cov_matrix):
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                return np.sum(np.square(contrib - contrib.mean()))
            
            # Initial guess: equal weights
            x0 = np.ones(n_assets) / n_assets
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
            ]
            bounds = [(0.01, 0.4) for _ in range(n_assets)]  # Min 1%, max 40%
            
            result = minimize(
                risk_budget_objective,
                x0,
                args=(cov_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                return result.x
            else:
                return x0  # Fallback to equal weights
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            return np.ones(cov_matrix.shape[0]) / cov_matrix.shape[0]


class AnomalyDetector:
    """Anomaly detection for market data quality."""
    
    def __init__(self):
        """Initialize anomaly detector."""
        self.model = None
        
    def fit(self, X: pd.DataFrame) -> None:
        """Fit anomaly detection model."""
        if not HAS_ADVANCED_LIBS:
            logger.warning("Anomaly detection requires 'sklearn' library")
            return
            
        try:
            self.model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X)
            logger.info("Anomaly detection model fitted")
        except Exception as e:
            logger.error(f"Error fitting anomaly detector: {e}")
            self.model = None
    
    def detect(self, X: pd.DataFrame) -> np.ndarray:
        """Detect anomalies (returns -1 for anomalies, 1 for normal)."""
        if self.model is None:
            return np.ones(len(X))  # All normal
        
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return np.ones(len(X))
    
    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores."""
        if self.model is None:
            return np.zeros(len(X))
        
        try:
            return self.model.decision_function(X)
        except Exception as e:
            logger.error(f"Error getting anomaly scores: {e}")
            return np.zeros(len(X)) 