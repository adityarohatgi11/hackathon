"""Energy price forecasting using Prophet and advanced time series methods."""

import pandas as pd
import numpy as np
from typing import Dict
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
try:
    from prophet import Prophet
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False
    logging.warning("Prophet/sklearn not available, using simple forecasting")

from .feature_engineering import FeatureEngineer

# Try to import advanced forecaster
try:
    from .advanced_forecaster import QuantitativeForecaster
    HAS_QUANTITATIVE = True
except ImportError:
    HAS_QUANTITATIVE = False

logger = logging.getLogger(__name__)


class Forecaster:
    """Energy price and volatility forecaster using Prophet and ensemble methods."""
    
    def __init__(self, use_prophet: bool = True, use_ensemble: bool = True):
        """Initialize the forecaster.
        
        Args:
            use_prophet: Whether to use Prophet for forecasting
            use_ensemble: Whether to use ensemble of multiple models
        """
        self.use_prophet = use_prophet and HAS_ML_LIBS
        self.use_ensemble = use_ensemble and HAS_ML_LIBS
        self.is_trained = False
        
        # Models
        self.prophet_model = None
        self.rf_model = None
        self.lr_model = None
        
        # Feature engineering
        self.feature_engineer = FeatureEngineer()
        self.selected_features = []
        
        # Model performance tracking
        self.model_metrics = {}
        
        logger.info(f"Forecaster initialized: Prophet={self.use_prophet}, Ensemble={self.use_ensemble}")
    
    def fit(self, prices: pd.DataFrame) -> None:
        """Fit the forecasting models on historical data.
        
        Args:
            prices: Historical price data with timestamp and price columns
        """
        logger.info("Training forecasting models")
        
        if len(prices) < 24:
            logger.warning("Insufficient data for training, need at least 24 hours")
            return
        
        # Engineer features
        df_features = self.feature_engineer.engineer_features(prices)
        
        # Prepare data for different models
        self._fit_prophet(prices)
        self._fit_ml_models(df_features)
        
        self.is_trained = True
        logger.info("Forecasting models trained successfully")
    
    def _fit_prophet(self, prices: pd.DataFrame) -> None:
        """Fit Prophet model."""
        if not self.use_prophet:
            return
            
        try:
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': prices['timestamp'],
                'y': prices['price']
            })
            
            # Configure Prophet with energy market characteristics
            self.prophet_model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,  # Not enough data typically
                seasonality_mode='additive',
                changepoint_prior_scale=0.05,  # More flexible for energy prices
                seasonality_prior_scale=10,
                holidays_prior_scale=10,
                interval_width=0.8
            )
            
            # Add custom seasonalities for energy markets
            self.prophet_model.add_seasonality(
                name='hourly', period=1, fourier_order=8
            )
            
            # Fit the model
            self.prophet_model.fit(prophet_df)
            logger.info("Prophet model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {e}")
            self.prophet_model = None
    
    def _fit_ml_models(self, df_features: pd.DataFrame) -> None:
        """Fit ML models (Random Forest and Linear Regression)."""
        if not self.use_ensemble:
            return
            
        try:
            # Select features
            self.selected_features = self.feature_engineer.select_features(
                df_features, target_col='price', max_features=30
            )
            
            # Prepare training data
            X, y = self.feature_engineer.prepare_forecast_data(
                df_features, target_col='price', feature_cols=self.selected_features
            )
            
            # Skip if insufficient data
            if len(X) < 48:  # At least 2 days
                logger.warning("Insufficient data for ML models")
                return
            
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Fit Random Forest
            self.rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.rf_model.fit(X_train, y_train)
            
            # Fit Linear Regression
            self.lr_model = LinearRegression()
            self.lr_model.fit(X_train, y_train)
            
            # Evaluate models
            rf_pred = self.rf_model.predict(X_test)
            lr_pred = self.lr_model.predict(X_test)
            
            self.model_metrics = {
                'rf_mae': mean_absolute_error(y_test, rf_pred),
                'rf_rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
                'lr_mae': mean_absolute_error(y_test, lr_pred),
                'lr_rmse': np.sqrt(mean_squared_error(y_test, lr_pred))
            }
            
            logger.info(f"ML models fitted: RF MAE={self.model_metrics['rf_mae']:.2f}, LR MAE={self.model_metrics['lr_mae']:.2f}")
            
        except Exception as e:
            logger.error(f"Error fitting ML models: {e}")
            self.rf_model = None
            self.lr_model = None
    
    def predict_next(self, prices: pd.DataFrame, periods: int = 24) -> pd.DataFrame:
        """Predict next period energy prices and uncertainty.
        
        Args:
            prices: Historical price data with timestamp and price columns
            periods: Number of periods to forecast (default 24 hours)
            
        Returns:
            DataFrame with predictions, upper/lower bounds, and uncertainty metrics
        """
        logger.info(f"Generating {periods}-hour forecast")
        
        # CRITICAL: Handle edge cases
        if periods <= 0:
            return pd.DataFrame(columns=[
                'timestamp', 'predicted_price', 'lower_bound', 'upper_bound',
                'σ_energy', 'σ_hash', 'σ_token', 'method'
            ])
        
        # Fit models if not already trained
        if not self.is_trained:
            self.fit(prices)
        
        # Generate forecasts from different models
        forecasts = {}
        
        if self.prophet_model is not None:
            forecasts['prophet'] = self._predict_prophet(prices, periods)
        
        if self.rf_model is not None and self.lr_model is not None:
            forecasts['ensemble'] = self._predict_ensemble(prices, periods)
        
        if not forecasts:
            # Fallback to simple forecasting
            return self._predict_simple(prices, periods)
        
        # Combine forecasts
        return self._combine_forecasts(forecasts, prices, periods)
    
    def _predict_prophet(self, prices: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Generate Prophet forecast."""
        try:
            # Create future dataframe
            last_time = prices['timestamp'].iloc[-1]
            future_times = pd.date_range(
                start=last_time + pd.Timedelta(hours=1),
                periods=periods,
                freq='H'
            )
            
            future_df = pd.DataFrame({'ds': future_times})
            
            # Generate forecast
            forecast = self.prophet_model.predict(future_df)
            
            return pd.DataFrame({
                'timestamp': future_times,
                'predicted_price': forecast['yhat'].values,
                'lower_bound': forecast['yhat_lower'].values,
                'upper_bound': forecast['yhat_upper'].values,
                'method': 'prophet'
            })
            
        except Exception as e:
            logger.error(f"Error in Prophet prediction: {e}")
            return pd.DataFrame()
    
    def _predict_ensemble(self, prices: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Generate ensemble forecast using ML models."""
        try:
            # Engineer features for the full dataset
            df_features = self.feature_engineer.engineer_features(prices)
            
            # Generate future timestamps
            last_time = prices['timestamp'].iloc[-1]
            future_times = pd.date_range(
                start=last_time + pd.Timedelta(hours=1),
                periods=periods,
                freq='H'
            )
            
            predictions = []
            current_df = df_features.copy()
            
            # Generate iterative predictions
            for i, future_time in enumerate(future_times):
                # Get most recent features
                if len(current_df) > 0:
                    last_row = current_df.iloc[-1:][self.selected_features].fillna(0)
                    
                    # Make predictions
                    rf_pred = self.rf_model.predict(last_row)[0]
                    lr_pred = self.lr_model.predict(last_row)[0]
                    
                    # Ensemble prediction (weighted average)
                    rf_weight = 0.7 if 'rf_mae' in self.model_metrics else 0.5
                    ensemble_pred = rf_weight * rf_pred + (1 - rf_weight) * lr_pred
                    
                    predictions.append(ensemble_pred)
                    
                    # Update dataframe for next prediction (simplified)
                    new_row = current_df.iloc[-1:].copy()
                    new_row['timestamp'] = future_time
                    new_row['price'] = ensemble_pred
                    
                    # Simple feature updates for next iteration
                    new_row = self.feature_engineer.engineer_features(
                        pd.concat([current_df.tail(168), new_row])  # Keep last week + new row
                    ).tail(1)
                    
                    current_df = pd.concat([current_df, new_row])
                else:
                    # Fallback
                    predictions.append(prices['price'].iloc[-1])
            
            # Estimate uncertainty based on model performance
            base_uncertainty = self.model_metrics.get('rf_mae', 5.0)
            uncertainty = np.array(predictions) * 0.1 + base_uncertainty
            
            return pd.DataFrame({
                'timestamp': future_times,
                'predicted_price': predictions,
                'lower_bound': np.array(predictions) - uncertainty,
                'upper_bound': np.array(predictions) + uncertainty,
                'method': 'ensemble'
            })
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return pd.DataFrame()
    
    def _predict_simple(self, prices: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Simple fallback forecasting method."""
        logger.info("Using simple forecasting method")
        
        # Handle empty price data gracefully
        if len(prices) == 0 or 'timestamp' not in prices.columns or 'price' not in prices.columns:
            last_time = pd.Timestamp.now()
            last_price = 50.0
        else:
            last_time = prices['timestamp'].iloc[-1]
            last_price = prices['price'].iloc[-1]
        
        # Generate future timestamps early for reuse
        future_times = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=periods, freq='H')

        # Estimate trend from recent data (if available)
        if len(prices) >= 24 and 'price' in prices.columns:
            recent_trend = (prices['price'].tail(24).mean() - prices['price'].head(24).mean()) / 24
        else:
            recent_trend = 0
        
        # Seasonal pattern (daily cycle)
        hours = np.array([t.hour for t in future_times])
        seasonal = 10 * np.sin(2 * np.pi * hours / 24)
        
        # Generate predictions
        trend_component = np.arange(periods) * recent_trend
        noise = np.random.normal(0, 2, periods)
        
        predictions = last_price + trend_component + seasonal + noise
        # CRITICAL: Ensure all predictions are positive and reasonable
        predictions = np.maximum(predictions, 0.01)
        predictions = np.minimum(predictions, 500.0)  # Cap extreme values
        
        uncertainty = np.abs(predictions * 0.15)  # 15% uncertainty
        
        # CRITICAL: Cap uncertainty to reasonable bounds (max $50/MWh)
        uncertainty = np.minimum(uncertainty, 50.0)
        
        return pd.DataFrame({
            'timestamp': future_times,
            'predicted_price': predictions,
            'lower_bound': predictions - uncertainty,
            'upper_bound': predictions + uncertainty,
            'σ_energy': uncertainty,
            'σ_hash': uncertainty * 0.5,
            'σ_token': uncertainty * 0.3,
            'method': 'simple'
        })
    
    def _combine_forecasts(self, forecasts: Dict[str, pd.DataFrame], 
                          prices: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Combine multiple forecasts into final prediction."""
        # CRITICAL: Handle empty forecasts
        if not forecasts:
            return self._predict_simple(prices, periods)
            
        if len(forecasts) == 1:
            forecast_df = list(forecasts.values())[0]
        else:
            # Weighted ensemble of forecasts
            prophet_weight = 0.6 if 'prophet' in forecasts else 0.0
            ensemble_weight = 0.4 if 'ensemble' in forecasts else 0.0
            
            # Normalize weights
            total_weight = prophet_weight + ensemble_weight
            if total_weight > 0:
                prophet_weight /= total_weight
                ensemble_weight /= total_weight
            
            # CRITICAL: Check if forecasts have data
            valid_forecasts = {k: v for k, v in forecasts.items() if len(v) > 0 and 'predicted_price' in v.columns}
            if not valid_forecasts:
                return self._predict_simple(prices, periods)
            
            # Combine predictions
            combined_pred = np.zeros(periods)
            combined_lower = np.zeros(periods)
            combined_upper = np.zeros(periods)
            
            if 'prophet' in valid_forecasts and 'predicted_price' in valid_forecasts['prophet'].columns:
                # Ensure proper dtype conversion for numpy operations
                prophet_pred = np.asarray(valid_forecasts['prophet']['predicted_price'].values, dtype=np.float64)
                prophet_lower = np.asarray(valid_forecasts['prophet']['lower_bound'].values, dtype=np.float64)
                prophet_upper = np.asarray(valid_forecasts['prophet']['upper_bound'].values, dtype=np.float64)
                
                combined_pred += prophet_weight * prophet_pred
                combined_lower += prophet_weight * prophet_lower
                combined_upper += prophet_weight * prophet_upper

            if 'ensemble' in valid_forecasts and 'predicted_price' in valid_forecasts['ensemble'].columns:
                # Ensure proper dtype conversion for numpy operations
                ensemble_pred = np.asarray(valid_forecasts['ensemble']['predicted_price'].values, dtype=np.float64)
                ensemble_lower = np.asarray(valid_forecasts['ensemble']['lower_bound'].values, dtype=np.float64)
                ensemble_upper = np.asarray(valid_forecasts['ensemble']['upper_bound'].values, dtype=np.float64)
                
                combined_pred += ensemble_weight * ensemble_pred
                combined_lower += ensemble_weight * ensemble_lower
                combined_upper += ensemble_weight * ensemble_upper
            
            forecast_df = pd.DataFrame({
                'timestamp': valid_forecasts[list(valid_forecasts.keys())[0]]['timestamp'],
                'predicted_price': combined_pred,
                'lower_bound': combined_lower,
                'upper_bound': combined_upper,
                'method': 'combined'
            })
        
        # CRITICAL: Ensure all predictions are positive and reasonable
        forecast_df['predicted_price'] = np.maximum(forecast_df['predicted_price'], 0.01)
        forecast_df['lower_bound'] = np.maximum(forecast_df['lower_bound'], 0.01)
        forecast_df['upper_bound'] = np.maximum(forecast_df['upper_bound'], forecast_df['lower_bound'])
        
        # CRITICAL: Cap extreme predictions (energy prices rarely exceed $500/MWh)
        max_reasonable_price = 500.0
        forecast_df['predicted_price'] = np.minimum(forecast_df['predicted_price'], max_reasonable_price)
        forecast_df['upper_bound'] = np.minimum(forecast_df['upper_bound'], max_reasonable_price)
        forecast_df['lower_bound'] = np.minimum(forecast_df['lower_bound'], forecast_df['upper_bound'])
        
        # Add uncertainty metrics for compatibility
        uncertainty = (forecast_df['upper_bound'] - forecast_df['lower_bound']) / 2
        
        # CRITICAL: Cap uncertainty to reasonable bounds (max 20% of price or $50/MWh)
        max_uncertainty = np.minimum(forecast_df['predicted_price'] * 0.2, 50.0)
        uncertainty = np.minimum(uncertainty, max_uncertainty)
        
        forecast_df['σ_energy'] = uncertainty
        forecast_df['σ_hash'] = uncertainty * 0.5
        forecast_df['σ_token'] = uncertainty * 0.3
        
        logger.info(f"Generated forecast using {forecast_df['method'].iloc[0]} method")
        return forecast_df
    
    def predict_volatility(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Predict price volatility metrics.
        
        Args:
            prices: Historical price data
            
        Returns:
            DataFrame with volatility predictions
        """
        if len(prices) == 0:
            return pd.DataFrame()
        
        # Calculate rolling volatility (allowing smaller sample sizes)
        prices_copy = prices.copy()
        prices_copy['returns'] = prices_copy['price'].pct_change(fill_method=None)
        
        # Multiple volatility measures with minimum periods to avoid NaNs
        vol_6h = prices_copy['returns'].rolling(window=6, min_periods=3).std()
        vol_24h = prices_copy['returns'].rolling(window=24, min_periods=6).std()
        vol_7d = prices_copy['returns'].rolling(window=168, min_periods=24).std()
        
        # GARCH-like volatility persistence
        alpha = 0.1  # Short-term weight
        beta = 0.85   # Persistence
        
        # Latest vol (fallback to non-NaN)
        latest_vol = vol_24h.dropna().iloc[-1] if vol_24h.dropna().any() else 0.05
        long_vol = vol_7d.dropna().iloc[-1] if vol_7d.dropna().any() else latest_vol
        
        forecast_vol = alpha * latest_vol + beta * long_vol
        
        return pd.DataFrame({
            'timestamp': prices['timestamp'],
            'volatility_6h': vol_6h,
            'volatility_24h': vol_24h,
            'volatility_7d': vol_7d,
            'vol_forecast': forecast_vol
        })
    
    def feature_importance(self) -> Dict[str, float]:
        """Return feature importance for model interpretability.
        
        Returns:
            Dictionary with feature names and importance scores
        """
        importance_dict = {}
        
        if self.rf_model is not None and self.selected_features:
            # Get Random Forest feature importance
            rf_importance = dict(zip(
                self.selected_features,
                self.rf_model.feature_importances_
            ))
            importance_dict.update(rf_importance)
        
        if not importance_dict:
            # Default importance for basic features
            importance_dict = {
                'hour_of_day': 0.25,
                'day_of_week': 0.15,
                'price_lag_24h': 0.20,
                'price_ma_24h': 0.18,
                'price_volatility_24h': 0.12,
                'is_peak_hours': 0.10
            }
        
        return importance_dict
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get model performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.model_metrics.copy()
    
    def forecast(self, prices: pd.DataFrame, horizon_hours: int = 24) -> Dict:
        """Generate forecast in dictionary format for test compatibility.
        
        Args:
            prices: Historical price data
            horizon_hours: Number of hours to forecast
            
        Returns:
            Dictionary with forecast data
        """
        forecast_df = self.predict_next(prices, periods=horizon_hours)
        
        # Ensure non-negative prices
        energy_prices = np.maximum(forecast_df['predicted_price'].values, 0.01)
        
        return {
            'timestamps': forecast_df['timestamp'].tolist(),
            'energy_price': energy_prices.tolist(),
            'sigma_energy': forecast_df['σ_energy'].tolist(),
            'sigma_hash': forecast_df['σ_hash'].tolist(),
            'sigma_token': forecast_df['σ_token'].tolist()
        }


def create_advanced_forecaster(**kwargs) -> 'Forecaster':
    """Factory function to create the most advanced forecaster available.
    
    Returns:
        QuantitativeForecaster if advanced libs available, else basic Forecaster
    """
    if HAS_QUANTITATIVE:
        logger.info("Creating QuantitativeForecaster with advanced models")
        return QuantitativeForecaster(**kwargs)
    else:
        logger.info("Creating basic Forecaster (advanced models not available)")
        return Forecaster(**kwargs)


# Backwards compatibility - use advanced forecaster by default if available
def get_forecaster(**kwargs) -> 'Forecaster':
    """Get the best available forecaster."""
    return create_advanced_forecaster(**kwargs)


def create_forecaster(**kwargs) -> 'Forecaster':
    """Alias for create_advanced_forecaster for test compatibility."""
    return create_advanced_forecaster(**kwargs) 