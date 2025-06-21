"""Feature engineering for energy price forecasting."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for energy price forecasting."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_names = []
        self.scaler_params = {}
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive features for forecasting.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features for forecasting")
        
        # Ensure we have a copy to avoid modifying original
        df_features = df.copy()
        
        # Temporal features
        df_features = self._add_temporal_features(df_features)
        
        # Price-based features
        df_features = self._add_price_features(df_features)
        
        # Volume-based features
        df_features = self._add_volume_features(df_features)
        
        # Technical indicators
        df_features = self._add_technical_indicators(df_features)
        
        # Market regime features
        df_features = self._add_market_regime_features(df_features)
        
        # Lag features
        df_features = self._add_lag_features(df_features)
        
        # Interaction features
        df_features = self._add_interaction_features(df_features)
        
        logger.info(f"Feature engineering complete: {len(df_features.columns)} features")
        return df_features
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = df.copy()
        
        # Basic temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Cyclic encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Peak/off-peak indicators
        df['is_peak_morning'] = ((df['hour'] >= 6) & (df['hour'] <= 10)).astype(int)
        df['is_peak_evening'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)
        df['is_peak_hours'] = (df['is_peak_morning'] | df['is_peak_evening']).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18) & (df['day_of_week'] < 5)).astype(int)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        df = df.copy()
        
        # Rolling statistics
        windows = [3, 6, 12, 24, 48, 168]  # 3h, 6h, 12h, 1d, 2d, 1w
        
        for window in windows:
            df[f'price_ma_{window}h'] = df['price'].rolling(window=window, min_periods=1).mean()
            df[f'price_std_{window}h'] = df['price'].rolling(window=window, min_periods=1).std()
            df[f'price_min_{window}h'] = df['price'].rolling(window=window, min_periods=1).min()
            df[f'price_max_{window}h'] = df['price'].rolling(window=window, min_periods=1).max()
            
            # Relative position in range
            price_range = df[f'price_max_{window}h'] - df[f'price_min_{window}h']
            df[f'price_position_{window}h'] = (df['price'] - df[f'price_min_{window}h']) / (price_range + 1e-8)
        
        # Price momentum
        df['price_change_1h'] = df['price'].diff(1)
        df['price_change_3h'] = df['price'].diff(3)
        df['price_change_24h'] = df['price'].diff(24)
        
        # Price returns - fix FutureWarning
        df['price_return_1h'] = df['price'].pct_change(1, fill_method=None)
        df['price_return_24h'] = df['price'].pct_change(24, fill_method=None)
        
        # Volatility measures
        df['price_volatility_6h'] = df['price_return_1h'].rolling(window=6).std()
        df['price_volatility_24h'] = df['price_return_1h'].rolling(window=24).std()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        if 'volume' not in df.columns:
            return df
            
        df = df.copy()
        
        # Volume statistics
        windows = [6, 24, 168]
        for window in windows:
            df[f'volume_ma_{window}h'] = df['volume'].rolling(window=window, min_periods=1).mean()
            df[f'volume_std_{window}h'] = df['volume'].rolling(window=window, min_periods=1).std()
        
        # Volume-price relationships
        df['price_volume_ratio'] = df['price'] / (df['volume'] + 1e-8)
        df['volume_weighted_price'] = (df['price'] * df['volume']).rolling(window=24).sum() / df['volume'].rolling(window=24).sum()
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators."""
        df = df.copy()
        
        # Bollinger Bands
        bb_window = 24
        bb_std = 2
        bb_ma = df['price'].rolling(window=bb_window).mean()
        bb_std_dev = df['price'].rolling(window=bb_window).std()
        df['bb_upper'] = bb_ma + (bb_std_dev * bb_std)
        df['bb_lower'] = bb_ma - (bb_std_dev * bb_std)
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # RSI (Relative Strength Index)
        df['price_delta'] = df['price'].diff()
        df['gain'] = df['price_delta'].where(df['price_delta'] > 0, 0)
        df['loss'] = -df['price_delta'].where(df['price_delta'] < 0, 0)
        
        rsi_window = 14
        avg_gain = df['gain'].rolling(window=rsi_window).mean()
        avg_loss = df['loss'].rolling(window=rsi_window).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['price'].ewm(span=12).mean()
        ema_26 = df['price'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime identification features."""
        df = df.copy()
        
        # Volatility regimes
        vol_24h = df['price'].rolling(window=24).std()
        vol_median = vol_24h.rolling(window=168).median()  # Weekly median
        df['high_volatility_regime'] = (vol_24h > vol_median * 1.5).astype(int)
        
        # Price level regimes
        price_24h_ma = df['price'].rolling(window=24).mean()
        price_weekly_ma = df['price'].rolling(window=168).mean()
        df['high_price_regime'] = (price_24h_ma > price_weekly_ma * 1.1).astype(int)
        
        # Trend identification
        short_ma = df['price'].rolling(window=6).mean()
        long_ma = df['price'].rolling(window=24).mean()
        df['uptrend'] = (short_ma > long_ma).astype(int)
        df['trend_strength'] = np.abs(short_ma - long_ma) / long_ma
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        df = df.copy()
        
        # Important lags based on energy market dynamics
        price_lags = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h to 1 week
        
        for lag in price_lags:
            df[f'price_lag_{lag}h'] = df['price'].shift(lag)
            if lag <= 24:  # Only for shorter lags
                df[f'price_return_lag_{lag}h'] = df['price_return_1h'].shift(lag)
        
        # Seasonal lags (same hour yesterday, same hour last week)
        df['price_lag_same_hour_yesterday'] = df['price'].shift(24)
        df['price_lag_same_hour_last_week'] = df['price'].shift(168)
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features."""
        df = df.copy()
        
        # Time-price interactions
        df['hour_price_interaction'] = df['hour'] * df['price']
        df['weekend_price_interaction'] = df['is_weekend'] * df['price']
        df['peak_hour_price_interaction'] = df['is_peak_hours'] * df['price']
        
        # Volatility-time interactions
        if 'price_volatility_24h' in df.columns:
            df['volatility_weekend'] = df['price_volatility_24h'] * df['is_weekend']
            df['volatility_peak'] = df['price_volatility_24h'] * df['is_peak_hours']
        
        return df
    
    def select_features(self, df: pd.DataFrame, target_col: str = 'price', 
                       max_features: int = 50) -> List[str]:
        """Select most important features using correlation and variance analysis.
        
        Args:
            df: DataFrame with features
            target_col: Target variable column name
            max_features: Maximum number of features to select
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting top {max_features} features")
        
        # Get feature columns (exclude timestamp and target)
        feature_cols = [col for col in df.columns if col not in ['timestamp', target_col]]
        
        # Remove features with low variance
        feature_df = df[feature_cols].fillna(0)
        variances = feature_df.var()
        high_var_features = variances[variances > 0.001].index.tolist()
        
        # Calculate correlations with target
        target_corrs = abs(df[high_var_features + [target_col]].corr()[target_col])
        target_corrs = target_corrs.drop(target_col).sort_values(ascending=False)
        
        # Select features with highest correlation
        selected_features = target_corrs.head(max_features).index.tolist()
        
        logger.info(f"Selected {len(selected_features)} features")
        return selected_features
    
    def prepare_forecast_data(self, df: pd.DataFrame, target_col: str = 'price',
                            feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for forecasting models.
        
        Args:
            df: DataFrame with features
            target_col: Target variable column name
            feature_cols: List of feature columns to use
            
        Returns:
            Tuple of (features_df, target_series)
        """
        if feature_cols is None:
            feature_cols = self.select_features(df, target_col)
        
        # Prepare features and target
        X = df[feature_cols].ffill().fillna(0)
        y = df[target_col]
        
        logger.info(f"Prepared forecast data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y 