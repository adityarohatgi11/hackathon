"""Forecasting module for GridPilot-GT energy price prediction."""

from .forecaster import Forecaster, create_advanced_forecaster, get_forecaster
from .feature_engineering import FeatureEngineer

# Try to import advanced forecaster class
try:
    from .advanced_forecaster import QuantitativeForecaster  # type: ignore
    __all__ = ['Forecaster', 'FeatureEngineer', 'QuantitativeForecaster', 'create_advanced_forecaster', 'get_forecaster']
except ImportError:
    __all__ = ['Forecaster', 'FeatureEngineer', 'create_advanced_forecaster', 'get_forecaster'] 