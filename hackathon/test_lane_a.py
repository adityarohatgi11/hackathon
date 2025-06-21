#!/usr/bin/env python3
"""Test script to showcase Lane A enhanced functionality."""

from api_client import get_prices, get_inventory
from forecasting import Forecaster, FeatureEngineer

def test_lane_a():
    """Test Lane A functionality."""
    print("🧪 Testing Lane A: Data & Forecasting")
    print("=" * 50)
    
    # Test enhanced API client
    print("\n📊 Testing Enhanced API Client:")
    prices = get_prices()
    inventory = get_inventory()
    
    print(f"  • Price data: {len(prices)} records, {len(prices.columns)} features")
    print(f"  • Sample features: {list(prices.columns)[:8]}...")
    print(f"  • Price range: ${prices['price'].min():.2f} - ${prices['price'].max():.2f}")
    print(f"  • Inventory: {inventory['power_total']:.0f}kW total, {inventory['battery_soc']:.1%} SOC")
    
    # Test feature engineering
    print("\n🔧 Testing Feature Engineering:")
    fe = FeatureEngineer()
    enhanced_data = fe.engineer_features(prices)
    selected_features = fe.select_features(enhanced_data, max_features=10)
    
    print(f"  • Features engineered: {len(enhanced_data.columns)}")
    print(f"  • Top features: {selected_features[:5]}")
    
    # Test forecasting
    print("\n🔮 Testing Advanced Forecasting:")
    forecaster = Forecaster()
    forecast = forecaster.predict_next(prices, periods=12)  # 12 hour forecast
    
    print(f"  • Forecast periods: {len(forecast)}")
    print(f"  • Price forecast range: ${forecast['predicted_price'].min():.2f} - ${forecast['predicted_price'].max():.2f}")
    print(f"  • Forecast method: {forecast['method'].iloc[0]}")
    print(f"  • Uncertainty (σ_energy): ±${forecast['σ_energy'].mean():.2f}")
    
    # Feature importance
    print("\n📈 Model Insights:")
    importance = forecaster.feature_importance()
    print(f"  • Top 3 important features:")
    for i, (feature, score) in enumerate(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]):
        print(f"    {i+1}. {feature}: {score:.3f}")
    
    # Performance metrics
    performance = forecaster.get_model_performance()
    if performance:
        print(f"  • Model performance: RF MAE={performance.get('rf_mae', 'N/A'):.2f}")
    
    print("\n✅ Lane A testing complete! All systems operational.")
    return True

if __name__ == "__main__":
    test_lane_a() 