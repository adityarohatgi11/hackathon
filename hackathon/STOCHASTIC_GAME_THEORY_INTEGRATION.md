# Stochastic Game Theory Integration Guide

## Overview

This guide demonstrates how to combine the new advanced stochastic methods with GridPilot-GT's existing game theory strategies for superior optimization results.

## 🎯 **Enhanced Game Theory with Stochastic Forecasting**

### **1. Stochastic VCG Auction Enhancement**

Instead of deterministic price forecasts, use probabilistic scenarios:

```python
from forecasting.stochastic_models import create_stochastic_forecaster, create_monte_carlo_engine
from game_theory.bid_generators import build_bid_vector
from game_theory.vcg_auction import vcg_allocate

# Create stochastic price model
sde_model = create_stochastic_forecaster("mean_reverting")
sde_model.fit(historical_prices)

# Generate multiple price scenarios for robust bidding
price_scenarios = sde_model.simulate(n_steps=24, n_paths=100)

# Create ensemble forecast with uncertainty quantification
forecast_mean = np.mean(price_scenarios, axis=0)
forecast_std = np.std(price_scenarios, axis=0)

stochastic_forecast = pd.DataFrame({
    'timestamp': pd.date_range(start=datetime.now(), periods=24, freq='H'),
    'predicted_price': forecast_mean,
    'σ_energy': forecast_std,
    'σ_hash': forecast_std * 0.5,
    'σ_token': forecast_std * 0.3,
    'lower_bound': forecast_mean - 1.96 * forecast_std,
    'upper_bound': forecast_mean + 1.96 * forecast_std,
    'method': 'stochastic_ensemble'
})

# Generate risk-aware bids
uncertainty_df = stochastic_forecast[['σ_energy', 'σ_hash', 'σ_token']]
bids = build_bid_vector(
    current_price=current_price,
    forecast=stochastic_forecast,
    uncertainty=uncertainty_df,
    soc=0.5,
    lambda_deg=0.0002
)

# Run VCG auction with stochastic bids
allocation_result = vcg_allocate(bids, capacity_kw=1000)
```

### **2. Monte Carlo MPC Optimization**

Enhance Model Predictive Control with scenario-based optimization:

```python
from game_theory.mpc_controller import MPCController

# Initialize Monte Carlo engine
mc_engine = create_monte_carlo_engine(n_simulations=1000)

# Define MPC strategy for scenario analysis
def mpc_allocation_strategy(forecast_df, initial_conditions):
    """MPC-based allocation strategy."""
    mpc = MPCController(horizon=24, lambda_deg=0.0002)
    current_state = {
        "soc": initial_conditions.get("soc", 0.5),
        "available_power_kw": initial_conditions.get("power_available", 1000.0)
    }
    
    result = mpc.optimize_horizon(forecast_df, current_state)
    return {"energy_allocation": result["energy_bids"]}

# Run Monte Carlo scenario analysis
scenario_results = mc_engine.scenario_analysis(
    price_model=sde_model,
    allocation_strategy=mpc_allocation_strategy,
    horizon=24,
    initial_conditions={"soc": 0.5, "power_available": 1000.0}
)

# Extract risk-adjusted metrics
expected_revenue = scenario_results['revenue_statistics']['expected_return']
revenue_var_95 = scenario_results['revenue_statistics']['var']
revenue_cvar = scenario_results['revenue_statistics']['cvar']

print(f"Expected Revenue: ${expected_revenue:.2f}")
print(f"Value at Risk (95%): ${revenue_var_95:.2f}")
print(f"Conditional VaR: ${revenue_cvar:.2f}")
```

### **3. Reinforcement Learning Game Strategy**

Use RL agents to adapt game theory strategies:

```python
from forecasting.stochastic_models import create_rl_agent

# Create RL agent for adaptive strategy selection
rl_agent = create_rl_agent(state_size=64, action_size=5)

# Define market state for RL decision
def get_market_state(prices_df, current_soc, market_volatility):
    """Extract market state for RL agent."""
    current_price = prices_df['price'].iloc[-1]
    price_trend = prices_df['price'].pct_change().tail(24).mean()
    
    return {
        "price": current_price,
        "soc": current_soc,
        "volatility": market_volatility,
        "demand": min(max(price_trend + 0.5, 0), 1)  # Normalize demand proxy
    }

# Get adaptive strategy parameters
market_state = get_market_state(prices_df, current_soc=0.6, market_volatility=0.15)
strategy_params = rl_agent.get_bidding_strategy(market_state)

print(f"RL Strategy: Aggressiveness={strategy_params['aggressiveness']:.2f}")
print(f"Risk Tolerance: {strategy_params['risk_tolerance']:.2f}")

# Apply RL strategy to bid generation
adaptive_lambda_deg = 0.0002 * (2 - strategy_params['aggressiveness'])  # More aggressive = lower degradation cost
risk_multiplier = 1 + strategy_params['risk_tolerance']  # Higher risk tolerance = larger bids

# Generate adaptive bids
adaptive_bids = build_bid_vector(
    current_price=current_price,
    forecast=stochastic_forecast,
    uncertainty=uncertainty_df * risk_multiplier,  # Scale uncertainty by risk tolerance
    soc=0.6,
    lambda_deg=adaptive_lambda_deg
)
```

## 🚀 **Advanced Integration Examples**

### **Example 1: Stochastic Nash Equilibrium**

Solve for equilibrium strategies under price uncertainty:

```python
# This would use the advanced_game_theory.py module when fully implemented
# For now, we can simulate the concept:

def simulate_stochastic_nash_equilibrium(price_scenarios, n_players=3):
    """Simulate Nash equilibrium under price uncertainty."""
    
    # Generate strategies for each player under different scenarios
    player_strategies = {}
    
    for player_id in range(n_players):
        player_allocations = []
        
        for scenario_idx in range(len(price_scenarios)):
            prices = price_scenarios[scenario_idx]
            
            # Each player optimizes against expected competitor behavior
            base_allocation = 1000 / n_players  # Equal starting point
            price_adjustment = (prices - prices.mean()) / prices.std()
            
            # Player-specific risk preferences
            risk_aversion = 0.5 + player_id * 0.2  # Players have different risk preferences
            allocation = base_allocation * (1 + price_adjustment * (1 - risk_aversion))
            allocation = np.clip(allocation, 0, 800)  # Capacity constraints
            
            player_allocations.append(allocation)
        
        # Average allocation across scenarios
        player_strategies[player_id] = np.mean(player_allocations, axis=0)
    
    return player_strategies

# Run stochastic Nash equilibrium
nash_strategies = simulate_stochastic_nash_equilibrium(price_scenarios[:10])  # Use 10 scenarios

for player_id, strategy in nash_strategies.items():
    total_allocation = np.sum(strategy)
    print(f"Player {player_id}: Total allocation = {total_allocation:.1f} kW")
```

### **Example 2: Risk-Adjusted Portfolio Optimization**

Combine CVaR optimization with game theory:

```python
def risk_adjusted_portfolio_optimization(price_scenarios, risk_limit=0.05):
    """Portfolio optimization with CVaR constraints."""
    
    n_scenarios, horizon = price_scenarios.shape
    revenues = []
    
    # Test different allocation strategies
    allocation_strategies = [
        ("Conservative", lambda p: np.full(len(p), 200)),  # Constant 200 kW
        ("Aggressive", lambda p: p / p.mean() * 400),      # Price-following
        ("Balanced", lambda p: 300 + (p - p.mean()) * 5), # Mixed strategy
    ]
    
    strategy_results = {}
    
    for strategy_name, allocation_func in allocation_strategies:
        strategy_revenues = []
        
        for scenario in price_scenarios:
            allocation = allocation_func(scenario)
            allocation = np.clip(allocation, 0, 800)  # Capacity limits
            
            revenue = np.sum(scenario * allocation) * 0.001  # Convert to $
            strategy_revenues.append(revenue)
        
        # Calculate risk metrics
        strategy_revenues = np.array(strategy_revenues)
        expected_return = np.mean(strategy_revenues)
        var_95 = np.percentile(strategy_revenues, 5)
        cvar_95 = np.mean(strategy_revenues[strategy_revenues <= var_95])
        
        strategy_results[strategy_name] = {
            "expected_return": expected_return,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "sharpe_ratio": expected_return / np.std(strategy_revenues)
        }
    
    return strategy_results

# Run risk-adjusted optimization
portfolio_results = risk_adjusted_portfolio_optimization(price_scenarios)

print("Risk-Adjusted Portfolio Analysis:")
print("=" * 50)
for strategy, metrics in portfolio_results.items():
    print(f"{strategy}:")
    print(f"  Expected Return: ${metrics['expected_return']:.2f}")
    print(f"  VaR (95%): ${metrics['var_95']:.2f}")
    print(f"  CVaR (95%): ${metrics['cvar_95']:.2f}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print()
```

### **Example 3: Dynamic Strategy Adaptation**

Real-time strategy switching based on market conditions:

```python
def dynamic_strategy_adaptation(current_prices, price_scenarios, market_regime="normal"):
    """Adapt strategy based on current market conditions."""
    
    # Detect market regime
    current_volatility = np.std(current_prices.tail(24))
    price_trend = current_prices.pct_change().tail(24).mean()
    
    if current_volatility > 0.2:
        market_regime = "high_volatility"
    elif abs(price_trend) > 0.1:
        market_regime = "trending"
    else:
        market_regime = "normal"
    
    print(f"Detected market regime: {market_regime}")
    
    # Select appropriate strategy
    if market_regime == "high_volatility":
        # Conservative strategy during high volatility
        strategy = "mean_reverting"
        risk_multiplier = 0.5
        lambda_deg = 0.0005  # Higher degradation cost = more conservative
        
    elif market_regime == "trending":
        # Momentum strategy during trends
        strategy = "gbm"
        risk_multiplier = 1.5
        lambda_deg = 0.0001  # Lower degradation cost = more aggressive
        
    else:
        # Balanced strategy for normal conditions
        strategy = "jump_diffusion"
        risk_multiplier = 1.0
        lambda_deg = 0.0002
    
    # Create appropriate SDE model
    adaptive_sde = create_stochastic_forecaster(strategy)
    adaptive_sde.fit(current_prices)
    
    # Generate forecast with regime-specific model
    adaptive_scenarios = adaptive_sde.simulate(n_steps=24, n_paths=100)
    adaptive_forecast = pd.DataFrame({
        'timestamp': pd.date_range(start=datetime.now(), periods=24, freq='H'),
        'predicted_price': np.mean(adaptive_scenarios, axis=0),
        'σ_energy': np.std(adaptive_scenarios, axis=0) * risk_multiplier,
        'σ_hash': np.std(adaptive_scenarios, axis=0) * risk_multiplier * 0.5,
        'σ_token': np.std(adaptive_scenarios, axis=0) * risk_multiplier * 0.3,
        'method': f'adaptive_{strategy}'
    })
    
    return adaptive_forecast, lambda_deg, market_regime

# Example usage
adaptive_forecast, adaptive_lambda, regime = dynamic_strategy_adaptation(
    prices_df['price'], price_scenarios
)

print(f"Adaptive forecast for {regime} regime:")
print(f"Mean 24h price: ${adaptive_forecast['predicted_price'].iloc[23]:.2f}")
print(f"Uncertainty: ${adaptive_forecast['σ_energy'].iloc[23]:.2f}")
print(f"Degradation cost: {adaptive_lambda:.6f}")
```

## 📊 **Performance Comparison**

### **Traditional vs. Stochastic Approaches**

```python
def compare_traditional_vs_stochastic():
    """Compare traditional deterministic vs. new stochastic approaches."""
    
    # Traditional approach (existing GridPilot-GT)
    traditional_forecast = basic_forecaster.predict_next(prices_df, periods=24)
    traditional_bids = build_bid_vector(
        current_price=current_price,
        forecast=traditional_forecast,
        uncertainty=pd.DataFrame({
            'σ_energy': [5.0] * 24,
            'σ_hash': [2.5] * 24,
            'σ_token': [1.5] * 24
        }),
        soc=0.5,
        lambda_deg=0.0002
    )
    
    # Stochastic approach (new implementation)
    stochastic_bids = build_bid_vector(
        current_price=current_price,
        forecast=stochastic_forecast,
        uncertainty=uncertainty_df,
        soc=0.5,
        lambda_deg=0.0002
    )
    
    # Compare results
    traditional_revenue = np.sum(traditional_forecast['predicted_price'] * traditional_bids['inference'])
    stochastic_revenue = np.sum(stochastic_forecast['predicted_price'] * stochastic_bids['inference'])
    
    improvement = (stochastic_revenue - traditional_revenue) / traditional_revenue * 100
    
    print("Performance Comparison:")
    print("=" * 30)
    print(f"Traditional Revenue: ${traditional_revenue:.2f}")
    print(f"Stochastic Revenue: ${stochastic_revenue:.2f}")
    print(f"Improvement: {improvement:.1f}%")
    
    return improvement

# Run comparison
performance_improvement = compare_traditional_vs_stochastic()
```

## 🎯 **Key Benefits of Integration**

### **1. Robust Decision Making**
- **Uncertainty Awareness**: Decisions account for forecast uncertainty
- **Risk Management**: CVaR constraints prevent excessive losses
- **Scenario Planning**: Multiple outcomes considered simultaneously

### **2. Adaptive Strategies**
- **Market Regime Detection**: Automatic strategy switching
- **Learning from Experience**: RL agents improve over time
- **Dynamic Risk Adjustment**: Real-time parameter updates

### **3. Competitive Advantages**
- **Strategic Robustness**: Performance maintained under competitor actions
- **Mathematical Sophistication**: Advanced models beyond typical energy trading
- **Quantitative Edge**: Superior forecasting and optimization

## 🚀 **Next Steps**

1. **Complete RL Integration**: Fix NaN handling in reinforcement learning
2. **Advanced Game Theory**: Implement full stochastic Nash equilibrium solver
3. **Real-time Adaptation**: Continuous model updating and strategy switching
4. **Performance Monitoring**: Track improvement metrics in production

The integration of stochastic methods with game theory provides GridPilot-GT with state-of-the-art quantitative capabilities that significantly enhance performance in uncertain, competitive energy markets. 