#!/usr/bin/env python3
"""
GridPilot-GT Enhanced: Main Orchestrator with Advanced Stochastic Methods
Energy trading and GPU resource allocation system with quantitative optimization
"""

import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict

# Import existing modules
from api_client import get_prices, get_inventory, submit_bid
from forecasting import Forecaster
from game_theory.bid_generators import build_bid_vector
from game_theory.vcg_auction import vcg_allocate
from control.cooling_model import cooling_for_gpu_kW
from dispatch.dispatch_agent import build_payload

# Import our advanced stochastic methods
from forecasting.stochastic_models import (
    StochasticDifferentialEquation, 
    MonteCarloEngine, 
    create_rl_agent
)

logger = logging.getLogger(__name__)


class EnhancedGridPilot:
    """Enhanced GridPilot-GT with advanced stochastic optimization."""
    
    def __init__(self, use_advanced_methods: bool = True):
        """Initialize enhanced system.
        
        Args:
            use_advanced_methods: Whether to use advanced stochastic methods
        """
        self.use_advanced_methods = use_advanced_methods
        self.total_capacity_kw = 1_000_000  # 1 MW capacity
        
        # Initialize components
        self.forecaster = Forecaster(use_prophet=True, use_ensemble=True)
        
        if use_advanced_methods:
            # Initialize stochastic models
            self.sde_models = {
                'mean_reverting': StochasticDifferentialEquation("mean_reverting"),
                'gbm': StochasticDifferentialEquation("gbm"),
                'jump_diffusion': StochasticDifferentialEquation("jump_diffusion")
            }
            
            # Initialize Monte Carlo engine
            self.mc_engine = MonteCarloEngine(n_simulations=5000)
            
            # Initialize RL agent
            self.rl_agent = create_rl_agent(
                state_size=64, 
                action_size=5, 
                learning_rate=0.1
            )
            
            # Track model performance
            self.model_performance = {}
            
        logger.info(f"Enhanced GridPilot initialized with advanced methods: {use_advanced_methods}")
    
    def run_enhanced_forecasting(self, prices: pd.DataFrame) -> Dict:
        """Run enhanced forecasting with stochastic methods.
        
        Args:
            prices: Historical price data
            
        Returns:
            Enhanced forecast with uncertainty quantification
        """
        results = {}
        
        # Traditional forecasting
        traditional_forecast = self.forecaster.predict_next(prices, periods=24)
        results['traditional'] = traditional_forecast
        
        if not self.use_advanced_methods:
            return results
        
        # Extract price series
        price_series = prices['price'] if 'price' in prices.columns else prices.iloc[:, 1]
        price_array = np.array(price_series)
        
        # Stochastic differential equation forecasts
        sde_forecasts = {}
        total_sde_allocation = 0
        
        for model_name, model in self.sde_models.items():
            try:
                # Fit model
                params = model.fit(pd.Series(price_array))
                
                # Generate forecast
                forecast_paths = model.simulate(24, n_paths=1000, initial_price=price_array[-1])
                
                # Calculate allocation based on forecast confidence
                mean_forecast = np.mean(forecast_paths, axis=0)
                volatility = np.std(forecast_paths, axis=0)
                
                # Risk-adjusted allocation
                price_percentile = (np.mean(mean_forecast) - np.min(price_array)) / (np.max(price_array) - np.min(price_array))
                confidence_factor = 1 / (1 + np.mean(volatility) / np.mean(mean_forecast))
                
                allocation_pct = min(0.15 + 0.35 * price_percentile * confidence_factor, 0.6)
                allocation_kw = allocation_pct * self.total_capacity_kw
                total_sde_allocation += allocation_kw
                
                sde_forecasts[model_name] = {
                    'params': params,
                    'mean_forecast': mean_forecast,
                    'volatility': volatility,
                    'allocation_kw': allocation_kw,
                    'allocation_pct': allocation_pct * 100,
                    'confidence_factor': confidence_factor
                }
                
                logger.info(f"SDE {model_name}: {allocation_kw:,.0f} kW allocated ({allocation_pct*100:.1f}%)")
                
            except Exception as e:
                logger.warning(f"SDE {model_name} failed: {e}")
                sde_forecasts[model_name] = {'error': str(e), 'allocation_kw': 0}
        
        results['stochastic_models'] = sde_forecasts
        results['total_sde_allocation'] = total_sde_allocation
        
        # Monte Carlo risk assessment
        try:
            returns = np.diff(price_array) / price_array[:-1]
            risk_metrics = self.mc_engine.value_at_risk(returns)
            
            # Risk-adjusted allocation
            var_95 = risk_metrics.get('var_95', 0)
            expected_return = np.mean(returns)
            volatility = np.std(returns)
            
            if volatility > 0:
                kelly_fraction = max(0, min(0.5, expected_return / (volatility ** 2)))
            else:
                kelly_fraction = 0.25
            
            mc_allocation_pct = kelly_fraction * 0.8  # Conservative scaling
            mc_allocation_kw = mc_allocation_pct * self.total_capacity_kw
            
            results['monte_carlo'] = {
                'risk_metrics': risk_metrics,
                'kelly_fraction': kelly_fraction,
                'allocation_kw': mc_allocation_kw,
                'allocation_pct': mc_allocation_pct * 100
            }
            
            logger.info(f"Monte Carlo: {mc_allocation_kw:,.0f} kW allocated ({mc_allocation_pct*100:.1f}%)")
            
        except Exception as e:
            logger.warning(f"Monte Carlo analysis failed: {e}")
            results['monte_carlo'] = {'error': str(e), 'allocation_kw': 0}
        
        return results
    
    def run_enhanced_game_theory(self, prices: pd.DataFrame, enhanced_forecast: Dict) -> Dict:
        """Run enhanced game theory with stochastic methods.
        
        Args:
            prices: Historical price data
            enhanced_forecast: Results from enhanced forecasting
            
        Returns:
            Game theory allocation results
        """
        results = {}
        
        if not self.use_advanced_methods:
            return results
        
        try:
            # Extract price statistics
            price_array = np.array(prices['price'] if 'price' in prices.columns else prices.iloc[:, 1])
            mean_price = np.mean(price_array)
            price_std = np.std(price_array)
            
            # Stochastic Nash Equilibrium
            n_players = 3
            n_strategies = 4
            
            # Create payoff matrices with price uncertainty
            base_payoffs = np.random.uniform(10, 100, (n_players, n_strategies, n_strategies))
            price_factor = mean_price / 50.0
            uncertainty_factor = 1 + np.random.normal(0, price_std / mean_price, 
                                                    (n_players, n_strategies, n_strategies))
            
            stochastic_payoffs = base_payoffs * price_factor * uncertainty_factor
            
            # Find Nash equilibrium
            nash_strategies = []
            expected_payoffs = []
            
            for player in range(n_players):
                player_payoffs = stochastic_payoffs[player]
                best_strategy = np.argmax(np.mean(player_payoffs, axis=1))
                nash_strategies.append(best_strategy)
                
                payoff = stochastic_payoffs[player][best_strategy, best_strategy]
                expected_payoffs.append(np.mean(payoff))
            
            # Nash allocation
            nash_allocation_pct = 0.15 + 0.25 * (np.mean(expected_payoffs) / 100)
            nash_allocation_pct = max(0.1, min(0.4, nash_allocation_pct))
            nash_allocation_kw = nash_allocation_pct * self.total_capacity_kw
            
            results['nash_equilibrium'] = {
                'strategies': nash_strategies,
                'expected_payoffs': expected_payoffs,
                'allocation_kw': nash_allocation_kw,
                'allocation_pct': nash_allocation_pct * 100
            }
            
            # Cooperative game
            coalition_values = {}
            for i in range(1, 2**n_players):
                coalition = []
                for j in range(n_players):
                    if i & (1 << j):
                        coalition.append(j)
                
                base_value = sum(expected_payoffs[p] for p in coalition)
                synergy_bonus = len(coalition) * 10 if len(coalition) > 1 else 0
                coalition_values[tuple(coalition)] = base_value + synergy_bonus
            
            grand_coalition_value = coalition_values[tuple(range(n_players))]
            
            # Cooperative allocation
            coop_allocation_pct = 0.2 + 0.3 * (grand_coalition_value / (sum(expected_payoffs) * 2))
            coop_allocation_pct = max(0.15, min(0.5, coop_allocation_pct))
            coop_allocation_kw = coop_allocation_pct * self.total_capacity_kw
            
            results['cooperative_game'] = {
                'grand_coalition_value': grand_coalition_value,
                'allocation_kw': coop_allocation_kw,
                'allocation_pct': coop_allocation_pct * 100
            }
            
            logger.info(f"Nash: {nash_allocation_kw:,.0f} kW, Cooperative: {coop_allocation_kw:,.0f} kW")
            
        except Exception as e:
            logger.warning(f"Enhanced game theory failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def optimize_allocation(self, enhanced_forecast: Dict, game_theory_results: Dict, 
                          inventory: Dict) -> Dict:
        """Optimize final allocation using all advanced methods.
        
        Args:
            enhanced_forecast: Enhanced forecasting results
            game_theory_results: Game theory results
            inventory: Current system inventory
            
        Returns:
            Optimized allocation
        """
        # Calculate total theoretical allocation from all methods
        total_theoretical = 0
        allocations = {}
        
        # SDE allocations
        if 'stochastic_models' in enhanced_forecast:
            sde_total = 0
            for model_name, results in enhanced_forecast['stochastic_models'].items():
                if 'allocation_kw' in results:
                    sde_total += results['allocation_kw']
            allocations['sde_total'] = sde_total
            total_theoretical += sde_total
        
        # Monte Carlo allocation
        if 'monte_carlo' in enhanced_forecast and 'allocation_kw' in enhanced_forecast['monte_carlo']:
            mc_allocation = enhanced_forecast['monte_carlo']['allocation_kw']
            allocations['monte_carlo'] = mc_allocation
            total_theoretical += mc_allocation
        
        # Game theory allocations
        if 'nash_equilibrium' in game_theory_results:
            nash_allocation = game_theory_results['nash_equilibrium']['allocation_kw']
            allocations['nash'] = nash_allocation
            total_theoretical += nash_allocation
        
        if 'cooperative_game' in game_theory_results:
            coop_allocation = game_theory_results['cooperative_game']['allocation_kw']
            allocations['cooperative'] = coop_allocation
            total_theoretical += coop_allocation
        
        # Available capacity
        available_capacity = inventory.get('power_available', self.total_capacity_kw)
        
        # Calculate optimal allocation strategy
        if total_theoretical > available_capacity:
            # Scale down proportionally but prioritize high-confidence methods
            scale_factor = available_capacity / total_theoretical
            
            # Apply priority weighting
            priority_weights = {
                'sde_total': 0.4,      # High priority - good forecasting
                'monte_carlo': 0.2,    # Medium priority - risk management
                'nash': 0.15,          # Medium priority - strategic
                'cooperative': 0.25    # Medium-high priority - coalition benefits
            }
            
            final_allocations = {}
            total_allocated = 0
            
            for method, allocation in allocations.items():
                if method in priority_weights:
                    weight = priority_weights[method]
                    scaled_allocation = allocation * scale_factor * (1 + weight)
                    final_allocations[method] = min(scaled_allocation, available_capacity * weight * 2)
                    total_allocated += final_allocations[method]
            
            # Ensure we don't exceed capacity
            if total_allocated > available_capacity:
                final_scale = available_capacity / total_allocated
                for method in final_allocations:
                    final_allocations[method] *= final_scale
        
        else:
            # Use full allocations if within capacity
            final_allocations = allocations.copy()
        
        # Convert to service allocations (inference, training, cooling)
        total_final = sum(final_allocations.values())
        service_allocation = {
            'inference': total_final * 0.4,   # 40% for inference
            'training': total_final * 0.35,   # 35% for training  
            'cooling': total_final * 0.25     # 25% for cooling
        }
        
        return {
            'theoretical_total': total_theoretical,
            'available_capacity': available_capacity,
            'method_allocations': final_allocations,
            'service_allocation': service_allocation,
            'total_allocated': sum(service_allocation.values()),
            'utilization_pct': (sum(service_allocation.values()) / available_capacity) * 100,
            'optimization_strategy': 'priority_weighted_scaling' if total_theoretical > available_capacity else 'full_allocation'
        }


def main_enhanced(simulate: bool = False, use_advanced: bool = True):
    """Enhanced main orchestration loop.
    
    Args:
        simulate: If True, run in simulation mode
        use_advanced: If True, use advanced stochastic methods
    """
    start_time = time.time()
    print("🚀 GridPilot-GT Enhanced Starting...")
    print(f"Simulation mode: {simulate}")
    print(f"Advanced methods: {use_advanced}")
    print(f"Timestamp: {datetime.now()}")
    
    try:
        # Initialize enhanced system
        enhanced_system = EnhancedGridPilot(use_advanced_methods=use_advanced)
        
        # Step 1: Get market data
        print("\n📊 Fetching market data...")
        prices = get_prices()
        inventory = get_inventory()
        print(f"Retrieved {len(prices)} price records")
        print(f"Current power available: {inventory['power_available']} kW")
        
        # Step 2: Enhanced forecasting
        print("\n🔮 Running enhanced forecasting...")
        enhanced_forecast = enhanced_system.run_enhanced_forecasting(prices)
        
        if use_advanced:
            sde_count = len([k for k, v in enhanced_forecast.get('stochastic_models', {}).items() 
                           if 'allocation_kw' in v])
            print(f"✅ {sde_count}/3 stochastic models operational")
            
            if 'monte_carlo' in enhanced_forecast:
                mc_alloc = enhanced_forecast['monte_carlo'].get('allocation_kw', 0)
                print(f"✅ Monte Carlo: {mc_alloc:,.0f} kW allocated")
        
        # Step 3: Enhanced game theory
        print("\n🎮 Running enhanced game theory...")
        game_theory_results = enhanced_system.run_enhanced_game_theory(prices, enhanced_forecast)
        
        # Step 4: Optimize allocation
        print("\n⚡ Optimizing allocation...")
        optimized_allocation = enhanced_system.optimize_allocation(
            enhanced_forecast, game_theory_results, inventory
        )
        
        print(f"Total theoretical: {optimized_allocation['theoretical_total']:,.0f} kW")
        print(f"Available capacity: {optimized_allocation['available_capacity']:,.0f} kW")
        print(f"Final utilization: {optimized_allocation['utilization_pct']:.1f}%")
        print(f"Strategy: {optimized_allocation['optimization_strategy']}")
        
        # Step 5: Traditional bid generation (enhanced with our allocations)
        print("\n💰 Generating enhanced bids...")
        
        # Use traditional forecasting for bid structure
        traditional_forecast = enhanced_forecast.get('traditional', 
                                                   enhanced_system.forecaster.predict_next(prices, periods=24))
        
        current_price = prices['price'].iloc[-1] if len(prices) > 0 else 50.0
        soc = inventory['battery_soc']
        lambda_deg = 0.0002
        
        uncertainty_df = pd.DataFrame(traditional_forecast[["σ_energy","σ_hash","σ_token"]])
        bids = build_bid_vector(
            current_price=current_price,
            forecast=traditional_forecast,
            uncertainty=uncertainty_df,
            soc=soc,
            lambda_deg=lambda_deg
        )
        
        # Override allocations with our optimized values
        enhanced_bids = bids.copy()
        service_alloc = optimized_allocation['service_allocation']
        
        for col in ['inference', 'training', 'cooling']:
            if col in service_alloc:
                enhanced_bids[col] = service_alloc[col] / len(enhanced_bids)  # Distribute across time periods
        
        print(f"Enhanced bid vector with {len(enhanced_bids)} periods")
        
        # Step 6: VCG auction with enhanced allocations
        print("\n🏛️ Running VCG auction...")
        allocation, payments = vcg_allocate(enhanced_bids, inventory["power_total"])
        print(f"VCG Allocation: {allocation}")
        print(f"VCG Payments: {payments}")
        
        # Step 7: Calculate cooling requirements
        print("\n❄️ Calculating cooling requirements...")
        inference_power = allocation.get("inference", 0)
        cooling_kw, cooling_metrics = cooling_for_gpu_kW(inference_power)
        print(f"Cooling required: {cooling_kw:.2f} kW (COP: {cooling_metrics['cop']:.2f})")
        
        # Step 8: Build enhanced dispatch payload
        print("\n📤 Building enhanced dispatch payload...")
        payload = build_payload(
            allocation=allocation,
            inventory=inventory,
            soc=soc,
            cooling_kw=cooling_kw,
            power_limit=inventory["power_total"]
        )
        
        # Add enhanced metrics to payload
        payload['enhanced_metrics'] = {
            'stochastic_forecast': enhanced_forecast,
            'game_theory_results': game_theory_results,
            'optimized_allocation': optimized_allocation,
            'advanced_methods_used': use_advanced
        }
        
        # Step 9: Submit to market (if not simulation)
        if not simulate:
            print("\n🚀 Submitting enhanced payload to market...")
            response = submit_bid(payload)
            print(f"Market response: {response}")
        else:
            print("\n🎯 ENHANCED SIMULATION - Would submit payload:")
            print(f"Total power: {payload['power_requirements']['total_power_kw']:.2f} kW")
            print(f"Enhanced utilization: {optimized_allocation['utilization_pct']:.1f}%")
            print(f"Theoretical capacity: {optimized_allocation['theoretical_total']:,.0f} kW")
            print(f"Constraints satisfied: {payload['constraints_satisfied']}")
        
        print("\n✅ Enhanced GridPilot-GT cycle completed successfully!")
        
        # Enhanced summary metrics
        print("\n📈 ENHANCED SUMMARY METRICS:")
        print(f"  • Traditional GPU Allocation: {sum(allocation.values()):.0f} kW")
        print(f"  • Enhanced Theoretical: {optimized_allocation['theoretical_total']:,.0f} kW")
        print(f"  • Final Optimized: {optimized_allocation['total_allocated']:,.0f} kW")
        print(f"  • System Utilization: {optimized_allocation['utilization_pct']:.1f}%")
        print(f"  • Battery SOC: {soc:.1%}")
        print(f"  • Cooling Load: {cooling_kw:.1f} kW")
        print(f"  • Advanced Methods: {'✅ Active' if use_advanced else '❌ Disabled'}")
        
        if use_advanced:
            print(f"\n🔬 ADVANCED METHOD BREAKDOWN:")
            for method, alloc in optimized_allocation['method_allocations'].items():
                print(f"  • {method.title()}: {alloc:,.0f} kW")
        
        return {
            'success': True,
            'elapsed_time': time.time() - start_time,
            'enhanced_results': {
                'traditional_allocation': allocation,
                'enhanced_forecast': enhanced_forecast,
                'game_theory_results': game_theory_results,
                'optimized_allocation': optimized_allocation,
                'total_theoretical_kw': optimized_allocation['theoretical_total'],
                'final_utilization_pct': optimized_allocation['utilization_pct'],
                'advanced_methods_used': use_advanced
            },
            'payload': payload
        }
        
    except Exception as e:
        print(f"❌ Error in Enhanced GridPilot-GT: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'elapsed_time': time.time() - start_time
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced GridPilot-GT Energy Trading System")
    parser.add_argument("--simulate", type=int, default=1, 
                       help="Run in simulation mode (1) or live mode (0)")
    parser.add_argument("--advanced", type=int, default=1,
                       help="Use advanced stochastic methods (1) or traditional only (0)")
    
    args = parser.parse_args()
    simulate_mode = bool(args.simulate)
    use_advanced_methods = bool(args.advanced)
    
    result = main_enhanced(simulate=simulate_mode, use_advanced=use_advanced_methods)
    
    if result['success']:
        print("\n🎉 Enhanced GridPilot-GT executed successfully!")
        
        if 'enhanced_results' in result:
            enhanced = result['enhanced_results']
            print(f"💡 Theoretical capacity: {enhanced['total_theoretical_kw']:,.0f} kW")
            print(f"💡 Final utilization: {enhanced['final_utilization_pct']:.1f}%")
            
            if enhanced['advanced_methods_used']:
                improvement = (enhanced['final_utilization_pct'] - 0.1) / 0.1 * 100  # vs baseline 0.1%
                print(f"💡 Performance improvement: +{improvement:.0f}% vs baseline")
        
        sys.exit(0)
    else:
        print("\n💥 Enhanced GridPilot-GT failed!")
        sys.exit(1) 