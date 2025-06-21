#!/usr/bin/env python3
"""
GridPilot-GT: Main Orchestrator
Energy trading and GPU resource allocation system
"""

import sys
import argparse
import pandas as pd
from datetime import datetime

# Import all modules
from api_client import get_prices, get_inventory, submit_bid, register_site
from forecasting import Forecaster
from game_theory.bid_generators import build_bid_vector
from game_theory.vcg_auction import vcg_allocate
from control.cooling_model import cooling_for_gpu_kW
from dispatch.dispatch_agent import build_payload


def main(simulate: bool = False):
    """Main orchestration loop for GridPilot-GT.
    
    Args:
        simulate: If True, run in simulation mode with mock data
    """
    print("🚀 GridPilot-GT Starting...")
    print(f"Simulation mode: {simulate}")
    print(f"Timestamp: {datetime.now()}")
    
    try:
        # Step 1: Get market data
        print("\n📊 Fetching market data...")
        prices = get_prices()
        inventory = get_inventory()
        print(f"Retrieved {len(prices)} price records")
        print(f"Current power available: {inventory['power_available']} kW")
        
        # Step 2: Generate forecasts
        print("\n🔮 Generating forecasts...")
        forecaster = Forecaster()
        forecast = forecaster.predict_next(prices)
        print(f"Generated {len(forecast)} hour forecast")
        
        # Step 3: Optimize bids
        print("\n💰 Optimizing bids...")
        current_price = prices['price'].iloc[-1] if len(prices) > 0 else 50.0
        soc = inventory['battery_soc']
        lambda_deg = 0.0002  # Battery degradation cost
        
        bids = build_bid_vector(
            current_price=current_price,
            forecast=forecast,
            uncertainty=forecast[["σ_energy","σ_hash","σ_token"]],
            soc=soc,
            lambda_deg=lambda_deg
        )
        print(f"Generated bid vector with {len(bids)} periods")
        
        # Step 4: Run VCG auction
        print("\n🏛️ Running VCG auction...")
        allocation, payments = vcg_allocate(bids, inventory["power_total"])
        print(f"Allocation: {allocation}")
        print(f"Payments: {payments}")
        
        # Step 5: Calculate cooling requirements
        print("\n❄️ Calculating cooling requirements...")
        inference_power = allocation.get("inference", 0) * 1000  # Convert to kW
        cooling_kw, cooling_metrics = cooling_for_gpu_kW(inference_power)
        print(f"Cooling required: {cooling_kw:.2f} kW (COP: {cooling_metrics['cop']:.2f})")
        
        # Step 6: Build dispatch payload
        print("\n📤 Building dispatch payload...")
        payload = build_payload(
            allocation=allocation,
            inventory=inventory,
            soc=soc,
            cooling_kw=cooling_kw,
            power_limit=inventory["power_total"]
        )
        
        # Step 7: Submit to market (if not simulation)
        if not simulate:
            print("\n🚀 Submitting to market...")
            response = submit_bid(payload)
            print(f"Market response: {response}")
        else:
            print("\n🎯 SIMULATION - Would submit payload:")
            print(f"Total power: {payload['power_requirements']['total_power_kw']:.2f} kW")
            print(f"Constraints satisfied: {payload['constraints_satisfied']}")
            print(f"System utilization: {payload['system_state']['utilization']:.1%}")
        
        print("\n✅ GridPilot-GT cycle completed successfully!")
        
        # Summary metrics
        print("\n📈 Summary Metrics:")
        print(f"  • GPU Allocation: {sum(allocation.values()):.3f}")
        print(f"  • Total Power: {payload['power_requirements']['total_power_kw']:.1f} kW")
        print(f"  • Battery SOC: {soc:.1%}")
        print(f"  • Cooling Load: {cooling_kw:.1f} kW")
        print(f"  • Efficiency: {cooling_metrics['efficiency']:.1%}")
        
        return payload
        
    except Exception as e:
        print(f"\n❌ Error in GridPilot-GT: {e}")
        if not simulate:
            raise
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridPilot-GT Energy Trading System")
    parser.add_argument("--simulate", type=int, default=1, 
                       help="Run in simulation mode (1) or live mode (0)")
    
    args = parser.parse_args()
    simulate_mode = bool(args.simulate)
    
    result = main(simulate=simulate_mode)
    
    if result:
        print(f"\n🎉 GridPilot-GT executed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 GridPilot-GT failed!")
        sys.exit(1) 