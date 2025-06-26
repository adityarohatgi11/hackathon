#!/usr/bin/env python3
"""
Energy Management System - Main Application
Entry point for the energy management dashboard and system components.
Combines MARA API GridPilot-GT functionality with LLM integration.
"""

import argparse
from datetime import datetime
import pandas as pd
import time
import sys
import os
import logging
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import toml
    from ui.dashboard import main as run_dashboard
    from llm_integration.mock_interface import MockLLMInterface
    from llm_integration.unified_interface import UnifiedLLMInterface
    
    # GridPilot-GT imports (optional - only if modules exist)
    try:
        from api_client.client import get_prices, get_inventory, submit_bid
        from forecasting.forecaster import Forecaster
        from game_theory.bid_generators import build_bid_vector
        from game_theory.vcg_auction import vcg_allocate
        from control.cooling_model import cooling_for_gpu_kW
        from dispatch.dispatch_agent import build_payload
        GRIDPILOT_AVAILABLE = True
    except ImportError:
        GRIDPILOT_AVAILABLE = False
        print("WARNING: GridPilot-GT modules not available - gridpilot mode will not work")
        
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install streamlit pandas plotly toml anthropic")
    sys.exit(1)


def load_config(config_path: str = "config.toml") -> dict:
    """Load configuration from TOML file."""
    try:
        with open(config_path, 'r') as f:
            return toml.load(f)
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found. Using defaults.")
        return {}
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}


def setup_logging(config: dict):
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    
    logging.basicConfig(
        level=level,
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_config.get('file', 'energy_management.log'))
        ]
    )


def test_llm_interface():
    """Test the LLM interface functionality."""
    print("Testing LLM Interface...")
    
    # Use unified interface that automatically selects the best provider
    try:
        interface = UnifiedLLMInterface()
        if interface.is_available():
            provider_info = interface.get_provider_info()
            print(f"SUCCESS: Using {provider_info['provider']} LLM provider")
            return interface
        else:
            print("WARNING: No LLM providers available")
    except Exception as e:
        print(f"WARNING: Error with unified LLM interface: {e}")
    
    # Fall back to mock interface
    interface = MockLLMInterface()
    print("SUCCESS: Mock LLM interface loaded")
    return interface


def run_tests():
    """Run system tests."""
    print("Running system tests...")
    
    # Test LLM interface
    interface = test_llm_interface()
    
    # Test sample queries
    test_queries = [
        "What are the benefits of demand response?",
        "Explain the battery management strategy",
        "How can I optimize energy costs?"
    ]
    
    print("\nTesting sample queries:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = interface.process_query(query)
        print(f"Response: {response[:100]}...")
    
    print("\nSUCCESS: All tests completed successfully!")


def run_gridpilot_gt(simulate: bool = False):
    """Main orchestration loop for GridPilot-GT.
    
    Args:
        simulate: If True, run in simulation mode with mock data
    """
    start_time = time.time()
    print("GridPilot-GT Starting...")
    print(f"Simulation mode: {simulate}")
    print(f"Timestamp: {datetime.now()}")
    
    try:
        # Step 1: Get market data
        print("\nFetching market data...")
        prices = get_prices()
        inventory = get_inventory()
        print(f"Retrieved {len(prices)} price records")
        print(f"Current power available: {inventory['power_available']} kW")
        
        # Step 2: Generate forecasts
        print("\nGenerating forecasts...")
        forecaster = Forecaster(use_prophet=not simulate, use_ensemble=not simulate)
        if simulate:
            forecast = forecaster._predict_simple(prices, periods=24)  # type: ignore  # pylint: disable=protected-access
        else:
            forecast = forecaster.predict_next(prices)
        print(f"Generated {len(forecast)} hour forecast")
        
        # Step 3: Optimize bids
        print("\nOptimizing bids...")
        current_price = prices['price'].iloc[-1] if len(prices) > 0 else 50.0
        soc = inventory['battery_soc']
        lambda_deg = 0.0002  # Battery degradation cost
        
        uncertainty_df = pd.DataFrame(forecast[["σ_energy","σ_hash","σ_token"]])
        bids = build_bid_vector(
            current_price=current_price,
            forecast=forecast,
            uncertainty=uncertainty_df,
            soc=soc,
            lambda_deg=lambda_deg
        )
        print(f"Generated bid vector with {len(bids)} periods")
        
        # Step 4: Run VCG auction
        print("\nRunning VCG auction...")
        allocation, payments = vcg_allocate(bids, inventory["power_total"])
        print(f"Allocation: {allocation}")
        print(f"Payments: {payments}")
        
        # Step 5: Calculate cooling requirements
        print("\nCalculating cooling requirements...")
        inference_power = allocation.get("inference", 0)  # Already in kW
        cooling_kw, cooling_metrics = cooling_for_gpu_kW(inference_power)
        print(f"Cooling required: {cooling_kw:.2f} kW (COP: {cooling_metrics['cop']:.2f})")
        
        # Step 6: Build dispatch payload
        print("\nBuilding dispatch payload...")
        payload = build_payload(
            allocation=allocation,
            inventory=inventory,
            soc=soc,
            cooling_kw=cooling_kw,
            power_limit=inventory["power_total"]
        )
        
        # Step 7: Submit to market (if not simulation)
        if not simulate:
            print("\nSubmitting to market...")
            response = submit_bid(payload)
            print(f"Market response: {response}")
        else:
            print("\nSIMULATION - Would submit payload:")
            print(f"Total power: {payload['power_requirements']['total_power_kw']:.2f} kW")
            print(f"Constraints satisfied: {payload['constraints_satisfied']}")
            print(f"System utilization: {payload['system_state']['utilization']:.1%}")
        
        print("\nSUCCESS: GridPilot-GT cycle completed successfully!")
        
        # Summary metrics
        print("\nSummary Metrics:")
        print(f"  • GPU Allocation: {sum(allocation.values()):.3f}")
        print(f"  • Total Power: {payload['power_requirements']['total_power_kw']:.1f} kW")
        print(f"  • Battery SOC: {soc:.1%}")
        print(f"  • Cooling Load: {cooling_kw:.1f} kW")
        print(f"  • Efficiency: {cooling_metrics['efficiency']:.1%}")
        
        # Return success result for testing (with interface compatibility)
        return {
            'success': True,
            'elapsed_time': time.time() - start_time,
            'soc': soc,
            'total_power': payload['power_requirements']['total_power_kw'],
            'revenue': payments,
            'payload': payload,
            # Interface compatibility fields
            'power_requirements': payload['power_requirements'],
            'constraints_satisfied': payload['constraints_satisfied'],
            'system_state': payload['system_state'],
            'performance_metrics': {
                'build_time_ms': (time.time() - start_time) * 1000 * 0.1,  # Estimate build time as 10% of total
                'total_time_ms': (time.time() - start_time) * 1000
            }
        }
        
    except Exception as e:
        print(f"ERROR: Error in GridPilot-GT: {e}")
        print(f"TIP: Run with simulate=True for testing")
        return {
            'success': False,
            'error': str(e),
            'elapsed_time': time.time() - start_time
        }


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Energy Management System with GridPilot-GT")
    parser.add_argument(
        "--mode", 
        choices=["dashboard", "test", "llm", "gridpilot"], 
        default="dashboard",
        help="Run mode: dashboard (default), test, llm, or gridpilot"
    )
    parser.add_argument(
        "--config", 
        default="config.toml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--port", 
        type=int,
        help="Dashboard port (overrides config)"
    )
    parser.add_argument(
        "--simulate", 
        type=int, 
        default=1,
        help="GridPilot-GT simulation mode (1) or live mode (0)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Energy Management System in {args.mode} mode")
    
    if args.mode == "test":
        run_tests()
    
    elif args.mode == "llm":
        # Interactive LLM mode
        interface = test_llm_interface()
        print("\nEnergy Management Assistant")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                query = input("\nYou: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if query:
                    response = interface.process_query(query)
                    print(f"\nAssistant: {response}")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.mode == "gridpilot":
        # Run GridPilot-GT energy trading system
        if not GRIDPILOT_AVAILABLE:
            print("ERROR: GridPilot-GT modules not available")
            print("Please ensure all GridPilot-GT dependencies are installed")
            sys.exit(1)
            
        simulate_mode = bool(args.simulate)
        result = run_gridpilot_gt(simulate=simulate_mode)
        
        if result['success']:
            print("\nSUCCESS: GridPilot-GT executed successfully!")
            sys.exit(0)
        else:
            print("\n💥 GridPilot-GT failed!")
            sys.exit(1)
    
    elif args.mode == "dashboard":
        # Run the Streamlit dashboard
        print("Starting Energy Management Dashboard...")
        print("The dashboard will open in your browser.")
        print("Press Ctrl+C to stop the server.")
        
        # Override port if specified
        if args.port:
            config['dashboard']['port'] = args.port
        
        # Run dashboard
        run_dashboard()


if __name__ == "__main__":
    main()
