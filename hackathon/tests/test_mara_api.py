#!/usr/bin/env python3
"""
Test script for MARA Hackathon API integration.

This script tests connectivity and authentication with the MARA API
and provides diagnostics for troubleshooting any issues.
"""

import json
import logging
from datetime import datetime
from api_client.client import test_mara_api_connection, get_prices, get_inventory, get_market_status

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run comprehensive MARA API tests."""
    print("🚀 GridPilot-GT MARA API Integration Test")
    print("=" * 50)
    
    # Test 1: API Connection and Authentication
    print("\n📡 Testing MARA API Connection...")
    connection_test = test_mara_api_connection()
    
    print(f"API Status: {connection_test['overall_status'].upper()}")
    print(f"API Key Configured: {'✅' if connection_test['api_key_configured'] else '❌'}")
    print(f"Prices Available: {'✅' if connection_test.get('prices_available') else '❌'}")
    print(f"Authentication: {'✅' if connection_test.get('authentication') == 'success' else '❌'}")
    
    if connection_test.get('recommendations'):
        print("\n💡 Recommendations:")
        for rec in connection_test['recommendations']:
            print(f"   {rec}")
    
    # Test 2: Real-time Pricing Data
    print("\n💰 Testing Real-time Pricing...")
    try:
        prices_df = get_prices()
        latest_price = prices_df.iloc[-1]
        
        print(f"✅ Retrieved {len(prices_df)} price records")
        print(f"   Latest Energy Price: ${latest_price['price']:.4f}/MWh")
        print(f"   Latest Hash Price: ${latest_price.get('hash_price', 'N/A'):.4f}")
        print(f"   Latest Token Price: ${latest_price.get('token_price', 'N/A'):.4f}")
        print(f"   Timestamp: {latest_price['timestamp']}")
        
    except Exception as e:
        print(f"❌ Pricing test failed: {e}")
    
    # Test 3: Inventory Management
    print("\n🏭 Testing Inventory Data...")
    try:
        inventory = get_inventory()
        
        print(f"✅ Inventory retrieved successfully")
        print(f"   Total Power: {inventory['power_total']:.1f} kW")
        print(f"   Available Power: {inventory['power_available']:.1f} kW")
        print(f"   Power Utilization: {(inventory['power_used']/inventory['power_total']*100):.1f}%")
        print(f"   Battery SOC: {inventory['battery_soc']*100:.1f}%")
        print(f"   GPU Utilization: {inventory['gpu_utilization']*100:.1f}%")
        print(f"   Status: {inventory['status']}")
        
        if inventory.get('alerts'):
            print(f"   🚨 Alerts: {', '.join(inventory['alerts'])}")
        
    except Exception as e:
        print(f"❌ Inventory test failed: {e}")
    
    # Test 4: Market Status
    print("\n📊 Testing Market Status...")
    try:
        market_status = get_market_status()
        
        print(f"✅ Market status retrieved")
        print(f"   Market Open: {'✅' if market_status.get('market_open') else '❌'}")
        print(f"   Grid Frequency: {market_status.get('grid_frequency', 'N/A')} Hz")
        print(f"   Emergency Status: {market_status.get('emergency_status', 'N/A')}")
        
        if 'total_power' in market_status:
            print(f"   Total Power Cost: ${market_status['total_power']:.2f}")
        if 'total_revenue' in market_status:
            print(f"   Total Revenue: ${market_status['total_revenue']:.2f}")
        
    except Exception as e:
        print(f"❌ Market status test failed: {e}")
    
    # Test 5: Sample Machine Allocation
    print("\n🤖 Testing Machine Allocation...")
    try:
        from api_client.client import submit_bid
        
        # Create a small test allocation
        test_allocation = {
            'allocation': {
                'air_miners': 1,
                'inference': 2,  # Will be scaled to asic_compute
                'training': 1,   # Will be scaled to gpu_compute
                'hydro_miners': 0,
                'immersion_miners': 1
            },
            'power_requirements': {'total_power_kw': 100},
            'system_state': {'status': 'test'}
        }
        
        result = submit_bid(test_allocation)
        
        if result['status'] == 'success':
            print(f"✅ Test allocation submitted successfully")
            print(f"   Allocation ID: {result.get('bid_id', 'N/A')}")
            print(f"   Accepted: {'✅' if result.get('allocation_accepted') else '❌'}")
        else:
            print(f"⚠️ Test allocation failed: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ Allocation test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Test Summary")
    
    if connection_test['overall_status'] == 'operational':
        print("✅ MARA API integration is READY for the hackathon!")
        print("   Your system can fetch real-time data and submit allocations.")
    else:
        print("⚠️ MARA API integration needs attention:")
        if connection_test.get('recommendations'):
            for rec in connection_test['recommendations']:
                print(f"   • {rec}")
    
    print(f"\nTest completed at: {datetime.now().isoformat()}")
    print("\n💡 Next steps:")
    print("   1. Update your API key in config.toml if needed")
    print("   2. Run the main GridPilot-GT system: python main.py")
    print("   3. Monitor real-time performance during the hackathon")


if __name__ == "__main__":
    main() 