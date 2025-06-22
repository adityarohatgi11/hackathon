#!/usr/bin/env python3
"""
Test script to verify AI analysis functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_integration.unified_interface import UnifiedLLMInterface
import json

def test_ai_analysis():
    """Test AI analysis functionality for different result types."""
    print("=" * 60)
    print("AI Analysis Functionality Test")
    print("=" * 60)
    
    # Create LLM interface
    print("1. Creating LLM interface...")
    llm_interface = UnifiedLLMInterface()
    print(f"   ✓ Provider: {llm_interface.provider}")
    print(f"   ✓ Available: {llm_interface.is_available()}")
    
    # Test different analysis types
    test_cases = [
        {
            "type": "qlearning",
            "data": {
                "episodes": 250,
                "avg_reward": 19.587,
                "best_reward": 33.856,
                "convergence": "achieved"
            },
            "prompt": """
            Analyze these Q-Learning training results and provide executive insights:
            
            Results: Episodes: 250, Average Reward: 19.587, Best Reward: 33.856
            
            Please provide:
            1. Performance assessment of the training
            2. What these metrics indicate about the learning process
            3. Recommendations for optimization
            4. Business implications for energy trading
            5. Next steps for improvement
            """
        },
        {
            "type": "stochastic",
            "data": {
                "model": "Mean Reverting",
                "accuracy": 85.2,
                "var_95": -12.5,
                "volatility": 0.15
            },
            "prompt": """
            Analyze these stochastic forecasting results and provide strategic insights:
            
            Results: Model: Mean Reverting, Accuracy: 85.2%, VaR(95%): -12.5%, Volatility: 15%
            
            Please provide:
            1. Risk assessment and market implications
            2. Forecasting accuracy and reliability
            3. Trading strategy recommendations
            4. Risk management suggestions
            5. Portfolio optimization insights
            """
        },
        {
            "type": "performance",
            "data": {
                "efficiency": 92.5,
                "utilization": 68.9,
                "cost_savings": 15000,
                "uptime": 99.2
            },
            "prompt": """
            Analyze these system performance metrics and provide operational insights:
            
            Results: Efficiency: 92.5%, Utilization: 68.9%, Cost Savings: $15,000, Uptime: 99.2%
            
            Please provide:
            1. System health and efficiency assessment
            2. Performance bottleneck identification
            3. Optimization recommendations
            4. Operational improvements
            5. Cost-benefit analysis
            """
        }
    ]
    
    print("\n2. Testing AI analysis for different result types...")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['type'].title()} Analysis ---")
        
        try:
            # Generate analysis
            insights = llm_interface.generate_insights(test_case['prompt'])
            
            print(f"✓ Analysis generated successfully")
            print(f"✓ Response length: {len(insights)} characters")
            print(f"✓ Preview: {insights[:150]}...")
            
            # Verify it contains expected sections
            expected_sections = ["**", "Analysis", "Recommendations", "Strategic"]
            found_sections = [section for section in expected_sections if section in insights]
            print(f"✓ Found {len(found_sections)}/{len(expected_sections)} expected sections")
            
        except Exception as e:
            print(f"✗ Analysis failed: {str(e)}")
    
    print("\n3. Testing service availability...")
    print(f"   ✓ Service available: {llm_interface.is_service_available()}")
    
    if hasattr(llm_interface, 'get_provider_info'):
        provider_info = llm_interface.get_provider_info()
        print(f"   ✓ Provider info: {provider_info}")
    
    print("\n4. Testing edge cases...")
    
    # Test empty prompt
    try:
        result = llm_interface.generate_insights("")
        print(f"   ✓ Empty prompt handled: {len(result)} chars")
    except Exception as e:
        print(f"   ✗ Empty prompt failed: {e}")
    
    # Test very long prompt
    try:
        long_prompt = "Analyze this energy data: " + "data point, " * 100
        result = llm_interface.generate_insights(long_prompt)
        print(f"   ✓ Long prompt handled: {len(result)} chars")
    except Exception as e:
        print(f"   ✗ Long prompt failed: {e}")
    
    print("\n" + "=" * 60)
    print("AI Analysis Test Complete!")
    print("=" * 60)
    
    # Test the specific functionality used in the dashboard
    print("\n5. Testing dashboard integration...")
    
    # Simulate dashboard button press
    sample_result_data = {
        "efficiency": 92.5,
        "utilization": 68.9,
        "revenue": 25000
    }
    
    try:
        # This mimics what happens when the Generate AI Analysis button is pressed
        prompt = f"""
        Analyze these results and provide comprehensive insights:
        
        Results: {sample_result_data}
        
        Please provide detailed analysis with actionable recommendations.
        """
        
        dashboard_result = llm_interface.generate_insights(prompt)
        print(f"   ✓ Dashboard simulation successful: {len(dashboard_result)} chars")
        print(f"   ✓ Analysis preview: {dashboard_result[:100]}...")
        
    except Exception as e:
        print(f"   ✗ Dashboard simulation failed: {e}")

if __name__ == "__main__":
    test_ai_analysis() 