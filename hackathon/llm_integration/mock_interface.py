"""
Mock LLM Interface for Testing
Provides simulated responses to test the LLM integration functionality
without requiring the actual llama-cpp-python installation.
"""

import logging
import os
from typing import Optional, Dict, Any, List
import pandas as pd
import random
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockLLM:
    """Mock LLM class that simulates llama-cpp-python functionality for testing."""
    
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        logger.info(f"Mock LLM initialized with model path: {model_path}")
    
    def create_completion(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, **kwargs):
        """Mock completion that returns simulated responses."""
        # Extract the actual query from the prompt (after "Query: ")
        query_start = prompt.find("Query: ")
        if query_start != -1:
            query = prompt[query_start + 7:].strip()
        else:
            query = prompt
        
        # Generate appropriate response based on query content
        query_lower = query.lower()
        
        if "demand response" in query_lower:
            response = "Demand response programs help reduce peak electricity usage by shifting consumption to off-peak hours. This lowers costs, improves grid stability, and reduces the need for expensive peaking power plants."
        elif "battery" in query_lower or "soc" in query_lower:
            response = "Battery management involves monitoring state of charge, optimizing charge and discharge cycles, and maintaining battery health. Proper management extends battery life and maximizes energy storage efficiency."
        elif "optimize" in query_lower or "cost" in query_lower:
            response = "Energy cost optimization includes load shifting to off-peak hours, demand response participation, and efficient equipment operation. This can reduce electricity bills by 10 to 30 percent."
        elif "risk" in query_lower:
            response = "Current energy usage patterns show several risks: high peak demand during business hours increases costs, limited flexibility to respond to price signals, and dependence on grid power without backup options."
        elif "explain" in query_lower:
            response = "This is a simulated explanation of the decision-making process. In a real scenario, the model would analyze the energy data, consider factors like demand patterns, grid conditions, and optimization goals to provide detailed reasoning for the recommended actions."
        else:
            response = "This is a simulated response from the Claude API. The model would analyze the provided context and generate relevant insights, explanations, or recommendations based on the energy management scenario."
        
        return {
            "choices": [{
                "text": response,
                "finish_reason": "length"
            }]
        }


class MockLLMInterface:
    """Mock LLM interface for testing and demonstration purposes."""
    
    def __init__(self, model_path: str = "mock-model"):
        self.model_path = model_path
        self.is_available = True
        logger.info(f"Mock LLM initialized with model path: {model_path}")
        
        # Enhanced response templates for different analysis types
        self.response_templates = {
            'qlearning': [
                """**Q-Learning Performance Analysis**

**Training Assessment:**
The training results show {convergence_status} with an average reward of {avg_reward}. This indicates {performance_level} learning efficiency for the energy trading environment.

**Learning Process Insights:**
• The agent successfully explored the state-action space with {episodes} episodes
• Convergence pattern suggests {learning_pattern} 
• Exploration-exploitation balance appears {balance_assessment}

**Business Implications:**
• Potential cost savings of 8-15% through optimized energy scheduling
• Improved market response time by 40-60%
• Enhanced risk management through predictive decision-making

**Recommendations:**
1. Increase training episodes to 500+ for better convergence
2. Implement dynamic learning rate scheduling
3. Add market volatility features to state representation
4. Deploy A/B testing framework for live validation

**Next Steps:**
Deploy the trained model in simulation environment before live trading implementation.""",

                """**Advanced Q-Learning Analysis**

**Performance Evaluation:**
The training metrics demonstrate {performance_category} learning with {reward_trend} reward progression. The agent achieved {convergence_quality} convergence patterns.

**Strategic Insights:**
• Market adaptation capability: {adaptation_level}
• Risk-adjusted returns show {risk_assessment} profile
• Operational efficiency gains: 12-18% projected

**Optimization Opportunities:**
1. Multi-agent coordination for complex scenarios
2. Transfer learning from similar market conditions
3. Ensemble methods for robust decision-making

**Implementation Roadmap:**
Phase 1: Simulation validation (2 weeks)
Phase 2: Limited live deployment (1 month)
Phase 3: Full-scale implementation (3 months)"""
            ],
            
            'stochastic': [
                """**Stochastic Forecasting Analysis**

**Model Performance:**
The {model_type} model achieved {accuracy}% forecast accuracy with {confidence_level} confidence intervals. This represents {performance_grade} predictive capability for energy market conditions.

**Risk Assessment:**
• VaR analysis indicates {var_interpretation} downside risk exposure
• Expected returns show {return_assessment} potential
• Volatility patterns suggest {volatility_insight} market conditions

**Market Implications:**
• Price forecasting reliability: {forecast_reliability}
• Optimal trading windows identified for next {horizon} hours
• Risk-adjusted position sizing recommendations available

**Strategic Recommendations:**
1. Implement dynamic hedging based on volatility forecasts
2. Optimize portfolio allocation using Monte Carlo insights
3. Establish stop-loss triggers at {risk_threshold}% levels
4. Consider correlation trading opportunities

**Risk Management:**
Deploy real-time monitoring with {monitoring_frequency} model updates to maintain forecast accuracy.""",

                """**Advanced Stochastic Model Insights**

**Forecast Reliability:**
The stochastic differential equation model shows {reliability_score} reliability with {error_bounds} prediction error bounds.

**Market Dynamics:**
• Mean reversion strength: {reversion_strength}
• Jump diffusion probability: {jump_probability}
• Volatility clustering: {clustering_pattern}

**Trading Strategy:**
Optimal execution strategy suggests {execution_timing} with {position_sizing} risk management.

**Performance Optimization:**
1. Calibrate model parameters weekly
2. Implement regime-switching enhancements  
3. Add macroeconomic indicators to feature set
4. Develop ensemble forecasting framework"""
            ],
            
            'auction': [
                """**Auction Mechanism Analysis**

**Efficiency Assessment:**
The {auction_type} auction achieved {efficiency_score}% efficiency with {revenue_performance} revenue generation. Market dynamics show {competition_level} competitive environment.

**Strategic Insights:**
• Bidding strategy effectiveness: {bidding_effectiveness}
• Price discovery mechanism: {price_discovery_quality}
• Market liquidity: {liquidity_assessment}

**Revenue Optimization:**
• Current auction design captures {revenue_capture}% of theoretical maximum
• Reserve price optimization could improve returns by {improvement_potential}%
• Multi-round mechanisms show {multi_round_benefit} advantages

**Competitive Positioning:**
1. Implement dynamic reserve pricing
2. Develop bidder behavior analytics
3. Optimize auction timing for maximum participation
4. Consider combinatorial auction formats for complex assets

**Market Intelligence:**
Bidder analysis reveals {bidder_patterns} with {market_concentration} concentration levels.""",

                """**Advanced Auction Strategy**

**Market Efficiency:**
Auction results demonstrate {market_efficiency} price discovery with {fairness_score} fairness metrics.

**Revenue Analysis:**
• Seller surplus: {seller_surplus}
• Buyer surplus: {buyer_surplus}  
• Total welfare: {total_welfare}

**Mechanism Design:**
The {mechanism_type} mechanism optimizes for {optimization_target} with {trade_off_analysis} trade-offs.

**Strategic Recommendations:**
1. Implement second-price sealed-bid for truthful bidding
2. Add participation incentives for increased liquidity
3. Consider dynamic auction formats for volatile markets
4. Develop reputation systems for repeat participants"""
            ],
            
            'mpc': [
                """**Model Predictive Control Analysis**

**Optimization Performance:**
The MPC controller achieved {cost_reduction}% cost reduction with {efficiency_gain}% efficiency improvement over the {horizon}-hour horizon.

**Control Strategy:**
• Optimal energy allocation: {energy_allocation}
• Peak demand management: {peak_management}
• Battery degradation mitigation: {degradation_control}

**Operational Insights:**
• System response time: {response_time} 
• Constraint satisfaction: {constraint_satisfaction}%
• Robustness to disturbances: {robustness_level}

**Cost-Benefit Analysis:**
• Annual savings projection: ${savings_projection:,}
• ROI timeline: {roi_timeline} months
• Risk-adjusted NPV: ${npv_projection:,}

**Recommendations:**
1. Extend prediction horizon during stable periods
2. Implement adaptive constraint handling
3. Add weather forecast integration
4. Develop emergency response protocols

**Parameter Tuning:**
Optimal degradation weight: {optimal_weight} for current market conditions.""",

                """**Advanced MPC Optimization**

**Control Performance:**
The predictive controller demonstrates {control_quality} performance with {tracking_accuracy}% setpoint tracking accuracy.

**Economic Optimization:**
• Operating cost reduction: {cost_optimization}%
• Energy efficiency gain: {efficiency_improvement}%
• Demand charge savings: {demand_savings}%

**Technical Analysis:**
• Constraint violation rate: {violation_rate}%
• Computational efficiency: {computation_time}ms per cycle
• Model prediction accuracy: {prediction_accuracy}%

**Strategic Recommendations:**
1. Implement model-plant mismatch detection
2. Add robust optimization for uncertainty handling
3. Develop adaptive control strategies
4. Integrate machine learning for parameter tuning"""
            ],
            
            'performance': [
                """**System Performance Analysis**

**Overall Assessment:**
Current system performance achieves {overall_score}% efficiency with {performance_trend} operational trends. The energy management system demonstrates {efficiency_assessment} performance across key metrics.

**Key Performance Indicators:**
• Energy Efficiency: {energy_efficiency}
• Cost Optimization: {cost_optimization}
• Revenue Generation: {revenue_generation}
• System Reliability: {reliability_score}

**Performance Analysis:**
• Efficiency Assessment: {efficiency_assessment}
• Cost Management: {cost_assessment}
• Revenue Performance: {revenue_assessment}
• Reliability Status: {reliability_assessment}

**Optimization Opportunities:**
• Primary Focus: {primary_bottleneck} optimization
• Secondary Opportunity: {secondary_opportunity} enhancement
• Improvement Strategy: {optimization_strategy} implementation

**Financial Impact:**
• ROI Projection: {roi_projection}% annual return
• Payback Period: {payback_period} months
• Performance Index: {performance_index}

**Strategic Recommendations:**
1. Target {improvement_area} through {improvement_method}
2. Implement {monitoring_solution} for enhanced visibility
3. Establish {benchmark_framework} for continuous improvement
4. Focus on {target_metric} optimization for maximum impact"""
            ],
            
            'general': [
                """**Energy Management Analysis**

**Current System Status:**
The energy management system demonstrates {performance_level} operational characteristics with {trend_direction} performance trends. Overall system efficiency maintains {confidence_level} reliability levels.

**Key Insights:**
• System performance indicators show {performance_trend} patterns
• Energy utilization efficiency remains within optimal parameters
• Cost optimization strategies are delivering expected results
• Risk management protocols are functioning effectively

**Strategic Recommendations:**
1. Continue monitoring key performance indicators for optimization opportunities
2. Implement predictive analytics for enhanced decision-making capabilities
3. Consider advanced automation for improved operational efficiency
4. Develop comprehensive reporting dashboards for stakeholder visibility

**Market Analysis:**
Current energy market conditions suggest {trend_direction} pricing trends with moderate volatility. Recommended strategies include demand response optimization and strategic energy procurement timing.

**Next Steps:**
1. Conduct detailed performance review with operational teams
2. Implement enhanced monitoring and alerting systems
3. Develop long-term optimization roadmap
4. Schedule regular system health assessments

**Business Impact:**
The analysis indicates potential for 10-15% efficiency improvements through targeted optimization strategies and enhanced monitoring capabilities.""",

                """**Comprehensive Energy Insights**

**Executive Summary:**
Energy management operations demonstrate solid performance with identifiable optimization opportunities. Current systems maintain {performance_level} efficiency levels with {trend_direction} operational trends.

**Operational Excellence:**
• System reliability maintains industry-leading standards
• Energy efficiency metrics exceed baseline expectations
• Cost management strategies deliver consistent results
• Risk mitigation protocols provide adequate protection

**Strategic Opportunities:**
1. Advanced analytics implementation for predictive insights
2. Machine learning integration for automated optimization
3. Real-time monitoring enhancement for proactive management
4. Stakeholder dashboard development for improved visibility

**Financial Performance:**
Current operations demonstrate strong financial performance with opportunities for 12-18% efficiency improvements through strategic optimization initiatives.

**Innovation Roadmap:**
Phase 1: Enhanced monitoring and analytics (3 months)
Phase 2: Automation and optimization (6 months)  
Phase 3: Advanced AI integration (12 months)"""
            ]
        }
    
    def generate_insights(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate contextual insights based on the prompt and analysis type."""
        try:
            # Determine analysis type from prompt
            analysis_type = self._detect_analysis_type(prompt)
            
            # Extract data from prompt for contextualization
            data_context = self._extract_data_context(prompt)
            
            # Select appropriate template
            templates = self.response_templates.get(analysis_type, self.response_templates['performance'])
            template = random.choice(templates)
            
            # Generate contextual values
            context_values = self._generate_context_values(analysis_type, data_context)
            
            # Format template with context
            response = template.format(**context_values)
            
            logger.info(f"Generated {analysis_type} insights with {len(context_values)} context variables")
            return response
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return self._fallback_response(prompt)
    
    def _detect_analysis_type(self, prompt: str) -> str:
        """Detect the type of analysis from the prompt."""
        prompt_lower = prompt.lower()
        
        if 'q-learning' in prompt_lower or 'qlearning' in prompt_lower:
            return 'qlearning'
        elif 'stochastic' in prompt_lower or 'forecast' in prompt_lower:
            return 'stochastic'
        elif 'auction' in prompt_lower or 'bidding' in prompt_lower:
            return 'auction'
        elif 'mpc' in prompt_lower or 'predictive control' in prompt_lower:
            return 'mpc'
        elif 'performance' in prompt_lower or 'metrics' in prompt_lower:
            return 'performance'
        else:
            return 'general'
    
    def _extract_data_context(self, prompt: str) -> Dict[str, Any]:
        """Extract relevant data context from the prompt."""
        context = {}
        
        # Extract numerical values
        import re
        numbers = re.findall(r'\d+\.?\d*', prompt)
        if numbers:
            context['extracted_numbers'] = [float(n) for n in numbers[:5]]
        
        # Extract key terms
        if 'Episodes' in prompt:
            context['has_episodes'] = True
        if 'Reward' in prompt:
            context['has_rewards'] = True
        if 'Efficiency' in prompt:
            context['has_efficiency'] = True
            
        return context
    
    def _generate_context_values(self, analysis_type: str, data_context: Dict[str, Any]) -> Dict[str, str]:
        """Generate contextual values for template formatting."""
        base_values = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'confidence_level': random.choice(['95%', '90%', '99%']),
            'performance_level': random.choice(['excellent', 'good', 'satisfactory', 'needs improvement']),
            'trend_direction': random.choice(['improving', 'stable', 'declining']),
        }
        
        if analysis_type == 'qlearning':
            base_values.update({
                'convergence_status': random.choice(['strong convergence', 'moderate convergence', 'slow convergence']),
                'avg_reward': f"{random.uniform(18, 25):.2f}",
                'episodes': random.choice(['100', '250', '500']),
                'learning_pattern': random.choice(['rapid initial learning', 'steady improvement', 'plateau behavior']),
                'balance_assessment': random.choice(['well-balanced', 'exploration-heavy', 'exploitation-focused']),
                'performance_category': random.choice(['strong', 'moderate', 'developing']),
                'reward_trend': random.choice(['increasing', 'stabilizing', 'fluctuating']),
                'convergence_quality': random.choice(['excellent', 'good', 'acceptable']),
                'adaptation_level': random.choice(['high', 'moderate', 'developing']),
                'risk_assessment': random.choice(['conservative', 'balanced', 'aggressive'])
            })
        
        elif analysis_type == 'stochastic':
            base_values.update({
                'model_type': random.choice(['Mean Reverting', 'Geometric Brownian', 'Jump Diffusion', 'Heston']),
                'accuracy': f"{random.uniform(78, 92):.1f}",
                'performance_grade': random.choice(['excellent', 'strong', 'satisfactory']),
                'var_interpretation': random.choice(['moderate', 'elevated', 'controlled']),
                'return_assessment': random.choice(['attractive', 'moderate', 'conservative']),
                'volatility_insight': random.choice(['stable', 'elevated', 'decreasing']),
                'forecast_reliability': random.choice(['high', 'moderate', 'developing']),
                'horizon': random.choice(['24', '48', '72']),
                'risk_threshold': f"{random.uniform(5, 15):.1f}",
                'monitoring_frequency': random.choice(['hourly', 'daily', 'real-time']),
                'reliability_score': random.choice(['high', 'moderate', 'acceptable']),
                'error_bounds': f"±{random.uniform(3, 8):.1f}%",
                'reversion_strength': random.choice(['strong', 'moderate', 'weak']),
                'jump_probability': f"{random.uniform(0.1, 0.3):.2f}",
                'clustering_pattern': random.choice(['significant', 'moderate', 'minimal']),
                'execution_timing': random.choice(['immediate', 'delayed', 'scheduled']),
                'position_sizing': random.choice(['conservative', 'moderate', 'aggressive'])
            })
        
        elif analysis_type == 'auction':
            base_values.update({
                'auction_type': random.choice(['Second Price', 'First Price', 'VCG', 'Combinatorial']),
                'efficiency_score': f"{random.uniform(85, 95):.1f}",
                'revenue_performance': random.choice(['strong', 'moderate', 'developing']),
                'competition_level': random.choice(['high', 'moderate', 'limited']),
                'bidding_effectiveness': random.choice(['optimal', 'good', 'suboptimal']),
                'price_discovery_quality': random.choice(['excellent', 'good', 'fair']),
                'liquidity_assessment': random.choice(['high', 'moderate', 'limited']),
                'revenue_capture': f"{random.uniform(80, 95):.1f}",
                'improvement_potential': f"{random.uniform(5, 15):.1f}",
                'multi_round_benefit': random.choice(['significant', 'moderate', 'minimal']),
                'bidder_patterns': random.choice(['strategic', 'competitive', 'conservative']),
                'market_concentration': random.choice(['low', 'moderate', 'high']),
                'market_efficiency': random.choice(['high', 'moderate', 'developing']),
                'fairness_score': random.choice(['excellent', 'good', 'acceptable']),
                'seller_surplus': f"${random.uniform(1000, 3000):,.0f}",
                'buyer_surplus': f"${random.uniform(500, 1500):,.0f}",
                'total_welfare': f"${random.uniform(2000, 5000):,.0f}",
                'mechanism_type': random.choice(['sealed-bid', 'open-outcry', 'Dutch', 'English']),
                'optimization_target': random.choice(['revenue', 'efficiency', 'fairness']),
                'trade_off_analysis': random.choice(['acceptable', 'optimal', 'suboptimal'])
            })
        
        elif analysis_type == 'mpc':
            base_values.update({
                'cost_reduction': f"{random.uniform(10, 20):.1f}",
                'efficiency_gain': f"{random.uniform(5, 15):.1f}",
                'horizon': random.choice(['12', '24', '48']),
                'energy_allocation': random.choice(['optimal', 'near-optimal', 'suboptimal']),
                'peak_management': random.choice(['excellent', 'good', 'adequate']),
                'degradation_control': random.choice(['effective', 'moderate', 'limited']),
                'response_time': f"{random.uniform(0.1, 0.5):.2f}s",
                'constraint_satisfaction': f"{random.uniform(95, 99):.1f}",
                'robustness_level': random.choice(['high', 'moderate', 'acceptable']),
                'savings_projection': f"{random.uniform(50000, 150000):.0f}",
                'roi_timeline': random.choice(['6-8', '8-12', '12-18']),
                'npv_projection': f"{random.uniform(200000, 500000):.0f}",
                'optimal_weight': f"{random.uniform(0.3, 0.7):.2f}",
                'control_quality': random.choice(['excellent', 'good', 'satisfactory']),
                'tracking_accuracy': f"{random.uniform(92, 98):.1f}",
                'cost_optimization': f"{random.uniform(8, 18):.1f}",
                'efficiency_improvement': f"{random.uniform(10, 20):.1f}",
                'demand_savings': f"{random.uniform(15, 25):.1f}",
                'violation_rate': f"{random.uniform(0.1, 2.0):.1f}",
                'computation_time': f"{random.uniform(10, 50):.0f}",
                'prediction_accuracy': f"{random.uniform(88, 96):.1f}"
            })
        
        elif analysis_type == 'performance':
            base_values.update({
                'overall_score': f"{random.uniform(85, 95):.1f}",
                'performance_trend': random.choice(['improving', 'stable', 'mixed']),
                'energy_efficiency': f"{random.uniform(88, 95):.1f}%",
                'cost_optimization': f"{random.uniform(85, 92):.1f}%",
                'revenue_generation': f"{random.uniform(90, 97):.1f}%",
                'reliability_score': f"{random.uniform(98, 99.5):.1f}%",
                'efficiency_assessment': random.choice(['excellent', 'strong', 'good']),
                'cost_assessment': random.choice(['optimized', 'efficient', 'acceptable']),
                'revenue_assessment': random.choice(['outstanding', 'strong', 'satisfactory']),
                'reliability_assessment': random.choice(['exceptional', 'excellent', 'very good']),
                'primary_bottleneck': random.choice(['energy efficiency', 'cost optimization', 'system latency']),
                'secondary_opportunity': random.choice(['demand forecasting', 'battery management', 'grid integration']),
                'improvement_area': random.choice(['cost optimization', 'energy efficiency', 'system reliability']),
                'improvement_method': random.choice(['advanced algorithms', 'hardware upgrades', 'process optimization']),
                'optimization_strategy': random.choice(['machine learning', 'predictive analytics', 'real-time optimization']),
                'target_metric': random.choice(['efficiency', 'cost reduction', 'reliability']),
                'monitoring_solution': random.choice(['IoT sensors', 'analytics dashboard', 'AI monitoring']),
                'benchmark_framework': random.choice(['KPI tracking', 'performance scorecards', 'continuous monitoring']),
                'roi_projection': f"{random.uniform(15, 35):.0f}",
                'payback_period': random.choice(['6-8', '8-12', '12-18']),
                'performance_index': f"{random.uniform(0.85, 0.95):.2f}",
                'benchmark_comparison': random.choice(['above average', 'industry leading', 'competitive']),
                'availability': f"{random.uniform(98, 99.8):.1f}",
                'efficiency_score': f"{random.uniform(88, 96):.1f}",
                'cost_per_kwh': f"{random.uniform(0.08, 0.15):.3f}",
                'revenue_per_mw': f"{random.uniform(50, 80):.0f}",
                'trajectory_analysis': random.choice(['positive', 'stable', 'improving']),
                'maintenance_forecast': random.choice(['on schedule', 'optimized', 'predictive']),
                'capacity_insights': random.choice(['well-utilized', 'optimization potential', 'balanced'])
            })
        
        # For general analysis type, use performance values as baseline
        elif analysis_type == 'general':
            base_values.update({
                'overall_score': f"{random.uniform(85, 95):.1f}",
                'performance_trend': random.choice(['improving', 'stable', 'mixed']),
                'energy_efficiency': f"{random.uniform(88, 95):.1f}%",
                'cost_optimization': f"{random.uniform(85, 92):.1f}%",
                'revenue_generation': f"{random.uniform(90, 97):.1f}%",
                'reliability_score': f"{random.uniform(98, 99.5):.1f}%",
                'efficiency_assessment': random.choice(['excellent', 'strong', 'good']),
                'cost_assessment': random.choice(['optimized', 'efficient', 'acceptable']),
                'revenue_assessment': random.choice(['outstanding', 'strong', 'satisfactory']),
                'reliability_assessment': random.choice(['exceptional', 'excellent', 'very good'])
            })
        
        return base_values
    
    def _fallback_response(self, prompt: str) -> str:
        """Provide a fallback response when analysis fails."""
        return f"""**Analysis Summary**

Based on the provided data, the system shows operational performance within expected parameters.

**Key Observations:**
• Current metrics indicate stable system operation
• Performance trends suggest continued reliability
• Optimization opportunities exist for enhanced efficiency

**Recommendations:**
1. Continue monitoring key performance indicators
2. Implement regular system health checks
3. Consider advanced analytics for deeper insights
4. Establish baseline metrics for future comparisons

**Next Steps:**
Schedule detailed analysis with domain experts for comprehensive evaluation.

*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"""
    
    def is_service_available(self) -> bool:
        """Check if the mock service is available."""
        return self.is_available
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            "model_name": "Mock Analysis Engine",
            "version": "1.0.0",
            "capabilities": [
                "Q-Learning Analysis",
                "Stochastic Forecasting",
                "Auction Mechanism Analysis", 
                "MPC Optimization Analysis",
                "Performance Analytics"
            ],
            "status": "operational"
        }


def test_mock_interface():
    """Test the mock LLM interface with sample data."""
    print("Testing Mock LLM Interface...")
    
    # Initialize interface
    interface = MockLLMInterface()
    print(f"Model available: {interface.is_service_available()}")
    
    # Test basic query
    print("\n--- Testing Basic Query ---")
    response = interface.generate_insights("What are the benefits of demand response in energy management?")
    print(f"Response: {response}")
    
    # Test with context
    print("\n--- Testing with Context ---")
    context = {
        "current_demand": "500 kW",
        "grid_price": "$0.15/kWh",
        "battery_charge": "80%"
    }
    response = interface.generate_insights("Should we discharge the battery now?", context)
    print(f"Response: {response}")
    
    # Test insights generation
    print("\n--- Testing Insights Generation ---")
    # Create sample energy data
    energy_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=24, freq='H'),
        'consumption': [100 + i*10 for i in range(24)],
        'demand': [150 + i*5 for i in range(24)]
    })
    
    insights = interface.generate_insights("Analyze this energy data and provide key insights about consumption patterns, efficiency opportunities, and optimization recommendations.", energy_data)
    print(f"Insights: {insights}")
    
    # Test decision explanation
    print("\n--- Testing Decision Explanation ---")
    decision_context = {
        "decision": "Discharge battery to reduce grid demand",
        "current_price": "$0.20/kWh",
        "battery_efficiency": "95%",
        "demand_reduction": "50 kW"
    }
    
    explanation = interface.generate_insights("Explain why the following decision was made: Discharge battery", decision_context)
    print(f"Explanation: {explanation}")
    
    # Test different query types
    print("\n--- Testing Different Query Types ---")
    queries = [
        "What are the current energy prices?",
        "How can we optimize our energy consumption?",
        "Explain the battery management strategy",
        "What are the risks of current energy usage patterns?"
    ]
    
    for query in queries:
        response = interface.generate_insights(query)
        print(f"Q: {query}")
        print(f"A: {response}\n")


if __name__ == "__main__":
    test_mock_interface() 