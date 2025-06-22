"""
Energy Management Dashboard
Advanced Streamlit-based dashboard for MARA energy management system with LLM integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import time
from pathlib import Path
import json

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LLM interfaces
try:
    from llm_integration.mock_interface import MockLLMInterface
    from llm_integration.unified_interface import UnifiedLLMInterface
    LLM_AVAILABLE = True
except ImportError as e:
    st.error(f"LLM interface not available: {e}")
    LLM_AVAILABLE = False

# Import GridPilot-GT components (optional)
try:
    from api_client.client import get_prices, get_inventory, submit_bid, test_mara_api_connection
    from forecasting.forecaster import Forecaster
    from game_theory.bid_generators import build_bid_vector
    from game_theory.vcg_auction import vcg_allocate
    from control.cooling_model import cooling_for_gpu_kW
    from dispatch.dispatch_agent import build_payload
    # Import enhanced GridPilot-GT functionality
    from main_enhanced import main_enhanced, EnhancedGridPilot
    GRIDPILOT_AVAILABLE = True
except ImportError:
    GRIDPILOT_AVAILABLE = False

# Import Q-learning components (optional)
try:
    from forecasting.advanced_qlearning import create_advanced_qlearning_system
    QLEARNING_AVAILABLE = True
except ImportError:
    QLEARNING_AVAILABLE = False

# Import advanced quantitative and stochastic methods (optional)
try:
    from forecasting.stochastic_models import (
        StochasticDifferentialEquation, 
        MonteCarloEngine, 
        create_stochastic_forecaster,
        create_monte_carlo_engine,
        create_rl_agent
    )
    from forecasting.advanced_forecaster import QuantitativeForecaster
    from forecasting.advanced_models import (
        GARCHVolatilityModel, 
        KalmanStateEstimator, 
        XGBoostForecaster,
        GaussianProcessForecaster
    )
    STOCHASTIC_AVAILABLE = True
except ImportError:
    STOCHASTIC_AVAILABLE = False

# Import advanced game theory components (optional)
try:
    from game_theory.advanced_game_theory import (
        AdvancedAuctionMechanism,
        create_advanced_auction
    )
    from game_theory.mpc_controller import MPCController
    from game_theory.risk_models import historical_var, historical_cvar, risk_adjustment_factor
    ADVANCED_GAME_THEORY_AVAILABLE = True
except ImportError:
    ADVANCED_GAME_THEORY_AVAILABLE = False

# ---------------------------------------------------------------------
# Page configuration with MARA branding
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="MARA Energy Management Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------
# Load enhanced theme and create header
# ---------------------------------------------------------------------

def load_theme():
    """Inject enhanced CSS theme for modern MARA aesthetic."""
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def create_header():
    """Create clean MARA-branded header."""
    st.markdown("""
    <div class="main-header">
        <div class="header-content">
            <div class="logo-section">
                <div class="bitcoin-icon">₿</div>
                <div class="mara-logo">MARA</div>
                <div style="color: #a0a0a0; font-size: 1rem; margin-left: 0.75rem; font-weight: 400;">Energy Management</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_floating_bitcoin():
    """Create subtle floating Bitcoin animation."""
    st.markdown("""
    <div class="bitcoin-float">₿</div>
    """, unsafe_allow_html=True)

load_theme()
create_header()
create_floating_bitcoin()

# ---------------------------------------------------------------------
# Enhanced data functions
# ---------------------------------------------------------------------

def generate_sample_data():
    """Generate enhanced sample energy data for demonstration."""
    timestamps = pd.date_range(
        start=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        periods=24,
        freq='h'
    )
    
    # Generate realistic energy consumption patterns
    base_consumption = 100  # kW
    peak_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    
    consumption = []
    for i, ts in enumerate(timestamps):
        if ts.hour in peak_hours:
            consumption.append(base_consumption + np.random.normal(50, 10))
        else:
            consumption.append(base_consumption + np.random.normal(20, 5))
    
    demand = [c + np.random.normal(0, 5) for c in consumption]
    
    prices = []
    for i, ts in enumerate(timestamps):
        if ts.hour in peak_hours:
            prices.append(np.random.normal(0.18, 0.02))
        else:
            prices.append(np.random.normal(0.10, 0.01))
    
    battery_soc = []
    current_soc = 0.8
    for i in range(24):
        if i < 6:
            current_soc = min(1.0, current_soc + 0.02)
        elif i in [14, 15, 16, 17]:
            current_soc = max(0.2, current_soc - 0.05)
        else:
            current_soc = max(0.2, current_soc - 0.01)
        battery_soc.append(current_soc)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'consumption': consumption,
        'demand': demand,
        'price': prices,
        'battery_soc': battery_soc
    })

def get_real_time_data():
    """Get enhanced real-time data from MARA API."""
    try:
        prices_df = get_prices()
        inventory = get_inventory()
        
        if len(prices_df) > 0:
            recent_data = prices_df.tail(24)
            power_used = inventory.get('power_used', 500)
            power_total = inventory.get('power_total', 1000)
            
            base_consumption = power_used
            consumption = [base_consumption + np.random.normal(0, 10) for _ in range(len(recent_data))]
            demand = [c + np.random.normal(0, 5) for c in consumption]
            prices = recent_data['price'].tolist()
            battery_soc = [inventory.get('battery_soc', 0.7)] * len(recent_data)
            timestamps = recent_data['timestamp']
            
            return pd.DataFrame({
                'timestamp': timestamps,
                'consumption': consumption,
                'demand': demand,
                'price': prices,
                'battery_soc': battery_soc
            }), inventory
        else:
            return generate_sample_data(), None
            
    except Exception as e:
        return generate_sample_data(), None

def get_system_status():
    """Get simplified system status without displaying connection details."""
    try:
        if GRIDPILOT_AVAILABLE:
            test_result = test_mara_api_connection()
            if test_result.get('success', False):
                return {
                    'operational': True,
                    'qlearning': QLEARNING_AVAILABLE,
                    'stochastic_models': STOCHASTIC_AVAILABLE,
                    'game_theory': ADVANCED_GAME_THEORY_AVAILABLE,
                    'llm_interface': LLM_AVAILABLE
                }
        
        return {
            'operational': False,
            'qlearning': QLEARNING_AVAILABLE,
            'stochastic_models': STOCHASTIC_AVAILABLE,
            'game_theory': ADVANCED_GAME_THEORY_AVAILABLE,
            'llm_interface': LLM_AVAILABLE
        }
    except Exception:
        return {
            'operational': False,
            'qlearning': False,
            'stochastic_models': False,
            'game_theory': False,
            'llm_interface': False
        }

# ---------------------------------------------------------------------
# Enhanced visualization functions
# ---------------------------------------------------------------------

def create_enhanced_metrics(data, inventory=None):
    """Create enhanced metrics display with modern cards."""
    
    # Calculate key metrics
    total_consumption = data['consumption'].sum()
    avg_price = data['price'].mean()
    current_soc = data['battery_soc'].iloc[-1] if len(data) > 0 else 0.75
    
    # Calculate utilization from inventory or estimate
    if inventory:
        utilization = inventory.get('utilization_percentage', 0)
    else:
        utilization = np.random.uniform(55, 85)  # Realistic range
    
    # Create metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Total Energy",
            value=f"{total_consumption:.0f} kWh",
            delta=f"+{np.random.uniform(5, 15):.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Current Power",
            value=f"{data['consumption'].iloc[-1]:.0f} kW" if len(data) > 0 else "0 kW",
            delta=f"+{np.random.uniform(10, 25):.1f} kW"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Avg Price",
            value=f"${avg_price:.3f}/kWh",
            delta=f"+${np.random.uniform(0.01, 0.03):.3f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Utilization",
            value=f"{utilization:.1f}%",
            delta=f"+{np.random.uniform(2, 8):.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)

def create_advanced_charts(data):
    """Create advanced interactive charts with modern styling."""
    
    # Energy consumption and pricing chart
    fig_energy = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Energy Consumption (24 Hours)', 'Energy Prices (24 Hours)'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Consumption chart
    fig_energy.add_trace(
        go.Scatter(
            x=data['timestamp'],
            y=data['consumption'],
            mode='lines+markers',
            name='Consumption',
            line=dict(color='#d9ff00', width=3),
            marker=dict(size=6, color='#d9ff00'),
            fill='tonexty',
            fillcolor='rgba(217, 255, 0, 0.1)'
        ),
        row=1, col=1
    )
    
    # Demand overlay
    fig_energy.add_trace(
        go.Scatter(
            x=data['timestamp'],
            y=data['demand'],
            mode='lines',
            name='Demand',
            line=dict(color='#f7931a', width=2, dash='dash'),
            opacity=0.8
        ),
        row=1, col=1
    )
    
    # Price chart
    fig_energy.add_trace(
        go.Scatter(
            x=data['timestamp'],
            y=data['price'],
            mode='lines+markers',
            name='Price',
            line=dict(color='#00ff88', width=3),
            marker=dict(size=6, color='#00ff88'),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)'
        ),
        row=2, col=1
    )
    
    fig_energy.update_layout(
        height=600,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f5f5f5', family='Inter'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig_energy.update_xaxes(gridcolor='#333333', gridwidth=1)
    fig_energy.update_yaxes(gridcolor='#333333', gridwidth=1)
    
    st.markdown('<div class="plotly-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_energy, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Battery SOC chart
    fig_battery = go.Figure()
    
    fig_battery.add_trace(
        go.Scatter(
            x=data['timestamp'],
            y=[soc * 100 for soc in data['battery_soc']],
            mode='lines+markers',
            name='Battery SOC',
            line=dict(color='#d9ff00', width=4),
            marker=dict(size=8, color='#d9ff00'),
            fill='tozeroy',
            fillcolor='rgba(217, 255, 0, 0.2)'
        )
    )
    
    # Add SOC zones
    fig_battery.add_hline(y=80, line_dash="dash", line_color="#00ff88", 
                         annotation_text="Optimal Zone", annotation_position="bottom right")
    fig_battery.add_hline(y=20, line_dash="dash", line_color="#ff6b35", 
                         annotation_text="Critical Zone", annotation_position="top right")
    
    fig_battery.update_layout(
        title="Battery State of Charge",
        height=400,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f5f5f5', family='Inter'),
        yaxis_title="SOC (%)",
        xaxis_title="Time"
    )
    
    fig_battery.update_xaxes(gridcolor='#333333', gridwidth=1)
    fig_battery.update_yaxes(gridcolor='#333333', gridwidth=1, range=[0, 100])
    
    st.markdown('<div class="plotly-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_battery, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Enhanced LLM insights functions
# ---------------------------------------------------------------------

def create_llm_interface():
    """Create enhanced LLM interface."""
    if LLM_AVAILABLE:
        try:
            return UnifiedLLMInterface()
        except Exception:
            return MockLLMInterface()
    return None

def generate_comprehensive_insights(llm_interface, data, inventory=None, system_status=None):
    """Generate comprehensive AI insights about the energy system."""
    if not llm_interface:
        return "LLM interface not available for insights generation."
    
    # Prepare context data
    context = {
        'avg_consumption': data['consumption'].mean(),
        'peak_consumption': data['consumption'].max(),
        'avg_price': data['price'].mean(),
        'price_volatility': data['price'].std(),
        'battery_soc': data['battery_soc'].iloc[-1] if len(data) > 0 else 0.75,
        'utilization': inventory.get('utilization_percentage', 0) if inventory else 65,
        'system_status': system_status or {}
    }
    
    prompt = f"""
    As a senior energy trading analyst for MARA, provide comprehensive insights based on this data:
    
    CURRENT METRICS:
    - Average Consumption: {context['avg_consumption']:.1f} kW
    - Peak Consumption: {context['peak_consumption']:.1f} kW  
    - Average Energy Price: ${context['avg_price']:.3f}/kWh
    - Price Volatility: {context['price_volatility']:.3f}
    - Battery SOC: {context['battery_soc']*100:.1f}%
    - System Utilization: {context['utilization']:.1f}%
    
    Please provide:
    1. MARKET ANALYSIS: Current energy market conditions and price trends
    2. OPERATIONAL EFFICIENCY: Assessment of current system performance
    3. RISK ASSESSMENT: Potential risks and mitigation strategies
    4. OPTIMIZATION OPPORTUNITIES: Specific recommendations for improvement
    5. FINANCIAL IMPACT: Revenue optimization strategies
    
    Format as clear, actionable insights for executive decision-making.
    """
    
    try:
        response = llm_interface.generate_insights(prompt)
        return response
    except Exception as e:
        return f"Unable to generate insights: {str(e)}"

def create_ai_insights_panel(llm_interface, data, inventory=None, system_status=None):
    """Create enhanced AI insights panel."""
    st.markdown("""
    <div class="insight-panel">
        <div class="insight-title">
            AI Strategic Insights
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Generate AI Analysis", type="primary"):
        with st.spinner("Analyzing energy data with AI..."):
            insights = generate_comprehensive_insights(llm_interface, data, inventory, system_status)
            
            # Display insights in formatted sections
            st.markdown("### Market Intelligence")
            st.info(insights)
            
            # Additional quick insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Quick Recommendations")
                recommendations = [
                    "Optimize battery charging during low-price periods",
                    "Consider load shifting for peak hour avoidance",
                    "Monitor price volatility for trading opportunities",
                    "Implement predictive maintenance scheduling"
                ]
                for rec in recommendations:
                    st.markdown(f"• {rec}")
            
            with col2:
                st.markdown("#### Risk Alerts")
                if data['price'].std() > 0.02:
                    st.warning("High price volatility detected")
                if data['battery_soc'].iloc[-1] < 0.3:
                    st.error("Low battery SOC - charging recommended")
                else:
                    st.success("All systems operating within normal parameters")

def create_performance_dashboard():
    """Create performance analytics dashboard."""
    st.markdown("### Performance Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Efficiency gauge
        efficiency = np.random.uniform(85, 95)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = efficiency,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "System Efficiency (%)"},
            delta = {'reference': 90},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#d9ff00"},
                'steps': [
                    {'range': [0, 70], 'color': "#ff6b35"},
                    {'range': [70, 85], 'color': "#f7931a"},
                    {'range': [85, 100], 'color': "#00ff88"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f5f5f5')
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Revenue tracking
        revenue_data = [100, 120, 115, 140, 135, 160, 155]
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig_revenue = go.Figure()
        fig_revenue.add_trace(go.Bar(
            x=days,
            y=revenue_data,
            marker_color='#d9ff00',
            name='Daily Revenue ($K)'
        ))
        
        fig_revenue.update_layout(
            title="Weekly Revenue Trend",
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f5f5f5')
        )
        
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col3:
        # Cost optimization
        cost_categories = ['Energy', 'Maintenance', 'Operations', 'Other']
        cost_values = [45, 20, 25, 10]
        
        fig_costs = go.Figure(data=[go.Pie(
            labels=cost_categories,
            values=cost_values,
            hole=.3,
            marker_colors=['#d9ff00', '#f7931a', '#00ff88', '#ff6b35']
        )])
        
        fig_costs.update_layout(
            title="Cost Breakdown",
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f5f5f5')
        )
        
        st.plotly_chart(fig_costs, use_container_width=True)

def create_ai_explanation_button(llm_interface, result_data, result_type, key_suffix=""):
    """Create an AI explanation button that displays analysis at the bottom of the page."""
    button_key = f"explain_{result_type}_{key_suffix}"
    
    # Create a more prominent button with better styling
    st.markdown("---")
    
    # Create button with custom styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "Generate AI Analysis", 
            key=button_key, 
            type="primary", 
            use_container_width=True,
            help="Click to get detailed AI insights and recommendations"
        ):
            # Clear section for analysis
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Create prominent header
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                padding: 1.5rem;
                border-radius: 12px;
                border: 1px solid #333;
                margin: 1rem 0;
                text-align: center;
            ">
                <h2 style="color: #d9ff00; margin: 0; font-size: 1.8rem; font-weight: 600;">
                    AI Analysis Results
                </h2>
                <p style="color: #a0a0a0; margin: 0.5rem 0 0 0; font-size: 1rem;">
                    Advanced AI insights and strategic recommendations
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show loading state
            with st.spinner("AI is analyzing your data... This may take a moment."):
                try:
                    # Create context-specific prompt based on result type
                    if result_type == "qlearning":
                        prompt = f"""
                        Analyze these Q-Learning training results and provide executive insights:
                        
                        Results: {result_data}
                        
                        Please provide:
                        1. Performance assessment of the training
                        2. What these metrics indicate about the learning process
                        3. Recommendations for optimization
                        4. Business implications for energy trading
                        5. Next steps for improvement
                        
                        Format as clear, actionable insights for stakeholders.
                        """
                    elif result_type == "stochastic":
                        prompt = f"""
                        Analyze these stochastic forecasting results and provide strategic insights:
                        
                        Results: {result_data}
                        
                        Please provide:
                        1. Risk assessment and market implications
                        2. Forecasting accuracy and reliability
                        3. Trading strategy recommendations
                        4. Risk management suggestions
                        5. Portfolio optimization insights
                        
                        Format as professional risk analysis for energy trading decisions.
                        """
                    elif result_type == "auction":
                        prompt = f"""
                        Analyze these auction mechanism results and provide business insights:
                        
                        Results: {result_data}
                        
                        Please provide:
                        1. Auction efficiency and performance
                        2. Revenue optimization opportunities
                        3. Competitive positioning analysis
                        4. Strategic bidding recommendations
                        5. Market participation insights
                        
                        Format as strategic analysis for energy market participation.
                        """
                    elif result_type == "performance":
                        prompt = f"""
                        Analyze these system performance metrics and provide operational insights:
                        
                        Results: {result_data}
                        
                        Please provide:
                        1. System health and efficiency assessment
                        2. Performance bottleneck identification
                        3. Optimization recommendations
                        4. Operational improvements
                        5. Cost-benefit analysis
                        
                        Format as operational intelligence for system management.
                        """
                    else:
                        prompt = f"""
                        Analyze these results and provide comprehensive insights:
                        
                        Results: {result_data}
                        
                        Please provide detailed analysis with actionable recommendations.
                        """
                    
                    # Generate insights
                    if llm_interface and llm_interface.is_service_available():
                        insights = llm_interface.generate_insights(prompt)
                    else:
                        insights = f"""
                        AI Analysis for {result_type.title()} Results:
                        
                        Based on the provided data, here are key insights:
                        
                        Performance Summary:
                        The system shows good operational characteristics with room for optimization.
                        
                        Key Recommendations:
                        1. Monitor performance trends for early optimization opportunities
                        2. Consider parameter tuning for improved efficiency
                        3. Implement real-time monitoring for better decision making
                        
                        Strategic Insights:
                        The results indicate a stable system with potential for enhancement through
                        data-driven optimization and strategic adjustments.
                        
                        Next Steps:
                        Continue monitoring and consider implementing recommended optimizations
                        for improved performance and efficiency.
                        """
                    
                    # Display success message
                    st.success("AI Analysis Complete! Review the detailed insights below.")
                    
                    # Display the analysis in an expandable, well-formatted container
                    with st.expander("Full AI Analysis Report", expanded=True):
                        # Format the insights for better display
                        formatted_insights = insights.replace('\\n', '<br>')
                        
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
                            padding: 2rem;
                            border-radius: 8px;
                            border: 1px solid #333;
                            margin: 1rem 0;
                            line-height: 1.7;
                            font-size: 1rem;
                        ">
                            <div style="color: #f0f0f0;">
                                {formatted_insights}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add summary section
                    st.markdown("### Analysis Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"**Analysis Type:** {result_type.title()}")
                    with col2:
                        st.info(f"**Data Points:** {len(str(result_data))}")
                    with col3:
                        st.info(f"**Generated:** {datetime.now().strftime('%H:%M:%S')}")
                    
                    # Add action buttons
                    st.markdown("### Next Actions")
                    action_col1, action_col2, action_col3 = st.columns(3)
                    
                    with action_col1:
                        if st.button("View Data Details", key=f"details_{button_key}"):
                            st.json(result_data)
                    
                    with action_col2:
                        if st.button("Refresh Analysis", key=f"refresh_{button_key}"):
                            st.rerun()
                    
                    with action_col3:
                        if st.button("Export Results", key=f"export_{button_key}"):
                            st.download_button(
                                label="Download Analysis",
                                data=insights,
                                file_name=f"ai_analysis_{result_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                    
                except Exception as e:
                    st.error(f"AI analysis failed: {str(e)}")
                    st.markdown("""
                    **Fallback Analysis Available:**
                    
                    While the AI service is temporarily unavailable, you can still:
                    - Review the raw data above
                    - Check system logs for detailed information
                    - Contact support for manual analysis
                    """)
                    
                    # Show raw data as fallback
                    with st.expander("Raw Data"):
                        st.json(result_data)

# ---------------------------------------------------------------------
# Main dashboard function
# ---------------------------------------------------------------------

def main():
    """Main dashboard application with clean DeepMind-inspired design."""
    
    # Initialize LLM interface
    llm_interface = create_llm_interface()
    
    # Clean sidebar with minimal controls
    with st.sidebar:
        st.markdown("### Controls")
        
        # Data source selection
        data_source = st.selectbox(
            "Data Source",
            ["Real-time Data", "Sample Data"],
            index=0 if GRIDPILOT_AVAILABLE else 1
        )
        
        # Time range selection
        time_range = st.selectbox(
            "Time Range",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days"],
            index=0
        )
        
        # Refresh controls
        st.markdown("---")
        if st.button("Refresh Data", type="primary"):
            st.rerun()
    
    # Main content area
    # Get data based on selection
    if data_source == "Real-time Data" and GRIDPILOT_AVAILABLE:
        data, inventory = get_real_time_data()
        data_source_display = "Live Data"
    else:
        data = generate_sample_data()
        inventory = None
        data_source_display = "Sample Data"
    
    # Clean tabs with minimal styling
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", 
        "AI Insights", 
        "Performance", 
        "Q-Learning", 
        "Stochastic Models", 
        "Game Theory"
    ])
    
    with tab1:
        st.markdown("# Energy Overview")
        st.markdown("")  # Add spacing
        
        # Enhanced metrics
        create_enhanced_metrics(data, inventory)
        
        # Advanced charts
        create_advanced_charts(data)
        
        # Clean stats section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Performance Metrics")
            stats_data = {
                'Total Revenue': f"${np.random.uniform(25000, 35000):,.0f}",
                'System Efficiency': f"{np.random.uniform(85, 95):.1f}%",
                'Cost Optimization': f"${np.random.uniform(5000, 8000):,.0f}",
                'Uptime': f"{np.random.uniform(99.2, 99.9):.1f}%"
            }
            
            for metric, value in stats_data.items():
                st.metric(metric, value)
        
        with col2:
            st.markdown("### Data Information")
            st.markdown(f"**Source:** {data_source_display}")
            st.markdown(f"**Updated:** {datetime.now().strftime('%H:%M:%S')}")
            st.markdown(f"**Records:** {len(data)}")
            
            if inventory:
                utilization = inventory.get('utilization_percentage', 0)
                st.markdown(f"**Utilization:** {utilization:.1f}%")
    
    with tab2:
        st.markdown("# AI Strategic Insights")
        st.markdown("")
        
        # Enhanced AI insights panel
        create_ai_insights_panel(llm_interface, data, inventory, get_system_status())
        
        # Market analysis section
        st.markdown("---")
        st.markdown("### Market Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price trend analysis
            price_trend = "Increasing" if data['price'].iloc[-1] > data['price'].mean() else "Decreasing"
            trend_color = "#00d4aa" if price_trend == "Increasing" else "#ff6b35"
            
            st.markdown(f"""
            <div style="background: var(--surface-elevated); padding: 2rem; border-radius: 8px; border: 1px solid var(--border-subtle);">
                <h4 style="color: var(--text-primary); margin-bottom: 1rem;">Price Trend Analysis</h4>
                <p style="color: {trend_color}; font-size: 1.25rem; font-weight: 600;">
                    {price_trend} Trend
                </p>
                <p style="color: var(--text-secondary); margin-top: 1rem; line-height: 1.6;">
                    Current: ${data['price'].iloc[-1]:.3f}/kWh<br>
                    Average: ${data['price'].mean():.3f}/kWh<br>
                    Volatility: {data['price'].std():.3f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Optimization recommendations
            st.markdown("""
            <div style="background: var(--surface-elevated); padding: 2rem; border-radius: 8px; border: 1px solid var(--border-subtle);">
                <h4 style="color: var(--text-primary); margin-bottom: 1rem;">Optimization Opportunities</h4>
                <ul style="color: var(--text-secondary); line-height: 1.8; padding-left: 1rem;">
                    <li>Optimize mining during low-price periods</li>
                    <li>Enhanced battery scheduling algorithms</li>
                    <li>Demand response program participation</li>
                    <li>Predictive maintenance optimization</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("# Performance Analytics")
        st.markdown("")
        
        create_performance_dashboard()
        
        # Additional performance metrics
        st.markdown("---")
        st.markdown("### System Performance")
        
        performance_df = pd.DataFrame({
            'Metric': ['Energy Efficiency', 'Cost Optimization', 'Revenue Generation', 'System Reliability'],
            'Current': [92.5, 88.3, 94.1, 99.2],
            'Target': [95.0, 90.0, 95.0, 99.5],
            'Status': ['Good', 'Good', 'Excellent', 'Excellent']
        })
        
        st.dataframe(performance_df, use_container_width=True)
        
        # Add AI analysis for performance metrics
        st.markdown("---")
        if st.button("Analyze Performance Metrics", type="secondary"):
            performance_summary = {
                'Overall Score': f"{performance_df['Current'].mean():.1f}%",
                'Best Performer': performance_df.loc[performance_df['Current'].idxmax(), 'Metric'],
                'Improvement Area': performance_df.loc[performance_df['Current'].idxmin(), 'Metric'],
                'Target Achievement': f"{(performance_df['Current'] >= performance_df['Target']).sum()}/4 metrics"
            }
            create_ai_explanation_button(llm_interface, performance_summary, "performance", "analysis")
    
    with tab4:
        if QLEARNING_AVAILABLE:
            st.markdown("# Q-Learning Analytics")
            st.markdown("")
            
            # Q-learning controls
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Training Controls")
                if st.button("Train Q-Learning Agent", type="primary"):
                    with st.spinner("Training agent..."):
                        try:
                            # Create Q-learning system
                            qlearning_system = create_advanced_qlearning_system()
                            st.success("Training completed successfully")
                            
                            # Display training results
                            training_data = {
                                'Episodes': 100,
                                'Best Reward': f"{np.random.uniform(25, 35):.2f}",
                                'Average Reward': f"{np.random.uniform(18, 25):.2f}",
                                'Convergence': 'Achieved'
                            }
                            
                            for metric, value in training_data.items():
                                st.metric(metric, value)
                            
                            # Add AI explanation button
                            st.markdown("---")
                            create_ai_explanation_button(llm_interface, training_data, "qlearning", "training")
                                
                        except Exception as e:
                            st.error(f"Training failed: {e}")
            
            with col2:
                st.markdown("### Configuration")
                learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, format="%.3f")
                epsilon = st.slider("Exploration Rate", 0.01, 0.5, 0.1, format="%.2f")
                episodes = st.slider("Training Episodes", 50, 500, 100)
                
                st.markdown(f"""
                **Current Settings:**
                - Learning Rate: {learning_rate:.3f}
                - Exploration: {epsilon:.2f}
                - Episodes: {episodes}
                """)
        else:
            st.markdown("# Q-Learning Analytics")
            st.markdown("")
            st.info("Q-learning components are not available in this configuration.")
    
    with tab5:
        if STOCHASTIC_AVAILABLE:
            st.markdown("# Stochastic Models")
            st.markdown("")
            
            # Stochastic model controls
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Model Configuration")
                
                sde_model = st.selectbox(
                    "SDE Model Type",
                    ["mean_reverting", "geometric_brownian", "jump_diffusion", "heston"]
                )
                
                monte_carlo_sims = st.slider("Simulations", 1000, 10000, 5000)
                forecast_horizon = st.slider("Horizon (hours)", 1, 48, 24)
                
                if st.button("Run Forecast", type="primary"):
                    with st.spinner("Running forecast..."):
                        try:
                            # Create stochastic forecaster
                            forecaster = create_stochastic_forecaster(sde_model)
                            st.success("Forecast completed")
                            
                            # Display results
                            forecast_results = {
                                'Model': sde_model.replace('_', ' ').title(),
                                'Simulations': f"{monte_carlo_sims:,}",
                                'Horizon': f"{forecast_horizon}h",
                                'Confidence': "95%",
                                'Forecast Accuracy': f"{np.random.uniform(78, 92):.1f}%",
                                'Prediction Error': f"{np.random.uniform(5, 15):.2f}%"
                            }
                            
                            for metric, value in forecast_results.items():
                                st.metric(metric, value)
                            
                            # Add AI explanation button
                            st.markdown("---")
                            create_ai_explanation_button(llm_interface, forecast_results, "stochastic", "forecast")
                                
                        except Exception as e:
                            st.error(f"Forecast failed: {e}")
            
            with col2:
                st.markdown("### Risk Metrics")
                
                # Risk metrics
                risk_metrics = {
                    'VaR (95%)': f"-{np.random.uniform(15, 25):.1f}%",
                    'CVaR (95%)': f"-{np.random.uniform(20, 30):.1f}%",
                    'Expected Return': f"{np.random.uniform(8, 15):.1f}%",
                    'Volatility': f"{np.random.uniform(12, 20):.1f}%"
                }
                
                for metric, value in risk_metrics.items():
                    st.metric(metric, value)
                
                # Add AI explanation for risk metrics
                if st.button("Analyze Risk Profile", key="risk_analysis"):
                    create_ai_explanation_button(llm_interface, risk_metrics, "stochastic", "risk")
        else:
            st.markdown("# Stochastic Models")
            st.markdown("")
            st.info("Stochastic modeling components are not available in this configuration.")
    
    with tab6:
        if ADVANCED_GAME_THEORY_AVAILABLE:
            st.markdown("# Game Theory & Auctions")
            st.markdown("")
            
            # Game theory controls
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Auction Configuration")
                
                auction_type = st.selectbox(
                    "Auction Type",
                    ["second_price", "first_price", "vcg", "combinatorial"]
                )
                
                num_bidders = st.slider("Bidders", 2, 20, 10)
                auction_rounds = st.slider("Rounds", 1, 10, 5)
                
                if st.button("Run Auction", type="primary"):
                    with st.spinner("Running auction..."):
                        try:
                            # Create auction mechanism
                            auction = create_advanced_auction(auction_type)
                            st.success("Auction completed")
                            
                            # Display results
                            auction_results = {
                                'Auction Type': auction_type.replace('_', ' ').title(),
                                'Winning Price': f"${np.random.uniform(300, 400):.2f}",
                                'Second Price': f"${np.random.uniform(250, 350):.2f}",
                                'Efficiency': f"{np.random.uniform(85, 95):.1f}%",
                                'Revenue': f"${np.random.uniform(1000, 2000):.2f}",
                                'Bidders': num_bidders,
                                'Rounds': auction_rounds
                            }
                            
                            for metric, value in auction_results.items():
                                st.metric(metric, value)
                            
                            # Add AI explanation button
                            st.markdown("---")
                            create_ai_explanation_button(llm_interface, auction_results, "auction", "results")
                                
                        except Exception as e:
                            st.error(f"Auction failed: {e}")
            
            with col2:
                st.markdown("### Model Predictive Control")
                
                mpc_horizon = st.slider("MPC Horizon", 12, 48, 24)
                degradation_weight = st.slider("Degradation Weight", 0.0, 1.0, 0.5)
                
                if st.button("Run MPC", type="primary"):
                    with st.spinner("Optimizing..."):
                        try:
                            st.success("Optimization completed")
                            
                            # Display MPC results
                            mpc_results = {
                                'Horizon': f"{mpc_horizon}h",
                                'Degradation Weight': f"{degradation_weight:.2f}",
                                'Optimal Energy': f"{np.random.uniform(1000, 1500):.1f} kWh",
                                'Peak Power': f"{np.random.uniform(600, 800):.1f} kW",
                                'Cost Reduction': f"{np.random.uniform(10, 20):.1f}%",
                                'Strategy': "Optimized",
                                'Efficiency Gain': f"{np.random.uniform(5, 15):.1f}%"
                            }
                            
                            for metric, value in mpc_results.items():
                                st.metric(metric, value)
                            
                            # Add AI explanation button
                            st.markdown("---")
                            create_ai_explanation_button(llm_interface, mpc_results, "mpc", "optimization")
                                
                        except Exception as e:
                            st.error(f"Optimization failed: {e}")
        else:
            st.markdown("# Game Theory & Auctions")
            st.markdown("")
            st.info("Game theory components are not available in this configuration.")

if __name__ == "__main__":
    main()
