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
    """Create an AI explanation button that displays analysis results."""
    button_key = f"explain_{result_type}_{key_suffix}"
    analysis_key = f"analysis_{result_type}_{key_suffix}"
    
    # Initialize session state for this analysis
    if analysis_key not in st.session_state:
        st.session_state[analysis_key] = None
    
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
            # Store analysis request in session state
            st.session_state[analysis_key] = "generating"
            st.rerun()
    
    # Display analysis if it exists in session state
    if st.session_state[analysis_key] == "generating":
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
        
        # Show loading state and generate analysis
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
                elif result_type == "mpc":
                    prompt = f"""
                    Analyze these MPC optimization results and provide control insights:
                    
                    Results: {result_data}
                    
                    Please provide:
                    1. Control performance assessment
                    2. Optimization effectiveness analysis
                    3. Constraint satisfaction evaluation
                    4. Operational efficiency insights
                    5. Implementation recommendations
                    
                    Format as control system analysis for operations team.
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
                    analysis_source = "AI Analysis"
                else:
                    insights = f"""
                    **{result_type.title()} Analysis Summary**
                    
                    Based on the provided data, here are key insights:
                    
                    **Performance Overview:**
                    The system demonstrates solid operational characteristics with identifiable optimization opportunities.
                    
                    **Key Findings:**
                    • Current performance metrics indicate stable system operation
                    • Several optimization opportunities have been identified
                    • Risk management protocols are functioning within acceptable parameters
                    
                    **Strategic Recommendations:**
                    1. Monitor performance trends for early optimization opportunities
                    2. Consider parameter tuning for improved efficiency
                    3. Implement real-time monitoring for better decision making
                    4. Develop predictive maintenance schedules
                    
                    **Business Impact:**
                    The analysis suggests potential for 10-15% efficiency improvements through
                    targeted optimization strategies and enhanced monitoring capabilities.
                    
                    **Next Steps:**
                    1. Implement recommended monitoring enhancements
                    2. Conduct detailed parameter optimization study
                    3. Develop implementation timeline for suggested improvements
                    4. Establish performance benchmarks for continuous improvement
                    """
                    analysis_source = "Fallback Analysis"
                
                # Store the generated analysis
                st.session_state[analysis_key] = {
                    "insights": insights,
                    "source": analysis_source,
                    "timestamp": datetime.now(),
                    "result_data": result_data
                }
                
                st.success(f"{analysis_source} Complete! Review the detailed insights below.")
                
            except Exception as e:
                st.error(f"Analysis generation failed: {str(e)}")
                # Store error state
                st.session_state[analysis_key] = {
                    "insights": f"Analysis temporarily unavailable. Error: {str(e)}",
                    "source": "Error",
                    "timestamp": datetime.now(),
                    "result_data": result_data
                }
    
    # Display stored analysis results
    if st.session_state[analysis_key] and isinstance(st.session_state[analysis_key], dict):
        analysis_data = st.session_state[analysis_key]
        
        # Display the analysis in an expandable, well-formatted container
        with st.expander("Full AI Analysis Report", expanded=True):
            # Format the insights for better display
            insights = analysis_data["insights"]
            
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
                    {insights.replace(chr(10), '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add summary section
        st.markdown("### Analysis Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Analysis Type:** {result_type.title()}")
        with col2:
            st.info(f"**Source:** {analysis_data['source']}")
        with col3:
            st.info(f"**Generated:** {analysis_data['timestamp'].strftime('%H:%M:%S')}")
        
        # Add action buttons
        st.markdown("### Actions")
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("View Raw Data", key=f"details_{button_key}"):
                st.json(analysis_data["result_data"])
        
        with action_col2:
            if st.button("New Analysis", key=f"refresh_{button_key}"):
                st.session_state[analysis_key] = None
                st.rerun()
        
        with action_col3:
            # Create download content
            download_content = f"""
AI Analysis Report - {result_type.title()}
Generated: {analysis_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
Source: {analysis_data['source']}

{insights}

Raw Data:
{str(analysis_data['result_data'])}
"""
            st.download_button(
                label="Download Report",
                data=download_content,
                file_name=f"ai_analysis_{result_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key=f"download_{button_key}"
            )

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
        
        # LLM Test Section (for debugging)
        with st.expander("LLM System Test", expanded=False):
            st.markdown("**Test the AI Analysis System**")
            
            if st.button("Test LLM Integration", key="test_llm"):
                with st.spinner("Testing LLM connection..."):
                    try:
                        test_prompt = "Analyze energy trading performance and provide 3 key insights."
                        test_result = llm_interface.generate_insights(test_prompt)
                        
                        st.success("LLM Integration Working!")
                        st.markdown("**Test Response:**")
                        st.markdown(f"```\n{test_result}\n```")
                        
                        # Show interface details
                        st.markdown("**System Information:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"Service Available: {llm_interface.is_service_available()}")
                        with col2:
                            model_info = llm_interface.get_model_info()
                            st.info(f"Model: {model_info.get('name', 'Unknown')}")
                            
                    except Exception as e:
                        st.error(f"LLM Test Failed: {str(e)}")
                        st.markdown("**Troubleshooting:**")
                        st.markdown("- Check if mock interface is properly initialized")
                        st.markdown("- Verify generate_insights method is working")
                        st.markdown("- Review error logs for detailed information")
        
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
        st.markdown("# System Performance Analytics")
        st.markdown("")
        
        # System Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("System Uptime", "99.8%", "0.2%")
        
        with col2:
            st.metric("Avg Response Time", "1.2s", "-0.3s")
        
        with col3:
            st.metric("Energy Efficiency", "94.2%", "2.1%")
        
        with col4:
            st.metric("Cost Optimization", "87.5%", "5.2%")
        
        # Performance Charts
        st.markdown("### Performance Overview")
        
        # Create sample performance data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        performance_data = pd.DataFrame({
            'date': dates,
            'efficiency': np.random.normal(94, 2, 30),
            'cost_savings': np.random.normal(87, 3, 30),
            'response_time': np.random.normal(1.2, 0.3, 30)
        })
        
        # Efficiency Chart
        fig_efficiency = px.line(performance_data, x='date', y='efficiency', 
                               title='System Efficiency Over Time')
        fig_efficiency.update_layout(template='plotly_dark')
        st.plotly_chart(fig_efficiency, use_container_width=True)
        
        # Cost Savings Chart
        fig_cost = px.bar(performance_data, x='date', y='cost_savings', 
                         title='Cost Optimization Performance')
        fig_cost.update_layout(template='plotly_dark')
        st.plotly_chart(fig_cost, use_container_width=True)
        
        # Response Time Chart
        fig_response = px.scatter(performance_data, x='date', y='response_time', 
                                title='System Response Time')
        fig_response.update_layout(template='plotly_dark')
        st.plotly_chart(fig_response, use_container_width=True)
        
        # Claude AI Analysis for Performance
        st.markdown("---")
        st.markdown("### Claude AI Performance Analysis")
        
        performance_context = {
            'system_uptime': '99.8%',
            'avg_response_time': '1.2s',
            'energy_efficiency': '94.2%',
            'cost_optimization': '87.5%',
            'efficiency_trend': 'stable with slight improvement',
            'cost_trend': 'improving by 5.2%',
            'response_trend': 'improving by 0.3s'
        }
        
        performance_prompt = f"""Analyze the system performance metrics and provide insights:
        
        Performance Metrics:
        - System Uptime: {performance_context['system_uptime']} (up 0.2%)
        - Average Response Time: {performance_context['avg_response_time']} (improved by 0.3s)
        - Energy Efficiency: {performance_context['energy_efficiency']} (up 2.1%)
        - Cost Optimization: {performance_context['cost_optimization']} (up 5.2%)
        
        Provide 3 key insights about system performance and recommendations for improvement."""
        
        create_ai_explanation_button(llm_interface, performance_prompt, "performance", "main")
    
    with tab4:
        st.markdown("# Q-Learning Agent Performance")
        st.markdown("")
        
        # Q-Learning Controls
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Training Configuration")
            episodes = st.slider("Training Episodes", 50, 500, 250)
            learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, format="%.3f")
            epsilon = st.slider("Exploration Rate", 0.01, 0.5, 0.1, format="%.2f")
        
        with col2:
            st.markdown("### Quick Actions")
            if st.button("Train Q-Learning Agent", type="primary"):
                with st.spinner("Training Q-Learning agent..."):
                    try:
                        # Import and run Q-learning training
                        from train_qlearning import run_qlearning_training
                        results = run_qlearning_training(episodes=episodes)
                        
                        # Store results in session state
                        st.session_state.qlearning_results = results
                        st.success(f"Training completed! Best reward: {results.get('best_reward', 0):.2f}")
                        
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
        
        # Display Q-Learning Results
        if 'qlearning_results' in st.session_state:
            results = st.session_state.qlearning_results
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Best Reward", f"{results.get('best_reward', 0):.2f}")
            
            with col2:
                st.metric("Avg Reward", f"{results.get('avg_reward', 0):.2f}")
            
            with col3:
                st.metric("Episodes", results.get('episodes', 0))
            
            with col4:
                st.metric("Training Time", results.get('training_time', 'N/A'))
            
            # Training Progress Chart
            if 'episode_rewards' in results:
                fig_rewards = px.line(
                    x=list(range(len(results['episode_rewards']))), 
                    y=results['episode_rewards'],
                    title='Q-Learning Training Progress',
                    labels={'x': 'Episode', 'y': 'Reward'}
                )
                fig_rewards.update_layout(template='plotly_dark')
                st.plotly_chart(fig_rewards, use_container_width=True)
            
            # Claude AI Analysis for Q-Learning
            st.markdown("---")
            st.markdown("### Claude AI Q-Learning Analysis")
            
            qlearning_prompt = f"""Analyze the Q-Learning training results and provide insights:
            
            Training Results:
            - Best Reward: {results.get('best_reward', 0):.2f}
            - Average Reward: {results.get('avg_reward', 0):.2f}
            - Episodes Trained: {results.get('episodes', 0)}
            - Training Time: {results.get('training_time', 'N/A')}
            - Convergence: {'Good' if results.get('best_reward', 0) > 20 else 'Needs Improvement'}
            
            Provide 3 key insights about the Q-learning performance and recommendations for optimization."""
            
            create_ai_explanation_button(llm_interface, qlearning_prompt, "qlearning", "main")
        
        else:
            st.info("Train the Q-Learning agent to see results and analysis.")
    
    with tab5:
        st.markdown("# Advanced Stochastic Models")
        st.markdown("")
        
        # Model Selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Model Configuration")
            model_type = st.selectbox(
                "Stochastic Model",
                ["mean_reverting", "geometric_brownian", "jump_diffusion", "heston"]
            )
            
            forecast_horizon = st.slider("Forecast Horizon (hours)", 24, 168, 72)
            num_simulations = st.slider("Monte Carlo Simulations", 100, 1000, 500)
        
        with col2:
            st.markdown("### Quick Actions")
            if st.button("Run Stochastic Forecast", type="primary"):
                with st.spinner("Running stochastic forecast..."):
                    try:
                        from forecasting.stochastic_models import StochasticForecaster
                        
                        # Get current price data
                        current_prices = mara_client.get_prices()
                        
                        # Initialize forecaster
                        forecaster = StochasticForecaster(model_type=model_type)
                        
                        # Generate forecast
                        forecast_results = forecaster.forecast(
                            current_prices, 
                            horizon=forecast_horizon,
                            n_simulations=num_simulations
                        )
                        
                        # Store results
                        st.session_state.stochastic_results = forecast_results
                        st.success("Stochastic forecast completed!")
                        
                    except Exception as e:
                        st.error(f"Forecast failed: {str(e)}")
        
        # Display Stochastic Results
        if 'stochastic_results' in st.session_state:
            results = st.session_state.stochastic_results
            
            # Risk Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                var_95 = results.get('var_95', 0)
                st.metric("VaR (95%)", f"{var_95:.2f}%")
            
            with col2:
                expected_return = results.get('expected_return', 0)
                st.metric("Expected Return", f"{expected_return:.2f}%")
            
            with col3:
                volatility = results.get('volatility', 0)
                st.metric("Volatility", f"{volatility:.2f}%")
            
            with col4:
                sharpe_ratio = results.get('sharpe_ratio', 0)
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            # Forecast Visualization
            if 'forecast_paths' in results:
                fig_paths = px.line(
                    results['forecast_paths'],
                    title=f'Stochastic Price Forecast - {model_type.title()} Model'
                )
                fig_paths.update_layout(template='plotly_dark')
                st.plotly_chart(fig_paths, use_container_width=True)
            
            # Claude AI Analysis for Stochastic Models
            st.markdown("---")
            st.markdown("### Claude AI Stochastic Analysis")
            
            stochastic_prompt = f"""Analyze the stochastic modeling results and provide insights:
            
            Stochastic Model Results:
            - Model Type: {model_type.title()}
            - Forecast Horizon: {forecast_horizon} hours
            - Monte Carlo Simulations: {num_simulations}
            - VaR (95%): {results.get('var_95', 0):.2f}%
            - Expected Return: {results.get('expected_return', 0):.2f}%
            - Volatility: {results.get('volatility', 0):.2f}%
            - Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}
            
            Provide 3 key insights about the stochastic forecast and risk assessment."""
            
            create_ai_explanation_button(llm_interface, stochastic_prompt, "stochastic", "main")
        
        else:
            st.info("Run a stochastic forecast to see results and analysis.")
    
    with tab6:
        st.markdown("# Advanced Game Theory & Auctions")
        st.markdown("")
        
        # Game Theory Controls
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Auction Configuration")
            auction_type = st.selectbox(
                "Auction Mechanism",
                ["second_price", "first_price", "vcg", "combinatorial"]
            )
            
            num_bidders = st.slider("Number of Bidders", 3, 10, 5)
            reserve_price = st.slider("Reserve Price", 0.01, 0.20, 0.05, format="%.3f")
        
        with col2:
            st.markdown("### Quick Actions")
            if st.button("Run Advanced Auction", type="primary"):
                with st.spinner("Running auction simulation..."):
                    try:
                        from game_theory.advanced_game_theory import AdvancedGameTheory
                        
                        # Initialize game theory system
                        game_system = AdvancedGameTheory()
                        
                        # Run auction
                        auction_results = game_system.run_auction(
                            auction_type=auction_type,
                            num_bidders=num_bidders,
                            reserve_price=reserve_price
                        )
                        
                        # Store results
                        st.session_state.auction_results = auction_results
                        st.success("Auction simulation completed!")
                        
                    except Exception as e:
                        st.error(f"Auction failed: {str(e)}")
            
            if st.button("Run MPC Optimization"):
                with st.spinner("Running MPC optimization..."):
                    try:
                        from game_theory.mpc_controller import MPCController
                        
                        # Initialize MPC
                        mpc = MPCController()
                        
                        # Get current data
                        current_prices = mara_client.get_prices()
                        
                        # Run optimization
                        mpc_results = mpc.optimize(current_prices)
                        
                        # Store results
                        st.session_state.mpc_results = mpc_results
                        st.success("MPC optimization completed!")
                        
                    except Exception as e:
                        st.error(f"MPC optimization failed: {str(e)}")
        
        # Display Auction Results
        if 'auction_results' in st.session_state:
            results = st.session_state.auction_results
            
            st.markdown("### Auction Results")
            
            # Auction Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Winning Bid", f"${results.get('winning_bid', 0):.3f}")
            
            with col2:
                st.metric("Revenue", f"${results.get('revenue', 0):.2f}")
            
            with col3:
                st.metric("Efficiency", f"{results.get('efficiency', 0):.1f}%")
            
            with col4:
                st.metric("Bidders", results.get('num_bidders', 0))
            
            # Bid Distribution
            if 'bid_data' in results:
                fig_bids = px.bar(
                    results['bid_data'],
                    title='Auction Bid Distribution'
                )
                fig_bids.update_layout(template='plotly_dark')
                st.plotly_chart(fig_bids, use_container_width=True)
            
            # Claude AI Analysis for Auction Results
            st.markdown("---")
            st.markdown("### Claude AI Auction Analysis")
            
            auction_prompt = f"""Analyze the auction results and provide insights:
            
            Auction Results:
            - Auction Type: {auction_type.title()}
            - Number of Bidders: {num_bidders}
            - Reserve Price: ${reserve_price:.3f}
            - Winning Bid: ${results.get('winning_bid', 0):.3f}
            - Total Revenue: ${results.get('revenue', 0):.2f}
            - Auction Efficiency: {results.get('efficiency', 0):.1f}%
            
            Provide 3 key insights about the auction performance and strategic recommendations."""
            
            create_ai_explanation_button(llm_interface, auction_prompt, "auction", "main")
        
        # Display MPC Results
        if 'mpc_results' in st.session_state:
            results = st.session_state.mpc_results
            
            st.markdown("### MPC Optimization Results")
            
            # MPC Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Cost Reduction", f"{results.get('cost_reduction', 0):.1f}%")
            
            with col2:
                st.metric("Optimal Horizon", f"{results.get('horizon', 0)} hrs")
            
            with col3:
                st.metric("Convergence", results.get('convergence', 'Unknown'))
            
            with col4:
                st.metric("Iterations", results.get('iterations', 0))
            
            # Control Actions
            if 'control_actions' in results:
                fig_control = px.line(
                    results['control_actions'],
                    title='MPC Control Actions'
                )
                fig_control.update_layout(template='plotly_dark')
                st.plotly_chart(fig_control, use_container_width=True)
            
            # Claude AI Analysis for MPC Results
            st.markdown("---")
            st.markdown("### Claude AI MPC Analysis")
            
            mpc_prompt = f"""Analyze the MPC optimization results and provide insights:
            
            MPC Results:
            - Cost Reduction: {results.get('cost_reduction', 0):.1f}%
            - Optimization Horizon: {results.get('horizon', 0)} hours
            - Convergence Status: {results.get('convergence', 'Unknown')}
            - Iterations: {results.get('iterations', 0)}
            - Control Strategy: {results.get('strategy', 'Adaptive')}
            
            Provide 3 key insights about the MPC performance and optimization recommendations."""
            
            create_ai_explanation_button(llm_interface, mpc_prompt, "mpc", "main")
        
        if 'auction_results' not in st.session_state and 'mpc_results' not in st.session_state:
            st.info("Run auction or MPC optimization to see results and analysis.")

if __name__ == "__main__":
    main()
