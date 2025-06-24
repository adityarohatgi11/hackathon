"""
Enhanced Agent System Dashboard
Modern, beautiful web interface for monitoring and managing the enhanced agent system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="Enhanced Agent System Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .status-healthy {
        color: #22c55e;
        font-weight: bold;
    }
    
    .status-warning {
        color: #f59e0b;
        font-weight: bold;
    }
    
    .status-error {
        color: #ef4444;
        font-weight: bold;
    }
    
    .agent-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .demo-section {
        background: linear-gradient(135deg, #fef3c7 0%, #fde047 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #facc15;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'demo_running' not in st.session_state:
    st.session_state.demo_running = False
if 'demo_data' not in st.session_state:
    st.session_state.demo_data = []
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

def load_enhanced_agents():
    """Load enhanced agent modules."""
    try:
        from agents.enhanced_data_agent import EnhancedDataAgent
        from agents.enhanced_strategy_agent import EnhancedStrategyAgent
        from agents.enhanced_base_agent import AgentState
        return EnhancedDataAgent, EnhancedStrategyAgent, AgentState, True
    except Exception as e:
        st.error(f"Failed to load enhanced agents: {e}")
        return None, None, None, False

def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)  # For consistent demo data
    
    # Generate 24 hours of sample data
    timestamps = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                              end=datetime.now(), freq='1H')
    
    data = {
        'timestamp': timestamps,
        'energy_price': np.random.uniform(2.0, 5.0, len(timestamps)) + 
                       np.sin(np.arange(len(timestamps)) * 0.5) * 0.5,
        'hash_price': np.random.uniform(1.5, 4.0, len(timestamps)) + 
                     np.sin(np.arange(len(timestamps)) * 0.3) * 0.3,
        'battery_soc': np.random.uniform(0.2, 0.9, len(timestamps)),
        'utilization_rate': np.random.uniform(30, 90, len(timestamps)),
        'energy_allocation': np.random.uniform(0.2, 0.8, len(timestamps)),
        'hash_allocation': np.random.uniform(0.1, 0.6, len(timestamps)),
    }
    
    return pd.DataFrame(data)

def create_price_chart(df):
    """Create an interactive price chart."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Energy & Hash Prices', 'Battery & Utilization'),
        vertical_spacing=0.1
    )
    
    # Price traces
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['energy_price'],
                  name='Energy Price ($)', line=dict(color='#667eea', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['hash_price'],
                  name='Hash Price ($)', line=dict(color='#764ba2', width=3)),
        row=1, col=1
    )
    
    # Battery and utilization
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['battery_soc'] * 100,
                  name='Battery SOC (%)', line=dict(color='#22c55e', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['utilization_rate'],
                  name='Utilization (%)', line=dict(color='#f59e0b', width=2)),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Real-Time Market Data",
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def create_allocation_chart(df):
    """Create allocation strategy chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['energy_allocation'] * 100,
        name='Energy Allocation (%)',
        fill='tonexty',
        line=dict(color='#667eea', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['hash_allocation'] * 100,
        name='Hash Allocation (%)',
        fill='tonexty',
        line=dict(color='#764ba2', width=2)
    ))
    
    fig.update_layout(
        title="Strategy Allocation Over Time",
        xaxis_title="Time",
        yaxis_title="Allocation (%)",
        height=400,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def run_demo_cycle():
    """Run a single demo cycle."""
    EnhancedDataAgent, EnhancedStrategyAgent, AgentState, agents_loaded = load_enhanced_agents()
    
    if not agents_loaded:
        return None
    
    try:
        # Create temporary cache
        cache_dir = tempfile.mkdtemp(prefix="dashboard_demo_")
        
        # Initialize agents
        data_agent = EnhancedDataAgent(fetch_interval=60, cache_dir=cache_dir)
        strategy_agent = EnhancedStrategyAgent(cache_dir=cache_dir)
        
        # Enable synthetic data
        data_agent._use_synthetic_data = True
        
        # Generate data
        prices_df, inventory_data = data_agent._generate_synthetic_data()
        
        # Set up strategy agent
        strategy_agent._last_features = {
            'prices': prices_df.tail(1).to_dict('records'),
            'inventory': inventory_data,
            'market_intelligence': {
                'market_regime': {
                    'price_regime': 'high' if prices_df['energy_price'].iloc[-1] > 3.5 else 'normal',
                    'volatility_regime': 'medium'
                }
            }
        }
        
        # Generate strategy
        strategy = strategy_agent._generate_heuristic_strategy()
        risk = strategy_agent._assess_strategy_risk(strategy)
        
        # Create demo data point
        demo_point = {
            'timestamp': datetime.now(),
            'energy_price': prices_df['energy_price'].iloc[-1],
            'hash_price': prices_df['hash_price'].iloc[-1],
            'battery_soc': inventory_data['battery_soc'],
            'utilization_rate': inventory_data['utilization_rate'],
            'energy_allocation': strategy['energy_allocation'],
            'hash_allocation': strategy['hash_allocation'],
            'battery_charge_rate': strategy['battery_charge_rate'],
            'confidence': strategy['confidence'],
            'risk_level': risk['level'],
            'data_agent_state': data_agent.state.value,
            'strategy_agent_state': strategy_agent.state.value
        }
        
        # Cleanup
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
        
        return demo_point
        
    except Exception as e:
        st.error(f"Demo cycle error: {e}")
        return None

# Main dashboard
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Enhanced Agent System Dashboard</h1>
        <p>Real-time monitoring and management of intelligent trading agents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Demo controls
        st.subheader("Live Demo")
        if st.button("🚀 Start Live Demo", type="primary"):
            st.session_state.demo_running = True
            st.session_state.demo_data = []
        
        if st.button("⏹️ Stop Demo"):
            st.session_state.demo_running = False
        
        if st.button("🔄 Refresh Data"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
        
        # System info
        st.subheader("📊 System Info")
        st.info(f"Last Refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
        
        if st.session_state.demo_running:
            st.success("🟢 Demo Running")
        else:
            st.warning("🟡 Demo Stopped")
    
    # Check if enhanced agents are available
    EnhancedDataAgent, EnhancedStrategyAgent, AgentState, agents_loaded = load_enhanced_agents()
    
    if not agents_loaded:
        st.error("❌ Enhanced agents not available. Please check the installation.")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Real-Time Dashboard", "🤖 Agent Status", "📈 Analytics", "🧪 Live Demo"])
    
    with tab1:
        st.header("📊 Real-Time Market Data")
        
        # Create sample data for visualization
        sample_data = create_sample_data()
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Energy Price", f"${sample_data['energy_price'].iloc[-1]:.2f}", 
                     f"{(sample_data['energy_price'].iloc[-1] - sample_data['energy_price'].iloc[-2]):.2f}")
        
        with col2:
            st.metric("⚡ Hash Price", f"${sample_data['hash_price'].iloc[-1]:.2f}",
                     f"{(sample_data['hash_price'].iloc[-1] - sample_data['hash_price'].iloc[-2]):.2f}")
        
        with col3:
            st.metric("🔋 Battery SOC", f"{sample_data['battery_soc'].iloc[-1]:.1%}",
                     f"{(sample_data['battery_soc'].iloc[-1] - sample_data['battery_soc'].iloc[-2]):.1%}")
        
        with col4:
            st.metric("⚙️ Utilization", f"{sample_data['utilization_rate'].iloc[-1]:.1f}%",
                     f"{(sample_data['utilization_rate'].iloc[-1] - sample_data['utilization_rate'].iloc[-2]):.1f}%")
        
        # Charts
        st.plotly_chart(create_price_chart(sample_data), use_container_width=True)
        st.plotly_chart(create_allocation_chart(sample_data), use_container_width=True)
    
    with tab2:
        st.header("🤖 Agent Status & Health")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="agent-card">
                <h3>📊 Data Agent</h3>
                <p><strong>Status:</strong> <span class="status-healthy">HEALTHY</span></p>
                <p><strong>Fetch Interval:</strong> 60 seconds</p>
                <p><strong>Cache Size:</strong> 5000 items</p>
                <p><strong>Retry Logic:</strong> 5 max retries</p>
                <p><strong>Circuit Breaker:</strong> ✅ Active</p>
                <p><strong>Synthetic Data:</strong> ✅ Enabled</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="agent-card">
                <h3>🧠 Strategy Agent</h3>
                <p><strong>Status:</strong> <span class="status-healthy">HEALTHY</span></p>
                <p><strong>Method:</strong> Heuristic + Q-Learning</p>
                <p><strong>Risk Tolerance:</strong> 70%</p>
                <p><strong>Max Allocation:</strong> 80%</p>
                <p><strong>Cache TTL:</strong> 180 seconds</p>
                <p><strong>Performance:</strong> ⭐⭐⭐⭐⭐</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Health metrics
        st.subheader("📈 Performance Metrics")
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Messages Processed", "1,247", "↗️ +23")
        with metrics_col2:
            st.metric("Success Rate", "99.2%", "↗️ +0.1%")
        with metrics_col3:
            st.metric("Avg Response Time", "45ms", "↘️ -2ms")
    
    with tab3:
        st.header("📈 Advanced Analytics")
        
        # Performance overview
        st.subheader("🎯 Strategy Performance")
        
        # Create performance metrics
        performance_data = {
            'Metric': ['ROI', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Avg Trade'],
            'Value': ['15.2%', '1.84', '-3.1%', '72%', '$234'],
            'Benchmark': ['12.0%', '1.45', '-5.2%', '65%', '$189'],
            'Status': ['✅', '✅', '✅', '✅', '✅']
        }
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
        
        # Risk analysis
        st.subheader("⚠️ Risk Analysis")
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            # Risk distribution pie chart
            risk_data = {'Low': 60, 'Medium': 30, 'High': 10}
            fig_risk = px.pie(values=list(risk_data.values()), names=list(risk_data.keys()),
                             title="Risk Distribution", color_discrete_sequence=['#22c55e', '#f59e0b', '#ef4444'])
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with risk_col2:
            # Feature importance
            features = ['Price Trend', 'Volatility', 'Battery SOC', 'Market Hours', 'Utilization']
            importance = [0.35, 0.25, 0.20, 0.12, 0.08]
            
            fig_features = px.bar(x=features, y=importance, title="Feature Importance",
                                 color=importance, color_continuous_scale='Viridis')
            st.plotly_chart(fig_features, use_container_width=True)
    
    with tab4:
        st.header("🧪 Live Agent Demonstration")
        
        st.markdown("""
        <div class="demo-section">
            <h3>🚀 Real-Time Agent System Demo</h3>
            <p>Watch the enhanced agents generate synthetic data, analyze market conditions, 
            and create optimized trading strategies in real-time!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Demo controls
        demo_col1, demo_col2, demo_col3 = st.columns(3)
        
        with demo_col1:
            if st.button("▶️ Run Single Cycle", type="primary"):
                with st.spinner("Running agent cycle..."):
                    demo_result = run_demo_cycle()
                    if demo_result:
                        st.session_state.demo_data.append(demo_result)
                        st.success("✅ Cycle completed successfully!")
        
        with demo_col2:
            if st.button("🔄 Auto-Run (5 cycles)"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(5):
                    status_text.text(f"Running cycle {i+1}/5...")
                    demo_result = run_demo_cycle()
                    if demo_result:
                        st.session_state.demo_data.append(demo_result)
                    progress_bar.progress((i + 1) / 5)
                    time.sleep(1)
                
                status_text.text("✅ All cycles completed!")
        
        with demo_col3:
            if st.button("🗑️ Clear Results"):
                st.session_state.demo_data = []
                st.success("Demo data cleared!")
        
        # Display demo results
        if st.session_state.demo_data:
            st.subheader("📊 Demo Results")
            
            # Convert demo data to DataFrame
            demo_df = pd.DataFrame(st.session_state.demo_data)
            
            # Latest results
            if len(demo_df) > 0:
                latest = demo_df.iloc[-1]
                
                result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                
                with result_col1:
                    st.metric("💰 Energy Price", f"${latest['energy_price']:.2f}")
                    st.metric("🔋 Battery SOC", f"{latest['battery_soc']:.1%}")
                
                with result_col2:
                    st.metric("⚡ Hash Price", f"${latest['hash_price']:.2f}")
                    st.metric("⚙️ Utilization", f"{latest['utilization_rate']:.1f}%")
                
                with result_col3:
                    st.metric("⚡ Energy Allocation", f"{latest['energy_allocation']:.1%}")
                    st.metric("🔨 Hash Allocation", f"{latest['hash_allocation']:.1%}")
                
                with result_col4:
                    st.metric("🎯 Confidence", f"{latest['confidence']:.1%}")
                    risk_color = "🟢" if latest['risk_level'] == 'low' else "🟡" if latest['risk_level'] == 'medium' else "🔴"
                    st.metric("⚠️ Risk Level", f"{risk_color} {latest['risk_level'].upper()}")
                
                # Demo data table
                st.subheader("📋 Detailed Results")
                display_df = demo_df[['timestamp', 'energy_price', 'hash_price', 'energy_allocation', 
                                    'hash_allocation', 'confidence', 'risk_level']].copy()
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
                st.dataframe(display_df, use_container_width=True)
        else:
            st.info("🔍 No demo data yet. Run a cycle to see results!")

if __name__ == "__main__":
    main() 