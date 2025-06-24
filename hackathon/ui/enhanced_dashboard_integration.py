"""
Enhanced Agent Dashboard Integration
Connects the enhanced agent system with the existing MARA dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import time
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_enhanced_agents():
    """Load enhanced agent modules with error handling."""
    try:
        from agents.enhanced_data_agent import EnhancedDataAgent
        from agents.enhanced_strategy_agent import EnhancedStrategyAgent
        from agents.enhanced_system_manager import EnhancedSystemManager, SystemConfig
        from agents.enhanced_base_agent import AgentState, AgentConfig
        return {
            'EnhancedDataAgent': EnhancedDataAgent,
            'EnhancedStrategyAgent': EnhancedStrategyAgent,
            'EnhancedSystemManager': EnhancedSystemManager,
            'SystemConfig': SystemConfig,
            'AgentState': AgentState,
            'AgentConfig': AgentConfig,
            'available': True
        }
    except Exception as e:
        st.error(f"Enhanced agents not available: {e}")
        return {'available': False}

class EnhancedAgentIntegration:
    """Integration class for enhanced agents with dashboard."""
    
    def __init__(self):
        self.agents = load_enhanced_agents()
        self.cache_dir = tempfile.mkdtemp(prefix="dashboard_agents_")
        self.data_agent = None
        self.strategy_agent = None
        self.system_manager = None
        self._initialized = False
    
    def initialize_agents(self):
        """Initialize enhanced agents for dashboard use."""
        if not self.agents['available']:
            return False
        
        try:
            # Create agent configurations
            data_config = self.agents['AgentConfig'](
                cache_size=1000,
                cache_ttl=300.0,
                enable_caching=True,
                enable_metrics=True,
                max_retries=3,
                retry_delay=1.0
            )
            
            strategy_config = self.agents['AgentConfig'](
                cache_size=500,
                cache_ttl=180.0,
                enable_caching=True,
                enable_metrics=True,
                max_retries=2,
                retry_delay=0.5
            )
            
            # Initialize agents
            self.data_agent = self.agents['EnhancedDataAgent'](
                fetch_interval=60, 
                cache_dir=self.cache_dir
            )
            self.data_agent.config = data_config
            self.data_agent._use_synthetic_data = True
            
            self.strategy_agent = self.agents['EnhancedStrategyAgent'](
                cache_dir=self.cache_dir
            )
            self.strategy_agent.config = strategy_config
            
            # Initialize system manager
            system_config = self.agents['SystemConfig'](
                data_fetch_interval=60,
                enable_monitoring=True,
                restart_on_failure=True
            )
            self.system_manager = self.agents['EnhancedSystemManager'](system_config)
            
            self._initialized = True
            return True
            
        except Exception as e:
            st.error(f"Failed to initialize enhanced agents: {e}")
            return False
    
    def get_real_time_data(self):
        """Get real-time data from enhanced data agent."""
        if not self._initialized:
            if not self.initialize_agents():
                return None
        
        try:
            # Generate synthetic data
            prices_df, inventory_data = self.data_agent._generate_synthetic_data()
            
            # Add timestamp and format for dashboard
            current_time = datetime.now()
            data = {
                'timestamp': [current_time - timedelta(hours=i) for i in range(23, -1, -1)],
                'energy_price': prices_df['energy_price'].values,
                'hash_price': prices_df['hash_price'].values,
                'token_price': prices_df.get('token_price', prices_df['energy_price'] * 1.2).values,
                'battery_soc': [inventory_data['battery_soc']] * 24,
                'utilization_rate': [inventory_data['utilization_rate']] * 24,
                'power_available': [inventory_data.get('power_available', 500)] * 24,
            }
            
            return pd.DataFrame(data)
            
        except Exception as e:
            st.error(f"Error getting real-time data: {e}")
            return None
    
    def get_strategy_recommendations(self, market_data: Optional[pd.DataFrame] = None):
        """Get strategy recommendations from enhanced strategy agent."""
        if not self._initialized:
            if not self.initialize_agents():
                return None
        
        try:
            if market_data is None:
                market_data = self.get_real_time_data()
            
            if market_data is None:
                return None
            
            # Prepare data for strategy agent
            latest_data = market_data.iloc[-1]
            
            self.strategy_agent._last_features = {
                'prices': [{
                    'energy_price': latest_data['energy_price'],
                    'hash_price': latest_data['hash_price'],
                    'token_price': latest_data.get('token_price', latest_data['energy_price'] * 1.2)
                }],
                'inventory': {
                    'battery_soc': latest_data['battery_soc'],
                    'utilization_rate': latest_data['utilization_rate'],
                    'power_available': latest_data.get('power_available', 500)
                },
                'market_intelligence': {
                    'market_regime': {
                        'price_regime': 'high' if latest_data['energy_price'] > 3.5 else 'normal',
                        'volatility_regime': 'medium'
                    }
                }
            }
            
            # Generate strategy
            strategy = self.strategy_agent._generate_heuristic_strategy()
            risk_assessment = self.strategy_agent._assess_strategy_risk(strategy)
            
            return {
                'strategy': strategy,
                'risk_assessment': risk_assessment,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Error getting strategy recommendations: {e}")
            return None
    
    def get_agent_health(self):
        """Get health status of enhanced agents."""
        if not self._initialized:
            return None
        
        try:
            data_health = self.data_agent.health
            strategy_health = self.strategy_agent.health
            
            return {
                'data_agent': {
                    'state': self.data_agent.state.value,
                    'messages_processed': data_health.messages_processed,
                    'messages_failed': data_health.messages_failed,
                    'success_rate': data_health.success_rate,
                    'avg_processing_time': data_health.avg_processing_time,
                    'circuit_open': self.data_agent._circuit_open,
                    'cache_size': len(self.data_agent._cache)
                },
                'strategy_agent': {
                    'state': self.strategy_agent.state.value,
                    'messages_processed': strategy_health.messages_processed,
                    'messages_failed': strategy_health.messages_failed,
                    'success_rate': strategy_health.success_rate,
                    'avg_processing_time': strategy_health.avg_processing_time,
                    'circuit_open': self.strategy_agent._circuit_open,
                    'cache_size': len(self.strategy_agent._cache)
                }
            }
            
        except Exception as e:
            st.error(f"Error getting agent health: {e}")
            return None
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir, ignore_errors=True)
        except:
            pass

def create_enhanced_agent_panel():
    """Create enhanced agent monitoring panel for the dashboard."""
    st.header("🚀 Enhanced Agent System")
    
    # Initialize integration
    if 'enhanced_integration' not in st.session_state:
        st.session_state.enhanced_integration = EnhancedAgentIntegration()
    
    integration = st.session_state.enhanced_integration
    
    if not integration.agents['available']:
        st.error("❌ Enhanced agents not available. Please check installation.")
        return
    
    # Control panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Data", type="primary"):
            st.rerun()
    
    with col2:
        if st.button("🧠 Get Strategy"):
            with st.spinner("Generating strategy..."):
                strategy_data = integration.get_strategy_recommendations()
                if strategy_data:
                    st.session_state.latest_strategy = strategy_data
    
    with col3:
        if st.button("💊 Health Check"):
            with st.spinner("Checking agent health..."):
                health_data = integration.get_agent_health()
                if health_data:
                    st.session_state.agent_health = health_data
    
    # Real-time data display
    st.subheader("📊 Real-Time Market Data")
    market_data = integration.get_real_time_data()
    
    if market_data is not None:
        # Key metrics
        latest = market_data.iloc[-1]
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            prev_energy = market_data.iloc[-2]['energy_price']
            delta_energy = latest['energy_price'] - prev_energy
            st.metric("💰 Energy Price", f"${latest['energy_price']:.2f}", f"{delta_energy:+.2f}")
        
        with metric_col2:
            prev_hash = market_data.iloc[-2]['hash_price']
            delta_hash = latest['hash_price'] - prev_hash
            st.metric("⚡ Hash Price", f"${latest['hash_price']:.2f}", f"{delta_hash:+.2f}")
        
        with metric_col3:
            st.metric("🔋 Battery SOC", f"{latest['battery_soc']:.1%}")
        
        with metric_col4:
            st.metric("⚙️ Utilization", f"{latest['utilization_rate']:.1f}%")
        
        # Price chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Market Prices', 'System Status'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=market_data['timestamp'], y=market_data['energy_price'],
                      name='Energy Price', line=dict(color='#667eea', width=3)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=market_data['timestamp'], y=market_data['hash_price'],
                      name='Hash Price', line=dict(color='#764ba2', width=3)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=market_data['timestamp'], y=market_data['battery_soc'] * 100,
                      name='Battery SOC (%)', line=dict(color='#22c55e', width=2)),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=market_data['timestamp'], y=market_data['utilization_rate'],
                      name='Utilization (%)', line=dict(color='#f59e0b', width=2)),
            row=2, col=1
        )
        
        fig.update_layout(height=500, showlegend=True, title_text="Enhanced Agent Data")
        st.plotly_chart(fig, use_container_width=True)
    
    # Strategy recommendations
    if 'latest_strategy' in st.session_state:
        st.subheader("🧠 Latest Strategy Recommendations")
        strategy_data = st.session_state.latest_strategy
        
        strat_col1, strat_col2, strat_col3 = st.columns(3)
        
        with strat_col1:
            st.metric("⚡ Energy Allocation", 
                     f"{strategy_data['strategy']['energy_allocation']:.1%}")
            st.metric("🔨 Hash Allocation", 
                     f"{strategy_data['strategy']['hash_allocation']:.1%}")
        
        with strat_col2:
            st.metric("🔋 Battery Action", 
                     f"{strategy_data['strategy']['battery_charge_rate']:.1%}")
            st.metric("🎯 Confidence", 
                     f"{strategy_data['strategy']['confidence']:.1%}")
        
        with strat_col3:
            risk_level = strategy_data['risk_assessment']['level']
            risk_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk_level, "⚪")
            st.metric("⚠️ Risk Level", f"{risk_color} {risk_level.upper()}")
            st.caption(f"Generated: {strategy_data['timestamp'].strftime('%H:%M:%S')}")
    
    # Agent health status
    if 'agent_health' in st.session_state:
        st.subheader("🏥 Agent Health Status")
        health_data = st.session_state.agent_health
        
        health_col1, health_col2 = st.columns(2)
        
        with health_col1:
            st.markdown("**📊 Data Agent**")
            data_health = health_data['data_agent']
            st.write(f"State: **{data_health['state'].upper()}**")
            st.write(f"Messages: {data_health['messages_processed']} processed, {data_health['messages_failed']} failed")
            st.write(f"Success Rate: {data_health['success_rate']:.1%}")
            st.write(f"Cache Size: {data_health['cache_size']} items")
            st.write(f"Circuit Breaker: {'🔴 OPEN' if data_health['circuit_open'] else '🟢 CLOSED'}")
        
        with health_col2:
            st.markdown("**🧠 Strategy Agent**")
            strategy_health = health_data['strategy_agent']
            st.write(f"State: **{strategy_health['state'].upper()}**")
            st.write(f"Messages: {strategy_health['messages_processed']} processed, {strategy_health['messages_failed']} failed")
            st.write(f"Success Rate: {strategy_health['success_rate']:.1%}")
            st.write(f"Cache Size: {strategy_health['cache_size']} items")
            st.write(f"Circuit Breaker: {'🔴 OPEN' if strategy_health['circuit_open'] else '🟢 CLOSED'}")

# Demo function for standalone use
def main():
    """Main function for standalone enhanced agent dashboard."""
    st.set_page_config(
        page_title="Enhanced Agent Dashboard",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Enhanced Agent System Dashboard")
    create_enhanced_agent_panel()

if __name__ == "__main__":
    main() 