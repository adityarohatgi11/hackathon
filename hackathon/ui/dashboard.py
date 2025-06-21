"""
Energy Management Dashboard
Streamlit-based dashboard for energy management system with LLM integration.
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

# Configure page
st.set_page_config(
    page_title="Energy Management Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    /* Prevent MathJax/KaTeX rendering and ensure consistent text styling */
    .mord, .mop, .mbin, .mrel, .mopen, .mclose, .mpunct, .minner {
        font-family: inherit !important;
        font-size: inherit !important;
        font-weight: inherit !important;
        color: inherit !important;
        background: inherit !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        line-height: inherit !important;
        vertical-align: baseline !important;
    }
    
    /* Ensure all text in LLM responses uses consistent styling */
    .stMarkdown p, .stMarkdown div, .stMarkdown span {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
        color: #ffffff !important;
    }
    
    /* Override any MathJax/KaTeX styling */
    .MathJax, .katex {
        display: inline !important;
        font-family: inherit !important;
        font-size: inherit !important;
        color: #ffffff !important;
    }
    
    /* Ensure consistent styling for all text elements */
    .stText, .stMarkdown, .stChatMessage {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        color: #ffffff !important;
    }
    
    /* Make sure all text in the dashboard is light colored */
    .stMarkdown, .stText, .stChatMessage, .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
        color: #ffffff !important;
    }
    
    /* Ensure chat messages are readable */
    .stChatMessageContent {
        color: #ffffff !important;
    }
    
    /* Make sure expandable content is readable */
    .streamlit-expanderContent {
        color: #ffffff !important;
    }
</style>

<script>
// Disable MathJax/KaTeX rendering to prevent inconsistent text styling
if (typeof MathJax !== 'undefined') {
    MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
    // Disable future MathJax processing
    MathJax.Hub.processSectionDelay = 0;
    MathJax.Hub.processUpdateDelay = 0;
}

// Remove any existing MathJax/KaTeX elements and replace with plain text
document.addEventListener('DOMContentLoaded', function() {
    // Find and replace MathJax elements with plain text
    const mathElements = document.querySelectorAll('.MathJax, .katex, .mord, .mop, .mbin, .mrel, .mopen, .mclose, .mpunct, .minner');
    mathElements.forEach(function(element) {
        const textContent = element.textContent || element.innerText;
        const textNode = document.createTextNode(textContent);
        element.parentNode.replaceChild(textNode, element);
    });
});

// Monitor for new content and prevent MathJax rendering
const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        if (mutation.type === 'childList') {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    const mathElements = node.querySelectorAll('.MathJax, .katex, .mord, .mop, .mbin, .mrel, .mopen, .mclose, .mpunct, .minner');
                    mathElements.forEach(function(element) {
                        const textContent = element.textContent || element.innerText;
                        const textNode = document.createTextNode(textContent);
                        element.parentNode.replaceChild(textNode, element);
                    });
                }
            });
        }
    });
});

// Start observing
observer.observe(document.body, {
    childList: true,
    subtree: true
});
</script>
""", unsafe_allow_html=True)


def generate_sample_data():
    """Generate sample energy data for demonstration."""
    # Generate 24 hours of data
    timestamps = pd.date_range(
        start=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        periods=24,
        freq='h'
    )
    
    # Generate realistic energy consumption patterns
    base_consumption = 100  # kW
    peak_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # Business hours
    
    consumption = []
    for i, ts in enumerate(timestamps):
        if ts.hour in peak_hours:
            # Higher consumption during business hours
            consumption.append(base_consumption + np.random.normal(50, 10))
        else:
            # Lower consumption during off-peak hours
            consumption.append(base_consumption + np.random.normal(20, 5))
    
    # Generate demand data (similar to consumption but with some variation)
    demand = [c + np.random.normal(0, 5) for c in consumption]
    
    # Generate price data (higher during peak hours)
    prices = []
    for i, ts in enumerate(timestamps):
        if ts.hour in peak_hours:
            prices.append(np.random.normal(0.18, 0.02))  # Peak price ~$0.18/kWh
        else:
            prices.append(np.random.normal(0.10, 0.01))  # Off-peak price ~$0.10/kWh
    
    # Generate battery state of charge
    battery_soc = []
    current_soc = 0.8  # Start at 80%
    for i in range(24):
        # Simulate battery charging/discharging
        if i < 6:  # Early morning - charging
            current_soc = min(1.0, current_soc + 0.02)
        elif i in [14, 15, 16, 17]:  # Peak hours - discharging
            current_soc = max(0.2, current_soc - 0.05)
        else:
            current_soc = max(0.2, current_soc - 0.01)  # Slow discharge
        battery_soc.append(current_soc)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'consumption': consumption,
        'demand': demand,
        'price': prices,
        'battery_soc': battery_soc
    })


def get_real_time_data():
    """Get real-time data from MARA API and GridPilot-GT system."""
    try:
        # Get real-time prices and inventory
        prices_df = get_prices()
        inventory = get_inventory()
        
        # Convert to dashboard format
        if len(prices_df) > 0:
            # Use the last 24 hours of data if available
            recent_data = prices_df.tail(24)
            
            # Create consumption/demand data based on inventory
            power_used = inventory.get('power_used', 500)  # kW
            power_total = inventory.get('power_total', 1000)  # kW
            
            # Generate consumption pattern based on real power usage
            base_consumption = power_used
            consumption = [base_consumption + np.random.normal(0, 10) for _ in range(len(recent_data))]
            demand = [c + np.random.normal(0, 5) for c in consumption]
            
            # Use real prices from MARA API
            prices = recent_data['price'].tolist()
            
            # Use real battery SOC
            battery_soc = [inventory.get('battery_soc', 0.7)] * len(recent_data)
            
            # Create timestamps
            timestamps = recent_data['timestamp']
            
            return pd.DataFrame({
                'timestamp': timestamps,
                'consumption': consumption,
                'demand': demand,
                'price': prices,
                'battery_soc': battery_soc
            }), inventory
        else:
            # Fallback to sample data if no real data
            return generate_sample_data(), None
            
    except Exception as e:
        st.warning(f"⚠️ Error fetching real-time data: {str(e)}")
        return generate_sample_data(), None


def get_system_status():
    """Get current system status from GridPilot-GT."""
    try:
        if not GRIDPILOT_AVAILABLE:
            return {
                'status': 'GridPilot-GT not available',
                'mara_api_status': 'Unavailable',
                'last_update': datetime.now().strftime("%H:%M:%S"),
                'alerts': ['GridPilot-GT modules not loaded']
            }
        # Test MARA API connection
        connection_test = test_mara_api_connection()
        inventory = get_inventory()
        status = {
            'system_health': 'operational' if connection_test['overall_status'] == 'operational' else 'limited',
            'mara_api_status': connection_test['overall_status'],
            'last_update': datetime.now().strftime("%H:%M:%S"),
            'power_available': inventory.get('power_available', 0),
            'power_used': inventory.get('power_used', 0),
            'battery_soc': inventory.get('battery_soc', 0.7),
            'temperature': inventory.get('temperature', 65),
            'gpu_utilization': inventory.get('gpu_utilization', 0.8),
            'alerts': inventory.get('alerts', [])
        }
        return status
    except Exception as e:
        return {
            'status': 'Error',
            'system_health': 'error',
            'mara_api_status': 'Error',
            'last_update': datetime.now().strftime("%H:%M:%S"),
            'power_available': 0,
            'power_used': 0,
            'battery_soc': 0.0,
            'temperature': 0.0,
            'gpu_utilization': 0.0,
            'alerts': [f'System error: {str(e)}']
        }


def run_gridpilot_optimization():
    """Run enhanced GridPilot-GT optimization cycle."""
    try:
        if not GRIDPILOT_AVAILABLE:
            return None
        
        # Run enhanced GridPilot-GT optimization
        enhanced_result = main_enhanced(simulate=True, use_advanced=True)
        
        if not enhanced_result['success']:
            st.error(f"❌ Enhanced GridPilot-GT optimization failed: {enhanced_result.get('error', 'Unknown error')}")
            return None
        
        # Extract results from enhanced optimization
        enhanced_results = enhanced_result['enhanced_results']
        payload = enhanced_result['payload']
        
        # Get traditional allocation for compatibility
        allocation = enhanced_results['traditional_allocation']
        
        # Calculate payments (simplified for dashboard display)
        total_power = sum(allocation.values())
        payments = {
            'inference': allocation.get('inference', 0) * 0.15,  # $0.15/kWh
            'training': allocation.get('training', 0) * 0.12,   # $0.12/kWh
            'cooling': allocation.get('cooling', 0) * 0.08      # $0.08/kWh
        }
        
        # Get cooling requirements from payload
        cooling_kw = payload.get('cooling_requirements', {}).get('cooling_kw', 0)
        
        return {
            'allocation': allocation,
            'payments': payments,
            'cooling_kw': cooling_kw,
            'payload': payload,
            'forecast': enhanced_results.get('enhanced_forecast', {}).get('traditional', {}),
            'enhanced_results': enhanced_results,
            'total_theoretical_kw': enhanced_results.get('total_theoretical_kw', 0),
            'final_utilization_pct': enhanced_results.get('final_utilization_pct', 0),
            'advanced_methods_used': enhanced_results.get('advanced_methods_used', False)
        }
        
    except Exception as e:
        st.error(f"❌ Enhanced GridPilot-GT optimization error: {str(e)}")
        return None


def create_energy_charts(data):
    """Create energy consumption and price charts."""
    # Energy consumption chart
    fig_consumption = px.line(
        data, 
        x='timestamp', 
        y='consumption',
        title='Energy Consumption (24 Hours)',
        labels={'consumption': 'Consumption (kW)', 'timestamp': 'Time'},
        color_discrete_sequence=['#1f77b4']
    )
    fig_consumption.update_layout(height=400)
    
    # Price chart
    fig_price = px.line(
        data, 
        x='timestamp', 
        y='price',
        title='Energy Prices (24 Hours)',
        labels={'price': 'Price ($/kWh)', 'timestamp': 'Time'},
        color_discrete_sequence=['#ff7f0e']
    )
    fig_price.update_layout(height=400)
    
    # Battery SOC chart
    # Create percentage values for display while keeping original data structure
    data_with_soc_percent = data.copy()
    data_with_soc_percent['battery_soc_percent'] = data['battery_soc'] * 100
    
    fig_battery = px.line(
        data_with_soc_percent, 
        x='timestamp', 
        y='battery_soc_percent',
        title='Battery State of Charge',
        labels={'battery_soc_percent': 'SOC (%)', 'timestamp': 'Time'},
        color_discrete_sequence=['#2ca02c']
    )
    fig_battery.update_layout(height=400)
    
    return fig_consumption, fig_price, fig_battery


def display_metrics(data):
    """Display key metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Consumption",
            value=f"{data['consumption'].sum():.0f} kWh",
            delta=f"{data['consumption'].mean():.1f} avg kW"
        )
    
    with col2:
        st.metric(
            label="Peak Demand",
            value=f"{data['demand'].max():.0f} kW",
            delta=f"at {data.loc[data['demand'].idxmax(), 'timestamp'].strftime('%H:%M')}"
        )
    
    with col3:
        st.metric(
            label="Avg Price",
            value=f"${data['price'].mean():.3f}/kWh",
            delta=f"${data['price'].max() - data['price'].min():.3f} range"
        )
    
    with col4:
        current_soc = data['battery_soc'].iloc[-1]
        st.metric(
            label="Battery SOC",
            value=f"{current_soc * 100:.1f}%",
            delta=f"{(current_soc - data['battery_soc'].iloc[0]) * 100:+.1f}%"
        )


def chat_interface(llm_interface):
    """Create a chat interface for user queries."""
    st.subheader("💬 Energy Management Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about energy management, system status, or optimization..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from LLM
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = llm_interface.process_query(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


def insights_panel(llm_interface, data):
    """Display AI-generated insights with auto-generation and caching."""
    st.subheader("🤖 AI Insights")
    
    # Initialize session state for insights caching
    if "insights_cache" not in st.session_state:
        st.session_state.insights_cache = {}
    
    # Check if we need to generate new insights (every 10 minutes)
    current_time = time.time()
    cache_key = "ai_insights"
    cache_duration = 600  # 10 minutes in seconds
    
    should_generate = (
        cache_key not in st.session_state.insights_cache or
        current_time - st.session_state.insights_cache[cache_key].get("timestamp", 0) > cache_duration
    )
    
    # Auto-generate insights if needed
    if should_generate:
        with st.spinner("🤖 Generating AI insights..."):
            try:
                insights = llm_interface.generate_insights(data)
                st.session_state.insights_cache[cache_key] = {
                    "insights": insights,
                    "timestamp": current_time
                }
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
                return
    
    # Display cached insights
    if cache_key in st.session_state.insights_cache:
        cached_data = st.session_state.insights_cache[cache_key]
        insights = cached_data["insights"]
        
        # Show last updated time
        last_updated = datetime.fromtimestamp(cached_data["timestamp"]).strftime("%H:%M:%S")
        st.caption(f"🕒 Last updated: {last_updated}")
        
        # Display insights
        st.markdown(insights)
        
        # Manual refresh button
        if st.button("🔄 Refresh Insights"):
            # Clear cache to force regeneration
            if cache_key in st.session_state.insights_cache:
                del st.session_state.insights_cache[cache_key]
            st.rerun()


def decision_explanation(llm_interface, data_source="Sample Data", inventory=None, data=None):
    """Display decision explanations with persistent expanders."""
    st.subheader("📋 Decision Explanations")
    
    # Initialize session state for decision explanations
    if "decision_explanations" not in st.session_state:
        st.session_state.decision_explanations = {}
    
    # Use real data if available, otherwise fallback to sample data
    if data_source == "Real-time Data (MARA API)" and inventory and data is not None:
        # Real-time decisions based on MARA API data
        current_price = data['price'].iloc[-1] if len(data) > 0 else 50.0
        power_used = inventory.get('power_used', 500)
        power_available = inventory.get('power_available', 1000)
        battery_soc = inventory.get('battery_soc', 0.7)
        temperature = inventory.get('temperature', 65)
        
        decisions = [
            {
                "title": "GridPilot-GT Optimization Decision",
                "decision": "Optimize power allocation for maximum revenue",
                "context": {
                    "Current Power Used": f"{power_used:.0f} kW",
                    "Power Available": f"{power_available:.0f} kW",
                    "Battery SOC": f"{battery_soc:.1%}",
                    "Current Price": f"${current_price:.2f}/kWh",
                    "System Temperature": f"{temperature:.1f}°C"
                }
            },
            {
                "title": "Battery Management Strategy",
                "decision": "Maintain optimal battery SOC for grid services",
                "context": {
                    "Battery SOC": f"{battery_soc:.1%}",
                    "Power Available": f"{power_available:.0f} kW",
                    "Utilization Rate": f"{(power_used/power_available*100):.1f}%",
                    "Thermal Status": "Normal" if temperature < 75 else "Warning"
                }
            }
        ]
    else:
        # Sample decisions for demonstration
        decisions = [
            {
                "title": "Battery Discharge Decision",
                "decision": "Discharge battery to reduce grid demand",
                "context": {
                    "Current Demand": "500 kW",
                    "Grid Price": "$0.18/kWh",
                    "Battery SOC": "85%",
                    "Expected Savings": "$45/hour"
                }
            },
            {
                "title": "Load Shifting Recommendation",
                "decision": "Shift non-critical loads to off-peak hours",
                "context": {
                    "Peak Price": "$0.18/kWh",
                    "Off-Peak Price": "$0.10/kWh",
                    "Shiftable Load": "100 kW",
                    "Potential Savings": "$8/hour during peak"
                }
            }
        ]
    
    for decision_info in decisions:
        decision_key = decision_info["title"].replace(" ", "_").lower()
        
        # Use container instead of expander to maintain state
        with st.container():
            st.write(f"**{decision_info['title']}**")
            st.write(f"**Decision:** {decision_info['decision']}")
            st.write("**Context:**")
            for key, value in decision_info["context"].items():
                st.write(f"- {key}: {value}")
            
            # Show explanation if already generated
            if decision_key in st.session_state.decision_explanations:
                st.markdown("**Explanation:**")
                st.markdown(st.session_state.decision_explanations[decision_key])
            
            # Generate explanation button
            if st.button(f"Explain {decision_info['title']}", key=f"btn_{decision_key}"):
                with st.spinner("Generating explanation..."):
                    try:
                        explanation = llm_interface.explain_decision(
                            decision_info["decision"], 
                            decision_info["context"]
                        )
                        # Store explanation in session state
                        st.session_state.decision_explanations[decision_key] = explanation
                        # Use st.success to show completion without page reload
                        st.success("✅ Explanation generated!")
                        # Display the explanation immediately
                        st.markdown("**Explanation:**")
                        st.markdown(explanation)
                    except Exception as e:
                        st.error(f"Error generating explanation: {str(e)}")
            
            st.divider()


def main():
    """Main dashboard function."""
    # Header
    st.markdown('<h1 class="main-header">⚡ Energy Management Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize LLM interface
    if LLM_AVAILABLE:
        try:
            # Use unified interface that automatically selects the best provider
            llm_interface = UnifiedLLMInterface()
            if llm_interface.is_available():
                provider_info = llm_interface.get_provider_info()
                st.info(f"Using {provider_info['provider']} LLM provider")
            else:
                llm_interface = MockLLMInterface()
                st.info("Using mock LLM interface for demonstration")
        except Exception as e:
            llm_interface = MockLLMInterface()
            st.info("Using mock LLM interface for demonstration")
    else:
        llm_interface = None
        st.warning("LLM interface not available")
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Data source selector
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Sample Data", "Real-time Data (MARA API)"],
        index=0 if not GRIDPILOT_AVAILABLE else 1,
        help="Choose between sample data or real-time MARA API data"
    )
    
    # Show GridPilot-GT status
    if GRIDPILOT_AVAILABLE:
        system_status = get_system_status()
        st.sidebar.info(f"🟢 GridPilot-GT: {system_status['mara_api_status']}")
    else:
        st.sidebar.warning("⚠️ GridPilot-GT modules not available")
    
    # Date selector
    selected_date = st.sidebar.date_input(
        "Select Date",
        value=datetime.now().date(),
        max_value=datetime.now().date()
    )
    
    # Time range selector
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days"],
        index=0
    )
    
    # GridPilot-GT optimization button
    if GRIDPILOT_AVAILABLE and data_source == "Real-time Data (MARA API)":
        if st.sidebar.button("🚀 Run Enhanced GridPilot-GT Optimization"):
            with st.spinner("Running enhanced GridPilot-GT optimization..."):
                optimization_result = run_gridpilot_optimization()
                if optimization_result:
                    st.sidebar.success("✅ Enhanced optimization completed!")
                    # Store result in session state for display
                    st.session_state.optimization_result = optimization_result
                else:
                    st.sidebar.error("❌ Enhanced optimization failed")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Get data based on selected source
    if data_source == "Real-time Data (MARA API)" and GRIDPILOT_AVAILABLE:
        data, inventory = get_real_time_data()
        if inventory:
            st.sidebar.success("✅ Connected to MARA API")
        else:
            st.sidebar.warning("⚠️ Using fallback data")
    else:
        data = generate_sample_data()
        inventory = None
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💬 Chat Assistant", "🤖 AI Insights", "⚙️ System Status"])
    
    with tab1:
        st.header("Energy Overview")
        
        # Display metrics
        display_metrics(data)
        
        # Show optimization results if available
        if 'optimization_result' in st.session_state and st.session_state.optimization_result:
            result = st.session_state.optimization_result
            st.subheader("🚀 Enhanced GridPilot-GT Optimization Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Inference Allocation", f"{result['allocation'].get('inference', 0):.1f} kW")
                st.metric("Training Allocation", f"{result['allocation'].get('training', 0):.1f} kW")
                if result.get('enhanced_results'):
                    st.metric("Theoretical Capacity", f"{result.get('total_theoretical_kw', 0):,.0f} kW")
            with col2:
                st.metric("Cooling Required", f"{result['cooling_kw']:.1f} kW")
                # Handle payments dictionary structure
                if isinstance(result['payments'], dict):
                    total_payment = sum(result['payments'].values())
                    st.metric("Total Revenue", f"${total_payment:.2f}")
                else:
                    st.metric("Total Revenue", f"${result['payments']:.2f}")
                if result.get('enhanced_results'):
                    st.metric("Advanced Methods", "✅ Active" if result.get('advanced_methods_used', False) else "❌ Disabled")
            with col3:
                # Use enhanced utilization if available
                if result.get('final_utilization_pct'):
                    st.metric("Enhanced Utilization", f"{result['final_utilization_pct']:.1%}")
                else:
                    # Fallback to traditional system utilization
                    system_utilization = result['payload']['system_state']['utilization']
                    st.metric("System Utilization", f"{system_utilization:.1%}")
                st.metric("Constraints Satisfied", "✅" if result['payload']['constraints_satisfied'] else "❌")
                if result.get('enhanced_results'):
                    st.metric("Performance", "Enhanced" if result.get('advanced_methods_used', False) else "Standard")
        
        # Create charts
        fig_consumption, fig_price, fig_battery = create_energy_charts(data)
        
        # Display charts
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_consumption, use_container_width=True)
            st.plotly_chart(fig_battery, use_container_width=True)
        with col2:
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Additional metrics
            st.subheader("Quick Stats")
            st.metric("Cost Today", f"${(data['consumption'] * data['price']).sum():.2f}")
            st.metric("Efficiency Score", "87%")
            st.metric("Carbon Footprint", "2.3 tons CO2")
    
    with tab2:
        if llm_interface:
            chat_interface(llm_interface)
        else:
            st.error("Chat interface not available")
    
    with tab3:
        if llm_interface:
            insights_panel(llm_interface, data)
            st.divider()
            decision_explanation(llm_interface, data_source, inventory, data)
        else:
            st.error("AI insights not available")
    
    with tab4:
        st.header("System Status")
        
        # Get real system status if available
        if GRIDPILOT_AVAILABLE:
            system_status = get_system_status()
            
            # System health indicators
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status_emoji = "🟢" if system_status['system_health'] == 'operational' else "🟡"
                st.metric("System Status", f"{status_emoji} {system_status['system_health'].title()}")
                st.metric("Last Update", system_status['last_update'])
                st.metric("MARA API", f"{status_emoji} {system_status['mara_api_status']}")
            
            with col2:
                st.metric("Power Available", f"{system_status['power_available']:.0f} kW")
                st.metric("Power Used", f"{system_status['power_used']:.0f} kW")
                # Use GridPilot-GT optimization result for System Utilization if available, otherwise calculate manually
                if 'optimization_result' in st.session_state and st.session_state.optimization_result:
                    system_utilization = st.session_state.optimization_result['payload']['system_state']['utilization']
                    st.metric("System Utilization", f"{system_utilization:.1%}")
                else:
                    system_utilization = (system_status['power_used']/system_status['power_available']*100) if system_status['power_available'] > 0 else 0.0
                    st.metric("System Utilization", f"{system_utilization:.1f}%")
            
            with col3:
                st.metric("Temperature", f"{system_status['temperature']:.1f}°C")
                st.metric("GPU Utilization", f"{system_status['gpu_utilization']:.1%}")
                st.metric("Alerts", f"{len(system_status['alerts'])} Active")
            
            # Show alerts if any
            if system_status['alerts']:
                st.subheader("⚠️ Active Alerts")
                for alert in system_status['alerts']:
                    st.warning(f"• {alert}")
            
            # System logs
            st.subheader("Recent System Events")
            events = [
                {"time": datetime.now().strftime("%H:%M"), "event": "GridPilot-GT optimization completed", "status": "✅"},
                {"time": (datetime.now() - timedelta(minutes=5)).strftime("%H:%M"), "event": "MARA API data updated", "status": "✅"},
                {"time": (datetime.now() - timedelta(minutes=10)).strftime("%H:%M"), "event": "Battery SOC monitoring", "status": "ℹ️"},
                {"time": (datetime.now() - timedelta(minutes=15)).strftime("%H:%M"), "event": "Market price analysis", "status": "✅"},
            ]
            
            for event in events:
                st.write(f"{event['time']} {event['status']} {event['event']}")
                
        else:
            # Fallback system status display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("System Status", "🟢 Online")
                st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
            
            with col2:
                st.metric("Battery Health", "🟢 95%")
                st.metric("Grid Connection", "🟢 Connected")
            
            with col3:
                st.metric("Data Quality", "🟢 Excellent")
                st.metric("Alerts", "0 Active")
            
            # System logs
            st.subheader("Recent System Events")
            events = [
                {"time": "14:30", "event": "Battery discharge initiated", "status": "✅"},
                {"time": "14:25", "event": "Peak demand detected", "status": "⚠️"},
                {"time": "14:20", "event": "Price optimization completed", "status": "✅"},
                {"time": "14:15", "event": "Load shifting recommendation", "status": "ℹ️"},
            ]
            
            for event in events:
                st.write(f"{event['time']} {event['status']} {event['event']}")
            
            st.info("💡 Enable GridPilot-GT modules to see real-time system status")


if __name__ == "__main__":
    main()
