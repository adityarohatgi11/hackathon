# GridPilot-GT: Cursor Development Contracts

## Project Overview
GridPilot-GT is a sophisticated energy trading and GPU resource allocation system that uses forecasting, game theory, and real-time optimization to maximize revenue in energy markets while managing cooling costs.

## Architecture Overview
```
main.py (orchestrator)
├── api_client/ (market data & API interactions)
├── forecasting/ (price & demand prediction)
├── game_theory/ (bidding strategies & auction mechanisms)
├── control/ (cooling models & constraints)
└── dispatch/ (real-time execution)
```

## Common Dependencies
- httpx==0.25.2 (async HTTP client)
- pandas==2.1.4 (data manipulation)
- prophet==1.1.5 (time series forecasting)
- numpy==1.24.4 (numerical computing)
- cvxpy==1.4.1 (convex optimization)
- streamlit==1.28.2 (web UI)
- llama-cpp-python==0.2.20 (LLM integration)

---

## Lane A: Data & Forecasting Module

### Responsibility
Build robust data ingestion and forecasting capabilities for energy prices, demand patterns, and market volatility.

### Key Files to Create
- `api_client/__init__.py`
- `api_client/client.py` 
- `forecasting/__init__.py`
- `forecasting/forecaster.py`
- `forecasting/feature_engineering.py`
- `tests/test_forecasting.py`

### Core Functions Required

#### api_client/client.py
```python
def register_site(site_name: str) -> dict:
    """Register new site and get API credentials"""
    
def get_prices(start_time: str = None, end_time: str = None) -> pd.DataFrame:
    """Fetch historical energy prices"""
    
def get_inventory() -> dict:
    """Get current system inventory and status"""
    
def submit_bid(payload: dict) -> dict:
    """Submit bid to energy market"""
```

#### forecasting/forecaster.py
```python
class Forecaster:
    def predict_next(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Predict next period energy prices and uncertainty"""
        
    def predict_volatility(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Predict price volatility metrics"""
        
    def feature_importance(self) -> dict:
        """Return feature importance for model interpretability"""
```

### Technical Requirements
- Use Prophet for time series forecasting
- Include uncertainty quantification (prediction intervals)
- Handle missing data and outliers gracefully
- Implement feature engineering for market signals
- Add comprehensive unit tests with 80%+ coverage

---

## Lane B: Bidding & MPC Module

### Responsibility
Implement sophisticated bidding strategies using Model Predictive Control and game theory optimization.

### Key Files to Create
- `game_theory/__init__.py`
- `game_theory/bid_generators.py`
- `game_theory/mpc_controller.py`
- `game_theory/risk_models.py`
- `tests/test_bidding.py`

### Core Functions Required

#### game_theory/bid_generators.py
```python
def build_bid_vector(current_price: float, forecast: pd.DataFrame, 
                    uncertainty: pd.DataFrame, soc: float, 
                    lambda_deg: float) -> pd.DataFrame:
    """Generate optimal bid vector using MPC"""
    
def portfolio_optimization(bids: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """Optimize bid portfolio considering risk constraints"""
    
def dynamic_pricing_strategy(market_conditions: dict) -> dict:
    """Adapt bidding strategy based on market conditions"""
```

#### game_theory/mpc_controller.py
```python
class MPCController:
    def optimize_horizon(self, forecast: pd.DataFrame, current_state: dict) -> dict:
        """Multi-step ahead optimization using MPC"""
        
    def update_constraints(self, new_constraints: dict):
        """Update system constraints dynamically"""
```

### Technical Requirements
- Use CVXPY for convex optimization
- Implement rolling horizon MPC with 24-hour lookahead
- Include battery degradation costs in objective function
- Handle multiple uncertainty scenarios
- Extensive backtesting framework

---

## Lane C: Auction & Dispatch Module

### Responsibility
Implement VCG auction mechanisms and real-time dispatch optimization.

### Key Files to Create
- `game_theory/vcg_auction.py`
- `dispatch/__init__.py`
- `dispatch/dispatch_agent.py`
- `dispatch/execution_engine.py`
- `tests/test_auction.py`

### Core Functions Required

#### game_theory/vcg_auction.py
```python
def vcg_allocate(bids: pd.DataFrame, total_capacity: float) -> tuple:
    """VCG auction allocation and payment calculation"""
    
def auction_efficiency_metrics(allocation: dict, bids: pd.DataFrame) -> dict:
    """Calculate auction efficiency and fairness metrics"""
```

#### dispatch/dispatch_agent.py
```python
def build_payload(allocation: dict, inventory: dict, soc: float, 
                 cooling_kw: float, power_limit: float) -> dict:
    """Build market submission payload"""
    
def real_time_adjustment(current_payload: dict, market_signal: dict) -> dict:
    """Adjust dispatch in real-time based on market signals"""
    
def emergency_response(system_state: dict) -> dict:
    """Handle emergency situations and constraint violations"""
```

### Technical Requirements
- Implement truthful VCG auction mechanism
- Real-time dispatch with <100ms response time
- Comprehensive safety checks and emergency protocols
- Integration with cooling system constraints
- Detailed logging and monitoring

---

## Lane D: UI & LLM Integration

### Responsibility
Build intuitive Streamlit dashboard and integrate LLM for natural language interfaces.

### Key Files to Create
- `ui/__init__.py`
- `ui/dashboard.py`
- `ui/components/`
- `llm_integration/__init__.py`
- `llm_integration/chat_interface.py`
- `tests/test_ui.py`

### Core Functions Required

#### ui/dashboard.py
```python
def main_dashboard():
    """Main Streamlit dashboard with real-time updates"""
    
def performance_analytics():
    """Performance analytics and visualization"""
    
def system_monitoring():
    """Real-time system health monitoring"""
```

#### llm_integration/chat_interface.py
```python
class LLMInterface:
    def process_query(self, query: str, context: dict) -> str:
        """Process natural language queries about system state"""
        
    def generate_insights(self, data: pd.DataFrame) -> str:
        """Generate insights from system data"""
        
    def explain_decisions(self, decision_log: dict) -> str:
        """Explain system decisions in natural language"""
```

### Technical Requirements
- Beautiful, responsive Streamlit interface
- Real-time data visualization with Plotly
- Local LLM integration (no API dependencies)
- Natural language query interface
- Mobile-responsive design
- Comprehensive user documentation

---

## Integration Specifications

### Data Flow
1. **api_client** fetches market data
2. **forecasting** generates predictions
3. **game_theory** optimizes bids
4. **dispatch** executes trades
5. **ui** displays results and LLM provides insights

### Configuration Management
All modules must read from `config.toml`:
```toml
api_key = "YOUR_API_KEY"
site_power_kw = 1000
BATTERY_CAP_MWH = 1.0
BATTERY_MAX_KW = 250.0
SOC_MIN = 0.15
SOC_MAX = 0.90
BATTERY_EFF = 0.94
LAMBDA_DEG = 0.0002
```

### Testing Requirements
- Minimum 80% test coverage per module
- Integration tests between modules
- Performance benchmarks
- Mock external API calls
- Continuous integration via GitHub Actions

### Documentation Standards
- Comprehensive docstrings (Google style)
- Type hints for all functions
- README per module with examples
- API documentation
- Performance optimization notes

---

## Development Workflow

1. **Setup**: Each engineer creates their feature branch
2. **Development**: Implement module following contracts
3. **Testing**: Write comprehensive tests
4. **Integration**: Test with other modules via main.py
5. **Review**: Code review and CI checks
6. **Merge**: Merge to main when all checks pass

## Success Metrics
- All modules integrate seamlessly
- System processes real market data
- UI provides clear insights
- Performance meets real-time requirements
- Code quality passes all linting and testing 