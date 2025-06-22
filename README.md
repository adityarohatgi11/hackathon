# GridPilot: Advanced Energy Trading & Optimization Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)
[![API](https://img.shields.io/badge/API-MARA%20Hackathon%202025-orange.svg)](https://mara-hackathon-api.onrender.com)

A sophisticated energy trading and optimization platform that integrates real-time market data, advanced quantitative forecasting, and intelligent resource allocation for mining and compute operations.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Integration](#api-integration)
- [Testing](#testing)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

GridPilot-GT is a comprehensive energy management system designed for the MARA Hackathon 2025. It provides real-time energy trading optimization, advanced forecasting capabilities, and intelligent resource allocation across mining and compute workloads.

### Key Capabilities

- **Real-time Market Integration**: Live pricing data from MARA API
- **Advanced Forecasting**: Multi-model ensemble with uncertainty quantification
- **Intelligent Optimization**: Model Predictive Control (MPC) and game theory
- **Resource Management**: Dynamic allocation across mining and compute assets
- **Interactive Dashboard**: Real-time monitoring and control interface

## ✨ Features

### 🏗️ Core System
- **Multi-lane Architecture**: Modular design across data, optimization, dispatch, and UI
- **Real-time Processing**: Sub-second decision making and allocation
- **Robust Error Handling**: Automatic fallback mechanisms and graceful degradation
- **Comprehensive Testing**: 30+ tests with 100% pass rate

### 📊 Advanced Analytics
- **Quantitative Models**: GARCH volatility, Kalman filtering, Wavelet analysis
- **Machine Learning Ensemble**: XGBoost, Gaussian Process, Prophet forecasting
- **Feature Engineering**: 102+ technical indicators and market patterns
- **Uncertainty Quantification**: Bootstrap sampling and model disagreement metrics

### 🎮 Optimization Engine
- **Model Predictive Control**: CVXPY-based power allocation optimization
- **Game Theory**: Strategic bidding with market equilibrium analysis
- **Portfolio Optimization**: Mean-variance and risk parity allocation
- **Constraint Management**: Battery SOC, thermal limits, grid requirements

## 🏗️ Architecture

The system operates across four integrated development lanes:

```
GridPilot-GT/
├── Lane A: Data & Forecasting     # Real-time data processing & ML models
├── Lane B: Bidding & MPC         # Optimization & game theory
├── Lane C: Auction & Dispatch    # Resource allocation & execution
└── Lane D: UI & LLM              # Dashboard & natural language interface
```

### Lane A: Data & Forecasting
- **MARA API Integration**: Real-time energy, hash, and token prices
- **Advanced Quantitative Models**: GARCH volatility, Kalman filtering, Wavelet analysis
- **Machine Learning Ensemble**: XGBoost + Gaussian Process + Prophet forecasting
- **Feature Engineering**: 102+ technical indicators and market patterns

### Lane B: Bidding & MPC
- **Model Predictive Control**: CVXPY optimization for power allocation
- **Game Theory**: Strategic bidding with market equilibrium analysis
- **Portfolio Optimization**: Mean-variance and risk parity allocation
- **Constraint Handling**: Battery SOC, thermal limits, grid requirements

### Lane C: Auction & Dispatch
- **VCG Auctions**: Vickrey-Clarke-Groves mechanism for fair allocation
- **Real-time Execution**: Sub-second decision making and allocation
- **Load Balancing**: Optimal distribution across mining and compute workloads

### Lane D: UI & LLM
- **Streamlit Dashboard**: Real-time monitoring and control
- **LLM Integration**: Natural language system interaction
- **Performance Analytics**: ROI tracking, efficiency metrics, market insights

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM (for advanced models)
- Internet connection (for MARA API integration)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd hackathon
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Access
```bash
# Edit config.toml with your MARA API credentials
cp config.toml.example config.toml
# Update with your API key and site configuration
```

### 4. Test Installation
```bash
python test_mara_api.py
```

### 5. Launch the System
```bash
python main.py
```

## 📦 Installation

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 8GB+ RAM recommended
- **Storage**: 2GB+ free space
- **Network**: Internet connection for API integration

### Dependencies

Core dependencies are managed through `requirements.txt`:

```bash
# Core framework
streamlit>=1.28.0
pandas>=2.0.0
numpy==1.24.4

# Quantitative modeling
arch==5.6.0           # GARCH models
filterpy==1.4.5       # Kalman filtering
PyWavelets==1.4.1     # Wavelet analysis
xgboost==2.0.3        # Gradient boosting
prophet==1.1.5        # Time series forecasting

# Optimization
cvxpy==1.4.1          # Convex optimization
scikit-learn==1.3.2   # Machine learning

# API and utilities
httpx==0.25.2         # HTTP client
requests==2.31.0      # HTTP library
toml>=0.10.2          # Configuration parsing

# Testing and development
pytest==7.4.3         # Testing framework
ruff==0.1.6           # Code linting
```

### Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Configuration

### API Configuration
Create or update `config.toml` with your MARA API credentials:

```toml
[api]
api_key = "YOUR_MARA_API_KEY_HERE"
site_name = "YourSiteName"
site_power_kw = 1000000  # 1MW default

[system]
debug = false
log_level = "INFO"
```

### Environment Variables
```bash
export MARA_API_KEY="your_api_key_here"
export SITE_NAME="your_site_name"
export SITE_POWER_KW="1000000"
```

## 🎮 Usage

### Starting the System
```bash
# Run the main application
python main.py

# Or run specific components
python -m hackathon.forecasting.advanced_forecaster
python -m hackathon.game_theory.mpc_controller
```

### Dashboard Access
Once running, access the Streamlit dashboard at:
```
http://localhost:8501
```

### API Testing
```bash
# Test MARA API integration
python test_mara_api.py

# Run comprehensive test suite
pytest tests/ -v
```

## 🔌 API Integration

### MARA Hackathon 2025 API

The system integrates with the MARA Hackathon 2025 API for real-time data:

#### Pricing Data (No Auth Required)
```bash
GET https://mara-hackathon-api.onrender.com/prices
# Returns: energy_price, hash_price, token_price, timestamp
```

#### Inventory Management (Auth Required)
```bash
GET https://mara-hackathon-api.onrender.com/inventory
# Returns: inference assets, miners, power allocation, tokens
```

#### Machine Allocation (Auth Required)
```bash
PUT https://mara-hackathon-api.onrender.com/machines
# Body: {"air_miners": 0, "asic_compute": 5, "gpu_compute": 30, ...}
```

#### Site Registration
```bash
POST https://mara-hackathon-api.onrender.com/sites
# Body: {"api_key": "XXX", "name": "SiteName", "power": 1000000}
```

### API Client Usage
```python
from hackathon.api_client.client import MARAClient

# Initialize client
client = MARAClient(api_key="your_key")

# Fetch real-time prices
prices = client.get_prices()

# Get inventory
inventory = client.get_inventory()

# Allocate machines
allocation = client.allocate_machines({
    "air_miners": 0,
    "asic_compute": 5,
    "gpu_compute": 30
})
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_lane_a.py -v        # Data & Forecasting
pytest tests/test_robustness.py -v    # Robustness & Edge Cases
pytest tests/test_basic.py -v         # Basic Integration

# Run with coverage
pytest tests/ --cov=hackathon --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component functionality
- **API Tests**: MARA API integration validation
- **Robustness Tests**: Error handling and edge cases
- **Performance Tests**: System performance validation

### Test Coverage
- **30+ comprehensive tests** with 100% pass rate
- **API integration testing** with fallback validation
- **Robustness testing** for missing data, extreme outliers
- **End-to-end simulation** with complete market cycles

## 🛠️ Development

### Project Structure
```
hackathon/
├── api_client/           # MARA API integration
├── control/              # System control and optimization
├── dispatch/             # Resource allocation and execution
├── forecasting/          # Advanced forecasting models
├── game_theory/          # Game theory and bidding strategies
├── llm_integration/      # LLM interface components
├── tests/                # Comprehensive test suite
├── ui/                   # Streamlit dashboard components
├── config.toml           # Configuration file
├── main.py              # Main application entry point
└── requirements.txt     # Python dependencies
```

### Code Quality
```bash
# Run linting
ruff check .

# Format code
ruff format .

# Type checking (if using mypy)
mypy hackathon/
```

### Development Workflow
1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Add** tests for new functionality
5. **Run** the test suite
6. **Submit** a pull request

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Contribution Guidelines
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/hackathon.git
cd hackathon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If available

# Run tests
pytest tests/ -v
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write comprehensive docstrings
- Include tests for new functionality

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MARA Hackathon 2025** for providing the API and challenge
- **Open source community** for the excellent libraries used in this project
- **Contributors** who have helped improve the system

## 📞 Support

For support and questions:
- **Issues**: [GitHub Issues](https://github.com/yourusername/hackathon/issues)
- **Documentation**: Check the [docs/](docs/) directory
- **Email**: [your-email@example.com]

---

**GridPilot-GT is ready for the MARA Hackathon 2025! 🚀**
