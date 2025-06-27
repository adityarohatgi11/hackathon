# GridPilot-GT Installation Guide

## System Requirements Check Results

**Your System Status:**
- **Python Version**: 3.12.4 (Compatible - 3.8+ required)
- **Package Manager**: pip
- **Conda**: Available
- **Homebrew**: 4.5.7
- **Git**: Installed
- **Operating System**: macOS (Darwin)

## Missing Dependencies (21 packages)


### Main Requirements (17 missing):
- `httpx==0.25.2` (you have 0.27.0 - version mismatch)
- `prophet==1.1.5` (you have 1.1.7 - version mismatch)  
- `numpy==1.24.4` (you have 1.26.4 - version mismatch)
- `cvxpy==1.4.1` (not installed)
- `llama-cpp-python>=0.2.0` (not installed)
- `scikit-learn==1.3.2` (you have 1.4.2 - version mismatch)
- `matplotlib==3.8.2` (you have 3.8.4 - version mismatch)
- `requests==2.31.0` (you have 2.32.2 - version mismatch)
- `pytest==7.4.3` (you have 7.4.4 - version mismatch)
- `ruff==0.1.6` (not installed)
- `arch==5.6.0` (you have 7.2.0 - version mismatch)
- `filterpy==1.4.5` (not installed)
- `PyWavelets==1.4.1` (you have 1.5.0 - version mismatch)
- `xgboost==2.0.3` (not installed)
- `redis==5.0.1` (not installed)
- `chromadb>=0.4.22` (not installed)
- `ray[rllib]>=2.9.3` (not installed)

### API Requirements (4 missing):
- `fastapi==0.104.1` (you have 0.115.13 - version mismatch)
- `uvicorn==0.24.0` (you have 0.34.3 - version mismatch)
- `websockets==12.0` (you have 15.0.1 - version mismatch)
- `pydantic==2.5.0` (you have 2.5.3 - version mismatch)

### Optional System Dependencies:
- `Redis Server` (not installed - needed for advanced features)

## Installation Options

### Option 1: Quick Install (Recommended)

Install all dependencies at once:
```bash
pip install -r hackathon/requirements.txt -r hackathon/requirements_api.txt
```

### Option 2: Targeted Install

Install only the missing packages:
```bash
pip install cvxpy==1.4.1 llama-cpp-python>=0.2.0 ruff==0.1.6 filterpy==1.4.5 xgboost==2.0.3 redis==5.0.1 chromadb>=0.4.22 ray[rllib]>=2.9.3
```

### Option 3: Step-by-Step Install

1. **Install Core ML/Data Science packages:**
```bash
pip install cvxpy==1.4.1 xgboost==2.0.3 filterpy==1.4.5
```

2. **Install LLM and Vector Database:**
```bash
pip install llama-cpp-python>=0.2.0 chromadb>=0.4.22
```

3. **Install Reinforcement Learning:**
```bash
pip install ray[rllib]>=2.9.3
```

4. **Install Development Tools:**
```bash
pip install ruff==0.1.6 redis==5.0.1
```

5. **Install API Dependencies:**
```bash
pip install -r hackathon/requirements_api.txt
```

## Install Redis Server (Optional but Recommended)

Redis is used for enhanced caching and performance:
```bash
brew install redis
brew services start redis
```

## Potential Issues and Solutions

### 1. **Version Conflicts**
If you encounter version conflicts, create a new conda environment:
```bash
conda create -n gridpilot python=3.11
conda activate gridpilot
pip install -r hackathon/requirements.txt -r hackathon/requirements_api.txt
```

### 2. **llama-cpp-python Installation Issues**
This package can be tricky on some systems:
```bash
# If you encounter issues, try:
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python>=0.2.0
```

### 3. **Ray Installation Issues**
Ray is a large package. If it fails:
```bash
# Try installing without extras first:
pip install ray>=2.9.3
# Then install rllib separately if needed
pip install ray[rllib]>=2.9.3
```

### 4. **CVXPY Solver Issues**
CVXPY might need additional solvers:
```bash
pip install cvxopt
```

## Verify Installation

After installation, run the dependency checker again:
```bash
python3 check_dependencies.py
```

## Test the System

1. **Basic functionality test:**
```bash
python hackathon/test_system.py
```

2. **Start the dashboard:**
```bash
streamlit run hackathon/ui/dashboard.py
```

3. **Test API integration:**
```bash
python hackathon/test_mara_api.py
```

## Configuration Setup

1. **Copy and edit configuration:**
```bash
cp hackathon/config.toml hackathon/config_local.toml
```

2. **Add your API keys to config_local.toml:**
   - MARA API key (if you have one)
   - Anthropic API key (for Claude AI features)

## Next Steps

Once all dependencies are installed:

1. **Run the complete system:**
```bash
python hackathon/start_complete_unified.py
```

2. **Access the dashboard at:** http://localhost:8503

3. **Try the simple demo:**
```bash
python hackathon/demo_simple.py
```

## Troubleshooting

If you encounter any issues:

1. **Check Python environment:**
```bash
which python
pip list | grep -E "(streamlit|pandas|numpy)"
```

2. **Clear pip cache:**
```bash
pip cache purge
```

3. **Update pip:**
```bash
pip install --upgrade pip
```

4. **Check for conflicts:**
```bash
pip check
```

## Support

If you continue to have issues:
- Check the project's main README.md
- Look at the test files for examples
- Ensure your environment has sufficient resources (8GB+ RAM recommended)

---

**Happy coding!** 
