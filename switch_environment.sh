#!/bin/bash

# GridPilot-GT Environment Manager
# Helps switch between conda environments and test dependencies


echo "======================================"
echo "Current Environment:"
echo "======================================"

if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo "Conda Environment: $CONDA_DEFAULT_ENV"
    echo "Python Path: $(which python)"
    echo "Python Version: $(python --version)"
else
    echo "Not in a conda environment"
fi

echo "Testing GridPilot Dependencies:"
python3 -c "
import sys
dependencies = ['pandas', 'numpy', 'streamlit', 'plotly', 'cvxpy', 'ray']
for pkg in dependencies:
    try:
        __import__(pkg)
        print(f'   {pkg}')
    except ImportError:
        print(f'   {pkg} - Missing')
"

echo ""
echo "======================================"
echo "Available Actions:"
echo "======================================"
echo "1. Stay in current environment"
echo "2. Create new GridPilot environment"
echo "3. Launch dashboard (current env)"
echo "4. Exit"
echo ""

read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo "Staying in current environment"
        ;;
    2)
        echo "Creating new GridPilot environment..."
        conda create -n gridpilot python=3.11 -y
        echo "Activate with: conda activate gridpilot"
        echo "Install deps with: pip install -r hackathon/requirements.txt"
        ;;
    3)
        echo "Launching dashboard..."
        cd hackathon
        echo "Starting Streamlit dashboard..."
        streamlit run ui/dashboard.py --server.port 8501 --server.address localhost
        ;;
    4)
        exit 0
        ;;
    *)
        echo "Invalid choice"
        ;;
esac 
