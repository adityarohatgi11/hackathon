#!/usr/bin/env python3
"""
GridPilot-GT Kernel Test Script
Tests Python environment and dependencies for Cursor IDE

"""

import sys
import os
from pathlib import Path
import subprocess

def main():
    print("GridPilot-GT Python Kernel Test")
    print("=" * 50)
    
    # Basic environment info
    print(f"Python Version: {sys.version}")
    print(f"Python Path: {sys.executable}")
    print(f"Current Directory: {os.getcwd()}")
    print(f"Working Directory: {Path.cwd()}")
    
    # Check conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'No conda environment detected')
    print(f"Conda Environment: {conda_env}")
    
    print("\nTesting GridPilot-GT Dependencies:")
    print("-" * 40)
    
    # Test core dependencies
    dependencies = [
        'pandas', 'numpy', 'streamlit', 'plotly', 'fastapi', 
        'httpx', 'requests', 'toml', 'cvxpy', 'xgboost', 
        'scikit-learn', 'prophet', 'redis', 'ray'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"{dep}")
        except ImportError as e:
            print(f"{dep} - {e}")
    
    print("\nTesting GridPilot-GT Modules:")
    print("-" * 40)
    
    # Test API client
    try:
        from hackathon.api_client.client import MARAClient
        print("API Client")
    except ImportError as e:
        print(f"API Client - {e}")
    
    # Test forecaster  
    try:
        from hackathon.forecasting.forecaster import Forecaster
        print("Forecaster")
    except ImportError as e:
        print(f"Forecaster - {e}")
    
    # Test LLM interface
    try:
        from hackathon.llm_integration.unified_interface import UnifiedLLMInterface
        print("LLM Interface")
    except ImportError as e:
        print(f"LLM Interface - {e}")
    
    # Final assessment
    if conda_env == 'base':
        print("Kernel is correctly set to conda base environment!")
        print("Ready to run GridPilot-GT in Cursor!")
    else:
        print("Consider switching to conda base environment")
        
    # Kernel selection instructions
    print(f"\nTo change kernel in Cursor:")
    print("1. Open Command Palette (Cmd+Shift+P)")
    print("2. Type 'Python: Select Interpreter'") 
    print("3. Choose the conda base environment:")
    print(f"   {sys.executable}")

if __name__ == "__main__":
    main() 
