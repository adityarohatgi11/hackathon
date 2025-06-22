#!/usr/bin/env python3
"""
Test script to verify Claude API setup
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_claude_setup():
    """Test Claude API setup and show what's needed."""
    print("=" * 60)
    print("Claude API Setup Test")
    print("=" * 60)
    
    # 1. Check anthropic package
    print("1. Checking anthropic package...")
    try:
        import anthropic
        print(f"   ✅ anthropic package installed (version: {anthropic.__version__})")
    except ImportError:
        print("   ❌ anthropic package NOT installed")
        print("   Run: pip install anthropic")
        return False
    
    # 2. Check API key
    print("2. Checking API key...")
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        print(f"   ✅ ANTHROPIC_API_KEY found (length: {len(api_key)} chars)")
        if api_key.startswith('sk-ant-'):
            print("   ✅ API key format looks correct")
        else:
            print("   ⚠️  API key format might be incorrect (should start with 'sk-ant-')")
    else:
        print("   ❌ ANTHROPIC_API_KEY environment variable NOT set")
        print("\n   To fix this:")
        print("   Option 1 - Temporary (current session only):")
        print("   export ANTHROPIC_API_KEY='your-api-key-here'")
        print("\n   Option 2 - Permanent (add to ~/.zshrc):")
        print("   echo 'export ANTHROPIC_API_KEY=\"your-api-key-here\"' >> ~/.zshrc")
        print("   source ~/.zshrc")
        print("\n   Option 3 - Create .env file in project:")
        print("   echo 'ANTHROPIC_API_KEY=your-api-key-here' > .env")
        return False
    
    # 3. Test Claude interface
    print("3. Testing Claude interface...")
    try:
        from llm_integration.claude_interface import ClaudeInterface
        claude = ClaudeInterface()
        if claude.is_available():
            print("   ✅ Claude interface initialized successfully")
            
            # Test a simple query
            print("4. Testing API call...")
            response = claude.process_query("Say 'Claude API is working' in exactly those words.")
            print(f"   Response: {response}")
            
            if "Claude API is working" in response:
                print("   ✅ Claude API is fully functional!")
                return True
            else:
                print("   ⚠️  API responded but with unexpected content")
                return True
        else:
            print("   ❌ Claude interface not available")
            return False
    except Exception as e:
        print(f"   ❌ Error testing Claude interface: {e}")
        return False

if __name__ == "__main__":
    success = test_claude_setup()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Claude API is ready to use!")
        print("The dashboard will now use Claude instead of mock interface.")
    else:
        print("❌ Claude API setup incomplete.")
        print("The dashboard will continue using mock interface until fixed.")
        print("\nGet your API key from: https://console.anthropic.com/")
    print("=" * 60) 