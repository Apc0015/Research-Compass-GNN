#!/usr/bin/env python3
"""
LLM Model Setup Helper
Automatically detects and configures available LLM models for Research Compass.
"""

import sys
import subprocess
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def test_ollama():
    """Test Ollama installation and list models."""
    print_header("🦙 Checking Ollama")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            
            if models:
                print("✅ Ollama is running!")
                print(f"   Found {len(models)} model(s):\n")
                for model in models:
                    name = model.get('name', 'unknown')
                    size = model.get('size', 0) / (1024**3)  # Convert to GB
                    print(f"   • {name} ({size:.1f} GB)")
                print("\n   Recommended: Use one of the above models")
                return True, models[0]['name']
            else:
                print("⚠️  Ollama is running but no models installed")
                print("\n   Quick setup:")
                print("   1. Install a model: ollama pull llama3.2")
                print("   2. Or try: ollama pull deepseek-r1:1.5b")
                print("   3. Then restart Research Compass")
                return False, None
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return False, None
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama")
        print("\n   Setup steps:")
        print("   1. Install Ollama: https://ollama.ai/download")
        print("   2. Start Ollama: ollama serve")
        print("   3. Pull a model: ollama pull llama3.2")
        return False, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None


def test_lmstudio():
    """Test LM Studio installation."""
    print_header("🎨 Checking LM Studio")
    
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('data', [])
            
            if models:
                print("✅ LM Studio is running!")
                print(f"   Found {len(models)} model(s):\n")
                for model in models:
                    name = model.get('id', 'unknown')
                    print(f"   • {name}")
                print("\n   Recommended: Use one of the above models")
                return True, models[0]['id']
            else:
                print("⚠️  LM Studio is running but no model loaded")
                print("\n   Quick setup:")
                print("   1. Open LM Studio")
                print("   2. Download a model (e.g., Llama 3.2, Mistral)")
                print("   3. Load the model in the 'Local Server' tab")
                print("   4. Start the server")
                return False, None
        else:
            print(f"❌ LM Studio returned status {response.status_code}")
            return False, None
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to LM Studio")
        print("\n   Setup steps:")
        print("   1. Download LM Studio: https://lmstudio.ai/")
        print("   2. Install and open it")
        print("   3. Download a model from the 'Discover' tab")
        print("   4. Load model in 'Local Server' tab and start server")
        return False, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None


def test_openrouter(api_key=None):
    """Test OpenRouter API."""
    print_header("🌐 Checking OpenRouter")
    
    if not api_key:
        print("ℹ️  OpenRouter API key not provided")
        print("\n   Setup steps:")
        print("   1. Sign up: https://openrouter.ai/")
        print("   2. Get API key from dashboard")
        print("   3. Add to .env: OPENROUTER_API_KEY=your_key_here")
        return False, None
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
        
        if response.status_code == 200:
            print("✅ OpenRouter API key is valid!")
            print("   Popular models available:")
            print("   • openai/gpt-4o")
            print("   • openai/gpt-4o-mini")
            print("   • anthropic/claude-3.5-sonnet")
            print("   • google/gemini-pro")
            print("   • meta-llama/llama-3.1-70b-instruct")
            return True, "openai/gpt-4o-mini"
        elif response.status_code == 401:
            print("❌ Invalid API key")
            return False, None
        else:
            print(f"❌ OpenRouter returned status {response.status_code}")
            return False, None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None


def test_openai(api_key=None):
    """Test OpenAI API."""
    print_header("🤖 Checking OpenAI")
    
    if not api_key:
        print("ℹ️  OpenAI API key not provided")
        print("\n   Setup steps:")
        print("   1. Sign up: https://platform.openai.com/")
        print("   2. Get API key from API keys section")
        print("   3. Add to .env: OPENAI_API_KEY=your_key_here")
        return False, None
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=5)
        
        if response.status_code == 200:
            print("✅ OpenAI API key is valid!")
            print("   Available models:")
            print("   • gpt-4o (Latest)")
            print("   • gpt-4o-mini (Fast & Affordable)")
            print("   • gpt-4-turbo")
            print("   • gpt-3.5-turbo")
            return True, "gpt-4o-mini"
        elif response.status_code == 401:
            print("❌ Invalid API key")
            return False, None
        else:
            print(f"❌ OpenAI returned status {response.status_code}")
            return False, None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None


def check_env_file():
    """Check if .env file exists and load API keys."""
    env_file = project_root / ".env"
    
    config = {
        'openrouter_key': None,
        'openai_key': None
    }
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('OPENROUTER_API_KEY='):
                    config['openrouter_key'] = line.split('=', 1)[1].strip('"\'')
                elif line.startswith('OPENAI_API_KEY='):
                    config['openai_key'] = line.split('=', 1)[1].strip('"\'')
    
    return config


def generate_recommendation():
    """Generate configuration recommendation."""
    print_header("📝 Configuration Recommendation")
    
    config = check_env_file()
    
    # Test all providers
    ollama_ok, ollama_model = test_ollama()
    lmstudio_ok, lmstudio_model = test_lmstudio()
    openrouter_ok, openrouter_model = test_openrouter(config['openrouter_key'])
    openai_ok, openai_model = test_openai(config['openai_key'])
    
    # Determine best option
    print_header("✨ Recommended Configuration")
    
    if ollama_ok:
        print("🎯 Best Option: Ollama (Local & Free)")
        print(f"\n   Add to your .env file:")
        print(f"   LLM_PROVIDER=ollama")
        print(f"   LLM_MODEL={ollama_model}")
        print(f"   OLLAMA_BASE_URL=http://localhost:11434")
        
    elif lmstudio_ok:
        print("🎯 Best Option: LM Studio (Local & Free)")
        print(f"\n   Add to your .env file:")
        print(f"   LLM_PROVIDER=lmstudio")
        print(f"   LLM_MODEL={lmstudio_model}")
        print(f"   LMSTUDIO_BASE_URL=http://localhost:1234")
        
    elif openai_ok:
        print("🎯 Best Option: OpenAI (Cloud, requires API key)")
        print(f"\n   Add to your .env file:")
        print(f"   LLM_PROVIDER=openai")
        print(f"   LLM_MODEL={openai_model}")
        print(f"   OPENAI_API_KEY=your_key_here")
        
    elif openrouter_ok:
        print("🎯 Best Option: OpenRouter (Cloud, requires API key)")
        print(f"\n   Add to your .env file:")
        print(f"   LLM_PROVIDER=openrouter")
        print(f"   LLM_MODEL={openrouter_model}")
        print(f"   OPENROUTER_API_KEY=your_key_here")
        
    else:
        print("⚠️  No LLM provider is currently available")
        print("\n   Recommended: Install Ollama (easiest)")
        print("\n   Quick setup:")
        print("   1. Install: https://ollama.ai/download")
        print("   2. Run: ollama pull llama3.2")
        print("   3. Add to .env:")
        print("      LLM_PROVIDER=ollama")
        print("      LLM_MODEL=llama3.2")
        print("      OLLAMA_BASE_URL=http://localhost:11434")
    
    print("\n" + "=" * 70 + "\n")


def main():
    """Main function."""
    print("\n" + "🧭 " * 20)
    print("   Research Compass - LLM Model Setup Helper")
    print("🧭 " * 20)
    
    generate_recommendation()
    
    print("ℹ️  After configuration, restart Research Compass with:")
    print("   python launcher.py\n")


if __name__ == "__main__":
    main()
