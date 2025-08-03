"""
Interactive setup script for Emergency Lighting Detection System.
Helps users choose and configure their preferred LLM backend.
"""

import subprocess
import sys
import os
from pathlib import Path


def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False


def check_gpu():
    """Check if CUDA/GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def create_env_file(backend_choice, api_key=None):
    """Create .env file with the chosen configuration."""
    env_content = """# Emergency Lighting Detection System Configuration

# Database
DATABASE_URL=sqlite:///emergency_lighting.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# File Processing
MAX_FILE_SIZE=50MB
SUPPORTED_FORMATS=pdf,png,jpg,jpeg

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Detection Settings
MIN_DETECTION_CONFIDENCE=0.5
BBOX_THRESHOLD=10
OCR_CONFIDENCE_THRESHOLD=30

"""

    if backend_choice == "openai" and api_key:
        env_content += f"""
# OpenAI Configuration
OPENAI_API_KEY={api_key}
OPENAI_MODEL=gpt-3.5-turbo
"""
    elif backend_choice == "gemini" and api_key:
        env_content += f"""
# Google Gemini Configuration  
GEMINI_API_KEY={api_key}
GEMINI_MODEL=gemini-1.5-flash
"""
    elif backend_choice == "ollama":
        env_content += """
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
"""
    elif backend_choice == "huggingface":
        env_content += """
# Hugging Face Configuration
HF_MODEL=microsoft/DialoGPT-medium
LOAD_HF_MODEL=true
"""
    else:
        env_content += """
# No LLM backend configured - using rule-based fallback
LLM_BACKEND=simple
"""

    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Created .env configuration file")


def main():
    print("🚀 Emergency Lighting Detection System Setup")
    print("=" * 50)
    
    # Check current directory
    if not Path("src").exists():
        print("❌ Please run this script from the project root directory")
        return
    
    print("This script will help you set up the Emergency Lighting Detection System.")
    print("You can choose from several LLM backends based on your needs and budget.\n")
    
    # Install core dependencies
    print("📦 Installing core dependencies...")
    core_packages = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0", 
        "opencv-python==4.8.1.78",
        "pytesseract==0.3.10",
        "easyocr==1.7.0",
        "pdf2image==1.16.3",
        "python-dotenv==1.0.0",
        "aiofiles==23.2.1",
        "sqlalchemy==2.0.23",
        "pydantic==2.5.0",
        "requests==2.31.0"
    ]
    
    for package in core_packages:
        if install_package(package):
            print(f"✅ Installed {package.split('==')[0]}")
        else:
            print(f"❌ Failed to install {package}")
    
    print("\n🤖 Choose your LLM backend:")
    print("1. 🆓 Google Gemini (Free tier available)")
    print("2. 🆓 Ollama (Completely free, runs locally)")
    print("3. 🆓 Hugging Face (Completely free, runs locally)")
    print("4. 💰 OpenAI (Paid, high quality)")
    print("5. 🔧 No LLM (Rule-based fallback only)")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-5): "))
            if 1 <= choice <= 5:
                break
            else:
                print("Please enter a number between 1 and 5")
        except ValueError:
            print("Please enter a valid number")
    
    # Handle the choice
    if choice == 1:  # Gemini
        print("\n🔑 Setting up Google Gemini...")
        if install_package("google-generativeai==0.3.2"):
            print("✅ Installed Google Gemini")
            print("\n📝 To get a free Gemini API key:")
            print("1. Go to: https://aistudio.google.com/app/apikey")
            print("2. Sign in with your Google account")
            print("3. Click 'Create API Key'")
            print("4. Copy the API key")
            
            api_key = input("\nEnter your Gemini API key (or press Enter to skip): ").strip()
            create_env_file("gemini", api_key if api_key else None)
            
            if api_key:
                print("✅ Gemini configured successfully!")
            else:
                print("⚠️  You can add your API key to the .env file later")
        
    elif choice == 2:  # Ollama
        print("\n🔧 Setting up Ollama...")
        print("Ollama runs AI models locally on your computer.")
        print("\nTo install Ollama:")
        print("1. Go to: https://ollama.ai/")
        print("2. Download and install Ollama for Windows")
        print("3. Open a terminal and run: ollama pull llama3.2:3b")
        print("4. This will download a 2GB model (good for your 4GB VRAM)")
        
        ollama_installed = input("\nHave you installed Ollama? (y/n): ").lower().startswith('y')
        
        if ollama_installed:
            print("Starting Ollama service...")
            try:
                subprocess.Popen(["ollama", "serve"], shell=True)
                print("✅ Ollama service started")
            except:
                print("⚠️  Please start Ollama manually: 'ollama serve'")
        
        create_env_file("ollama")
        print("✅ Ollama configured!")
        
    elif choice == 3:  # Hugging Face
        print("\n🤗 Setting up Hugging Face local models...")
        gpu_available = check_gpu()
        
        if gpu_available:
            print("✅ CUDA GPU detected - installing GPU support")
            packages = [
                "transformers==4.36.0",
                "torch==2.1.0",
                "accelerate==0.25.0"
            ]
        else:
            print("ℹ️  No GPU detected - installing CPU version")
            packages = [
                "transformers==4.36.0", 
                "torch==2.1.0+cpu",
                "accelerate==0.25.0"
            ]
        
        for package in packages:
            if install_package(package):
                print(f"✅ Installed {package.split('==')[0]}")
        
        create_env_file("huggingface")
        print("✅ Hugging Face configured!")
        print("⚠️  First run will download model files (~1GB)")
        
    elif choice == 4:  # OpenAI
        print("\n💰 Setting up OpenAI...")
        if install_package("openai==1.3.5"):
            print("✅ Installed OpenAI")
            print("\n📝 To get an OpenAI API key:")
            print("1. Go to: https://platform.openai.com/api-keys")
            print("2. Sign in or create an account")
            print("3. Click 'Create new secret key'")
            print("4. Copy the API key")
            print("⚠️  Note: OpenAI charges per API call")
            
            api_key = input("\nEnter your OpenAI API key (or press Enter to skip): ").strip()
            create_env_file("openai", api_key if api_key else None)
            
            if api_key:
                print("✅ OpenAI configured successfully!")
            else:
                print("⚠️  You can add your API key to the .env file later")
        
    else:  # No LLM
        print("\n🔧 Setting up rule-based fallback...")
        create_env_file("simple")
        print("✅ Configured for rule-based grouping only")
        print("ℹ️  You can add LLM backends later by editing .env")
    
    # Final setup
    print("\n🔧 Final setup...")
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("debug", exist_ok=True)
    print("✅ Created directories")
    
    # Initialize database
    try:
        subprocess.run([sys.executable, "src/database/init_db.py"], check=True)
        print("✅ Database initialized")
    except:
        print("⚠️  Database initialization failed - will create on first run")
    
    print("\n🎉 Setup complete!")
    print("\n🚀 Quick start:")
    print("1. Run demo: python demo.py")
    print("2. Start API: python main.py")
    print("3. View docs: http://localhost:8000/docs")
    print("\n📄 Check README.md for detailed documentation")


if __name__ == "__main__":
    main()
