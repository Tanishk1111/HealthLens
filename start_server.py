#!/usr/bin/env python3
"""
HealthLens API Server Startup Script

This script provides an easy way to start the HealthLens API server
with proper configuration and error handling.
"""

import os
import sys
import argparse
import uvicorn
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'PIL',
        'numpy',
        'requests',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_environment():
    """Check environment configuration"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("⚠️  No .env file found. Creating from template...")
        
        template_file = Path('env.example')
        if template_file.exists():
            import shutil
            shutil.copy(template_file, env_file)
            print("✅ Created .env file from template")
            print("📝 Please edit .env file with your configuration")
        else:
            print("❌ No env.example template found")
            return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for Perplexity API key
    perplexity_key = os.getenv('PERPLEXITY_API_KEY')
    if not perplexity_key or perplexity_key == 'your_perplexity_api_key_here':
        print("⚠️  Perplexity API key not configured")
        print("   Sonar integration will be disabled")
        print("   Add your API key to .env file to enable expert consultation")
    else:
        print("✅ Perplexity API key configured")
    
    return True

def check_models_directory():
    """Check and create models directory"""
    models_dir = Path('models')
    
    if not models_dir.exists():
        print("📁 Creating models directory...")
        models_dir.mkdir(exist_ok=True)
        print("✅ Models directory created")
        print("   Place your trained models in the 'models/' directory")
    else:
        model_files = list(models_dir.glob('*.pt')) + list(models_dir.glob('*.pth'))
        if model_files:
            print(f"✅ Found {len(model_files)} model files in models/ directory")
        else:
            print("📁 Models directory exists but no model files found")
            print("   The API will use default/placeholder models")
    
    return True

def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description='HealthLens API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--log-level', default='info', help='Log level')
    parser.add_argument('--skip-checks', action='store_true', help='Skip startup checks')
    
    args = parser.parse_args()
    
    print("🏥 HealthLens Medical AI Agent")
    print("=" * 40)
    
    if not args.skip_checks:
        print("🔍 Running startup checks...")
        
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
        
        # Check environment
        if not check_environment():
            sys.exit(1)
        
        # Check models directory
        if not check_models_directory():
            sys.exit(1)
        
        print("✅ All checks passed!")
        print()
    
    # Start server
    print(f"🚀 Starting HealthLens API server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Reload: {args.reload}")
    print(f"   Workers: {args.workers}")
    print(f"   Log level: {args.log_level}")
    print()
    print(f"📖 API Documentation will be available at:")
    print(f"   Swagger UI: http://{args.host}:{args.port}/docs")
    print(f"   ReDoc: http://{args.host}:{args.port}/redoc")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 40)
    
    try:
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,  # Workers > 1 incompatible with reload
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 HealthLens API server stopped")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 