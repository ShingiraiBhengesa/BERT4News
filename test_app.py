#!/usr/bin/env python
"""Simple test script to run the BERT4News application."""
import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set environment variables
os.environ['FLASK_ENV'] = 'development'
os.environ['FLASK_DEBUG'] = 'True'

try:
    # Import and initialize the Flask app
    from api.app import app, init_app
    
    print("🚀 Starting BERT4News application...")
    print("📍 Make sure you have run 'make setup' first to create the database")
    print()
    
    # Initialize the app
    init_app()
    
    print("✅ Application initialized successfully")
    print("🌐 Starting server on http://localhost:5000")
    print("📱 Open your browser and visit: http://localhost:5000")
    print()
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Run the Flask development server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Disable reloader to avoid path issues
    )
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Make sure you have installed the requirements:")
    print("   pip install -r requirements.txt")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Try running 'make setup' first to initialize the database")
    sys.exit(1)
