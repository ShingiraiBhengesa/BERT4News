"""WSGI entry point for production deployment."""
from app import app, init_app

# Initialize the application
init_app()

if __name__ == "__main__":
    app.run()
