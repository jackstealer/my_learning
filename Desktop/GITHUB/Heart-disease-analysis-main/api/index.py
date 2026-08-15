import sys
import os

# Add the flask_app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'flask_app'))

from app import app

# This is required for Vercel
app = app
