"""
API module initialization.
"""

from .app import app, create_app
from .models import *
from .routes import router

__all__ = ['app', 'create_app', 'router']
