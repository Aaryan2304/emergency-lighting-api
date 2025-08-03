"""
Source package initialization.
"""

from .api import app, create_app
from .core import ProcessingPipeline
from .database import DatabaseManager
from .detection import LightingDetector
from .extraction import OCREngine
from .llm import GroupingEngine
from .utils import config, get_logger

__version__ = "1.0.0"

__all__ = [
    'app', 'create_app',
    'ProcessingPipeline',
    'DatabaseManager',
    'LightingDetector',
    'OCREngine',
    'GroupingEngine',
    'config', 'get_logger'
]
