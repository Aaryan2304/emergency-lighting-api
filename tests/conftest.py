"""
Test configuration and fixtures.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import Config
from src.database import DatabaseManager
from src.api.app import create_app


@pytest.fixture
def test_config():
    """Create test configuration."""
    return Config(
        DATABASE_URL='sqlite:///test.db',
        DEBUG=True,
        LOG_LEVEL='DEBUG',
        UPLOAD_DIR='test_uploads',
        OUTPUT_DIR='test_outputs',
        DEBUG_DIR='test_debug'
    )


@pytest.fixture
async def test_db(test_config):
    """Create test database."""
    db_manager = DatabaseManager()
    db_manager.db_path = 'test.db'
    await db_manager.initialize()
    
    yield db_manager
    
    # Cleanup
    if os.path.exists('test.db'):
        os.remove('test.db')


@pytest.fixture
def test_app(test_config):
    """Create test FastAPI app."""
    app = create_app()
    return app


@pytest.fixture
def sample_pdf_path():
    """Create sample PDF file for testing."""
    # Create a minimal PDF-like file for testing
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(b'%PDF-1.4\nSample PDF content for testing\n%%EOF')
        yield f.name
    
    # Cleanup
    if os.path.exists(f.name):
        os.remove(f.name)


@pytest.fixture
def sample_image():
    """Create sample image for testing."""
    import numpy as np
    return np.zeros((100, 100, 3), dtype=np.uint8)


# Async test event loop setup
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
