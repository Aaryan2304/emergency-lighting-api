"""
Database initialization script.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database.db_manager import DatabaseManager
from src.utils import setup_logging, get_logger

logger = get_logger(__name__)


async def initialize_database():
    """Initialize the database with tables and indexes."""
    try:
        setup_logging()
        logger.info("Starting database initialization")
        
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(initialize_database())
