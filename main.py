"""
Main entry point for the emergency lighting detection system.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.api.app import main as api_main
from src.database.init_db import initialize_database
from src.utils import setup_logging, get_logger, config, file_handler

logger = get_logger(__name__)


async def init_system():
    """Initialize the system (database, directories, etc.)."""
    try:
        logger.info("Initializing Emergency Lighting Detection System")
        
        # Initialize database
        await initialize_database()
        
        # Clean up old files
        file_handler.cleanup_old_files(config.UPLOAD_DIR, max_age_hours=24)
        file_handler.cleanup_old_files(config.DEBUG_DIR, max_age_hours=48)
        
        logger.info("System initialization completed")
        
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        raise


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Emergency Lighting Detection from Construction Blueprints"
    )
    
    parser.add_argument(
        '--mode', 
        choices=['api', 'init', 'test'],
        default='api',
        help='Run mode: api (start API server), init (initialize system), test (run tests)'
    )
    
    parser.add_argument(
        '--host',
        default=config.API_HOST,
        help=f'API host (default: {config.API_HOST})'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=config.API_PORT,
        help=f'API port (default: {config.API_PORT})'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=config.LOG_LEVEL,
        help=f'Log level (default: {config.LOG_LEVEL})'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Update config if needed
    if args.debug:
        config.DEBUG = True
    if args.host != config.API_HOST:
        config.API_HOST = args.host
    if args.port != config.API_PORT:
        config.API_PORT = args.port
    
    # Run based on mode
    try:
        if args.mode == 'init':
            asyncio.run(init_system())
            
        elif args.mode == 'api':
            # Initialize system first
            asyncio.run(init_system())
            # Start API server
            api_main()
            
        elif args.mode == 'test':
            # Run tests
            import pytest
            sys.exit(pytest.main(['-v', 'tests/']))
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
