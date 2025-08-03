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
        file_handler.cleanup_old_files(config.OUTPUT_DIR, max_age_hours=48)
        
        logger.info("System initialization complete")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        return False


def run_pipeline(input_path: str, output_path: str = None):
    """Run the detection pipeline on a single file."""
    try:
        from src.core.pipeline import EmergencyLightingPipeline
        
        # Initialize pipeline
        pipeline = EmergencyLightingPipeline()
        
        # Process file
        results = asyncio.run(pipeline.process_file(input_path, output_path))
        
        if results:
            logger.info(f"Processing complete. Results: {results}")
            return results
        else:
            logger.error("Processing failed")
            return None
            
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Emergency Lighting Detection System')
    parser.add_argument('--mode', choices=['api', 'pipeline'], default='api',
                        help='Run mode: api server or pipeline processing')
    parser.add_argument('--input', type=str, help='Input file path (for pipeline mode)')
    parser.add_argument('--output', type=str, help='Output file path (for pipeline mode)')
    parser.add_argument('--host', type=str, default=config.API_HOST, help='API host')
    parser.add_argument('--port', type=int, default=config.API_PORT, help='API port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.debug else config.LOG_LEVEL
    setup_logging(log_level)
    
    try:
        if args.mode == 'api':
            logger.info("Starting API server mode")
            
            # Initialize system
            init_success = asyncio.run(init_system())
            if not init_success:
                logger.error("Failed to initialize system")
                sys.exit(1)
            
            # Update config with args
            config.API_HOST = args.host
            config.API_PORT = args.port
            config.DEBUG = args.debug
            
            # Start API server
            api_main()
            
        elif args.mode == 'pipeline':
            logger.info("Starting pipeline mode")
            
            if not args.input:
                logger.error("Input file path required for pipeline mode")
                sys.exit(1)
            
            # Initialize system
            init_success = asyncio.run(init_system())
            if not init_success:
                logger.error("Failed to initialize system")
                sys.exit(1)
            
            # Run pipeline
            results = run_pipeline(args.input, args.output)
            if results:
                logger.info("Pipeline completed successfully")
                # Print results summary
                print("\n=== PROCESSING RESULTS ===")
                print(f"Input: {args.input}")
                if args.output:
                    print(f"Output: {args.output}")
                print(f"Emergency Lights Detected: {len(results.get('emergency_lights', []))}")
                print(f"Table Data Extracted: {len(results.get('table_data', []))}")
                print("==========================\n")
            else:
                logger.error("Pipeline failed")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
