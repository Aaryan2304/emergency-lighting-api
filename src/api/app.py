"""
FastAPI application for emergency lighting detection API.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
from contextlib import asynccontextmanager

from src.api.routes import router
from src.utils import config, get_logger, setup_logging
from src.database.db_manager import DatabaseManager

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Emergency Lighting Detection API")
    
    # Initialize database
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Emergency Lighting Detection API")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Emergency Lighting Detection API",
        description="AI Vision pipeline for detecting emergency lighting fixtures from construction blueprints",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router, prefix="/api/v1")
    
    # Add global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "Emergency Lighting Detection API",
            "version": "1.0.0"
        }
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "Emergency Lighting Detection API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }
    
    return app


# Create app instance
app = create_app()


def main():
    """Main function to run the API server."""
    setup_logging()
    
    logger.info(f"Starting server on {config.API_HOST}:{config.API_PORT}")
    
    uvicorn.run(
        "src.api.app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()
