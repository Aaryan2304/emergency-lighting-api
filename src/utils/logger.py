"""
Logging utilities for the emergency lighting detection system.
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional
from datetime import datetime

from .config import config


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        console_output: Whether to output to console
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    # Use config defaults if not provided
    log_level = log_level or config.LOG_LEVEL
    log_file = log_file or config.LOG_FILE
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = ColoredFormatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler with rotation
    if log_file:
        try:
            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"Warning: Could not set up file logging: {e}")
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # Log initial setup message
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with parameters and execution time.
    
    Args:
        logger: Logger instance to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            # Log function entry
            logger.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Exiting {func.__name__} - Execution time: {execution_time:.3f}s")
                return result
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"Exception in {func.__name__} after {execution_time:.3f}s: {str(e)}")
                raise
                
        return wrapper
    return decorator


def log_performance(logger: logging.Logger, operation: str):
    """
    Context manager for logging performance metrics.
    
    Args:
        logger: Logger instance
        operation: Description of the operation
        
    Usage:
        with log_performance(logger, "PDF processing"):
            # Your code here
            pass
    """
    class PerformanceLogger:
        def __init__(self, logger, operation):
            self.logger = logger
            self.operation = operation
            self.start_time = None
        
        def __enter__(self):
            self.start_time = datetime.now()
            self.logger.info(f"Starting {self.operation}")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            execution_time = (datetime.now() - self.start_time).total_seconds()
            
            if exc_type is None:
                self.logger.info(f"Completed {self.operation} in {execution_time:.3f}s")
            else:
                self.logger.error(f"Failed {self.operation} after {execution_time:.3f}s: {exc_val}")
    
    return PerformanceLogger(logger, operation)


def sanitize_log_data(data: any) -> str:
    """
    Sanitize data for safe logging (remove sensitive information).
    
    Args:
        data: Data to sanitize
        
    Returns:
        Sanitized string representation
    """
    if isinstance(data, dict):
        sanitized = data.copy()
        
        # Remove sensitive keys
        sensitive_keys = ['password', 'api_key', 'token', 'secret', 'key']
        for key in list(sanitized.keys()):
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = '***REDACTED***'
        
        return str(sanitized)
    
    return str(data)


class LogContext:
    """Context manager for adding contextual information to logs."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.original_factory = None
    
    def __enter__(self):
        self.original_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.original_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.original_factory)


# Initialize logging on module import
setup_logging()
