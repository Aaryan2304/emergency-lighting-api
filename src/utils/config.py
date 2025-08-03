"""
Configuration management for the emergency lighting detection system.
"""

import os
from typing import Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Configuration class for the emergency lighting detection system."""
    
    # Database Configuration
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'sqlite:///emergency_lighting.db')
    
    # API Configuration
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '8000'))
    DEBUG: bool = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # LLM Configuration - Multiple Backends
    # OpenAI
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    
    # Google Gemini
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    GEMINI_MODEL: str = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
    
    # Ollama (local)
    OLLAMA_BASE_URL: str = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL: str = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
    
    # Hugging Face (local)
    HF_MODEL: str = os.getenv('HF_MODEL', 'microsoft/DialoGPT-medium')
    LOAD_HF_MODEL: bool = os.getenv('LOAD_HF_MODEL', 'false').lower() == 'true'
    
    # General LLM settings
    LLM_MAX_TOKENS: int = int(os.getenv('LLM_MAX_TOKENS', '1000'))
    LLM_BACKEND: str = os.getenv('LLM_BACKEND', 'auto')  # auto, openai, gemini, ollama, huggingface, simple
    
    # Processing Configuration
    MAX_FILE_SIZE: int = 52428800  # 50MB in bytes (use bytes directly)
    SUPPORTED_FORMATS: List[str] = field(default_factory=lambda: os.getenv('SUPPORTED_FORMATS', 'pdf,png,jpg,jpeg').split(','))
    PROCESSING_TIMEOUT: int = int(os.getenv('PROCESSING_TIMEOUT', '300'))  # 5 minutes
    
    # Redis Configuration
    REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.getenv('LOG_FILE', 'logs/app.log')
    
    # Storage Configuration
    UPLOAD_DIR: str = os.getenv('UPLOAD_DIR', 'uploads')
    OUTPUT_DIR: str = os.getenv('OUTPUT_DIR', 'outputs')
    DEBUG_DIR: str = os.getenv('DEBUG_DIR', 'debug')
    
    # OCR Configuration
    TESSERACT_CONFIG: str = os.getenv('TESSERACT_CONFIG', '--oem 3 --psm 6')
    OCR_LANGUAGE: str = os.getenv('OCR_LANGUAGE', 'eng')
    
    # Computer Vision Configuration
    MIN_CONTOUR_AREA: int = int(os.getenv('MIN_CONTOUR_AREA', '100'))
    DETECTION_THRESHOLD: float = float(os.getenv('DETECTION_THRESHOLD', '0.8'))
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._create_directories()
    
    def _validate_config(self):
        """Validate configuration values."""
        if not self.OPENAI_API_KEY and not self.DEBUG:
            raise ValueError("OPENAI_API_KEY is required for production use")
        
        if self.API_PORT < 1 or self.API_PORT > 65535:
            raise ValueError("API_PORT must be between 1 and 65535")
        
        if self.MAX_FILE_SIZE <= 0:
            raise ValueError("MAX_FILE_SIZE must be positive")
        
        if self.PROCESSING_TIMEOUT <= 0:
            raise ValueError("PROCESSING_TIMEOUT must be positive")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.UPLOAD_DIR,
            self.OUTPUT_DIR,
            self.DEBUG_DIR,
            os.path.dirname(self.LOG_FILE) if os.path.dirname(self.LOG_FILE) else 'logs'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def from_env_file(cls, env_file: str) -> 'Config':
        """
        Create configuration from specific environment file.
        
        Args:
            env_file: Path to environment file
            
        Returns:
            Config instance
        """
        load_dotenv(env_file)
        return cls()
    
    def get_db_url(self) -> str:
        """Get database URL with proper formatting."""
        return self.DATABASE_URL
    
    def get_redis_config(self) -> dict:
        """Get Redis configuration dictionary."""
        return {
            'url': self.REDIS_URL,
            'decode_responses': True
        }
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration dictionary."""
        return {
            'api_key': self.OPENAI_API_KEY,
            'model': self.LLM_MODEL,
            'max_tokens': self.LLM_MAX_TOKENS
        }
    
    def get_ocr_config(self) -> dict:
        """Get OCR configuration dictionary."""
        return {
            'tesseract_config': self.TESSERACT_CONFIG,
            'language': self.OCR_LANGUAGE
        }
    
    def get_cv_config(self) -> dict:
        """Get computer vision configuration dictionary."""
        return {
            'min_contour_area': self.MIN_CONTOUR_AREA,
            'detection_threshold': self.DETECTION_THRESHOLD
        }
    
    def is_file_supported(self, filename: str) -> bool:
        """
        Check if file format is supported.
        
        Args:
            filename: Name of the file
            
        Returns:
            True if supported, False otherwise
        """
        if not filename:
            return False
        
        file_ext = filename.lower().split('.')[-1]
        return file_ext in [fmt.strip().lower() for fmt in self.SUPPORTED_FORMATS]
    
    def get_max_file_size_mb(self) -> float:
        """Get maximum file size in MB."""
        return self.MAX_FILE_SIZE / (1024 * 1024)


# Global configuration instance
config = Config()
