"""
File handling utilities for the emergency lighting detection system.
"""

import os
import shutil
import tempfile
import hashlib
from pathlib import Path
from typing import List, Optional, Union, BinaryIO
import logging
from datetime import datetime
import mimetypes

from .config import config
from .logger import get_logger

logger = get_logger(__name__)


class FileHandler:
    """Utility class for file operations."""
    
    def __init__(self):
        self.upload_dir = Path(config.UPLOAD_DIR)
        self.output_dir = Path(config.OUTPUT_DIR)
        self.debug_dir = Path(config.DEBUG_DIR)
        
        # Ensure directories exist
        self.upload_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.debug_dir.mkdir(exist_ok=True)
    
    def save_upload(self, file_content: bytes, filename: str) -> str:
        """
        Save uploaded file to upload directory.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Path to saved file
        """
        try:
            # Validate file
            self._validate_file(file_content, filename)
            
            # Generate unique filename
            safe_filename = self._sanitize_filename(filename)
            unique_filename = self._generate_unique_filename(safe_filename)
            
            # Save file
            file_path = self.upload_dir / unique_filename
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"Uploaded file saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving upload: {str(e)}")
            raise
    
    def save_upload_stream(self, file_stream: BinaryIO, filename: str) -> str:
        """
        Save uploaded file from stream to upload directory.
        
        Args:
            file_stream: File stream
            filename: Original filename
            
        Returns:
            Path to saved file
        """
        try:
            # Generate unique filename
            safe_filename = self._sanitize_filename(filename)
            unique_filename = self._generate_unique_filename(safe_filename)
            
            # Save file
            file_path = self.upload_dir / unique_filename
            with open(file_path, 'wb') as dest:
                shutil.copyfileobj(file_stream, dest)
            
            # Validate saved file
            file_size = os.path.getsize(file_path)
            if file_size > config.MAX_FILE_SIZE:
                self.cleanup_file(str(file_path))
                raise ValueError(f"File size ({file_size}) exceeds maximum allowed ({config.MAX_FILE_SIZE})")
            
            logger.info(f"Uploaded file saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving upload stream: {str(e)}")
            raise
    
    def save_result(self, data: dict, pdf_name: str) -> str:
        """
        Save processing result to output directory.
        
        Args:
            data: Result data to save
            pdf_name: Original PDF name
            
        Returns:
            Path to saved result file
        """
        try:
            import json
            
            # Generate result filename
            base_name = Path(pdf_name).stem
            result_filename = f"{base_name}_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            result_path = self.output_dir / result_filename
            
            # Save result
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Result saved: {result_path}")
            return str(result_path)
            
        except Exception as e:
            logger.error(f"Error saving result: {str(e)}")
            raise
    
    def save_debug_data(self, data: any, stage: str, pdf_name: str) -> str:
        """
        Save debug data for troubleshooting.
        
        Args:
            data: Debug data to save
            stage: Processing stage name
            pdf_name: Original PDF name
            
        Returns:
            Path to saved debug file
        """
        try:
            import json
            
            # Generate debug filename
            base_name = Path(pdf_name).stem
            debug_filename = f"{base_name}_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            debug_path = self.debug_dir / debug_filename
            
            # Convert data to JSON if possible
            if hasattr(data, 'tolist'):  # numpy array
                json_data = data.tolist()
            elif isinstance(data, (dict, list, str, int, float, bool)):
                json_data = data
            else:
                json_data = str(data)
            
            # Save debug data
            with open(debug_path, 'w', encoding='utf-8') as f:
                if isinstance(json_data, str):
                    f.write(json_data)
                else:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Debug data saved: {debug_path}")
            return str(debug_path)
            
        except Exception as e:
            logger.warning(f"Error saving debug data: {str(e)}")
            return ""
    
    def create_temp_file(self, suffix: str = '', prefix: str = 'temp_') -> str:
        """
        Create a temporary file.
        
        Args:
            suffix: File suffix
            prefix: File prefix
            
        Returns:
            Path to temporary file
        """
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        os.close(fd)  # Close file descriptor
        return temp_path
    
    def cleanup_file(self, file_path: str) -> bool:
        """
        Clean up (delete) a file safely.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Error cleaning up file {file_path}: {str(e)}")
            return False
    
    def cleanup_old_files(self, directory: str, max_age_hours: int = 24) -> int:
        """
        Clean up old files from a directory.
        
        Args:
            directory: Directory to clean
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of files cleaned up
        """
        try:
            directory_path = Path(directory)
            if not directory_path.exists():
                return 0
            
            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
            cleaned_count = 0
            
            for file_path in directory_path.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up {file_path}: {str(e)}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old files from {directory}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error in cleanup_old_files: {str(e)}")
            return 0
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Get information about a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            path = Path(file_path)
            stat = path.stat()
            
            return {
                'name': path.name,
                'size': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'extension': path.suffix.lower(),
                'mime_type': mimetypes.guess_type(str(path))[0],
                'exists': path.exists(),
                'is_file': path.is_file()
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {str(e)}")
            return {}
    
    def _validate_file(self, file_content: bytes, filename: str):
        """
        Validate uploaded file.
        
        Args:
            file_content: File content
            filename: Filename
            
        Raises:
            ValueError: If file is invalid
        """
        # Check file size
        if len(file_content) > config.MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum allowed ({config.get_max_file_size_mb():.1f} MB)")
        
        # Check file format
        if not config.is_file_supported(filename):
            supported = ', '.join(config.SUPPORTED_FORMATS)
            raise ValueError(f"Unsupported file format. Supported: {supported}")
        
        # Basic content validation
        if len(file_content) == 0:
            raise ValueError("File is empty")
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent security issues.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove or replace dangerous characters
        dangerous_chars = '<>:"/\\|?*'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        
        return filename
    
    def _generate_unique_filename(self, filename: str) -> str:
        """
        Generate unique filename to prevent collisions.
        
        Args:
            filename: Base filename
            
        Returns:
            Unique filename
        """
        # Add timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(filename)
        
        unique_filename = f"{name}_{timestamp}{ext}"
        
        # Ensure uniqueness
        counter = 1
        while (self.upload_dir / unique_filename).exists():
            unique_filename = f"{name}_{timestamp}_{counter}{ext}"
            counter += 1
        
        return unique_filename
    
    def calculate_file_hash(self, file_path: str, algorithm: str = 'md5') -> str:
        """
        Calculate hash of a file.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256)
            
        Returns:
            Hex digest of file hash
        """
        try:
            hash_obj = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating file hash: {str(e)}")
            return ""
    
    def list_files(self, directory: str, pattern: str = '*') -> List[str]:
        """
        List files in directory matching pattern.
        
        Args:
            directory: Directory path
            pattern: File pattern (glob)
            
        Returns:
            List of file paths
        """
        try:
            directory_path = Path(directory)
            if not directory_path.exists():
                return []
            
            return [str(p) for p in directory_path.glob(pattern) if p.is_file()]
            
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            return []


# Global file handler instance
file_handler = FileHandler()
