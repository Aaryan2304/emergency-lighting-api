"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    UPLOADED = "uploaded"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


class UploadResponse(BaseModel):
    """Response model for file upload."""
    status: ProcessingStatus
    pdf_name: str
    message: str
    upload_id: Optional[str] = None


class LightingFixtureGroup(BaseModel):
    """Model for a group of lighting fixtures."""
    count: int = Field(ge=0, description="Number of fixtures in this group")
    description: str = Field(description="Description of the fixture type")


class ProcessingResult(BaseModel):
    """Response model for processing results."""
    pdf_name: str
    status: ProcessingStatus
    result: Optional[Dict[str, LightingFixtureGroup]] = None
    message: Optional[str] = None
    processing_time: Optional[float] = None
    
    @field_validator('pdf_name')
    @classmethod
    def validate_pdf_name(cls, v):
        """Validate PDF name format."""
        if not v or not v.strip():
            raise ValueError("PDF name cannot be empty")
        
        if not v.lower().endswith('.pdf'):
            v = v + '.pdf'
        
        return v.strip()


class DetectionDetails(BaseModel):
    """Detailed detection information."""
    bounding_box: List[int] = Field(description="Bounding box coordinates [x1, y1, x2, y2]")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score")
    symbol: Optional[str] = Field(description="Detected symbol/label")
    
    @field_validator('bounding_box')
    @classmethod
    def validate_bounding_box(cls, v):
        """Validate bounding box coordinates."""
        if len(v) != 4:
            raise ValueError("Bounding box must have exactly 4 coordinates")
        if not all(isinstance(coord, (int, float)) for coord in v):
            raise ValueError("All coordinates must be numbers")
        if v[0] >= v[2] or v[1] >= v[3]:
            raise ValueError("Invalid bounding box coordinates")
        return v


class DetailedResult(BaseModel):
    """Detailed processing results with detection information."""
    pdf_name: str
    status: ProcessingStatus
    detections: List[DetectionDetails] = []
    grouped_results: Optional[Dict[str, LightingFixtureGroup]] = None
    ocr_text: Optional[List[str]] = None
    processing_metadata: Optional[Dict[str, Any]] = None
    
    @field_validator('pdf_name')
    @classmethod
    def validate_pdf_name(cls, v):
        """Validate PDF name format."""
        if not v or not v.strip():
            raise ValueError("PDF name cannot be empty")
        
        if not v.lower().endswith('.pdf'):
            v = v + '.pdf'
        
        return v.strip()


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    code: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    dependencies: Dict[str, bool] = Field(default_factory=dict)


class ValidationRequest(BaseModel):
    """Request for validating PDF processing."""
    pdf_name: str
    
    @field_validator('pdf_name')
    @classmethod
    def validate_pdf_name(cls, v):
        """Validate PDF name format."""
        if not v or not v.strip():
            raise ValueError("PDF name cannot be empty")
        
        if not v.lower().endswith('.pdf'):
            v = v + '.pdf'
        
        return v.strip()


class ValidationResponse(BaseModel):
    """Response for validation request."""
    pdf_name: str
    is_valid: bool
    issues: List[str] = []
    suggestions: List[str] = []


class ProcessingStats(BaseModel):
    """Processing statistics."""
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    average_processing_time: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.now)


class SystemStatus(BaseModel):
    """System status information."""
    api_status: str
    database_status: str
    llm_status: str
    ocr_status: str
    queue_size: int = 0
    active_jobs: int = 0
    uptime: str
    memory_usage: Optional[float] = None


class BulkProcessingRequest(BaseModel):
    """Request for bulk processing multiple PDFs."""
    pdf_names: List[str]
    priority: int = Field(default=0, ge=0, le=10)


class ReprocessRequest(BaseModel):
    """Request to reprocess a PDF."""
    pdf_name: str
    force_reprocess: bool = False
    
    @field_validator('pdf_name')
    @classmethod
    def validate_pdf_name(cls, v):
        """Validate PDF name format."""
        if not v or not v.strip():
            raise ValueError("PDF name cannot be empty")
        
        if not v.lower().endswith('.pdf'):
            v = v + '.pdf'
        
        return v.strip()


class ConfigUpdateRequest(BaseModel):
    """Request to update system configuration."""
    config_key: str
    config_value: Any
