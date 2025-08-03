"""
Database models for the emergency lighting detection system.
"""

from sqlalchemy import Column, String, Float, Integer, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class ProcessingRecord(Base):
    """SQLAlchemy model for processing records."""
    
    __tablename__ = 'processing_records'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    pdf_name = Column(String, nullable=False, unique=True)
    file_path = Column(String, nullable=False)
    project_id = Column(String, nullable=True)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processing_time = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    result_data = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    detections = relationship("Detection", back_populates="processing_record", cascade="all, delete-orphan")
    rulebook_data = relationship("RulebookData", back_populates="processing_record", cascade="all, delete-orphan")


class Detection(Base):
    """SQLAlchemy model for individual detections."""
    
    __tablename__ = 'detections'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    processing_id = Column(String, ForeignKey('processing_records.id'), nullable=False)
    symbol = Column(String, nullable=True)
    bounding_box = Column(Text, nullable=True)  # JSON string
    text_nearby = Column(Text, nullable=True)   # JSON string
    confidence = Column(Float, nullable=True)
    source_page = Column(Integer, nullable=True)
    source_sheet = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    processing_record = relationship("ProcessingRecord", back_populates="detections")


class RulebookData(Base):
    """SQLAlchemy model for rulebook data."""
    
    __tablename__ = 'rulebook_data'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    processing_id = Column(String, ForeignKey('processing_records.id'), nullable=False)
    data_type = Column(String, nullable=False)  # 'notes', 'schedule', 'legend', etc.
    content = Column(Text, nullable=False)      # JSON string
    source_page = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    processing_record = relationship("ProcessingRecord", back_populates="rulebook_data")


# Database schema creation utility
def create_tables(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(engine)
