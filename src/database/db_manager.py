"""
Database manager for emergency lighting detection system.
Handles all database operations and data persistence.
"""

import sqlite3
import aiosqlite
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import asyncio

from ..utils import config, get_logger
from ..api.models import ProcessingStatus

logger = get_logger(__name__)


class DatabaseManager:
    """Database manager for handling all data persistence operations."""
    
    def __init__(self):
        self.db_url = config.get_db_url()
        self.db_path = self.db_url.replace('sqlite:///', '')
    
    async def initialize(self):
        """Initialize database and create tables."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Create processing records table
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS processing_records (
                        id TEXT PRIMARY KEY,
                        pdf_name TEXT NOT NULL UNIQUE,
                        file_path TEXT NOT NULL,
                        project_id TEXT,
                        status TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processing_time REAL,
                        confidence_score REAL,
                        result_data TEXT,
                        error_message TEXT
                    )
                ''')
                
                # Create detections table
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS detections (
                        id TEXT PRIMARY KEY,
                        processing_id TEXT NOT NULL,
                        symbol TEXT,
                        bounding_box TEXT,
                        text_nearby TEXT,
                        confidence REAL,
                        source_page INTEGER,
                        source_sheet TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (processing_id) REFERENCES processing_records (id)
                    )
                ''')
                
                # Create rulebook table
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS rulebook_data (
                        id TEXT PRIMARY KEY,
                        processing_id TEXT NOT NULL,
                        data_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source_page INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (processing_id) REFERENCES processing_records (id)
                    )
                ''')
                
                # Create indexes
                await db.execute('CREATE INDEX IF NOT EXISTS idx_pdf_name ON processing_records(pdf_name)')
                await db.execute('CREATE INDEX IF NOT EXISTS idx_status ON processing_records(status)')
                await db.execute('CREATE INDEX IF NOT EXISTS idx_project_id ON processing_records(project_id)')
                await db.execute('CREATE INDEX IF NOT EXISTS idx_processing_id ON detections(processing_id)')
                
                await db.commit()
                
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    async def create_processing_record(self, pdf_name: str, file_path: str, 
                                     project_id: Optional[str] = None,
                                     status: ProcessingStatus = ProcessingStatus.UPLOADED) -> str:
        """
        Create a new processing record.
        
        Args:
            pdf_name: Name of the PDF file
            file_path: Path to the uploaded file
            project_id: Optional project identifier
            status: Initial processing status
            
        Returns:
            Processing record ID
        """
        try:
            record_id = str(uuid.uuid4())
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    INSERT INTO processing_records 
                    (id, pdf_name, file_path, project_id, status)
                    VALUES (?, ?, ?, ?, ?)
                ''', (record_id, pdf_name, file_path, project_id, status.value))
                
                await db.commit()
            
            logger.info(f"Created processing record: {record_id} for {pdf_name}")
            return record_id
            
        except Exception as e:
            logger.error(f"Error creating processing record: {str(e)}")
            raise
    
    async def get_processing_record(self, pdf_name: str) -> Optional[Dict[str, Any]]:
        """
        Get processing record by PDF name.
        
        Args:
            pdf_name: Name of the PDF file
            
        Returns:
            Processing record or None if not found
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                async with db.execute('''
                    SELECT * FROM processing_records WHERE pdf_name = ?
                ''', (pdf_name,)) as cursor:
                    row = await cursor.fetchone()
                    
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Error getting processing record: {str(e)}")
            return None
    
    async def update_processing_status(self, record_id: str, status: ProcessingStatus,
                                     error_message: Optional[str] = None):
        """
        Update processing status.
        
        Args:
            record_id: Processing record ID
            status: New status
            error_message: Optional error message for failed status
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    UPDATE processing_records 
                    SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (status.value, error_message, record_id))
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Error updating processing status: {str(e)}")
            raise
    
    async def update_processing_result(self, record_id: str, status: ProcessingStatus,
                                     result_data: Dict[str, Any],
                                     processing_time: Optional[float] = None,
                                     confidence_score: Optional[float] = None):
        """
        Update processing result.
        
        Args:
            record_id: Processing record ID
            status: Processing status
            result_data: Complete result data
            processing_time: Processing time in seconds
            confidence_score: Overall confidence score
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Update main record
                await db.execute('''
                    UPDATE processing_records 
                    SET status = ?, result_data = ?, processing_time = ?, 
                        confidence_score = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (status.value, json.dumps(result_data), processing_time, 
                      confidence_score, record_id))
                
                # Save individual detections
                detections = result_data.get('detections', [])
                for detection in detections:
                    detection_id = str(uuid.uuid4())
                    await db.execute('''
                        INSERT INTO detections 
                        (id, processing_id, symbol, bounding_box, text_nearby, 
                         confidence, source_page, source_sheet)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        detection_id, record_id,
                        detection.get('symbol'),
                        json.dumps(detection.get('bounding_box', [])),
                        json.dumps(detection.get('text_nearby', [])),
                        detection.get('confidence'),
                        detection.get('source_page'),
                        detection.get('source_sheet')
                    ))
                
                # Save rulebook data
                rulebook = result_data.get('rulebook', {})
                for data_type, content in rulebook.items():
                    if content:  # Only save non-empty content
                        rulebook_id = str(uuid.uuid4())
                        await db.execute('''
                            INSERT INTO rulebook_data 
                            (id, processing_id, data_type, content)
                            VALUES (?, ?, ?, ?)
                        ''', (rulebook_id, record_id, data_type, json.dumps(content)))
                
                await db.commit()
                
            logger.info(f"Updated processing result for record: {record_id}")
            
        except Exception as e:
            logger.error(f"Error updating processing result: {str(e)}")
            raise
    
    async def list_processing_records(self, project_id: Optional[str] = None,
                                    status: Optional[ProcessingStatus] = None,
                                    limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List processing records with optional filtering.
        
        Args:
            project_id: Optional project ID filter
            status: Optional status filter
            limit: Maximum number of records
            offset: Number of records to skip
            
        Returns:
            List of processing records
        """
        try:
            query = "SELECT * FROM processing_records WHERE 1=1"
            params = []
            
            if project_id:
                query += " AND project_id = ?"
                params.append(project_id)
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
                    
        except Exception as e:
            logger.error(f"Error listing processing records: {str(e)}")
            return []
    
    async def delete_processing_record(self, pdf_name: str) -> bool:
        """
        Delete processing record and related data.
        
        Args:
            pdf_name: Name of the PDF file
            
        Returns:
            True if deleted, False if not found
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get record ID first
                async with db.execute('''
                    SELECT id FROM processing_records WHERE pdf_name = ?
                ''', (pdf_name,)) as cursor:
                    row = await cursor.fetchone()
                    
                if not row:
                    return False
                
                record_id = row[0]
                
                # Delete related data
                await db.execute('DELETE FROM detections WHERE processing_id = ?', (record_id,))
                await db.execute('DELETE FROM rulebook_data WHERE processing_id = ?', (record_id,))
                await db.execute('DELETE FROM processing_records WHERE id = ?', (record_id,))
                
                await db.commit()
                
            logger.info(f"Deleted processing record for: {pdf_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting processing record: {str(e)}")
            return False
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Processing statistics
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Total processed
                async with db.execute('SELECT COUNT(*) FROM processing_records') as cursor:
                    total_processed = (await cursor.fetchone())[0]
                
                # Successful
                async with db.execute('''
                    SELECT COUNT(*) FROM processing_records WHERE status = ?
                ''', (ProcessingStatus.COMPLETE.value,)) as cursor:
                    successful = (await cursor.fetchone())[0]
                
                # Failed
                async with db.execute('''
                    SELECT COUNT(*) FROM processing_records WHERE status = ?
                ''', (ProcessingStatus.FAILED.value,)) as cursor:
                    failed = (await cursor.fetchone())[0]
                
                # Average processing time
                async with db.execute('''
                    SELECT AVG(processing_time) FROM processing_records 
                    WHERE processing_time IS NOT NULL
                ''') as cursor:
                    avg_time = (await cursor.fetchone())[0] or 0.0
                
                # Last processed
                async with db.execute('''
                    SELECT MAX(updated_at) FROM processing_records 
                    WHERE status = ?
                ''', (ProcessingStatus.COMPLETE.value,)) as cursor:
                    last_processed = (await cursor.fetchone())[0]
                
                return {
                    'total_processed': total_processed,
                    'successful': successful,
                    'failed': failed,
                    'average_processing_time': avg_time,
                    'last_processed': datetime.fromisoformat(last_processed) if last_processed else None
                }
                
        except Exception as e:
            logger.error(f"Error getting processing stats: {str(e)}")
            return {
                'total_processed': 0,
                'successful': 0,
                'failed': 0,
                'average_processing_time': 0.0,
                'last_processed': None
            }
    
    async def get_queue_size(self) -> int:
        """Get number of items in processing queue."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute('''
                    SELECT COUNT(*) FROM processing_records 
                    WHERE status IN (?, ?)
                ''', (ProcessingStatus.UPLOADED.value, ProcessingStatus.IN_PROGRESS.value)) as cursor:
                    return (await cursor.fetchone())[0]
        except:
            return 0
    
    async def get_active_jobs_count(self) -> int:
        """Get number of actively processing jobs."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute('''
                    SELECT COUNT(*) FROM processing_records 
                    WHERE status = ?
                ''', (ProcessingStatus.IN_PROGRESS.value,)) as cursor:
                    return (await cursor.fetchone())[0]
        except:
            return 0
    
    async def health_check(self) -> bool:
        """Perform database health check."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('SELECT 1')
            return True
        except:
            return False
    
    async def cleanup_old_records(self, days_old: int = 30) -> int:
        """
        Clean up old processing records.
        
        Args:
            days_old: Number of days to keep records
            
        Returns:
            Number of records cleaned up
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Delete old completed/failed records
                result = await db.execute('''
                    DELETE FROM processing_records 
                    WHERE status IN (?, ?) 
                    AND created_at < datetime('now', '-{} days')
                '''.format(days_old), (ProcessingStatus.COMPLETE.value, ProcessingStatus.FAILED.value))
                
                await db.commit()
                
                cleaned_count = result.rowcount
                logger.info(f"Cleaned up {cleaned_count} old records")
                return cleaned_count
                
        except Exception as e:
            logger.error(f"Error cleaning up old records: {str(e)}")
            return 0
