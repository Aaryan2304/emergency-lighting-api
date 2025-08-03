"""
API routes for emergency lighting detection.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, List
import asyncio
import json

from src.api.models import (
    UploadResponse, ProcessingResult, DetailedResult, ErrorResponse,
    ProcessingStatus, ValidationRequest, ValidationResponse,
    BulkProcessingRequest, ReprocessRequest, ProcessingStats, SystemStatus
)
from src.core.pipeline import ProcessingPipeline
from src.database.db_manager import DatabaseManager
from src.utils import config, get_logger, file_handler

logger = get_logger(__name__)
router = APIRouter()

# Initialize components
pipeline = ProcessingPipeline()
db_manager = DatabaseManager()


@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "emergency-lighting-api"}


@router.post("/blueprints/upload", response_model=UploadResponse)
async def upload_blueprint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    project_id: Optional[str] = Query(None, description="Project grouping identifier")
):
    """
    Upload a PDF blueprint and trigger background processing.
    
    Args:
        file: PDF file to upload
        project_id: Optional project identifier for grouping
        background_tasks: FastAPI background tasks
        
    Returns:
        Upload response with processing status
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not config.is_file_supported(file.filename):
            supported = ', '.join(config.SUPPORTED_FORMATS)
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported: {supported}"
            )
        
        # Check file size
        content = await file.read()
        if len(content) > config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed ({config.get_max_file_size_mb():.1f} MB)"
            )
        
        # Save uploaded file
        file_path = file_handler.save_upload(content, file.filename)
        
        # Check if record already exists and delete it to allow re-upload
        existing_record = await db_manager.get_processing_record(file.filename)
        if existing_record:
            logger.info(f"Found existing record for {file.filename}, deleting to allow re-upload")
            await db_manager.delete_processing_record(file.filename)
        
        # Create processing record
        processing_id = await db_manager.create_processing_record(
            pdf_name=file.filename,
            file_path=file_path,
            project_id=project_id,
            status=ProcessingStatus.UPLOADED
        )
        
        # Schedule background processing
        background_tasks.add_task(
            process_blueprint_background,
            processing_id,
            file_path,
            file.filename
        )
        
        logger.info(f"File uploaded successfully: {file.filename} (ID: {processing_id})")
        
        return UploadResponse(
            status=ProcessingStatus.UPLOADED,
            pdf_name=file.filename,
            message="Processing started in background.",
            upload_id=processing_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload_blueprint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/blueprints/result", response_model=ProcessingResult)
async def get_processing_result(
    pdf_name: str = Query(..., description="Name of the uploaded PDF"),
    detailed: bool = Query(False, description="Include detailed detection information")
):
    """
    Retrieve processing results for a given PDF.
    
    Args:
        pdf_name: Name of the PDF file
        detailed: Whether to include detailed information
        
    Returns:
        Processing result
    """
    try:
        # Get processing record
        record = await db_manager.get_processing_record(pdf_name)
        
        if not record:
            raise HTTPException(
                status_code=404, 
                detail=f"No processing record found for {pdf_name}"
            )
        
        # Build response based on status
        if record['status'] == ProcessingStatus.COMPLETE:
            result_data = json.loads(record['result_data']) if record['result_data'] else {}
            
            return ProcessingResult(
                pdf_name=pdf_name,
                status=ProcessingStatus.COMPLETE,
                result=result_data.get('grouped_results', {}),
                processing_time=record.get('processing_time')
            )
            
        elif record['status'] == ProcessingStatus.FAILED:
            return ProcessingResult(
                pdf_name=pdf_name,
                status=ProcessingStatus.FAILED,
                message=record.get('error_message', 'Processing failed')
            )
            
        else:
            return ProcessingResult(
                pdf_name=pdf_name,
                status=ProcessingStatus.IN_PROGRESS,
                message="Processing is still in progress. Please try again later."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_processing_result: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/blueprints/result/detailed", response_model=DetailedResult)
async def get_detailed_result(
    pdf_name: str = Query(..., description="Name of the uploaded PDF")
):
    """
    Retrieve detailed processing results including individual detections.
    
    Args:
        pdf_name: Name of the PDF file
        
    Returns:
        Detailed processing result
    """
    try:
        record = await db_manager.get_processing_record(pdf_name)
        
        if not record:
            raise HTTPException(
                status_code=404,
                detail=f"No processing record found for {pdf_name}"
            )
        
        if record['status'] != ProcessingStatus.COMPLETE:
            return DetailedResult(
                pdf_name=pdf_name,
                status=record['status']
            )
        
        # Parse detailed results
        result_data = json.loads(record['result_data']) if record['result_data'] else {}
        
        return DetailedResult(
            pdf_name=pdf_name,
            status=ProcessingStatus.COMPLETE,
            summary=result_data.get('summary', {}),
            detections=result_data.get('detections', []),
            rulebook=result_data.get('rulebook', {}),
            processing_metadata=result_data.get('metadata', {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_detailed_result: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/blueprints/list")
async def list_processed_blueprints(
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    status: Optional[ProcessingStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip")
):
    """
    List processed blueprints with optional filtering.
    
    Args:
        project_id: Optional project ID filter
        status: Optional status filter
        limit: Maximum number of results
        offset: Number of results to skip
        
    Returns:
        List of processed blueprints
    """
    try:
        records = await db_manager.list_processing_records(
            project_id=project_id,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return {
            "blueprints": records,
            "total": len(records),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error in list_processed_blueprints: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/blueprints/reprocess", response_model=UploadResponse)
async def reprocess_blueprint(
    background_tasks: BackgroundTasks,
    request: ReprocessRequest
):
    """
    Reprocess an existing blueprint.
    
    Args:
        background_tasks: FastAPI background tasks
        request: Reprocessing request
        
    Returns:
        Reprocessing response
    """
    try:
        record = await db_manager.get_processing_record(request.pdf_name)
        
        if not record:
            raise HTTPException(
                status_code=404,
                detail=f"No processing record found for {request.pdf_name}"
            )
        
        if (record['status'] == ProcessingStatus.COMPLETE and 
            not request.force_reprocess):
            raise HTTPException(
                status_code=400,
                detail="Blueprint already processed. Use force_reprocess=true to reprocess."
            )
        
        # Update status to in progress
        await db_manager.update_processing_status(
            record['id'],
            ProcessingStatus.IN_PROGRESS
        )
        
        # Schedule reprocessing
        background_tasks.add_task(
            process_blueprint_background,
            record['id'],
            record['file_path'],
            request.pdf_name
        )
        
        return UploadResponse(
            status=ProcessingStatus.IN_PROGRESS,
            pdf_name=request.pdf_name,
            message="Reprocessing started in background.",
            upload_id=record['id']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in reprocess_blueprint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/blueprints/{pdf_name}")
async def delete_blueprint(pdf_name: str):
    """
    Delete a blueprint and its processing results.
    
    Args:
        pdf_name: Name of the PDF to delete
        
    Returns:
        Deletion confirmation
    """
    try:
        success = await db_manager.delete_processing_record(pdf_name)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"No processing record found for {pdf_name}"
            )
        
        return {"message": f"Blueprint {pdf_name} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_blueprint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/system/status", response_model=SystemStatus)
async def get_system_status():
    """
    Get system status and health information.
    
    Returns:
        System status information
    """
    try:
        # Check component status
        api_status = "healthy"
        
        # Check database
        try:
            await db_manager.health_check()
            database_status = "healthy"
        except:
            database_status = "unhealthy"
        
        # Check LLM (basic check)
        llm_status = "healthy" if config.OPENAI_API_KEY else "not_configured"
        
        # OCR is always available (built-in)
        ocr_status = "healthy"
        
        # Get processing queue info
        queue_size = await db_manager.get_queue_size()
        active_jobs = await db_manager.get_active_jobs_count()
        
        return SystemStatus(
            api_status=api_status,
            database_status=database_status,
            llm_status=llm_status,
            ocr_status=ocr_status,
            processing_queue_size=queue_size,
            active_jobs=active_jobs
        )
        
    except Exception as e:
        logger.error(f"Error in get_system_status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/system/stats", response_model=ProcessingStats)
async def get_processing_stats():
    """
    Get processing statistics.
    
    Returns:
        Processing statistics
    """
    try:
        stats = await db_manager.get_processing_stats()
        return ProcessingStats(**stats)
        
    except Exception as e:
        logger.error(f"Error in get_processing_stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def process_blueprint_background(processing_id: str, file_path: str, pdf_name: str):
    """
    Background task for processing blueprints.
    
    Args:
        processing_id: Processing record ID
        file_path: Path to uploaded file
        pdf_name: Original PDF name
    """
    try:
        logger.info(f"Starting background processing for {pdf_name}")
        
        # Update status to in progress
        await db_manager.update_processing_status(
            processing_id,
            ProcessingStatus.IN_PROGRESS
        )
        
        # Process the file
        result = await pipeline.process_pdf(file_path, pdf_name)
        
        # Save results
        await db_manager.update_processing_result(
            processing_id,
            ProcessingStatus.COMPLETE,
            result,
            processing_time=result.get('metadata', {}).get('processing_time'),
            confidence_score=result.get('metadata', {}).get('confidence_score')
        )
        
        logger.info(f"Completed background processing for {pdf_name}")
        
    except Exception as e:
        logger.error(f"Error in background processing for {pdf_name}: {str(e)}")
        
        # Update status to failed
        await db_manager.update_processing_status(
            processing_id,
            ProcessingStatus.FAILED,
            error_message=str(e)
        )
    
    finally:
        # Cleanup uploaded file
        try:
            file_handler.cleanup_file(file_path)
        except:
            pass
