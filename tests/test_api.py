"""
Test the API endpoints.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.api.app import create_app


class TestAPI:
    """Test cases for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
    
    @patch('src.api.routes.pipeline')
    @patch('src.api.routes.db_manager')
    def test_upload_blueprint_success(self, mock_db, mock_pipeline, client):
        """Test successful blueprint upload."""
        # Mock database operations
        mock_db.create_processing_record = AsyncMock(return_value="test-id")
        
        # Create test file
        test_file_content = b"%PDF-1.4\nTest PDF content\n%%EOF"
        
        response = client.post(
            "/api/v1/blueprints/upload",
            files={"file": ("test.pdf", test_file_content, "application/pdf")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "uploaded"
        assert data["pdf_name"] == "test.pdf"
        assert "upload_id" in data
    
    def test_upload_blueprint_no_file(self, client):
        """Test upload without file."""
        response = client.post("/api/v1/blueprints/upload")
        assert response.status_code == 422  # Validation error
    
    def test_upload_blueprint_unsupported_format(self, client):
        """Test upload with unsupported file format."""
        test_file_content = b"Test content"
        
        response = client.post(
            "/api/v1/blueprints/upload",
            files={"file": ("test.txt", test_file_content, "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]
    
    def test_upload_blueprint_large_file(self, client):
        """Test upload with file too large."""
        # Create file larger than limit (assuming 50MB limit)
        large_content = b"x" * (60 * 1024 * 1024)  # 60MB
        
        response = client.post(
            "/api/v1/blueprints/upload",
            files={"file": ("large.pdf", large_content, "application/pdf")}
        )
        
        assert response.status_code == 400
        assert "File size exceeds" in response.json()["detail"]
    
    @patch('src.api.routes.db_manager')
    def test_get_result_not_found(self, mock_db, client):
        """Test getting result for non-existent PDF."""
        mock_db.get_processing_record = AsyncMock(return_value=None)
        
        response = client.get("/api/v1/blueprints/result?pdf_name=nonexistent.pdf")
        
        assert response.status_code == 404
        assert "No processing record found" in response.json()["detail"]
    
    @patch('src.api.routes.db_manager')
    def test_get_result_in_progress(self, mock_db, client):
        """Test getting result for in-progress processing."""
        mock_record = {
            'pdf_name': 'test.pdf',
            'status': 'in_progress',
            'result_data': None
        }
        mock_db.get_processing_record = AsyncMock(return_value=mock_record)
        
        response = client.get("/api/v1/blueprints/result?pdf_name=test.pdf")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "in_progress"
        assert "still in progress" in data["message"]
    
    @patch('src.api.routes.db_manager')
    def test_get_result_complete(self, mock_db, client):
        """Test getting result for completed processing."""
        result_data = {
            'summary': {
                'A1E': {'count': 5, 'description': 'Emergency fixtures'}
            }
        }
        
        mock_record = {
            'pdf_name': 'test.pdf',
            'status': 'complete',
            'result_data': json.dumps(result_data),
            'processing_time': 45.5,
            'confidence_score': 0.85
        }
        mock_db.get_processing_record = AsyncMock(return_value=mock_record)
        
        response = client.get("/api/v1/blueprints/result?pdf_name=test.pdf")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "complete"
        assert data["result"]["A1E"]["count"] == 5
        assert data["processing_time"] == 45.5
        assert data["confidence_score"] == 0.85
    
    @patch('src.api.routes.db_manager')
    def test_get_result_failed(self, mock_db, client):
        """Test getting result for failed processing."""
        mock_record = {
            'pdf_name': 'test.pdf',
            'status': 'failed',
            'error_message': 'Processing failed due to invalid PDF'
        }
        mock_db.get_processing_record = AsyncMock(return_value=mock_record)
        
        response = client.get("/api/v1/blueprints/result?pdf_name=test.pdf")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "Processing failed" in data["message"]
    
    @patch('src.api.routes.db_manager')
    def test_get_detailed_result(self, mock_db, client):
        """Test getting detailed result."""
        result_data = {
            'summary': {'A1E': {'count': 5, 'description': 'Emergency fixtures'}},
            'detections': [
                {
                    'symbol': 'A1E',
                    'bounding_box': [10, 10, 30, 30],
                    'confidence': 0.9
                }
            ],
            'rulebook': {'notes': ['Emergency lighting requirements']},
            'metadata': {'processing_time': 45.5}
        }
        
        mock_record = {
            'pdf_name': 'test.pdf',
            'status': 'complete',
            'result_data': json.dumps(result_data)
        }
        mock_db.get_processing_record = AsyncMock(return_value=mock_record)
        
        response = client.get("/api/v1/blueprints/result/detailed?pdf_name=test.pdf")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "complete"
        assert len(data["detections"]) == 1
        assert data["detections"][0]["symbol"] == "A1E"
    
    @patch('src.api.routes.db_manager')
    def test_list_blueprints(self, mock_db, client):
        """Test listing blueprints."""
        mock_records = [
            {
                'pdf_name': 'test1.pdf',
                'status': 'complete',
                'created_at': '2023-01-01T00:00:00'
            },
            {
                'pdf_name': 'test2.pdf',
                'status': 'in_progress',
                'created_at': '2023-01-02T00:00:00'
            }
        ]
        mock_db.list_processing_records = AsyncMock(return_value=mock_records)
        
        response = client.get("/api/v1/blueprints/list")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["blueprints"]) == 2
        assert data["total"] == 2
    
    @patch('src.api.routes.db_manager')
    def test_list_blueprints_with_filters(self, mock_db, client):
        """Test listing blueprints with filters."""
        mock_db.list_processing_records = AsyncMock(return_value=[])
        
        response = client.get(
            "/api/v1/blueprints/list?status=complete&project_id=test-project&limit=10"
        )
        
        assert response.status_code == 200
        # Verify that the mock was called with correct parameters
        mock_db.list_processing_records.assert_called_once()
    
    @patch('src.api.routes.db_manager')
    def test_delete_blueprint_success(self, mock_db, client):
        """Test successful blueprint deletion."""
        mock_db.delete_processing_record = AsyncMock(return_value=True)
        
        response = client.delete("/api/v1/blueprints/test.pdf")
        
        assert response.status_code == 200
        data = response.json()
        assert "deleted successfully" in data["message"]
    
    @patch('src.api.routes.db_manager')
    def test_delete_blueprint_not_found(self, mock_db, client):
        """Test deleting non-existent blueprint."""
        mock_db.delete_processing_record = AsyncMock(return_value=False)
        
        response = client.delete("/api/v1/blueprints/nonexistent.pdf")
        
        assert response.status_code == 404
    
    @patch('src.api.routes.db_manager')
    def test_system_status(self, mock_db, client):
        """Test system status endpoint."""
        mock_db.health_check = AsyncMock(return_value=True)
        mock_db.get_queue_size = AsyncMock(return_value=5)
        mock_db.get_active_jobs_count = AsyncMock(return_value=2)
        
        response = client.get("/api/v1/system/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["api_status"] == "healthy"
        assert data["database_status"] == "healthy"
        assert data["processing_queue_size"] == 5
        assert data["active_jobs"] == 2
    
    @patch('src.api.routes.db_manager')
    def test_processing_stats(self, mock_db, client):
        """Test processing statistics endpoint."""
        mock_stats = {
            'total_processed': 100,
            'successful': 85,
            'failed': 15,
            'average_processing_time': 42.5,
            'last_processed': None
        }
        mock_db.get_processing_stats = AsyncMock(return_value=mock_stats)
        
        response = client.get("/api/v1/system/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 100
        assert data["successful"] == 85
        assert data["failed"] == 15
        assert data["average_processing_time"] == 42.5


if __name__ == "__main__":
    pytest.main([__file__])
