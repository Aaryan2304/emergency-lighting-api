"""
Test the main detection functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.detection import LightingDetector, ImageProcessor, BoundingBoxUtils
from src.utils import Config


class TestLightingDetector:
    """Test cases for LightingDetector."""
    
    def test_detector_initialization(self, test_config):
        """Test detector initialization."""
        detector = LightingDetector(test_config)
        assert detector.config == test_config
        assert detector.image_processor is not None
        assert detector.bbox_utils is not None
    
    def test_detect_emergency_lights_empty_image(self, test_config):
        """Test detection with empty image."""
        detector = LightingDetector(test_config)
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        detections = detector.detect_emergency_lights(empty_image)
        assert isinstance(detections, list)
    
    def test_detect_shaded_areas(self, test_config):
        """Test shaded area detection."""
        detector = LightingDetector(test_config)
        
        # Create image with a dark rectangle
        image = np.ones((200, 200), dtype=np.uint8) * 255
        image[50:100, 50:150] = 0  # Dark rectangle
        
        shaded_areas = detector._detect_shaded_areas(image)
        assert isinstance(shaded_areas, list)
    
    def test_calculate_confidence(self, test_config):
        """Test confidence calculation."""
        detector = LightingDetector(test_config)
        
        area = {
            'bbox': [10, 10, 50, 30],
            'area': 800
        }
        
        confidence = detector._calculate_confidence(area)
        assert 0.0 <= confidence <= 1.0
    
    def test_visualize_detections(self, test_config):
        """Test detection visualization."""
        detector = LightingDetector(test_config)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        detections = [
            {
                'symbol': 'A1E',
                'bounding_box': [10, 10, 30, 30],
                'confidence': 0.8
            }
        ]
        
        vis_image = detector.visualize_detections(image, detections)
        assert vis_image.shape == image.shape


class TestImageProcessor:
    """Test cases for ImageProcessor."""
    
    def test_processor_initialization(self, test_config):
        """Test processor initialization."""
        processor = ImageProcessor(test_config)
        assert processor.config == test_config
    
    def test_preprocess_grayscale(self, test_config):
        """Test preprocessing grayscale image."""
        processor = ImageProcessor(test_config)
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        processed = processor.preprocess(gray_image)
        assert processed.shape == gray_image.shape
        assert processed.dtype == np.uint8
    
    def test_preprocess_color(self, test_config):
        """Test preprocessing color image."""
        processor = ImageProcessor(test_config)
        color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        processed = processor.preprocess(color_image)
        assert len(processed.shape) == 2  # Should be converted to grayscale
    
    def test_resize_image(self, test_config):
        """Test image resizing."""
        processor = ImageProcessor(test_config)
        large_image = np.zeros((2000, 3000, 3), dtype=np.uint8)
        
        resized = processor.resize_image(large_image, max_width=1000, max_height=800)
        assert resized.shape[1] <= 1000  # Width should be <= max_width
        assert resized.shape[0] <= 800   # Height should be <= max_height
    
    def test_extract_regions_of_interest(self, test_config):
        """Test ROI extraction."""
        processor = ImageProcessor(test_config)
        image = np.zeros((200, 300), dtype=np.uint8)
        
        regions = processor.extract_regions_of_interest(image)
        
        assert 'full' in regions
        assert 'top_half' in regions
        assert 'bottom_half' in regions
        assert 'left_half' in regions
        assert 'right_half' in regions
        assert 'center' in regions


class TestBoundingBoxUtils:
    """Test cases for BoundingBoxUtils."""
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation with no overlap."""
        box1 = [0, 0, 10, 10]
        box2 = [20, 20, 30, 30]
        
        iou = BoundingBoxUtils.calculate_iou(box1, box2)
        assert iou == 0.0
    
    def test_calculate_iou_full_overlap(self):
        """Test IoU calculation with full overlap."""
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]
        
        iou = BoundingBoxUtils.calculate_iou(box1, box2)
        assert iou == 1.0
    
    def test_calculate_iou_partial_overlap(self):
        """Test IoU calculation with partial overlap."""
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        
        iou = BoundingBoxUtils.calculate_iou(box1, box2)
        assert 0.0 < iou < 1.0
    
    def test_calculate_distance(self):
        """Test distance calculation between boxes."""
        box1 = [0, 0, 10, 10]
        box2 = [20, 20, 30, 30]
        
        distance = BoundingBoxUtils.calculate_distance(box1, box2)
        expected = ((15-5)**2 + (25-5)**2)**0.5  # Distance between centers
        assert abs(distance - expected) < 0.01
    
    def test_expand_bbox(self):
        """Test bounding box expansion."""
        bbox = [10, 10, 20, 20]
        image_shape = (100, 100)
        margin = 5
        
        expanded = BoundingBoxUtils.expand_bbox(bbox, margin, image_shape)
        
        assert expanded[0] == 5   # x1 - margin
        assert expanded[1] == 5   # y1 - margin
        assert expanded[2] == 25  # x2 + margin
        assert expanded[3] == 25  # y2 + margin
    
    def test_expand_bbox_bounds_checking(self):
        """Test bounding box expansion with bounds checking."""
        bbox = [0, 0, 10, 10]
        image_shape = (50, 50)
        margin = 20
        
        expanded = BoundingBoxUtils.expand_bbox(bbox, margin, image_shape)
        
        assert expanded[0] >= 0    # Should not go below 0
        assert expanded[1] >= 0    # Should not go below 0
        assert expanded[2] <= 50   # Should not exceed image width
        assert expanded[3] <= 50   # Should not exceed image height
    
    def test_is_bbox_valid(self):
        """Test bounding box validation."""
        valid_bbox = [10, 10, 30, 30]
        invalid_bbox1 = [30, 30, 10, 10]  # x2 < x1, y2 < y1
        invalid_bbox2 = [10, 10, 15, 15]  # Too small
        
        assert BoundingBoxUtils.is_bbox_valid(valid_bbox)
        assert not BoundingBoxUtils.is_bbox_valid(invalid_bbox1)
        assert not BoundingBoxUtils.is_bbox_valid(invalid_bbox2, min_size=20)
    
    def test_calculate_bbox_area(self):
        """Test bounding box area calculation."""
        bbox = [0, 0, 10, 20]
        area = BoundingBoxUtils.calculate_bbox_area(bbox)
        assert area == 200  # 10 * 20
    
    def test_normalize_bbox(self):
        """Test bounding box normalization."""
        bbox = [10, 20, 30, 40]
        image_shape = (100, 200)  # height, width
        
        normalized = BoundingBoxUtils.normalize_bbox(bbox, image_shape)
        
        assert normalized[0] == 0.05  # 10/200
        assert normalized[1] == 0.2   # 20/100
        assert normalized[2] == 0.15  # 30/200
        assert normalized[3] == 0.4   # 40/100
    
    def test_filter_overlapping_boxes(self):
        """Test overlapping box filtering."""
        boxes = [
            {'bounding_box': [0, 0, 10, 10], 'confidence': 0.9},
            {'bounding_box': [5, 5, 15, 15], 'confidence': 0.7},  # Overlapping
            {'bounding_box': [20, 20, 30, 30], 'confidence': 0.8}  # Separate
        ]
        
        filtered = BoundingBoxUtils.filter_overlapping_boxes(boxes, iou_threshold=0.3)
        
        # Should keep highest confidence box from overlapping pair
        assert len(filtered) == 2
        assert filtered[0]['confidence'] == 0.9  # Highest confidence kept
        assert any(box['confidence'] == 0.8 for box in filtered)  # Separate box kept


if __name__ == "__main__":
    pytest.main([__file__])
