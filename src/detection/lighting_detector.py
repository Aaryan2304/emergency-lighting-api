"""
Main lighting detection module for emergency lighting fixtures.
Handles detection of shaded rectangular areas and associated symbols.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

from .image_processor import ImageProcessor
from .bbox_utils import BoundingBoxUtils
from ..utils.config import Config

logger = logging.getLogger(__name__)


class LightingDetector:
    """Main class for detecting emergency lighting fixtures in electrical drawings."""
    
    def __init__(self, config: Config):
        self.config = config
        self.image_processor = ImageProcessor(config)
        self.bbox_utils = BoundingBoxUtils()
        
    def detect_emergency_lights(self, image: np.ndarray) -> List[Dict]:
        """
        Detect emergency lighting fixtures in the given image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected fixtures with bounding boxes and metadata
        """
        try:
            # Preprocess the image
            processed_image = self.image_processor.preprocess(image)
            
            # Detect shaded rectangular areas
            shaded_areas = self._detect_shaded_areas(processed_image)
            
            # Detect symbols and text near shaded areas
            detections = []
            for area in shaded_areas:
                detection = self._analyze_area(area, image)
                if detection:
                    detections.append(detection)
                    
            logger.info(f"Detected {len(detections)} emergency lighting fixtures")
            return detections
            
        except Exception as e:
            logger.error(f"Error in emergency light detection: {str(e)}")
            return []
    
    def _detect_shaded_areas(self, image: np.ndarray) -> List[Dict]:
        """
        Detect shaded rectangular areas in the image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of detected shaded areas with coordinates
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply threshold to find dark/shaded areas
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shaded_areas = []
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.config.MIN_CONTOUR_AREA:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (rectangular shapes)
            aspect_ratio = w / h if h > 0 else 0
            if 0.3 <= aspect_ratio <= 3.0:  # Reasonable rectangle ratios
                shaded_areas.append({
                    'bbox': [x, y, x + w, y + h],
                    'area': area,
                    'contour': contour
                })
                
        return shaded_areas
    
    def _analyze_area(self, area: Dict, original_image: np.ndarray) -> Optional[Dict]:
        """
        Analyze a detected area to extract symbol and nearby text.
        
        Args:
            area: Detected area information
            original_image: Original image for text extraction
            
        Returns:
            Detection result with symbol, bbox, and nearby text
        """
        bbox = area['bbox']
        x1, y1, x2, y2 = bbox
        
        # Expand search area for nearby text
        margin = 50
        h, w = original_image.shape[:2]
        search_x1 = max(0, x1 - margin)
        search_y1 = max(0, y1 - margin)
        search_x2 = min(w, x2 + margin)
        search_y2 = min(h, y2 + margin)
        
        # Extract region for text analysis
        search_region = original_image[search_y1:search_y2, search_x1:search_x2]
        
        # This would be enhanced with OCR in a complete implementation
        # For now, return basic detection structure
        detection = {
            'symbol': self._extract_symbol(search_region),
            'bounding_box': bbox,
            'text_nearby': self._extract_nearby_text(search_region),
            'confidence': self._calculate_confidence(area),
            'area_size': area['area']
        }
        
        return detection
    
    def _extract_symbol(self, region: np.ndarray) -> str:
        """
        Extract symbol from the region (placeholder for OCR integration).
        
        Args:
            region: Image region to analyze
            
        Returns:
            Extracted symbol string
        """
        # Placeholder - would integrate with OCR engine
        # This could identify symbols like "A1E", "EM", etc.
        return "A1E"  # Default symbol for emergency lighting
    
    def _extract_nearby_text(self, region: np.ndarray) -> List[str]:
        """
        Extract nearby text from the region (placeholder for OCR integration).
        
        Args:
            region: Image region to analyze
            
        Returns:
            List of extracted text strings
        """
        # Placeholder - would integrate with OCR engine
        # This could identify text like "EM", "Exit", "Unswitched"
        return ["EM", "Exit"]
    
    def _calculate_confidence(self, area: Dict) -> float:
        """
        Calculate confidence score for the detection.
        
        Args:
            area: Detected area information
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple confidence based on area size and shape
        area_size = area['area']
        bbox = area['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Prefer medium-sized rectangular areas
        aspect_ratio = width / height if height > 0 else 0
        size_score = min(1.0, area_size / 1000)  # Normalize area
        shape_score = 1.0 - abs(aspect_ratio - 1.5) / 1.5  # Prefer 1.5:1 ratio
        
        return max(0.0, min(1.0, (size_score + shape_score) / 2))
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Visualize detections on the image for debugging.
        
        Args:
            image: Original image
            detections: List of detections
            
        Returns:
            Image with detection overlays
        """
        vis_image = image.copy()
        
        for detection in detections:
            bbox = detection['bounding_box']
            symbol = detection['symbol']
            confidence = detection.get('confidence', 0.0)
            
            # Draw bounding box
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         (0, 255, 0), 2)
            
            # Draw symbol and confidence
            label = f"{symbol} ({confidence:.2f})"
            cv2.putText(vis_image, label, (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis_image
