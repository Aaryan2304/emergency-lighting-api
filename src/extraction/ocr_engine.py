"""
OCR engine for text extraction from electrical drawings.
Supports multiple OCR backends and text preprocessing.
"""

import cv2
import numpy as np
import pytesseract
import easyocr
from typing import List, Dict, Optional, Tuple
import logging
import re
import os
import platform

from ..utils.config import Config

logger = logging.getLogger(__name__)

# Configure Tesseract path for Windows
if platform.system() == "Windows":
    # Use the confirmed working path
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    logger.info(f"Set Tesseract path to: {pytesseract.pytesseract.tesseract_cmd}")
    
    # Verify Tesseract is accessible
    try:
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract version: {version}")
    except Exception as e:
        logger.error(f"Tesseract not accessible: {e}")


class OCREngine:
    """OCR engine for extracting text from electrical drawings."""
    
    def __init__(self, config: Config):
        self.config = config
        self.tesseract_config = config.TESSERACT_CONFIG
        self.ocr_language = config.OCR_LANGUAGE
        
        # Initialize EasyOCR reader
        try:
            self.easyocr_reader = easyocr.Reader([self.ocr_language])
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR: {e}")
            self.easyocr_reader = None
    
    def extract_text_tesseract(self, image: np.ndarray, 
                              bbox: Optional[List[int]] = None) -> List[Dict]:
        """
        Extract text using Tesseract OCR.
        
        Args:
            image: Input image
            bbox: Optional bounding box to extract from [x1, y1, x2, y2]
            
        Returns:
            List of text detections with bounding boxes
        """
        try:
            # Extract region if bbox provided
            if bbox:
                x1, y1, x2, y2 = bbox
                roi = image[y1:y2, x1:x2]
            else:
                roi = image.copy()
                x1, y1 = 0, 0
            
            # Preprocess for better OCR
            processed = self._preprocess_for_ocr(roi)
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                processed, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            detections = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = int(data['conf'][i])
                
                if text and confidence > 30:  # Filter low confidence
                    x = data['left'][i] + x1
                    y = data['top'][i] + y1
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    detections.append({
                        'text': text,
                        'confidence': confidence / 100.0,  # Normalize to 0-1
                        'bounding_box': [x, y, x + w, y + h],
                        'method': 'tesseract'
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Tesseract OCR error: {str(e)}")
            return []
    
    def extract_text_easyocr(self, image: np.ndarray, 
                            bbox: Optional[List[int]] = None) -> List[Dict]:
        """
        Extract text using EasyOCR.
        
        Args:
            image: Input image
            bbox: Optional bounding box to extract from [x1, y1, x2, y2]
            
        Returns:
            List of text detections with bounding boxes
        """
        if not self.easyocr_reader:
            return []
        
        try:
            # Extract region if bbox provided
            if bbox:
                x1, y1, x2, y2 = bbox
                roi = image[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1
            else:
                roi = image.copy()
                offset_x, offset_y = 0, 0
            
            # Run EasyOCR
            results = self.easyocr_reader.readtext(roi)
            
            detections = []
            for result in results:
                bbox_points, text, confidence = result
                
                if confidence > 0.3:  # Filter low confidence
                    # Convert bbox points to rectangle
                    x_coords = [point[0] for point in bbox_points]
                    y_coords = [point[1] for point in bbox_points]
                    
                    x1 = int(min(x_coords)) + offset_x
                    y1 = int(min(y_coords)) + offset_y
                    x2 = int(max(x_coords)) + offset_x
                    y2 = int(max(y_coords)) + offset_y
                    
                    detections.append({
                        'text': text.strip(),
                        'confidence': confidence,
                        'bounding_box': [x1, y1, x2, y2],
                        'method': 'easyocr'
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"EasyOCR error: {str(e)}")
            return []
    
    def extract_text_combined(self, image: np.ndarray, 
                             bbox: Optional[List[int]] = None) -> List[Dict]:
        """
        Extract text using both OCR engines and combine results.
        
        Args:
            image: Input image
            bbox: Optional bounding box to extract from
            
        Returns:
            Combined text detections
        """
        # Get results from both engines
        tesseract_results = self.extract_text_tesseract(image, bbox)
        easyocr_results = self.extract_text_easyocr(image, bbox)
        
        # Combine and deduplicate results
        all_results = tesseract_results + easyocr_results
        return self._deduplicate_text_results(all_results)
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR performance.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Dilation to thicken text
        processed = cv2.dilate(processed, kernel, iterations=1)
        
        return processed
    
    def _deduplicate_text_results(self, results: List[Dict]) -> List[Dict]:
        """
        Remove duplicate text detections from multiple OCR engines.
        
        Args:
            results: List of text detection results
            
        Returns:
            Deduplicated results
        """
        if not results:
            return []
        
        # Sort by confidence
        sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        
        deduplicated = []
        for result in sorted_results:
            is_duplicate = False
            
            for existing in deduplicated:
                # Check for overlap and similar text
                if (self._calculate_bbox_overlap(result['bounding_box'], 
                                               existing['bounding_box']) > 0.5 and
                    self._text_similarity(result['text'], existing['text']) > 0.8):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(result)
        
        return deduplicated
    
    def _calculate_bbox_overlap(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate overlap ratio between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            Overlap ratio
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return intersection / min(area1, area2) if min(area1, area2) > 0 else 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using simple metrics.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if text1 == text2:
            return 1.0
        
        # Simple character-based similarity
        if len(text1) == 0 or len(text2) == 0:
            return 0.0
        
        common_chars = sum(1 for a, b in zip(text1, text2) if a == b)
        max_len = max(len(text1), len(text2))
        
        return common_chars / max_len
    
    def extract_symbols(self, detections: List[Dict]) -> List[Dict]:
        """
        Extract lighting symbols from OCR detections.
        
        Args:
            detections: List of text detections
            
        Returns:
            List of symbol detections
        """
        symbol_patterns = [
            r'^[A-Z]\d+[A-Z]?$',  # A1E, B2, etc.
            r'^EM$',              # Emergency
            r'^EXIT$',            # Exit
            r'^[A-Z]+\d*$'        # Other symbols
        ]
        
        symbols = []
        for detection in detections:
            text = detection['text'].upper().strip()
            
            for pattern in symbol_patterns:
                if re.match(pattern, text):
                    detection['symbol_type'] = self._classify_symbol(text)
                    symbols.append(detection)
                    break
        
        return symbols
    
    def _classify_symbol(self, symbol: str) -> str:
        """
        Classify the type of lighting symbol.
        
        Args:
            symbol: Symbol text
            
        Returns:
            Symbol classification
        """
        symbol = symbol.upper()
        
        if 'E' in symbol or 'EM' in symbol:
            return 'emergency'
        elif 'EXIT' in symbol:
            return 'exit'
        elif re.match(r'^[A-Z]\d+$', symbol):
            return 'fixture'
        else:
            return 'unknown'
