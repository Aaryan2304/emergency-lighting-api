"""
OCR engine for text extraction from electrical drawings.
Supports multiple OCR backends and text preprocessing.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import re
import os
import platform

# Import OCR libraries with error handling
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"pytesseract not available: {e}")
    PYTESSERACT_AVAILABLE = False
    pytesseract = None

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"easyocr not available: {e}")
    EASYOCR_AVAILABLE = False
    easyocr = None

from ..utils.config import Config

logger = logging.getLogger(__name__)

# Check Tesseract availability
TESSERACT_AVAILABLE = False

if PYTESSERACT_AVAILABLE and pytesseract:
    # Configure Tesseract path for Windows
    if platform.system() == "Windows":
        # Use the confirmed working path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        logger.info(f"Set Tesseract path to: {pytesseract.pytesseract.tesseract_cmd}")

    # Verify Tesseract is accessible
    try:
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract version: {version}")
        TESSERACT_AVAILABLE = True
    except Exception as e:
        logger.warning(f"Tesseract not accessible, will use EasyOCR only: {e}")
        TESSERACT_AVAILABLE = False
else:
    logger.warning("pytesseract module not available, will use EasyOCR only")
    TESSERACT_AVAILABLE = False


class OCREngine:
    """OCR engine for extracting text from electrical drawings."""
    
    def __init__(self, config: Config):
        self.config = config
        self.tesseract_config = config.TESSERACT_CONFIG
        self.ocr_language = config.OCR_LANGUAGE
        self.tesseract_available = TESSERACT_AVAILABLE
        
        # Initialize EasyOCR reader
        self.easyocr_available = EASYOCR_AVAILABLE
        self.easyocr_reader = None
        
        if EASYOCR_AVAILABLE and easyocr:
            try:
                self.easyocr_reader = easyocr.Reader([self.ocr_language])
                logger.info("EasyOCR reader initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR reader: {e}")
                self.easyocr_available = False
        else:
            logger.warning("EasyOCR not available")
            self.easyocr_available = False
        
        # Check if at least one OCR engine is available
        if not self.tesseract_available and not self.easyocr_available:
            logger.error("No OCR engines available! Text extraction will fail.")
        
        logger.info(f"OCR Engine initialized - Tesseract: {self.tesseract_available}, EasyOCR: {self.easyocr_available}")
    
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
        if not self.tesseract_available:
            logger.warning("Tesseract not available, returning empty results")
            return []
            
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
            logger.warning("EasyOCR not available, returning empty results")
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
        Extract text using available OCR engines and combine results.
        
        Args:
            image: Input image
            bbox: Optional bounding box to extract from
            
        Returns:
            Combined text detections
        """
        all_results = []
        
        # Get results from Tesseract if available
        if self.tesseract_available:
            tesseract_results = self.extract_text_tesseract(image, bbox)
            all_results.extend(tesseract_results)
        
        # Get results from EasyOCR if available
        if self.easyocr_reader:
            easyocr_results = self.extract_text_easyocr(image, bbox)
            all_results.extend(easyocr_results)
        
        # If no OCR engines are available, log error
        if not self.tesseract_available and not self.easyocr_reader:
            logger.error("No OCR engines available!")
            return []
        
        # Deduplicate results if we have multiple engines
        if self.tesseract_available and self.easyocr_reader:
            return self._deduplicate_text_results(all_results)
        else:
            return all_results
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
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
        
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        
        # Denoise
        gray = cv2.medianBlur(gray, 3)
        
        # Threshold to binary
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
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
        
        # Sort by confidence (descending)
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        deduplicated = []
        for result in results:
            is_duplicate = False
            
            for existing in deduplicated:
                # Check if this result overlaps significantly with an existing one
                overlap = self._calculate_bbox_overlap(
                    result['bounding_box'], 
                    existing['bounding_box']
                )
                
                # Check text similarity
                text_sim = self._text_similarity(result['text'], existing['text'])
                
                # Consider duplicate if high overlap AND similar text
                if overlap > 0.7 and text_sim > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(result)
        
        return deduplicated
    
    def _calculate_bbox_overlap(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate intersection over union (IoU) of two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU ratio (0-1)
        """
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using simple character-based approach.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity ratio (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # Simple character overlap ratio
        text1_set = set(text1.lower())
        text2_set = set(text2.lower())
        
        intersection = text1_set.intersection(text2_set)
        union = text1_set.union(text2_set)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def extract_symbols(self, detections: List[Dict]) -> List[Dict]:
        """
        Extract and classify symbols from text detections.
        
        Args:
            detections: List of text detections
            
        Returns:
            List of symbol detections with classifications
        """
        symbols = []
        
        for detection in detections:
            text = detection['text'].strip()
            
            # Skip empty or very short text
            if len(text) < 1:
                continue
            
            # Classify the symbol
            symbol_type = self._classify_symbol(text)
            
            if symbol_type != 'unknown':
                symbols.append({
                    'text': text,
                    'type': symbol_type,
                    'confidence': detection['confidence'],
                    'bounding_box': detection['bounding_box']
                })
        
        return symbols
    
    def _classify_symbol(self, symbol: str) -> str:
        """
        Classify a text symbol into predefined categories.
        
        Args:
            symbol: Text symbol to classify
            
        Returns:
            Symbol classification
        """
        symbol = symbol.upper().strip()
        
        # Emergency lighting patterns
        if 'EL' in symbol or 'EMERGENCY' in symbol:
            return 'emergency_light'
        elif 'EXIT' in symbol:
            return 'exit'
        elif re.match(r'^[A-Z]\d+$', symbol):
            return 'fixture'
        else:
            return 'unknown'
