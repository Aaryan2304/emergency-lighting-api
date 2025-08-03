"""
Table extraction utilities for lighting schedules and legends.
Handles detection and parsing of tabular data from electrical drawings.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import pandas as pd

from .ocr_engine import OCREngine
from ..utils.config import Config

logger = logging.getLogger(__name__)


class TableExtractor:
    """Extract tables from electrical drawings, particularly lighting schedules."""
    
    def __init__(self, config: Config):
        self.config = config
        self.ocr_engine = OCREngine(config)
    
    def extract_lighting_schedule(self, image: np.ndarray) -> Dict:
        """
        Extract lighting schedule table from the image.
        
        Args:
            image: Input image containing lighting schedule
            
        Returns:
            Structured lighting schedule data
        """
        try:
            # Detect table regions
            table_regions = self._detect_tables(image)
            
            if not table_regions:
                logger.warning("No tables detected in image")
                return {}
            
            # Extract the largest table (likely the main schedule)
            main_table = max(table_regions, key=lambda x: x['area'])
            
            # Extract table content
            table_data = self._extract_table_content(image, main_table)
            
            # Parse lighting schedule
            schedule = self._parse_lighting_schedule(table_data)
            
            return schedule
            
        except Exception as e:
            logger.error(f"Error extracting lighting schedule: {str(e)}")
            return {}
    
    def extract_legend(self, image: np.ndarray) -> Dict:
        """
        Extract legend information from the image.
        
        Args:
            image: Input image containing legend
            
        Returns:
            Legend data with symbol mappings
        """
        try:
            # Look for legend-specific patterns
            legend_regions = self._detect_legend_regions(image)
            
            legend_data = {}
            for region in legend_regions:
                content = self._extract_table_content(image, region)
                parsed = self._parse_legend_content(content)
                legend_data.update(parsed)
            
            return legend_data
            
        except Exception as e:
            logger.error(f"Error extracting legend: {str(e)}")
            return {}
    
    def _detect_tables(self, image: np.ndarray) -> List[Dict]:
        """
        Detect table regions in the image using line detection.
        
        Args:
            image: Input image
            
        Returns:
            List of detected table regions
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines to find table structure
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Find table contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Minimum table size
                x, y, w, h = cv2.boundingRect(contour)
                tables.append({
                    'bbox': [x, y, x + w, y + h],
                    'area': area,
                    'contour': contour
                })
        
        return tables
    
    def _detect_legend_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect regions that likely contain legend information.
        
        Args:
            image: Input image
            
        Returns:
            List of legend regions
        """
        # Look for text patterns that indicate legends
        text_detections = self.ocr_engine.extract_text_combined(image)
        
        legend_keywords = ['legend', 'lighting', 'fixture', 'schedule', 'symbol']
        legend_regions = []
        
        for detection in text_detections:
            text = detection['text'].lower()
            if any(keyword in text for keyword in legend_keywords):
                # Expand region around legend text
                bbox = detection['bounding_box']
                expanded_bbox = self._expand_region_for_table(bbox, image.shape)
                
                legend_regions.append({
                    'bbox': expanded_bbox,
                    'area': (expanded_bbox[2] - expanded_bbox[0]) * (expanded_bbox[3] - expanded_bbox[1]),
                    'keyword': text
                })
        
        return legend_regions
    
    def _expand_region_for_table(self, bbox: List[int], image_shape: Tuple[int, int]) -> List[int]:
        """
        Expand a text region to likely include the full table.
        
        Args:
            bbox: Original text bounding box
            image_shape: Image dimensions (height, width)
            
        Returns:
            Expanded bounding box
        """
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Expand significantly to capture table
        margin_x = 200
        margin_y = 300
        
        expanded = [
            max(0, x1 - margin_x),
            max(0, y1 - 50),  # Small margin above
            min(w, x2 + margin_x),
            min(h, y2 + margin_y)  # Large margin below for table
        ]
        
        return expanded
    
    def _extract_table_content(self, image: np.ndarray, table_region: Dict) -> List[List[str]]:
        """
        Extract text content from a table region.
        
        Args:
            image: Input image
            table_region: Table region information
            
        Returns:
            2D list representing table content
        """
        bbox = table_region['bbox']
        x1, y1, x2, y2 = bbox
        
        # Extract table region
        table_image = image[y1:y2, x1:x2]
        
        # Get all text in the region
        text_detections = self.ocr_engine.extract_text_combined(table_image)
        
        if not text_detections:
            return []
        
        # Group text by rows based on y-coordinates
        rows = self._group_text_by_rows(text_detections)
        
        # Sort rows by y-coordinate
        sorted_rows = sorted(rows.items(), key=lambda x: x[0])
        
        # Extract text for each row
        table_content = []
        for _, row_texts in sorted_rows:
            # Sort by x-coordinate for column order
            sorted_texts = sorted(row_texts, key=lambda x: x['bounding_box'][0])
            row_content = [text['text'] for text in sorted_texts]
            table_content.append(row_content)
        
        return table_content
    
    def _group_text_by_rows(self, text_detections: List[Dict], 
                           row_threshold: int = 20) -> Dict[int, List[Dict]]:
        """
        Group text detections by rows based on y-coordinates.
        
        Args:
            text_detections: List of text detections
            row_threshold: Threshold for grouping into same row
            
        Returns:
            Dictionary mapping row y-coordinates to text lists
        """
        rows = {}
        
        for detection in text_detections:
            bbox = detection['bounding_box']
            y_center = (bbox[1] + bbox[3]) // 2
            
            # Find existing row or create new one
            assigned_row = None
            for row_y in rows.keys():
                if abs(y_center - row_y) <= row_threshold:
                    assigned_row = row_y
                    break
            
            if assigned_row is None:
                assigned_row = y_center
                rows[assigned_row] = []
            
            rows[assigned_row].append(detection)
        
        return rows
    
    def _parse_lighting_schedule(self, table_content: List[List[str]]) -> Dict:
        """
        Parse lighting schedule table content into structured data.
        
        Args:
            table_content: 2D list of table content
            
        Returns:
            Structured lighting schedule
        """
        if not table_content:
            return {}
        
        # Try to identify header row
        headers = []
        data_rows = []
        
        for i, row in enumerate(table_content):
            if i == 0 or any(keyword in ' '.join(row).lower() 
                           for keyword in ['symbol', 'description', 'type', 'fixture']):
                headers = [cell.lower().strip() for cell in row]
            else:
                if len(row) >= len(headers):
                    data_rows.append(row[:len(headers)])
        
        # Map standard column names
        column_mapping = {
            'symbol': ['symbol', 'sym', 'mark'],
            'description': ['description', 'desc', 'type'],
            'mount': ['mount', 'mounting', 'installation'],
            'voltage': ['voltage', 'volt', 'v'],
            'lumens': ['lumens', 'lumen', 'lm'],
            'watts': ['watts', 'watt', 'w', 'power']
        }
        
        # Map headers to standard names
        header_map = {}
        for i, header in enumerate(headers):
            for standard_name, variants in column_mapping.items():
                if any(variant in header for variant in variants):
                    header_map[i] = standard_name
                    break
        
        # Build structured data
        schedule = {}
        for row in data_rows:
            if len(row) > 0 and row[0].strip():  # Valid row with symbol
                symbol = row[0].strip()
                fixture_data = {'symbol': symbol}
                
                for col_idx, value in enumerate(row[1:], 1):
                    if col_idx in header_map:
                        fixture_data[header_map[col_idx]] = value.strip()
                
                schedule[symbol] = fixture_data
        
        return schedule
    
    def _parse_legend_content(self, table_content: List[List[str]]) -> Dict:
        """
        Parse legend table content into symbol mappings.
        
        Args:
            table_content: 2D list of legend content
            
        Returns:
            Dictionary mapping symbols to descriptions
        """
        legend = {}
        
        for row in table_content:
            if len(row) >= 2:
                symbol = row[0].strip()
                description = ' '.join(row[1:]).strip()
                
                if symbol and description:
                    legend[symbol] = description
        
        return legend
    
    def extract_general_notes(self, image: np.ndarray) -> List[Dict]:
        """
        Extract general notes from the drawing.
        
        Args:
            image: Input image
            
        Returns:
            List of general notes with metadata
        """
        # Look for notes sections
        text_detections = self.ocr_engine.extract_text_combined(image)
        
        notes_keywords = ['notes', 'general', 'requirements', 'specifications']
        notes = []
        
        for detection in text_detections:
            text = detection['text'].lower()
            
            # Check if this might be a note
            if (len(text) > 20 and  # Substantial text
                any(keyword in text for keyword in notes_keywords) or
                any(char in text for char in ['.', ';', ':']) and len(text.split()) > 3):
                
                notes.append({
                    'text': detection['text'],
                    'bounding_box': detection['bounding_box'],
                    'confidence': detection['confidence'],
                    'type': 'note'
                })
        
        return notes
