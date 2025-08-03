"""
Text processing utilities for cleaning and enhancing OCR results.
"""

import re
import string
from typing import List, Dict, Set
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Utilities for processing and cleaning OCR text results."""
    
    def __init__(self):
        # Common electrical drawing symbols and abbreviations
        self.electrical_symbols = {
            'A1E', 'A2E', 'B1E', 'EM', 'EXIT', 'LED', 'W', 'WP',
            'REC', 'SURF', 'PEND', 'WALL', 'CEIL', 'FLOOR'
        }
        
        # Common words in electrical drawings
        self.electrical_terms = {
            'emergency', 'lighting', 'fixture', 'luminaire', 'exit',
            'recessed', 'surface', 'pendant', 'wall', 'ceiling',
            'mounted', 'voltage', 'lumens', 'watts', 'ballast'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text from OCR.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common OCR errors
        cleaned = self._fix_common_ocr_errors(cleaned)
        
        # Remove unwanted characters
        cleaned = self._remove_unwanted_chars(cleaned)
        
        return cleaned
    
    def _fix_common_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR misreads in electrical drawings.
        
        Args:
            text: Input text
            
        Returns:
            Corrected text
        """
        # Common OCR corrections for electrical symbols
        corrections = {
            '0': 'O',  # Zero to O
            'l': '1',  # lowercase l to 1
            'I': '1',  # uppercase I to 1
            'S': '5',  # S to 5 in certain contexts
            'G': '6',  # G to 6
            'B': '8',  # B to 8
        }
        
        # Apply corrections in specific contexts
        corrected = text
        
        # Fix symbol patterns like A1E, B2E
        symbol_pattern = r'([A-Z])([0lI])([A-Z]?)'
        def fix_symbol(match):
            letter1, middle, letter2 = match.groups()
            if middle in ['0', 'l', 'I']:
                middle = '1'
            return letter1 + middle + letter2
        
        corrected = re.sub(symbol_pattern, fix_symbol, corrected)
        
        return corrected
    
    def _remove_unwanted_chars(self, text: str) -> str:
        """
        Remove unwanted characters while preserving meaningful content.
        
        Args:
            text: Input text
            
        Returns:
            Filtered text
        """
        # Keep alphanumeric, spaces, and common punctuation
        allowed_chars = string.ascii_letters + string.digits + ' .-_()/'
        return ''.join(char for char in text if char in allowed_chars)
    
    def extract_symbols_from_text(self, text_list: List[str]) -> List[str]:
        """
        Extract lighting symbols from text list.
        
        Args:
            text_list: List of text strings
            
        Returns:
            List of extracted symbols
        """
        symbols = []
        symbol_patterns = [
            r'^[A-Z]\d+[A-Z]?$',  # A1E, B2, etc.
            r'^EM$',              # Emergency
            r'^EXIT$',            # Exit
        ]
        
        for text in text_list:
            cleaned = self.clean_text(text).upper()
            
            # Check against known symbols
            if cleaned in self.electrical_symbols:
                symbols.append(cleaned)
                continue
            
            # Check against patterns
            for pattern in symbol_patterns:
                if re.match(pattern, cleaned):
                    symbols.append(cleaned)
                    break
        
        return list(set(symbols))  # Remove duplicates
    
    def categorize_text(self, text: str) -> str:
        """
        Categorize text as symbol, description, or other.
        
        Args:
            text: Input text
            
        Returns:
            Category string
        """
        cleaned = self.clean_text(text).upper()
        
        # Check if it's a symbol
        if (len(cleaned) <= 5 and 
            (cleaned in self.electrical_symbols or
             re.match(r'^[A-Z]\d+[A-Z]?$', cleaned))):
            return 'symbol'
        
        # Check if it contains electrical terms
        if any(term in cleaned.lower() for term in self.electrical_terms):
            return 'description'
        
        # Check if it's numeric (voltage, lumens, etc.)
        if re.match(r'^\d+[VWL]?$', cleaned):
            return 'specification'
        
        return 'other'
    
    def group_related_text(self, text_detections: List[Dict], 
                          distance_threshold: float = 50) -> List[List[Dict]]:
        """
        Group text detections that are spatially related.
        
        Args:
            text_detections: List of text detection dictionaries
            distance_threshold: Maximum distance for grouping
            
        Returns:
            List of text groups
        """
        if not text_detections:
            return []
        
        groups = []
        used = [False] * len(text_detections)
        
        for i, detection in enumerate(text_detections):
            if used[i]:
                continue
            
            # Start new group
            group = [detection]
            used[i] = True
            
            # Find nearby text
            bbox1 = detection['bounding_box']
            center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
            
            for j, other_detection in enumerate(text_detections):
                if i == j or used[j]:
                    continue
                
                bbox2 = other_detection['bounding_box']
                center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
                
                # Calculate distance
                distance = ((center1[0] - center2[0])**2 + 
                           (center1[1] - center2[1])**2)**0.5
                
                if distance <= distance_threshold:
                    group.append(other_detection)
                    used[j] = True
            
            groups.append(group)
        
        return groups
    
    def extract_specifications(self, text_list: List[str]) -> Dict[str, str]:
        """
        Extract specifications like voltage, lumens, watts from text.
        
        Args:
            text_list: List of text strings
            
        Returns:
            Dictionary of specifications
        """
        specs = {}
        
        for text in text_list:
            cleaned = self.clean_text(text).upper()
            
            # Voltage patterns
            voltage_match = re.search(r'(\d+)\s*V(?:OLT)?S?', cleaned)
            if voltage_match:
                specs['voltage'] = voltage_match.group(1) + 'V'
            
            # Lumens patterns
            lumens_match = re.search(r'(\d+)\s*L(?:UMEN)?S?|(\d+)\s*LM', cleaned)
            if lumens_match:
                value = lumens_match.group(1) or lumens_match.group(2)
                specs['lumens'] = value + 'lm'
            
            # Watts patterns
            watts_match = re.search(r'(\d+)\s*W(?:ATT)?S?', cleaned)
            if watts_match:
                specs['watts'] = watts_match.group(1) + 'W'
            
            # Dimensions patterns
            dimension_match = re.search(r"(\d+)['\"]?\s*[Xx×]\s*(\d+)['\"]?", cleaned)
            if dimension_match:
                specs['dimensions'] = f"{dimension_match.group(1)}x{dimension_match.group(2)}"
        
        return specs
    
    def validate_text_quality(self, text: str, min_confidence: float = 0.5) -> bool:
        """
        Validate if text quality is sufficient for processing.
        
        Args:
            text: Input text
            min_confidence: Minimum confidence threshold
            
        Returns:
            True if text quality is acceptable
        """
        if not text or len(text.strip()) < 2:
            return False
        
        # Check for excessive special characters (OCR noise)
        special_char_ratio = sum(1 for char in text if not char.isalnum() and char != ' ') / len(text)
        if special_char_ratio > 0.5:
            return False
        
        # Check for reasonable character distribution
        alpha_ratio = sum(1 for char in text if char.isalpha()) / len(text)
        if alpha_ratio < 0.3:  # Too few letters
            return False
        
        return True
    
    def standardize_symbols(self, symbol: str) -> str:
        """
        Standardize symbol format for consistency.
        
        Args:
            symbol: Input symbol
            
        Returns:
            Standardized symbol
        """
        cleaned = self.clean_text(symbol).upper()
        
        # Standard emergency lighting symbols
        if cleaned in ['EM', 'EMERGENCY']:
            return 'EM'
        elif cleaned in ['EXIT', 'EX']:
            return 'EXIT'
        elif re.match(r'^[A-Z]\d+E$', cleaned):
            return cleaned  # Already standard format
        elif re.match(r'^[A-Z]\d+$', cleaned):
            return cleaned + 'E'  # Add emergency indicator
        
        return cleaned
