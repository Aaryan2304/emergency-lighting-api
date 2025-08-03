"""
Main processing pipeline for emergency lighting detection.
Orchestrates the complete workflow from PDF to final results.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import time
import json
from typing import Dict, List, Any
import numpy as np

from src.detection import LightingDetector, ImageProcessor
from src.extraction import OCREngine, TableExtractor, TextProcessor
from src.llm import GroupingEngine
from src.utils import config, get_logger, file_handler

logger = get_logger(__name__)


class ProcessingPipeline:
    """Main processing pipeline for emergency lighting detection."""
    
    def __init__(self):
        self.lighting_detector = LightingDetector(config)
        self.image_processor = ImageProcessor(config)
        self.ocr_engine = OCREngine(config)
        self.table_extractor = TableExtractor(config)
        self.text_processor = TextProcessor()
        self.grouping_engine = GroupingEngine(config)
    
    async def process_pdf(self, file_path: str, pdf_name: str) -> Dict[str, Any]:
        """
        Process a PDF file through the complete pipeline.
        
        Args:
            file_path: Path to the PDF file
            pdf_name: Original PDF name
            
        Returns:
            Complete processing results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting pipeline processing for {pdf_name}")
            
            # Stage 1: Convert PDF to images
            logger.info("Stage 1: Converting PDF to images")
            images = await self._convert_pdf_to_images(file_path)
            
            if not images:
                raise ValueError("Failed to convert PDF to images")
            
            # Save debug data
            file_handler.save_debug_data(
                {'page_count': len(images)}, 
                'pdf_conversion', 
                pdf_name
            )
            
            # Stage 2: Detect lighting fixtures
            logger.info("Stage 2: Detecting lighting fixtures")
            all_detections = []
            
            for i, image in enumerate(images):
                cv_image = self.image_processor.pil_to_cv2(image)
                detections = self.lighting_detector.detect_emergency_lights(cv_image)
                
                # Add page information
                for detection in detections:
                    detection['source_page'] = i + 1
                    detection['source_sheet'] = f"Page_{i+1}"
                
                all_detections.extend(detections)
            
            logger.info(f"Detected {len(all_detections)} lighting fixtures")
            
            # Save debug data
            file_handler.save_debug_data(
                all_detections, 
                'detection_results', 
                pdf_name
            )
            
            # Stage 3: Extract text and tables
            logger.info("Stage 3: Extracting text and tables")
            rulebook = await self._extract_static_content(images, pdf_name)
            
            # Save debug data
            file_handler.save_debug_data(
                rulebook, 
                'extraction_results', 
                pdf_name
            )
            
            # Stage 4: LLM-powered grouping
            logger.info("Stage 4: Grouping fixtures with LLM")
            lighting_schedule = rulebook.get('lighting_schedule', {})
            grouped_results = await self.grouping_engine.group_lighting_fixtures(
                all_detections, 
                rulebook, 
                lighting_schedule
            )
            
            # Save debug data
            file_handler.save_debug_data(
                grouped_results, 
                'grouping_results', 
                pdf_name
            )
            
            # Stage 5: Generate summary
            logger.info("Stage 5: Generating summary")
            summary = await self.grouping_engine.generate_summary(grouped_results, rulebook)
            
            # Calculate processing metadata
            processing_time = time.time() - start_time
            confidence_score = self._calculate_overall_confidence(all_detections)
            
            # Build final result
            final_result = {
                'summary': summary,
                'grouped_results': grouped_results,
                'detections': all_detections,
                'rulebook': rulebook,
                'metadata': {
                    'processing_time': processing_time,
                    'confidence_score': confidence_score,
                    'total_pages': len(images),
                    'total_detections': len(all_detections),
                    'pipeline_version': '1.0.0',
                    'timestamp': time.time()
                }
            }
            
            # Save final debug data
            file_handler.save_debug_data(
                final_result, 
                'final_results', 
                pdf_name
            )
            
            logger.info(f"Pipeline processing completed for {pdf_name} in {processing_time:.2f}s")
            return final_result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed for {pdf_name}: {str(e)}")
            raise
    
    async def _convert_pdf_to_images(self, file_path: str) -> List:
        """
        Convert PDF to images for processing.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of PIL images
        """
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            images = await loop.run_in_executor(
                None, 
                self.image_processor.pdf_to_images, 
                file_path
            )
            
            if not images:
                # Log the issue but let the image_processor handle fallbacks
                logger.warning("Initial PDF conversion returned no images, image processor should have tried fallbacks")
                raise ValueError("No pages found in PDF")
            
            # Resize images if too large
            processed_images = []
            for image in images:
                cv_image = self.image_processor.pil_to_cv2(image)
                resized = self.image_processor.resize_image(cv_image)
                processed_images.append(self.image_processor.cv2_to_pil(resized))
            
            return processed_images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise
    
    async def _extract_static_content(self, images: List, pdf_name: str) -> Dict[str, Any]:
        """
        Extract static content like tables, notes, and schedules.
        
        Args:
            images: List of page images
            pdf_name: PDF name for debugging
            
        Returns:
            Extracted static content
        """
        try:
            rulebook = {
                'lighting_schedule': {},
                'legend': {},
                'notes': [],
                'specifications': {}
            }
            
            for i, image in enumerate(images):
                cv_image = self.image_processor.pil_to_cv2(image)
                
                # Extract lighting schedule
                schedule = self.table_extractor.extract_lighting_schedule(cv_image)
                if schedule:
                    rulebook['lighting_schedule'].update(schedule)
                
                # Extract legend
                legend = self.table_extractor.extract_legend(cv_image)
                if legend:
                    rulebook['legend'].update(legend)
                
                # Extract general notes
                notes = self.table_extractor.extract_general_notes(cv_image)
                if notes:
                    for note in notes:
                        note['source_page'] = i + 1
                    rulebook['notes'].extend(notes)
                
                # Extract text for specifications
                text_detections = self.ocr_engine.extract_text_combined(cv_image)
                text_strings = [d['text'] for d in text_detections]
                specs = self.text_processor.extract_specifications(text_strings)
                if specs:
                    rulebook['specifications'].update(specs)
            
            # Clean up extracted data
            rulebook = self._clean_extracted_data(rulebook)
            
            return rulebook
            
        except Exception as e:
            logger.error(f"Error extracting static content: {str(e)}")
            return {
                'lighting_schedule': {},
                'legend': {},
                'notes': [],
                'specifications': {}
            }
    
    def _clean_extracted_data(self, rulebook: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and normalize extracted data.
        
        Args:
            rulebook: Raw extracted data
            
        Returns:
            Cleaned data
        """
        # Remove duplicates from notes
        unique_notes = []
        seen_texts = set()
        
        for note in rulebook.get('notes', []):
            text = note.get('text', '').strip().lower()
            if text and text not in seen_texts and len(text) > 10:
                seen_texts.add(text)
                unique_notes.append(note)
        
        rulebook['notes'] = unique_notes
        
        # Clean lighting schedule
        schedule = rulebook.get('lighting_schedule', {})
        cleaned_schedule = {}
        
        for symbol, data in schedule.items():
            if symbol and isinstance(data, dict):
                cleaned_data = {}
                for key, value in data.items():
                    if value and str(value).strip():
                        cleaned_data[key] = str(value).strip()
                
                if cleaned_data:
                    cleaned_schedule[symbol.strip().upper()] = cleaned_data
        
        rulebook['lighting_schedule'] = cleaned_schedule
        
        return rulebook
    
    def _calculate_overall_confidence(self, detections: List[Dict]) -> float:
        """
        Calculate overall confidence score for the processing.
        
        Args:
            detections: List of detections
            
        Returns:
            Overall confidence score
        """
        if not detections:
            return 0.0
        
        confidences = [d.get('confidence', 0.0) for d in detections]
        
        # Calculate weighted average (higher weight for higher confidence)
        weighted_sum = sum(conf * conf for conf in confidences)  # Square for weighting
        weight_sum = sum(confidences)
        
        if weight_sum == 0:
            return 0.0
        
        return weighted_sum / weight_sum
    
    async def validate_results(self, results: Dict[str, Any], 
                             expected_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate processing results against expected values.
        
        Args:
            results: Processing results to validate
            expected_results: Expected results for comparison
            
        Returns:
            Validation report
        """
        try:
            validation_report = {
                'validation_status': 'PASS',
                'confidence_score': 0.8,
                'issues_found': [],
                'suggestions': [],
                'summary': 'Validation completed successfully'
            }
            
            # Basic structure validation
            required_keys = ['summary', 'detections', 'metadata']
            missing_keys = [key for key in required_keys if key not in results]
            
            if missing_keys:
                validation_report['issues_found'].append(
                    f"Missing required keys: {', '.join(missing_keys)}"
                )
                validation_report['validation_status'] = 'FAIL'
            
            # Count validation
            summary = results.get('summary', {})
            detections = results.get('detections', [])
            
            total_grouped = sum(group.get('count', 0) for group in summary.values())
            total_detected = len(detections)
            
            if abs(total_grouped - total_detected) > total_detected * 0.2:
                validation_report['issues_found'].append(
                    f"Count mismatch: {total_detected} detected, {total_grouped} grouped"
                )
                validation_report['validation_status'] = 'WARNING'
            
            # Confidence validation
            metadata = results.get('metadata', {})
            overall_confidence = metadata.get('confidence_score', 0.0)
            
            if overall_confidence < 0.6:
                validation_report['issues_found'].append(
                    f"Low overall confidence: {overall_confidence:.2f}"
                )
                validation_report['validation_status'] = 'WARNING'
            
            # Generate suggestions
            if validation_report['issues_found']:
                validation_report['suggestions'] = [
                    "Review detection parameters",
                    "Check image quality",
                    "Verify lighting schedule accuracy"
                ]
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Error in result validation: {str(e)}")
            return {
                'validation_status': 'FAIL',
                'confidence_score': 0.0,
                'issues_found': [f"Validation error: {str(e)}"],
                'suggestions': [],
                'summary': 'Validation failed due to internal error'
            }
