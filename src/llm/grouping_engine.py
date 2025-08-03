"""
LLM-powered grouping engine for emergency lighting classification.
Uses multiple LLM backends to intelligently group and classify lighting fixtures.
"""

import json
import logging
from typing import Dict, List, Optional
import asyncio

from .prompt_templates import PromptTemplates
from .llm_backends import LLMManager
from ..utils.config import Config

logger = logging.getLogger(__name__)


class GroupingEngine:
    """LLM-powered engine for grouping and classifying emergency lighting fixtures."""
    
    def __init__(self, config: Config):
        self.config = config
        self.prompt_templates = PromptTemplates()
        
        # Initialize LLM manager with multiple backends
        self.llm_manager = LLMManager(config)
        self.llm_available = len(self.llm_manager.get_available_backends()) > 0
        
        if self.llm_available:
            logger.info(f"LLM backends available: {self.llm_manager.get_available_backends()}")
            logger.info(f"Primary backend: {self.llm_manager.get_primary_backend()}")
        else:
            logger.warning("No LLM backends available - using fallback")
    
    async def group_lighting_fixtures(self, detections: List[Dict], 
                                    rulebook: Dict, 
                                    lighting_schedule: Dict) -> Dict:
        """
        Group lighting fixtures using LLM analysis.
        
        Args:
            detections: List of detected lighting fixtures
            rulebook: Extracted rules and notes
            lighting_schedule: Lighting schedule data
            
        Returns:
            Grouped lighting classification results
        """
        try:
            # Prepare context for LLM
            context = self._prepare_context(detections, rulebook, lighting_schedule)
            
            # Generate prompt
            prompt = self.prompt_templates.get_grouping_prompt(context)
            
            # Call LLM
            response, backend_used = await self.llm_manager.generate_response(prompt)
            logger.info(f"Used {backend_used} backend for grouping")
            
            # Parse and validate response
            grouped_results = self._parse_llm_response(response)
            
            # Post-process and validate
            validated_results = self._validate_grouping(grouped_results, detections)
            
            logger.info(f"Successfully grouped {len(detections)} fixtures into {len(validated_results)} categories")
            return validated_results
            
        except Exception as e:
            logger.error(f"Error in LLM grouping: {str(e)}")
            return self._fallback_grouping(detections)
    
    async def classify_fixture_types(self, detections: List[Dict], 
                                   lighting_schedule: Dict) -> List[Dict]:
        """
        Classify individual fixture types using LLM.
        
        Args:
            detections: List of detected fixtures
            lighting_schedule: Reference lighting schedule
            
        Returns:
            List of fixtures with enhanced classifications
        """
        try:
            classified_fixtures = []
            
            for detection in detections:
                # Prepare context for individual classification
                context = {
                    'fixture': detection,
                    'schedule': lighting_schedule,
                    'nearby_fixtures': self._find_nearby_fixtures(detection, detections)
                }
                
                # Generate classification prompt
                prompt = self.prompt_templates.get_classification_prompt(context)
                
                # Get LLM classification
                response, _ = await self.llm_manager.generate_response(prompt)
                classification = self._parse_classification_response(response)
                
                # Enhance detection with classification
                enhanced_detection = detection.copy()
                enhanced_detection.update(classification)
                classified_fixtures.append(enhanced_detection)
            
            return classified_fixtures
            
        except Exception as e:
            logger.error(f"Error in fixture classification: {str(e)}")
            return detections  # Return original if classification fails
    
    async def generate_summary(self, grouped_results: Dict, 
                              rulebook: Dict) -> Dict:
        """
        Generate intelligent summary of lighting analysis.
        
        Args:
            grouped_results: Grouped lighting results
            rulebook: Extracted rules and requirements
            
        Returns:
            Comprehensive summary with insights
        """
        try:
            context = {
                'groups': grouped_results,
                'rules': rulebook,
                'total_fixtures': sum(group.get('count', 0) for group in grouped_results.values())
            }
            
            prompt = self.prompt_templates.get_summary_prompt(context)
            response, _ = await self.llm_manager.generate_response(prompt)
            
            summary = self._parse_summary_response(response)
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return self._generate_basic_summary(grouped_results)
    
    def _prepare_context(self, detections: List[Dict], 
                        rulebook: Dict, 
                        lighting_schedule: Dict) -> Dict:
        """
        Prepare structured context for LLM processing.
        
        Args:
            detections: Detection results
            rulebook: Rules and notes
            lighting_schedule: Schedule data
            
        Returns:
            Structured context dictionary
        """
        # Extract symbols and their frequencies
        symbol_counts = {}
        for detection in detections:
            symbol = detection.get('symbol', 'Unknown')
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        # Extract unique text patterns
        text_patterns = set()
        for detection in detections:
            nearby_text = detection.get('text_nearby', [])
            text_patterns.update(nearby_text)
        
        context = {
            'detected_symbols': symbol_counts,
            'text_patterns': list(text_patterns),
            'lighting_schedule': lighting_schedule,
            'general_notes': rulebook.get('notes', []),
            'total_detections': len(detections),
            'confidence_scores': [d.get('confidence', 0) for d in detections]
        }
        
        return context
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Make API call to language model using the best available backend.
        
        Args:
            prompt: Input prompt for LLM
            
        Returns:
            LLM response text
        """
        try:
            response, backend_used = await self.llm_manager.generate_response(prompt, max_tokens=1000)
            logger.info(f"Successfully used {backend_used} backend")
            return response
            
        except Exception as e:
            logger.error(f"LLM API call failed: {str(e)}")
            raise
    
    def _parse_llm_response(self, response: str) -> Dict:
        """
        Parse LLM response into structured format.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed grouping results
        """
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback parsing for non-JSON responses
                return self._parse_text_response(response)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {str(e)}")
            return self._parse_text_response(response)
    
    def _parse_text_response(self, response: str) -> Dict:
        """
        Parse text-based LLM response.
        
        Args:
            response: Text response
            
        Returns:
            Parsed results
        """
        # Simple pattern matching for grouping results
        groups = {}
        lines = response.split('\n')
        
        current_group = None
        for line in lines:
            line = line.strip()
            
            # Look for group headers
            if ':' in line and any(word in line.lower() for word in ['light', 'fixture', 'emergency']):
                parts = line.split(':')
                if len(parts) >= 2:
                    group_name = parts[0].strip()
                    description = parts[1].strip()
                    
                    # Extract count if present
                    count_match = None
                    for part in parts:
                        if any(char.isdigit() for char in part):
                            count_match = ''.join(filter(str.isdigit, part))
                            break
                    
                    groups[group_name] = {
                        'count': int(count_match) if count_match else 1,
                        'description': description
                    }
        
        return groups
    
    def _parse_classification_response(self, response: str) -> Dict:
        """
        Parse individual fixture classification response.
        
        Args:
            response: LLM response for classification
            
        Returns:
            Classification results
        """
        classification = {
            'fixture_type': 'unknown',
            'emergency_type': 'unknown',
            'mount_type': 'unknown',
            'confidence': 0.5
        }
        
        response_lower = response.lower()
        
        # Extract fixture type
        if 'recessed' in response_lower:
            classification['fixture_type'] = 'recessed'
        elif 'surface' in response_lower:
            classification['fixture_type'] = 'surface'
        elif 'pendant' in response_lower:
            classification['fixture_type'] = 'pendant'
        elif 'wall' in response_lower:
            classification['fixture_type'] = 'wall-mounted'
        
        # Extract emergency type
        if 'exit' in response_lower:
            classification['emergency_type'] = 'exit'
        elif 'emergency' in response_lower:
            classification['emergency_type'] = 'emergency'
        elif 'combo' in response_lower:
            classification['emergency_type'] = 'exit_emergency_combo'
        
        return classification
    
    def _parse_summary_response(self, response: str) -> Dict:
        """
        Parse summary generation response.
        
        Args:
            response: LLM summary response
            
        Returns:
            Structured summary
        """
        try:
            # Try JSON parsing first
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback to text parsing
        return {
            'overview': response[:200] + '...' if len(response) > 200 else response,
            'key_findings': [],
            'recommendations': []
        }
    
    def _validate_grouping(self, grouped_results: Dict, 
                          original_detections: List[Dict]) -> Dict:
        """
        Validate and correct LLM grouping results.
        
        Args:
            grouped_results: LLM grouping results
            original_detections: Original detection data
            
        Returns:
            Validated grouping results
        """
        validated = {}
        total_detected = len(original_detections)
        total_grouped = sum(group.get('count', 0) for group in grouped_results.values())
        
        # If counts don't match, redistribute
        if abs(total_detected - total_grouped) > total_detected * 0.2:  # 20% tolerance
            logger.warning(f"Count mismatch: detected {total_detected}, grouped {total_grouped}")
            return self._redistribute_counts(grouped_results, total_detected)
        
        # Clean up group names and descriptions
        for group_name, group_data in grouped_results.items():
            clean_name = self._clean_group_name(group_name)
            validated[clean_name] = {
                'count': max(0, group_data.get('count', 0)),
                'description': group_data.get('description', '').strip()
            }
        
        return validated
    
    def _clean_group_name(self, name: str) -> str:
        """
        Clean and standardize group names.
        
        Args:
            name: Original group name
            
        Returns:
            Cleaned group name
        """
        # Remove common prefixes/suffixes
        cleaned = name.strip()
        prefixes = ['group', 'type', 'category', 'lights']
        
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Capitalize properly
        return cleaned.title()
    
    def _redistribute_counts(self, groups: Dict, total_count: int) -> Dict:
        """
        Redistribute counts to match total detections.
        
        Args:
            groups: Original groups
            total_count: Target total count
            
        Returns:
            Redistributed groups
        """
        if not groups:
            return {}
        
        # Calculate scaling factor
        current_total = sum(group.get('count', 0) for group in groups.values())
        if current_total == 0:
            # Distribute evenly
            count_per_group = total_count // len(groups)
            remainder = total_count % len(groups)
            
            redistributed = {}
            for i, (name, data) in enumerate(groups.items()):
                count = count_per_group + (1 if i < remainder else 0)
                redistributed[name] = {
                    'count': count,
                    'description': data.get('description', '')
                }
            return redistributed
        
        # Scale proportionally
        scale_factor = total_count / current_total
        redistributed = {}
        
        for name, data in groups.items():
            scaled_count = int(data.get('count', 0) * scale_factor)
            redistributed[name] = {
                'count': scaled_count,
                'description': data.get('description', '')
            }
        
        return redistributed
    
    def _find_nearby_fixtures(self, target_fixture: Dict, 
                             all_fixtures: List[Dict]) -> List[Dict]:
        """
        Find fixtures near the target fixture.
        
        Args:
            target_fixture: Target fixture to find neighbors for
            all_fixtures: List of all fixtures
            
        Returns:
            List of nearby fixtures
        """
        nearby = []
        target_bbox = target_fixture.get('bounding_box', [])
        
        if len(target_bbox) != 4:
            return nearby
        
        target_center = [(target_bbox[0] + target_bbox[2]) / 2, 
                        (target_bbox[1] + target_bbox[3]) / 2]
        
        for fixture in all_fixtures:
            if fixture == target_fixture:
                continue
            
            fixture_bbox = fixture.get('bounding_box', [])
            if len(fixture_bbox) != 4:
                continue
            
            fixture_center = [(fixture_bbox[0] + fixture_bbox[2]) / 2,
                             (fixture_bbox[1] + fixture_bbox[3]) / 2]
            
            # Calculate distance
            distance = ((target_center[0] - fixture_center[0])**2 + 
                       (target_center[1] - fixture_center[1])**2)**0.5
            
            if distance <= 100:  # Within 100 pixels
                nearby.append(fixture)
        
        return nearby[:5]  # Return top 5 nearest
    
    def _fallback_grouping(self, detections: List[Dict]) -> Dict:
        """
        Fallback grouping when LLM fails.
        
        Args:
            detections: Original detections
            
        Returns:
            Basic grouping results
        """
        # Simple symbol-based grouping
        groups = {}
        
        for detection in detections:
            symbol = detection.get('symbol', 'Unknown')
            
            if symbol not in groups:
                groups[symbol] = {
                    'count': 0,
                    'description': f'Emergency lighting fixture type {symbol}'
                }
            
            groups[symbol]['count'] += 1
        
        return groups
    
    def _generate_basic_summary(self, grouped_results: Dict) -> Dict:
        """
        Generate basic summary when LLM summary fails.
        
        Args:
            grouped_results: Grouped results
            
        Returns:
            Basic summary
        """
        total_fixtures = sum(group.get('count', 0) for group in grouped_results.values())
        
        return {
            'overview': f'Detected {total_fixtures} emergency lighting fixtures across {len(grouped_results)} categories.',
            'total_fixtures': total_fixtures,
            'fixture_types': len(grouped_results),
            'groups': grouped_results
        }
