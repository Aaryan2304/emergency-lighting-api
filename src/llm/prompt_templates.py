"""
Prompt templates for LLM interactions in emergency lighting analysis.
"""

from typing import Dict, List
import json


class PromptTemplates:
    """Collection of prompt templates for LLM-powered analysis."""
    
    def get_grouping_prompt(self, context: Dict) -> str:
        """
        Generate prompt for lighting fixture grouping.
        
        Args:
            context: Analysis context with detections, schedule, and rules
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are an expert electrical engineer analyzing emergency lighting fixtures from construction blueprints.

DETECTED FIXTURES:
{self._format_detected_symbols(context.get('detected_symbols', {}))}

LIGHTING SCHEDULE REFERENCE:
{self._format_lighting_schedule(context.get('lighting_schedule', {}))}

NEARBY TEXT PATTERNS:
{', '.join(context.get('text_patterns', []))}

GENERAL NOTES:
{self._format_general_notes(context.get('general_notes', []))}

TASK: Group the detected emergency lighting fixtures into logical categories based on:
1. Symbol patterns (A1E, A2E, etc.)
2. Fixture descriptions from the schedule
3. Text patterns found near fixtures
4. Engineering best practices

Provide a JSON response with this structure:
{{
  "Lights01": {{
    "count": 12,
    "description": "2x4 LED Emergency Fixture"
  }},
  "Lights02": {{
    "count": 5,
    "description": "Exit/Emergency Combo Unit"
  }}
}}

Requirements:
- Total count should equal {context.get('total_detections', 0)} detected fixtures
- Use clear, descriptive names for fixture types
- Group similar fixtures together
- Consider mounting types (recessed, surface, wall-mounted)
- Distinguish between emergency-only and exit/emergency combo units

Response:"""
        
        return prompt
    
    def get_classification_prompt(self, context: Dict) -> str:
        """
        Generate prompt for individual fixture classification.
        
        Args:
            context: Context with fixture and schedule data
            
        Returns:
            Formatted prompt string
        """
        fixture = context.get('fixture', {})
        schedule = context.get('schedule', {})
        nearby = context.get('nearby_fixtures', [])
        
        prompt = f"""
You are analyzing a single emergency lighting fixture from an electrical blueprint.

FIXTURE DETAILS:
- Symbol: {fixture.get('symbol', 'Unknown')}
- Nearby Text: {', '.join(fixture.get('text_nearby', []))}
- Confidence: {fixture.get('confidence', 0.0):.2f}

REFERENCE SCHEDULE:
{self._format_lighting_schedule(schedule)}

NEARBY FIXTURES:
{self._format_nearby_fixtures(nearby)}

TASK: Classify this fixture and provide details in this format:

{{
  "fixture_type": "recessed|surface|pendant|wall-mounted",
  "emergency_type": "emergency|exit|exit_emergency_combo",
  "mount_type": "ceiling|wall|surface",
  "description": "Detailed description of the fixture",
  "confidence": 0.85
}}

Consider:
- Symbol patterns (E = Emergency, combinations indicate combo units)
- Mounting implications from nearby text
- Standard electrical drawing conventions
- Consistency with lighting schedule

Response:"""
        
        return prompt
    
    def get_summary_prompt(self, context: Dict) -> str:
        """
        Generate prompt for creating analysis summary.
        
        Args:
            context: Context with grouped results and rules
            
        Returns:
            Formatted prompt string
        """
        groups = context.get('groups', {})
        rules = context.get('rules', {})
        total = context.get('total_fixtures', 0)
        
        prompt = f"""
You are creating a comprehensive summary of emergency lighting analysis for a construction project.

DETECTED FIXTURE GROUPS:
{self._format_fixture_groups(groups)}

TOTAL FIXTURES: {total}

PROJECT RULES & REQUIREMENTS:
{self._format_general_notes(rules.get('notes', []))}

TASK: Create a professional summary with insights and recommendations.

Provide a JSON response with this structure:
{{
  "overview": "Brief overview of findings",
  "total_fixtures": {total},
  "fixture_breakdown": {{
    "emergency_only": 0,
    "exit_only": 0,
    "combination_units": 0
  }},
  "key_findings": [
    "List of important observations"
  ],
  "compliance_notes": [
    "Notes about code compliance"
  ],
  "recommendations": [
    "Suggestions for improvement"
  ]
}}

Focus on:
- Emergency lighting coverage adequacy
- Code compliance observations
- Fixture distribution patterns
- Potential issues or improvements

Response:"""
        
        return prompt
    
    def get_validation_prompt(self, detection_data: Dict, 
                             schedule_data: Dict) -> str:
        """
        Generate prompt for validating detection results.
        
        Args:
            detection_data: Detection results
            schedule_data: Lighting schedule data
            
        Returns:
            Validation prompt
        """
        prompt = f"""
You are validating emergency lighting detection results against the project schedule.

DETECTION RESULTS:
{json.dumps(detection_data, indent=2)}

OFFICIAL LIGHTING SCHEDULE:
{json.dumps(schedule_data, indent=2)}

TASK: Validate the detection results and identify:
1. Missing fixtures that should be present
2. Extra detections that don't match the schedule
3. Symbol mismatches or errors
4. Count discrepancies

Provide validation feedback in JSON format:
{{
  "validation_status": "PASS|FAIL|WARNING",
  "confidence_score": 0.85,
  "issues_found": [
    "List of validation issues"
  ],
  "suggestions": [
    "Recommended corrections"
  ],
  "summary": "Overall validation summary"
}}

Response:"""
        
        return prompt
    
    def _format_detected_symbols(self, symbols: Dict) -> str:
        """Format detected symbols for prompt inclusion."""
        if not symbols:
            return "No symbols detected"
        
        formatted = []
        for symbol, count in symbols.items():
            formatted.append(f"- {symbol}: {count} instances")
        
        return '\n'.join(formatted)
    
    def _format_lighting_schedule(self, schedule: Dict) -> str:
        """Format lighting schedule for prompt inclusion."""
        if not schedule:
            return "No lighting schedule available"
        
        formatted = []
        for symbol, data in schedule.items():
            description = data.get('description', 'Unknown fixture')
            mount = data.get('mount', 'Unknown mounting')
            voltage = data.get('voltage', 'Unknown voltage')
            
            formatted.append(f"- {symbol}: {description} ({mount}, {voltage})")
        
        return '\n'.join(formatted)
    
    def _format_general_notes(self, notes: List) -> str:
        """Format general notes for prompt inclusion."""
        if not notes:
            return "No general notes available"
        
        if isinstance(notes, list):
            return '\n'.join(f"- {note}" for note in notes)
        else:
            return str(notes)
    
    def _format_nearby_fixtures(self, fixtures: List[Dict]) -> str:
        """Format nearby fixtures for prompt inclusion."""
        if not fixtures:
            return "No nearby fixtures"
        
        formatted = []
        for fixture in fixtures[:3]:  # Limit to 3 nearest
            symbol = fixture.get('symbol', 'Unknown')
            nearby_text = ', '.join(fixture.get('text_nearby', []))
            formatted.append(f"- {symbol} (text: {nearby_text})")
        
        return '\n'.join(formatted)
    
    def _format_fixture_groups(self, groups: Dict) -> str:
        """Format fixture groups for prompt inclusion."""
        if not groups:
            return "No fixture groups defined"
        
        formatted = []
        for group_name, group_data in groups.items():
            count = group_data.get('count', 0)
            description = group_data.get('description', 'No description')
            formatted.append(f"- {group_name}: {count} fixtures ({description})")
        
        return '\n'.join(formatted)
    
    def get_error_analysis_prompt(self, errors: List[str], 
                                 context: Dict) -> str:
        """
        Generate prompt for analyzing detection errors.
        
        Args:
            errors: List of error descriptions
            context: Analysis context
            
        Returns:
            Error analysis prompt
        """
        prompt = f"""
You are analyzing errors in emergency lighting detection from construction blueprints.

DETECTED ERRORS:
{chr(10).join(f"- {error}" for error in errors)}

CONTEXT:
- Total attempted detections: {context.get('total_detections', 0)}
- Processing confidence: {context.get('avg_confidence', 0.0):.2f}
- Image quality indicators: {context.get('quality_metrics', {})}

TASK: Analyze the errors and provide recommendations for improvement.

Respond with:
{{
  "error_categories": [
    "List of error types identified"
  ],
  "root_causes": [
    "Likely causes of the errors"
  ],
  "improvement_suggestions": [
    "Specific recommendations"
  ],
  "confidence_impact": "How errors affect overall confidence"
}}

Response:"""
        
        return prompt
