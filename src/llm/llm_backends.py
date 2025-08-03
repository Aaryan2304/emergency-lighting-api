"""
Multiple LLM backends for emergency lighting detection.
Supports OpenAI, Google Gemini, Ollama (local), and Hugging Face models.
"""

import json
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import requests
from pathlib import Path

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..utils import get_logger

logger = get_logger(__name__)


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    async def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response from the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available and configured."""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI GPT backend."""
    
    def __init__(self, config):
        super().__init__(config)
        if OPENAI_AVAILABLE and hasattr(config, 'OPENAI_API_KEY'):
            openai.api_key = config.OPENAI_API_KEY
            self.model = getattr(config, 'OPENAI_MODEL', 'gpt-3.5-turbo')
    
    async def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using OpenAI API."""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return (OPENAI_AVAILABLE and 
                hasattr(self.config, 'OPENAI_API_KEY') and 
                self.config.OPENAI_API_KEY)


class GeminiBackend(LLMBackend):
    """Google Gemini backend (free tier available)."""
    
    def __init__(self, config):
        super().__init__(config)
        if GEMINI_AVAILABLE and hasattr(config, 'GEMINI_API_KEY'):
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(
                getattr(config, 'GEMINI_MODEL', 'gemini-1.5-flash')
            )
    
    async def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using Gemini API."""
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.1
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return (GEMINI_AVAILABLE and 
                hasattr(self.config, 'GEMINI_API_KEY') and 
                self.config.GEMINI_API_KEY)


class OllamaBackend(LLMBackend):
    """Ollama local model backend (completely free)."""
    
    def __init__(self, config):
        super().__init__(config)
        self.base_url = getattr(config, 'OLLAMA_BASE_URL', 'http://localhost:11434')
        self.model = getattr(config, 'OLLAMA_MODEL', 'llama3.2:3b')
    
    async def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using Ollama local API."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.1
                }
            }
            
            response = await asyncio.to_thread(
                requests.post,
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()['response'].strip()
            
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


class HuggingFaceBackend(LLMBackend):
    """Hugging Face local model backend (completely free)."""
    
    def __init__(self, config):
        super().__init__(config)
        self.model_name = getattr(config, 'HF_MODEL', 'microsoft/DialoGPT-medium')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Initialize model if requested
        if getattr(config, 'LOAD_HF_MODEL', False):
            self._load_model()
    
    def _load_model(self):
        """Load the Hugging Face model."""
        try:
            logger.info(f"Loading Hugging Face model: {self.model_name}")
            
            # For text generation models
            if 'gpt' in self.model_name.lower() or 'llama' in self.model_name.lower():
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    device_map="auto" if self.device == 'cuda' else None
                )
            else:
                # Fallback for other models
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                ).to(self.device)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load HF model: {str(e)}")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
    
    async def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using Hugging Face model."""
        if not self.is_available():
            raise RuntimeError("Hugging Face model not loaded")
        
        try:
            if self.pipeline:
                # Use pipeline for generation
                response = await asyncio.to_thread(
                    self.pipeline,
                    prompt,
                    max_length=len(prompt.split()) + max_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )
                generated_text = response[0]['generated_text']
                # Extract only the new generated part
                return generated_text[len(prompt):].strip()
            
            else:
                # Use model + tokenizer
                inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                with torch.no_grad():
                    outputs = await asyncio.to_thread(
                        self.model.generate,
                        inputs,
                        max_length=inputs.shape[1] + max_tokens,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response[len(prompt):].strip()
                
        except Exception as e:
            logger.error(f"HuggingFace generation error: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Check if Hugging Face model is available."""
        return (TRANSFORMERS_AVAILABLE and 
                (self.model is not None or self.pipeline is not None))


class SimpleLLMBackend(LLMBackend):
    """Simple rule-based fallback when no LLM is available."""
    
    def __init__(self, config):
        super().__init__(config)
    
    async def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate simple rule-based response."""
        
        # Extract lighting symbols from prompt
        common_symbols = ['A1E', 'EM', 'EXIT', 'LED', 'A1', 'W', 'WP', 'R', 'E']
        detected_symbols = []
        
        for symbol in common_symbols:
            if symbol in prompt.upper():
                detected_symbols.append(symbol)
        
        # Create simple grouping based on common patterns
        groups = {}
        
        for symbol in detected_symbols:
            if 'EXIT' in symbol or 'A1E' in symbol:
                groups[symbol] = {
                    'count': prompt.upper().count(symbol),
                    'description': 'Emergency Exit Light',
                    'type': 'exit_emergency'
                }
            elif 'EM' in symbol:
                groups[symbol] = {
                    'count': prompt.upper().count(symbol),
                    'description': 'Emergency Light',
                    'type': 'emergency'
                }
            elif 'LED' in symbol:
                groups[symbol] = {
                    'count': prompt.upper().count(symbol),
                    'description': 'LED Emergency Fixture',
                    'type': 'led_emergency'
                }
            else:
                groups[symbol] = {
                    'count': prompt.upper().count(symbol),
                    'description': f'{symbol} Emergency Fixture',
                    'type': 'general_emergency'
                }
        
        # If no symbols found, create generic response
        if not groups:
            groups = {
                'EMERGENCY_LIGHTS': {
                    'count': 1,
                    'description': 'Emergency Lighting Fixtures',
                    'type': 'general'
                }
            }
        
        return json.dumps(groups, indent=2)
    
    def is_available(self) -> bool:
        """Simple backend is always available."""
        return True


class LLMManager:
    """Manages multiple LLM backends with automatic fallback."""
    
    def __init__(self, config):
        self.config = config
        self.backends = []
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize available backends in order of preference."""
        
        # Define backend priority (most preferred first)
        backend_classes = [
            ('gemini', GeminiBackend),      # Free tier available
            ('ollama', OllamaBackend),      # Completely free, local
            ('openai', OpenAIBackend),      # Paid but high quality
            ('huggingface', HuggingFaceBackend),  # Free, local, requires setup
            ('simple', SimpleLLMBackend)    # Always available fallback
        ]
        
        for name, backend_class in backend_classes:
            try:
                backend = backend_class(self.config)
                if backend.is_available():
                    self.backends.append((name, backend))
                    logger.info(f"Initialized {name} backend")
                else:
                    logger.debug(f"{name} backend not available")
            except Exception as e:
                logger.warning(f"Failed to initialize {name} backend: {str(e)}")
        
        if not self.backends:
            logger.warning("No LLM backends available, using simple fallback")
            self.backends.append(('simple', SimpleLLMBackend(self.config)))
    
    async def generate_response(self, prompt: str, max_tokens: int = 1000) -> tuple[str, str]:
        """Generate response using the first available backend."""
        
        for name, backend in self.backends:
            try:
                logger.info(f"Attempting to use {name} backend")
                response = await backend.generate_response(prompt, max_tokens)
                logger.info(f"Successfully generated response using {name}")
                return response, name
            except Exception as e:
                logger.warning(f"{name} backend failed: {str(e)}")
                continue
        
        raise RuntimeError("All LLM backends failed")
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backend names."""
        return [name for name, _ in self.backends]
    
    def get_primary_backend(self) -> str:
        """Get the name of the primary (first) backend."""
        return self.backends[0][0] if self.backends else "none"
    
    async def group_fixtures(self, detections: List[Dict]) -> Dict[str, Any]:
        """Group lighting fixtures using LLM analysis."""
        
        if not detections:
            return {}
        
        # Extract unique symbols and their contexts
        symbol_data = {}
        for detection in detections:
            symbol = detection.get('symbol', 'Unknown')
            if symbol not in symbol_data:
                symbol_data[symbol] = {
                    'count': 0,
                    'contexts': [],
                    'specifications': []
                }
            
            symbol_data[symbol]['count'] += 1
            if 'text_nearby' in detection:
                symbol_data[symbol]['contexts'].extend(detection['text_nearby'])
            if 'specifications' in detection:
                symbol_data[symbol]['specifications'].append(detection['specifications'])
        
        # Create prompt for LLM grouping
        prompt = """You are an electrical engineering expert analyzing emergency lighting fixtures from construction drawings.

Based on the following detected symbols and their contexts, please group and classify these emergency lighting fixtures.

Detected Symbols:
"""
        
        for symbol, data in symbol_data.items():
            contexts = list(set(data['contexts']))[:5]  # Unique contexts, max 5
            prompt += f"\n- {symbol}: {data['count']} fixtures"
            if contexts:
                prompt += f" (contexts: {', '.join(contexts)})"
        
        prompt += """

Please provide a JSON response with fixture groupings in this format:
{
  "fixture_type_1": {
    "count": 10,
    "description": "Emergency Exit Light with LED",
    "symbols": ["A1E", "EXIT"],
    "mount_type": "Wall",
    "function": "Emergency egress lighting"
  },
  "fixture_type_2": {
    "count": 5,
    "description": "Emergency Unit Light",
    "symbols": ["EM"],
    "mount_type": "Ceiling", 
    "function": "Emergency illumination"
  }
}

Focus on:
1. Grouping similar fixtures by function and type
2. Providing clear descriptions
3. Identifying mounting types (Wall/Ceiling/Recessed)
4. Distinguishing between exit lights, emergency lights, and LED fixtures

Return only valid JSON, no other text."""

        try:
            # Generate response using available backend
            response, backend_used = await self.generate_response(prompt, max_tokens=1500)
            logger.info(f"Used {backend_used} for fixture grouping")
            
            # Try to parse JSON response
            try:
                import re
                # Extract JSON from response (in case there's extra text)
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    grouped_fixtures = json.loads(json_str)
                    return grouped_fixtures
                else:
                    logger.warning("No JSON found in LLM response, using fallback grouping")
                    return self._fallback_grouping(symbol_data)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM JSON response: {e}, using fallback grouping")
                return self._fallback_grouping(symbol_data)
                
        except Exception as e:
            logger.error(f"LLM grouping failed: {e}, using fallback grouping")
            return self._fallback_grouping(symbol_data)
    
    def _fallback_grouping(self, symbol_data: Dict) -> Dict[str, Any]:
        """Fallback rule-based grouping when LLM fails."""
        
        grouped = {}
        
        # Simple rule-based grouping
        exit_symbols = ['A1E', 'EXIT', 'EGRESS']
        emergency_symbols = ['EM', 'EMERGENCY']
        led_symbols = ['LED']
        
        exit_count = sum(data['count'] for symbol, data in symbol_data.items() 
                        if any(exit_sym in symbol.upper() for exit_sym in exit_symbols))
        
        emergency_count = sum(data['count'] for symbol, data in symbol_data.items() 
                            if any(em_sym in symbol.upper() for em_sym in emergency_symbols))
        
        led_count = sum(data['count'] for symbol, data in symbol_data.items() 
                       if any(led_sym in symbol.upper() for led_sym in led_symbols))
        
        if exit_count > 0:
            grouped['Emergency_Exit_Lights'] = {
                'count': exit_count,
                'description': 'Emergency Exit Lighting Fixtures',
                'mount_type': 'Wall',
                'function': 'Emergency egress lighting'
            }
        
        if emergency_count > 0:
            grouped['Emergency_Lights'] = {
                'count': emergency_count, 
                'description': 'Emergency Illumination Units',
                'mount_type': 'Ceiling',
                'function': 'Emergency illumination'
            }
        
        if led_count > 0:
            grouped['LED_Fixtures'] = {
                'count': led_count,
                'description': 'LED Emergency Fixtures', 
                'mount_type': 'Recessed',
                'function': 'Energy efficient emergency lighting'
            }
        
        # Group remaining fixtures
        other_count = sum(data['count'] for symbol, data in symbol_data.items()) - exit_count - emergency_count - led_count
        if other_count > 0:
            grouped['Other_Emergency_Fixtures'] = {
                'count': other_count,
                'description': 'Miscellaneous Emergency Lighting',
                'mount_type': 'Various',
                'function': 'Emergency lighting'
            }
        
        return grouped
