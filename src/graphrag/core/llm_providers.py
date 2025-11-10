"""
LLM Provider Implementations
Supports: Ollama, LM Studio, OpenRouter, OpenAI
"""

from typing import List, Dict, Optional, Tuple
import logging
import requests
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response."""
        pass

    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        """Test connection to the provider. Returns (success, message)."""
        pass

    @abstractmethod
    def list_models(self) -> List[Dict[str, str]]:
        """List available models. Returns list of {id, name} dicts."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int, timeout: float):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Auto-detect and fallback if model not available
        self._auto_detect_model()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using Ollama API with robust error handling."""
        url = f"{self.base_url}/api/generate"

        # Combine system and user prompts for Ollama
        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)

            if response.status_code == 200:
                try:
                    result = response.json()
                    return result.get("response", "No response generated")
                except ValueError as e:
                    raise Exception(f"Ollama returned invalid JSON: {e}")
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to Ollama at {self.base_url}. Is it running? (Try: ollama serve)")
        except requests.exceptions.Timeout:
            raise Exception(f"Ollama request timed out after {self.timeout}s. Try increasing timeout or check if model is loaded.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama request failed: {str(e)}")

    def test_connection(self) -> Tuple[bool, str]:
        """Test Ollama connection."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                num_models = len(data.get('models', []))
                return True, f"✓ Connected to Ollama ({num_models} models available)"
            else:
                return False, f"✗ Ollama returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "✗ Cannot connect to Ollama. Is it running? (Try: ollama serve)"
        except Exception as e:
            return False, f"✗ Ollama error: {str(e)}"

    def list_models(self) -> List[Dict[str, str]]:
        """List available Ollama models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get('models', []):
                    model_name = model.get('name', '')
                    models.append({
                        'id': model_name,
                        'name': model_name,
                        'size': model.get('size', 0)
                    })
                return models
            return []
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    def _auto_detect_model(self):
        """Auto-detect if requested model is available, fallback to first available."""
        try:
            models = self.list_models()
            if not models:
                logger.warning(f"No Ollama models found. Please install a model with: ollama pull {self.model}")
                return
            
            # Check if requested model exists
            available_model_ids = [m['id'] for m in models]
            
            if self.model not in available_model_ids:
                # Model not found, use first available
                fallback_model = models[0]['id']
                logger.warning(
                    f"Requested model '{self.model}' not found in Ollama. "
                    f"Auto-switching to '{fallback_model}'. "
                    f"Available models: {', '.join(available_model_ids)}"
                )
                self.model = fallback_model
            else:
                logger.info(f"✓ Ollama model '{self.model}' is available")
                
        except Exception as e:
            logger.warning(f"Could not auto-detect Ollama models: {e}")


class LMStudioProvider(LLMProvider):
    """LM Studio local LLM provider (OpenAI-compatible API)."""

    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int, timeout: float):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Auto-detect and fallback if model not available
        self._auto_detect_model()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using LM Studio API (OpenAI-compatible) with robust error handling."""
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)

            if response.status_code == 200:
                try:
                    result = response.json()
                    # Safe dictionary access with .get() fallbacks
                    choices = result.get("choices", [])
                    if choices and len(choices) > 0:
                        message = choices[0].get("message", {})
                        content = message.get("content", "")
                        if content:
                            return content
                        else:
                            raise Exception("LM Studio returned empty response")
                    else:
                        raise Exception("LM Studio returned no choices in response")
                except ValueError as e:
                    raise Exception(f"LM Studio returned invalid JSON: {e}")
            else:
                raise Exception(f"LM Studio API error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to LM Studio at {self.base_url}. Is the server running?")
        except requests.exceptions.Timeout:
            raise Exception(f"LM Studio request timed out after {self.timeout}s. Try increasing timeout.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"LM Studio request failed: {str(e)}")

    def test_connection(self) -> Tuple[bool, str]:
        """Test LM Studio connection."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                num_models = len(data.get('data', []))
                return True, f"✓ Connected to LM Studio ({num_models} models available)"
            else:
                return False, f"✗ LM Studio returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "✗ Cannot connect to LM Studio. Is the server running?"
        except Exception as e:
            return False, f"✗ LM Studio error: {str(e)}"

    def list_models(self) -> List[Dict[str, str]]:
        """List available LM Studio models."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get('data', []):
                    model_id = model.get('id', '')
                    models.append({
                        'id': model_id,
                        'name': model_id
                    })
                return models
            return []
        except Exception as e:
            logger.error(f"Failed to list LM Studio models: {e}")
            return []
    
    def _auto_detect_model(self):
        """Auto-detect if requested model is available, fallback to first available."""
        try:
            models = self.list_models()
            if not models:
                logger.warning(f"No LM Studio models found. Please load a model in LM Studio.")
                return
            
            # Check if requested model exists
            available_model_ids = [m['id'] for m in models]
            
            if self.model not in available_model_ids:
                # Model not found, use first available
                fallback_model = models[0]['id']
                logger.warning(
                    f"Requested model '{self.model}' not found in LM Studio. "
                    f"Auto-switching to '{fallback_model}'. "
                    f"Available models: {', '.join(available_model_ids)}"
                )
                self.model = fallback_model
            else:
                logger.info(f"✓ LM Studio model '{self.model}' is available")
                
        except Exception as e:
            logger.warning(f"Could not auto-detect LM Studio models: {e}")


class OpenRouterProvider(LLMProvider):
    """OpenRouter cloud LLM aggregator."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int, timeout: float):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = "https://openrouter.ai/api"
        
        # Auto-detect and validate model if API key is provided
        if self.api_key:
            self._auto_detect_model()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using OpenRouter API with robust error handling."""
        url = f"{self.base_url}/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/graphrag",
            "X-Title": "GraphRAG System"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

            if response.status_code == 200:
                try:
                    result = response.json()
                    # Safe dictionary access with .get() fallbacks
                    choices = result.get("choices", [])
                    if choices and len(choices) > 0:
                        message = choices[0].get("message", {})
                        content = message.get("content", "")
                        if content:
                            return content
                        else:
                            raise Exception("OpenRouter returned empty response")
                    else:
                        raise Exception("OpenRouter returned no choices in response")
                except ValueError as e:
                    raise Exception(f"OpenRouter returned invalid JSON: {e}")
            elif response.status_code == 401:
                raise Exception("OpenRouter authentication failed. Check your API key.")
            elif response.status_code == 429:
                raise Exception("OpenRouter rate limit exceeded. Please try again later.")
            else:
                raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to OpenRouter. Check your internet connection.")
        except requests.exceptions.Timeout:
            raise Exception(f"OpenRouter request timed out after {self.timeout}s. Try increasing timeout.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"OpenRouter request failed: {str(e)}")

    def test_connection(self) -> Tuple[bool, str]:
        """Test OpenRouter connection."""
        if not self.api_key:
            return False, "✗ OpenRouter API key not provided"

        try:
            # Test with models endpoint
            url = f"{self.base_url}/v1/models"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                data = response.json()
                num_models = len(data.get('data', []))
                return True, f"✓ Connected to OpenRouter ({num_models}+ models available)"
            elif response.status_code == 401:
                return False, "✗ Invalid OpenRouter API key"
            else:
                return False, f"✗ OpenRouter returned status {response.status_code}"
        except Exception as e:
            return False, f"✗ OpenRouter error: {str(e)}"

    def list_models(self) -> List[Dict[str, str]]:
        """List available OpenRouter models."""
        if not self.api_key:
            return []

        try:
            url = f"{self.base_url}/v1/models"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get('data', []):
                    model_id = model.get('id', '')
                    model_name = model.get('name', model_id)
                    # Add context length and pricing info if available
                    context = model.get('context_length', 'N/A')
                    models.append({
                        'id': model_id,
                        'name': f"{model_name} (ctx: {context})"
                    })
                return models
            return []
        except Exception as e:
            logger.error(f"Failed to list OpenRouter models: {e}")
            return []
    
    def _auto_detect_model(self):
        """Validate model availability, suggest popular alternatives if not found."""
        try:
            models = self.list_models()
            if not models:
                logger.warning("Could not fetch OpenRouter models. Using requested model anyway.")
                return
            
            # Check if requested model exists
            available_model_ids = [m['id'] for m in models]
            
            if self.model not in available_model_ids:
                # Suggest popular alternatives
                popular_models = [
                    'openai/gpt-4o',
                    'openai/gpt-4o-mini',
                    'anthropic/claude-3.5-sonnet',
                    'google/gemini-pro',
                    'meta-llama/llama-3.1-70b-instruct'
                ]
                
                # Find first available popular model
                fallback = next((m for m in popular_models if m in available_model_ids), None)
                
                if fallback:
                    logger.warning(
                        f"Requested model '{self.model}' not found in OpenRouter. "
                        f"Auto-switching to '{fallback}'."
                    )
                    self.model = fallback
                else:
                    logger.warning(
                        f"Requested model '{self.model}' not found. "
                        f"Will attempt to use it anyway. Popular options: {', '.join(popular_models[:3])}"
                    )
            else:
                logger.info(f"✓ OpenRouter model '{self.model}' is available")
                
        except Exception as e:
            logger.warning(f"Could not validate OpenRouter model: {e}")


class OpenAIProvider(LLMProvider):
    """OpenAI cloud LLM provider."""

    # Popular OpenAI models
    AVAILABLE_MODELS = [
        {"id": "gpt-4o", "name": "GPT-4o (Latest)"},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini (Fast & Affordable)"},
        {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
        {"id": "gpt-4", "name": "GPT-4"},
        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
        {"id": "gpt-3.5-turbo-16k", "name": "GPT-3.5 Turbo 16K"}
    ]

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int, timeout: float):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = "https://api.openai.com"
        
        # Validate model is in known list
        self._validate_model()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using OpenAI API with robust error handling."""
        url = f"{self.base_url}/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

            if response.status_code == 200:
                try:
                    result = response.json()
                    # Safe dictionary access with .get() fallbacks
                    choices = result.get("choices", [])
                    if choices and len(choices) > 0:
                        message = choices[0].get("message", {})
                        content = message.get("content", "")
                        if content:
                            return content
                        else:
                            raise Exception("OpenAI returned empty response")
                    else:
                        raise Exception("OpenAI returned no choices in response")
                except ValueError as e:
                    raise Exception(f"OpenAI returned invalid JSON: {e}")
            elif response.status_code == 401:
                raise Exception("OpenAI authentication failed. Check your API key.")
            elif response.status_code == 429:
                raise Exception("OpenAI rate limit exceeded. Please try again later or upgrade your plan.")
            elif response.status_code == 400:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", response.text)
                    raise Exception(f"OpenAI bad request: {error_msg}")
                except ValueError:
                    raise Exception(f"OpenAI bad request: {response.text}")
            else:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to OpenAI. Check your internet connection.")
        except requests.exceptions.Timeout:
            raise Exception(f"OpenAI request timed out after {self.timeout}s. Try increasing timeout.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"OpenAI request failed: {str(e)}")

    def test_connection(self) -> Tuple[bool, str]:
        """Test OpenAI connection."""
        if not self.api_key:
            return False, "✗ OpenAI API key not provided"

        try:
            # Test with models endpoint
            url = f"{self.base_url}/v1/models"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                return True, "✓ Connected to OpenAI"
            elif response.status_code == 401:
                return False, "✗ Invalid OpenAI API key"
            else:
                return False, f"✗ OpenAI returned status {response.status_code}"
        except Exception as e:
            return False, f"✗ OpenAI error: {str(e)}"

    def list_models(self) -> List[Dict[str, str]]:
        """List available OpenAI models (static list of popular models)."""
        return self.AVAILABLE_MODELS.copy()
    
    def _validate_model(self):
        """Validate model is in known models list, fallback to gpt-4o-mini if not."""
        known_model_ids = [m['id'] for m in self.AVAILABLE_MODELS]
        
        if self.model not in known_model_ids:
            fallback = "gpt-4o-mini"
            logger.warning(
                f"Model '{self.model}' not in known OpenAI models. "
                f"Auto-switching to '{fallback}'. "
                f"Known models: {', '.join(known_model_ids)}"
            )
            self.model = fallback
        else:
            logger.info(f"✓ Using OpenAI model '{self.model}'")
