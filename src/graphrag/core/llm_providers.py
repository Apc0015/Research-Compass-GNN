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

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using Ollama API."""
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

        response = requests.post(url, json=payload, timeout=self.timeout)

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response generated")
        else:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

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


class LMStudioProvider(LLMProvider):
    """LM Studio local LLM provider (OpenAI-compatible API)."""

    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int, timeout: float):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using LM Studio API (OpenAI-compatible)."""
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

        response = requests.post(url, json=payload, timeout=self.timeout)

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"LM Studio API error: {response.status_code} - {response.text}")

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


class OpenRouterProvider(LLMProvider):
    """OpenRouter cloud LLM aggregator."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int, timeout: float):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = "https://openrouter.ai/api"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using OpenRouter API."""
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

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

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

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using OpenAI API."""
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

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")

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
