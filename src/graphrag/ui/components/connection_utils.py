"""
Connection Testing and Model Detection Utilities.

This module provides helper functions for testing connections to various LLM providers
and database services, as well as detecting available models.

Extracted from unified_launcher.py as part of Phase 3 architecture improvements.
"""

import logging
import requests
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


# LLM Connection Testing Functions

def test_ollama_connection() -> Dict[str, Any]:
    """Test connection to Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return {"success": True, "message": "Connected to Ollama", "data": response.json()}
        return {"success": False, "message": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "message": "Cannot connect to Ollama. Is it running on localhost:11434?"}
    except requests.exceptions.Timeout:
        return {"success": False, "message": "Connection timeout (5s)"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


def test_lmstudio_connection() -> Dict[str, Any]:
    """Test connection to LM Studio."""
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            return {"success": True, "message": "Connected to LM Studio", "data": response.json()}
        return {"success": False, "message": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "message": "Cannot connect to LM Studio. Is it running on localhost:1234?"}
    except requests.exceptions.Timeout:
        return {"success": False, "message": "Connection timeout (5s)"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


def test_openrouter_connection(api_key: str) -> Dict[str, Any]:
    """Test connection to OpenRouter."""
    if not api_key or not api_key.strip():
        return {"success": False, "message": "API key is required for OpenRouter"}

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://research-compass.local",
            "X-Title": "Research Compass"
        }
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=5)
        if response.status_code == 200:
            return {"success": True, "message": "Connected to OpenRouter", "data": response.json()}
        elif response.status_code == 401:
            return {"success": False, "message": "Invalid API key"}
        return {"success": False, "message": f"HTTP {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"success": False, "message": "Connection timeout (5s)"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


def test_openai_connection(api_key: str) -> Dict[str, Any]:
    """Test connection to OpenAI."""
    if not api_key or not api_key.strip():
        return {"success": False, "message": "API key is required for OpenAI"}

    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=5)
        if response.status_code == 200:
            return {"success": True, "message": "Connected to OpenAI", "data": response.json()}
        elif response.status_code == 401:
            return {"success": False, "message": "Invalid API key"}
        return {"success": False, "message": f"HTTP {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"success": False, "message": "Connection timeout (5s)"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


# Model Detection Functions

def detect_ollama_models() -> List[str]:
    """Detect available models from Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            return [model.get("name", "") for model in models if model.get("name")]
        return []
    except Exception as e:
        logger.error(f"Error detecting Ollama models: {e}")
        return []


def detect_lmstudio_models() -> List[str]:
    """Detect available models from LM Studio."""
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            return [model.get("id", "") for model in models if model.get("id")]
        return []
    except Exception as e:
        logger.error(f"Error detecting LM Studio models: {e}")
        return []


def detect_openrouter_models(api_key: str) -> List[str]:
    """Detect available models from OpenRouter."""
    if not api_key or not api_key.strip():
        return []

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://research-compass.local",
            "X-Title": "Research Compass"
        }
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            # Return top popular models (limit to 50 for UI)
            return [model.get("id", "") for model in models[:50] if model.get("id")]
        return []
    except Exception as e:
        logger.error(f"Error detecting OpenRouter models: {e}")
        return []


def detect_openai_models(api_key: str) -> List[str]:
    """Detect available models from OpenAI."""
    if not api_key or not api_key.strip():
        return []

    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            # Filter for chat models
            return sorted([model.get("id", "") for model in models
                          if model.get("id") and ("gpt" in model.get("id", "").lower())])
        return []
    except Exception as e:
        logger.error(f"Error detecting OpenAI models: {e}")
        return []


# Database Connection Testing

def test_neo4j_connection(uri: str, username: str, password: str) -> Dict[str, Any]:
    """Test connection to Neo4j database."""
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            result.single()
        driver.close()

        return {"success": True, "message": "Successfully connected to Neo4j"}
    except ImportError:
        return {"success": False, "message": "neo4j package not installed. Install with: pip install neo4j"}
    except Exception as e:
        return {"success": False, "message": f"Connection failed: {str(e)}"}
