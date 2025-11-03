"""
Enhanced LLM Manager Module with Multi-Provider Support
Supports: Ollama, LM Studio, OpenRouter, OpenAI
"""

from typing import Optional, Dict, List, Tuple
import logging
from .llm_providers import (
    LLMProvider,
    OllamaProvider,
    LMStudioProvider,
    OpenRouterProvider,
    OpenAIProvider
)

logger = logging.getLogger(__name__)


class LLMManager:
    """Enhanced LLM Manager supporting multiple providers with auto-discovery."""

    def __init__(
        self,
        provider: str = "ollama",  # ollama, lmstudio, openrouter, openai
        model: str = "llama3.2",
        base_url: str = "http://localhost:1234",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        timeout: float = 30.0,
        max_retries: int = 2
    ):
        """
        Initialize the LLM Manager with specified provider.

        Args:
            provider: One of 'ollama', 'lmstudio', 'openrouter', 'openai'
            model: Model name/ID to use
            base_url: Base URL for local providers (Ollama, LM Studio)
            api_key: API key for cloud providers (OpenRouter, OpenAI)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
        """
        self.provider_name = provider.lower()
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize the appropriate provider
        self.provider = self._create_provider()

        logger.info(
            f"LLM Manager configured: {self.provider_name.upper()} - {self.model}"
        )

    def _create_provider(self) -> LLMProvider:
        """Create and return the appropriate provider instance."""
        if self.provider_name == "ollama":
            return OllamaProvider(
                base_url=self.base_url,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
        elif self.provider_name == "lmstudio":
            return LMStudioProvider(
                base_url=self.base_url,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
        elif self.provider_name == "openrouter":
            return OpenRouterProvider(
                api_key=self.api_key or "",
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
        elif self.provider_name == "openai":
            return OpenAIProvider(
                api_key=self.api_key or "",
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
        else:
            raise ValueError(
                f"Unsupported provider: {self.provider_name}. "
                f"Choose from: ollama, lmstudio, openrouter, openai"
            )

    def generate_answer(
        self,
        query: str,
        context: str,
        use_graph: bool = False,
        graph_context: str = "",
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate an answer using the LLM."""

        # Default system prompt
        if system_prompt is None:
            system_prompt = """You are a helpful research assistant. Answer questions based ONLY on the provided context.
If the context doesn't contain enough information, say so. Always cite which source or chunk your answer comes from.
Be concise but comprehensive."""

        # Build user prompt
        user_prompt = f"""Question: {query}

Context from documents:
{context}
"""

        if use_graph and graph_context:
            user_prompt += f"""

Additional context from knowledge graph:
{graph_context}
"""

        user_prompt += "\n\nPlease provide a detailed answer based on the above context."

        # Generate response
        try:
            return self.provider.generate(system_prompt, user_prompt)
        except Exception as e:
            error_msg = (
                f"Error generating answer: {str(e)}\n\n"
                f"Check your {self.provider_name.upper()} configuration."
            )
            logger.error(error_msg)
            return error_msg

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Multi-turn chat interface."""
        # For simplicity, just use the last message
        if messages:
            last_msg = messages[-1]
            return self.generate_answer(
                query=last_msg.get("content", ""),
                context="",
                system_prompt=system_prompt
            )
        return "No messages provided"

    def get_config(self) -> Dict:
        """Get current LLM configuration."""
        return {
            "provider": self.provider_name.upper(),
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "base_url": self.base_url,
            "api_key_set": bool(self.api_key)
        }

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test the LLM connection.

        Returns:
            Tuple of (success: bool, message: str)
        """
        return self.provider.test_connection()

    def list_available_models(self) -> List[Dict[str, str]]:
        """
        List available models from the current provider.

        Returns:
            List of dicts with 'id' and 'name' keys
        """
        return self.provider.list_models()

    def update_config(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Update LLM configuration and reinitialize provider.

        Args:
            provider: New provider name
            model: New model name
            base_url: New base URL (for local providers)
            api_key: New API key (for cloud providers)
            temperature: New temperature
            max_tokens: New max tokens
        """
        if provider is not None:
            self.provider_name = provider.lower()
        if model is not None:
            self.model = model
        if base_url is not None:
            self.base_url = base_url
        if api_key is not None:
            self.api_key = api_key
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens

        # Reinitialize provider with new config
        self.provider = self._create_provider()
        logger.info(f"LLM configuration updated: {self.provider_name.upper()} - {self.model}")