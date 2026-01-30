"""
Base LLM provider interface.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass


@dataclass
class Message:
    """A message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    tokens_used: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Provides a unified interface for different LLM APIs
    (Mistral, Ollama, etc.)
    """

    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific options

        Returns:
            LLMResponse object
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs,
    ) -> Iterator[str]:
        """
        Generate a streaming response from the LLM.

        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific options

        Yields:
            Text chunks as they are generated
        """
        pass

    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from this provider.

        Returns:
            List of model information dictionaries
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model details
        """
        pass

    def get_provider_name(self) -> str:
        """
        Get the name of this provider.

        Returns:
            Provider name string
        """
        return self.__class__.__name__.replace("Client", "").replace("Provider", "")
