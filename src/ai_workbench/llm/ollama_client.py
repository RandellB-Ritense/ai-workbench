"""
Ollama local LLM client.
"""
from typing import List, Dict, Any, Iterator
import ollama
from ai_workbench.llm.base import LLMProvider, Message, LLMResponse


class OllamaClient(LLMProvider):
    """
    LLM provider for Ollama local models.

    Supports llama2, mistral, phi, and other models running locally via Ollama.
    """

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize Ollama client.

        Args:
            model: Model name (e.g., "llama2", "mistral", "phi")
            base_url: Ollama API base URL
        """
        self.model = model
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response from Ollama.

        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Ollama parameters

        Returns:
            LLMResponse object
        """
        # Convert messages to Ollama format
        ollama_messages = [
            {
                "role": msg.role,
                "content": msg.content,
            }
            for msg in messages
        ]

        # Call Ollama API
        response = self.client.chat(
            model=self.model,
            messages=ollama_messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs,
            },
        )

        return LLMResponse(
            content=response["message"]["content"],
            model=response.get("model", self.model),
            tokens_used=response.get("eval_count", None),
            metadata={
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_duration": response.get("total_duration", 0),
            },
        )

    def generate_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs,
    ) -> Iterator[str]:
        """
        Generate a streaming response from Ollama.

        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Ollama parameters

        Yields:
            Text chunks as they are generated
        """
        # Convert messages to Ollama format
        ollama_messages = [
            {
                "role": msg.role,
                "content": msg.content,
            }
            for msg in messages
        ]

        # Stream from Ollama API
        stream = self.client.chat(
            model=self.model,
            messages=ollama_messages,
            stream=True,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs,
            },
        )

        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available Ollama models.

        Returns:
            List of model information
        """
        try:
            models = self.client.list()
            return [
                {
                    "id": model["name"],
                    "name": model["name"],
                    "size": model.get("size", "unknown"),
                    "modified": model.get("modified_at", "unknown"),
                }
                for model in models.get("models", [])
            ]
        except Exception as e:
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Model information dictionary
        """
        try:
            models = self.list_models()
            for model in models:
                if model["id"] == self.model or model["id"].startswith(self.model):
                    return model
        except Exception:
            pass

        return {
            "id": self.model,
            "name": self.model,
            "size": "unknown",
        }

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "Ollama"

    def pull_model(self, model: str) -> bool:
        """
        Pull a model from Ollama library.

        Args:
            model: Model name to pull

        Returns:
            True if successful
        """
        try:
            self.client.pull(model)
            return True
        except Exception:
            return False

    def is_running(self) -> bool:
        """
        Check if Ollama server is running.

        Returns:
            True if Ollama is accessible
        """
        try:
            self.client.list()
            return True
        except Exception:
            return False
