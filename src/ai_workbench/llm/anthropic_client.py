"""
Anthropic Claude LLM client.
"""
from typing import List, Dict, Any, Iterator
from anthropic import Anthropic
from ai_workbench.llm.base import LLMProvider, Message, LLMResponse


class AnthropicClient(LLMProvider):
    """
    LLM provider for Anthropic Claude models.

    Supports Claude 3.5 Sonnet, Claude 3 Opus, and other Anthropic models.
    """

    # Available Claude models
    MODELS = {
        "claude-3-5-sonnet-20241022": {
            "name": "Claude 3.5 Sonnet",
            "context_window": 200000,
            "output_tokens": 8192,
        },
        "claude-3-5-sonnet-20240620": {
            "name": "Claude 3.5 Sonnet (older)",
            "context_window": 200000,
            "output_tokens": 8192,
        },
        "claude-3-opus-20240229": {
            "name": "Claude 3 Opus",
            "context_window": 200000,
            "output_tokens": 4096,
        },
        "claude-3-sonnet-20240229": {
            "name": "Claude 3 Sonnet",
            "context_window": 200000,
            "output_tokens": 4096,
        },
        "claude-3-haiku-20240307": {
            "name": "Claude 3 Haiku",
            "context_window": 200000,
            "output_tokens": 4096,
        },
    }

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key
            model: Model name to use
        """
        self.api_key = api_key
        self.model = model
        self.client = Anthropic(api_key=api_key)

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response from Claude.

        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic API parameters

        Returns:
            LLMResponse object
        """
        # Convert messages to Anthropic format
        system_message = None
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        # Call Claude API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message,
            messages=api_messages,
            **kwargs,
        )

        # Extract content
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        return LLMResponse(
            content=content,
            model=response.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            metadata={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "stop_reason": response.stop_reason,
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
        Generate a streaming response from Claude.

        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic API parameters

        Yields:
            Text chunks as they are generated
        """
        # Convert messages to Anthropic format
        system_message = None
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        # Stream from Claude API
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message,
            messages=api_messages,
            **kwargs,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available Claude models.

        Returns:
            List of model information
        """
        return [
            {
                "id": model_id,
                "name": info["name"],
                "context_window": info["context_window"],
                "output_tokens": info["output_tokens"],
            }
            for model_id, info in self.MODELS.items()
        ]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Model information dictionary
        """
        if self.model in self.MODELS:
            return {
                "id": self.model,
                **self.MODELS[self.model],
            }
        return {
            "id": self.model,
            "name": self.model,
            "context_window": "unknown",
            "output_tokens": "unknown",
        }

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "Anthropic"
