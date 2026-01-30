"""
Mistral LLM client.
"""
from typing import List, Dict, Any, Iterator
from mistralai import Mistral
from ai_workbench.llm.base import LLMProvider, Message, LLMResponse


class MistralClient(LLMProvider):
    """
    LLM provider for Mistral models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-large-latest",
    ):
        self.model = model
        self.client = Mistral(api_key=api_key)

    def _messages_to_payload(self, messages: List[Message]) -> List[Dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs,
    ) -> LLMResponse:
        response = self.client.chat.complete(
            model=self.model,
            messages=self._messages_to_payload(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        content = ""
        if response and getattr(response, "choices", None):
            message = response.choices[0].message
            content = getattr(message, "content", "") or ""
            if isinstance(content, list):
                content = "".join([str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content])

        usage = getattr(response, "usage", None)
        tokens_used = None
        if usage is not None:
            tokens_used = getattr(usage, "total_tokens", None)

        return LLMResponse(
            content=content,
            model=getattr(response, "model", self.model),
            tokens_used=tokens_used,
            metadata={
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            },
        )

    def generate_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs,
    ) -> Iterator[str]:
        stream = self.client.chat.stream(
            model=self.model,
            messages=self._messages_to_payload(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        for event in stream:
            data = getattr(event, "data", None)
            if not data or not getattr(data, "choices", None):
                continue
            delta = data.choices[0].delta
            chunk = getattr(delta, "content", None)
            if not chunk:
                continue
            if isinstance(chunk, list):
                text = "".join([str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in chunk])
                if text:
                    yield text
            else:
                yield str(chunk)

    def list_models(self) -> List[Dict[str, Any]]:
        try:
            models = self.client.models.list()
            items = getattr(models, "data", None) or getattr(models, "models", None) or []
            results = []
            for model in items:
                model_id = getattr(model, "id", None) or getattr(model, "name", None) or model.get("id") or model.get("name")
                model_name = getattr(model, "name", None) or getattr(model, "id", None) or model.get("name") or model.get("id")
                if model_id or model_name:
                    results.append(
                        {
                            "id": model_id or model_name,
                            "name": model_name or model_id,
                        }
                    )
            return results
        except Exception:
            return []

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "id": self.model,
            "name": self.model,
        }

    def get_provider_name(self) -> str:
        return "Mistral"
