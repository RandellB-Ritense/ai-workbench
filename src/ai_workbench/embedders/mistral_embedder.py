"""
Mistral AI embedder implementation.
"""
import time
from typing import List
from mistralai import Mistral
from ai_workbench.embedders.base import Embedder


class MistralEmbedder(Embedder):
    """
    Embedder using Mistral AI's embedding models.

    Supports mistral-embed and other Mistral embedding models.
    Includes batch processing and retry logic for API failures.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-embed",
        batch_size: int = 100,
        max_retries: int = 3,
    ):
        """
        Initialize Mistral embedder.

        Args:
            api_key: Mistral API key
            model: Model name (default: mistral-embed)
            batch_size: Maximum texts to embed in one API call
            max_retries: Maximum retry attempts for failed requests
        """
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.client = Mistral(api_key=api_key)

        # Mistral embed model produces 1024-dimensional vectors
        self._embedding_dimension = 1024

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts in batch(es).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches to respect API limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch with exponential backoff retry.

        Args:
            texts: Batch of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    inputs=texts,
                )

                # Extract embeddings from response
                embeddings = [item.embedding for item in response.data]
                return embeddings

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(
                        f"Failed to generate embeddings after {self.max_retries} attempts: {e}"
                    )

                # Exponential backoff: 1s, 2s, 4s, etc.
                wait_time = 2**attempt
                time.sleep(wait_time)

        # Should not reach here, but just in case
        raise Exception("Failed to generate embeddings")

    def get_embedding_dimension(self) -> int:
        """
        Get embedding vector dimensionality.

        Returns:
            Number of dimensions (1024 for mistral-embed)
        """
        return self._embedding_dimension

    def get_model_name(self) -> str:
        """
        Get model name.

        Returns:
            Model name string
        """
        return self.model
