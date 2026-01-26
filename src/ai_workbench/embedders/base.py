"""
Base embedder interface for generating text embeddings.
"""
from abc import ABC, abstractmethod
from typing import List


class Embedder(ABC):
    """
    Abstract base class for text embedding models.

    Embedders convert text into dense vector representations
    that can be used for semantic search and similarity comparison.
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a batch.

        Batch processing is often more efficient than individual embeddings.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors, one per input text
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimensionality of the embedding vectors.

        Returns:
            Number of dimensions in embedding vectors
        """
        pass

    def get_model_name(self) -> str:
        """
        Get the name of the embedding model.

        Returns:
            Model name string
        """
        return self.__class__.__name__
