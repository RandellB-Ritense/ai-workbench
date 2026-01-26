"""
Base vector store interface for storing and querying embeddings.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class QueryResult:
    """Result from a vector similarity search."""
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class VectorStore(ABC):
    """
    Abstract base class for vector databases.

    Vector stores persist embeddings and enable similarity search
    for retrieval-augmented generation (RAG).
    """

    @abstractmethod
    def add(
        self,
        chunk_ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ):
        """
        Add embeddings to the vector store.

        Args:
            chunk_ids: Unique IDs for each chunk
            texts: Original text content for each chunk
            embeddings: Embedding vectors
            metadatas: Metadata dictionaries for each chunk
        """
        pass

    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        """
        Query the vector store for similar embeddings.

        Args:
            query_embedding: Query vector to search for
            top_k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of QueryResult objects sorted by similarity score
        """
        pass

    @abstractmethod
    def delete(self, chunk_ids: List[str]):
        """
        Delete chunks from the vector store.

        Args:
            chunk_ids: IDs of chunks to delete
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Get the number of vectors in the store.

        Returns:
            Total count of stored vectors
        """
        pass

    @abstractmethod
    def clear(self):
        """Delete all vectors from the store."""
        pass
