"""
ChromaDB vector store implementation.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from ai_workbench.vector_stores.base import VectorStore, QueryResult


class ChromaStore(VectorStore):
    """
    Vector store implementation using ChromaDB.

    ChromaDB provides persistent storage, fast similarity search,
    and metadata filtering capabilities.
    """

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "ai_workbench_docs",
        embedding_dimension: int = 1024,
    ):
        """
        Initialize ChromaDB store.

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
            embedding_dimension: Dimensionality of embeddings
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension

        # Ensure directory exists
        persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"embedding_dimension": embedding_dimension},
        )

    def add(
        self,
        chunk_ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ):
        """
        Add embeddings to ChromaDB.

        Args:
            chunk_ids: Unique IDs for each chunk
            texts: Original text content
            embeddings: Embedding vectors
            metadatas: Metadata dictionaries
        """
        if not chunk_ids:
            return

        # ChromaDB expects specific types for metadata values
        # Convert all metadata values to strings/ints/floats
        clean_metadatas = []
        for metadata in metadatas:
            clean_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                elif isinstance(value, dict):
                    # Flatten nested dicts
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, (str, int, float, bool)):
                            clean_metadata[f"{key}_{nested_key}"] = nested_value
                else:
                    # Convert other types to string
                    clean_metadata[key] = str(value)
            clean_metadatas.append(clean_metadata)

        # Add to collection
        self.collection.add(
            ids=chunk_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=clean_metadatas,
        )

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        """
        Query ChromaDB for similar vectors.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter: Optional metadata filter (ChromaDB where clause)

        Returns:
            List of QueryResult objects
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter,
        )

        # Parse results into QueryResult objects
        query_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                query_results.append(
                    QueryResult(
                        chunk_id=results["ids"][0][i],
                        score=1.0 - results["distances"][0][i],  # Convert distance to similarity
                        text=results["documents"][0][i],
                        metadata=results["metadatas"][0][i] if results["metadatas"][0] else {},
                    )
                )

        return query_results

    def delete(self, chunk_ids: List[str]):
        """
        Delete chunks from ChromaDB.

        Args:
            chunk_ids: IDs of chunks to delete
        """
        if not chunk_ids:
            return

        self.collection.delete(ids=chunk_ids)

    def count(self) -> int:
        """
        Get count of vectors in collection.

        Returns:
            Number of vectors stored
        """
        return self.collection.count()

    def clear(self):
        """Delete all vectors from collection."""
        # Delete the collection and recreate it
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"embedding_dimension": self.embedding_dimension},
        )

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection metadata
        """
        return {
            "name": self.collection_name,
            "count": self.count(),
            "persist_directory": str(self.persist_directory),
            "embedding_dimension": self.embedding_dimension,
        }
