"""
RAG retriever for querying vector stores and retrieving relevant documents.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ai_workbench.embedders.base import Embedder
from ai_workbench.vector_stores.base import VectorStore, QueryResult


@dataclass
class RetrievedDocument:
    """A document retrieved from RAG search."""
    text: str
    score: float
    source_url: str
    title: Optional[str]
    chunk_index: int
    metadata: Dict[str, Any]


class RAGRetriever:
    """
    Retriever for querying vector stores with text queries.

    Handles embedding the query, searching the vector store,
    and returning relevant documents with scores.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ):
        """
        Initialize RAG retriever.

        Args:
            embedder: Embedder to use for query embedding
            vector_store: Vector store to search
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score (0.0-1.0)
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query text
            top_k: Number of results (overrides default)
            metadata_filter: Optional metadata filter (e.g., {"source_url": "https://..."})

        Returns:
            List of RetrievedDocument objects sorted by relevance
        """
        k = top_k if top_k is not None else self.top_k

        # Embed the query
        query_embedding = self.embedder.embed_text(query)

        # Search vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=k,
            filter=metadata_filter,
        )

        # Filter by score threshold and convert to RetrievedDocument
        retrieved_docs = []
        for result in results:
            if result.score >= self.score_threshold:
                retrieved_docs.append(
                    RetrievedDocument(
                        text=result.text,
                        score=result.score,
                        source_url=result.metadata.get("source_url", "unknown"),
                        title=result.metadata.get("title", None),
                        chunk_index=result.metadata.get("chunk_index", 0),
                        metadata=result.metadata,
                    )
                )

        return retrieved_docs

    def retrieve_by_source(
        self,
        query: str,
        source_url: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents from a specific source URL.

        Args:
            query: Search query text
            source_url: Filter by this source URL
            top_k: Number of results

        Returns:
            List of RetrievedDocument objects from the specified source
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            metadata_filter={"source_url": source_url},
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics.

        Returns:
            Dictionary with retriever configuration and vector store info
        """
        return {
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "vector_store_count": self.vector_store.count(),
            "embedder": self.embedder.get_model_name(),
            "embedding_dimension": self.embedder.get_embedding_dimension(),
        }
