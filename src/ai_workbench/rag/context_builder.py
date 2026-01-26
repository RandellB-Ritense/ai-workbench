"""
Context builder for formatting retrieved documents for LLM consumption.
"""
import tiktoken
from typing import List, Dict, Any
from ai_workbench.rag.retriever import RetrievedDocument


class ContextBuilder:
    """
    Build formatted context from retrieved documents.

    Handles deduplication, token budget management,
    and formatting for LLM consumption.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        encoding_name: str = "cl100k_base",
        include_sources: bool = True,
    ):
        """
        Initialize context builder.

        Args:
            max_tokens: Maximum tokens for context
            encoding_name: Tiktoken encoding to use
            include_sources: Include source citations in context
        """
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.include_sources = include_sources

    def build_context(
        self,
        documents: List[RetrievedDocument],
        query: str,
    ) -> Dict[str, Any]:
        """
        Build formatted context from retrieved documents.

        Args:
            documents: List of retrieved documents
            query: Original query text

        Returns:
            Dictionary with formatted context and metadata
        """
        if not documents:
            return {
                "context": "No relevant documents found.",
                "documents_used": [],
                "tokens_used": 0,
                "truncated": False,
            }

        # Deduplicate documents (same source URL + chunk index)
        seen = set()
        unique_docs = []
        for doc in documents:
            key = (doc.source_url, doc.chunk_index)
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        # Build context within token budget
        formatted_docs = []
        tokens_used = 0
        documents_used = []
        truncated = False

        for i, doc in enumerate(unique_docs, 1):
            # Format document
            formatted = self._format_document(doc, i)
            doc_tokens = len(self.encoding.encode(formatted))

            # Check if adding this doc exceeds budget
            if tokens_used + doc_tokens > self.max_tokens:
                truncated = True
                break

            formatted_docs.append(formatted)
            tokens_used += doc_tokens
            documents_used.append({
                "source_url": doc.source_url,
                "title": doc.title,
                "score": doc.score,
                "chunk_index": doc.chunk_index,
            })

        # Combine into final context
        if self.include_sources:
            context = self._build_context_with_sources(formatted_docs, query)
        else:
            context = "\n\n".join(formatted_docs)

        return {
            "context": context,
            "documents_used": documents_used,
            "tokens_used": tokens_used,
            "truncated": truncated,
            "total_available": len(unique_docs),
        }

    def _format_document(self, doc: RetrievedDocument, index: int) -> str:
        """
        Format a single document.

        Args:
            doc: Document to format
            index: Document number

        Returns:
            Formatted document string
        """
        parts = []

        # Document header
        if doc.title:
            parts.append(f"## Document {index}: {doc.title}")
        else:
            parts.append(f"## Document {index}")

        # Source and score (optional metadata)
        if self.include_sources:
            parts.append(f"Source: {doc.source_url}")
            parts.append(f"Relevance: {doc.score:.2f}")

        # Document content
        parts.append("")
        parts.append(doc.text.strip())

        return "\n".join(parts)

    def _build_context_with_sources(
        self,
        formatted_docs: List[str],
        query: str,
    ) -> str:
        """
        Build context with source attribution.

        Args:
            formatted_docs: List of formatted document strings
            query: Original query

        Returns:
            Formatted context string with header
        """
        parts = [
            "# Retrieved Documentation",
            "",
            f"The following documents were retrieved to answer: \"{query}\"",
            "",
            "---",
            "",
        ]

        parts.extend(formatted_docs)

        return "\n".join(parts)

    def get_token_count(self, text: str) -> int:
        """
        Get token count for text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def estimate_context_tokens(
        self,
        documents: List[RetrievedDocument],
    ) -> int:
        """
        Estimate total tokens if all documents were included.

        Args:
            documents: List of documents

        Returns:
            Estimated token count
        """
        total = 0
        for i, doc in enumerate(documents, 1):
            formatted = self._format_document(doc, i)
            total += self.get_token_count(formatted)
        return total
