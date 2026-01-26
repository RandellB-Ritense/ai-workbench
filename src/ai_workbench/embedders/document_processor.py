"""
Document processor for chunking scraped content into embeddable segments.
"""
import json
import tiktoken
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """A chunk of text from a document with metadata."""
    chunk_id: str
    source_url: str
    chunk_text: str
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any]


class DocumentProcessor:
    """
    Process scraped documents into chunks suitable for embedding.

    Uses tiktoken for token-aware chunking to ensure chunks stay within
    embedding model limits and maintain semantic coherence.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base",  # GPT-4/Claude tokenizer
    ):
        """
        Initialize document processor.

        Args:
            chunk_size: Target number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            encoding_name: Tiktoken encoding to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def load_scraped_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load scraped content from JSON file.

        Args:
            file_path: Path to scraped JSON file (from web_scraper.py)

        Returns:
            List of page dictionaries
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        return data.get("pages", [])

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into token-aware segments.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # Encode text to tokens
        tokens = self.encoding.encode(text)

        # If text is smaller than chunk size, return as-is
        if len(tokens) <= self.chunk_size:
            return [text]

        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            # Get chunk of tokens
            end_idx = start_idx + self.chunk_size
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start index, accounting for overlap
            start_idx = end_idx - self.chunk_overlap

        return chunks

    def process_document(
        self,
        page: Dict[str, Any],
        doc_index: int,
    ) -> List[DocumentChunk]:
        """
        Process a single document into chunks.

        Args:
            page: Page dictionary from scraper output
            doc_index: Index of this document in the collection

        Returns:
            List of DocumentChunk objects
        """
        url = page.get("url", "unknown")
        markdown_content = page.get("markdown_content", "")
        title = page.get("title", "Untitled")

        # Chunk the markdown content
        text_chunks = self.chunk_text(markdown_content)

        # Create DocumentChunk objects
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"doc_{doc_index}_chunk_{i}"

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source_url=url,
                    chunk_text=chunk_text,
                    chunk_index=i,
                    total_chunks=len(text_chunks),
                    metadata={
                        "title": title,
                        "word_count": page.get("word_count", 0),
                        "scraped_at": page.get("scraped_at", ""),
                        "doc_metadata": page.get("metadata", {}),
                    },
                )
            )

        return chunks

    def process_scraped_file(self, file_path: Path) -> List[DocumentChunk]:
        """
        Process entire scraped JSON file into chunks.

        Args:
            file_path: Path to scraped JSON file

        Returns:
            List of all DocumentChunk objects from all pages
        """
        pages = self.load_scraped_json(file_path)

        all_chunks = []
        for doc_idx, page in enumerate(pages):
            # Skip pages with errors
            if page.get("error"):
                continue

            chunks = self.process_document(page, doc_idx)
            all_chunks.extend(chunks)

        return all_chunks

    def get_chunk_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Get statistics about processed chunks.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            Dictionary with statistics
        """
        total_text = "".join(chunk.chunk_text for chunk in chunks)
        total_tokens = len(self.encoding.encode(total_text))

        return {
            "total_chunks": len(chunks),
            "total_documents": len(set(chunk.source_url for chunk in chunks)),
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": total_tokens / len(chunks) if chunks else 0,
            "chunk_size_config": self.chunk_size,
            "chunk_overlap_config": self.chunk_overlap,
        }
