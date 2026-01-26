"""
Configuration management for AI Workbench.
Allows setting default paths and preferences via environment variables or config file.
"""
from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkbenchConfig(BaseSettings):
    """
    Configuration for AI Workbench.

    Can be configured via:
    1. Environment variables (prefixed with WORKBENCH_)
    2. .env file in the current directory
    3. Default values
    """

    # Default output directory
    default_output_dir: Path = Field(
        default=Path.home() / "ai-workbench-output",
        description="Default directory for output files",
    )

    # Crawler settings
    crawler_max_depth: int = Field(
        default=2,
        description="Default maximum crawl depth",
    )
    crawler_max_pages: int = Field(
        default=100,
        description="Default maximum pages to crawl",
    )
    crawler_timeout: int = Field(
        default=10,
        description="HTTP request timeout in seconds",
    )

    # General settings
    verbose: bool = Field(
        default=False,
        description="Enable verbose output",
    )

    # LLM Configuration
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key for Claude models",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    default_llm_provider: str = Field(
        default="anthropic",
        description="Default LLM provider (anthropic, ollama)",
    )
    default_llm_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Default LLM model name",
    )

    # Embedding Configuration
    mistral_api_key: Optional[str] = Field(
        default=None,
        description="Mistral API key for embeddings",
    )
    default_embedding_model: str = Field(
        default="mistral-embed",
        description="Default embedding model",
    )
    chunk_size: int = Field(
        default=500,
        description="Text chunk size for embeddings (in tokens)",
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between text chunks (in tokens)",
    )

    # Vector Store Configuration
    vector_store_path: Path = Field(
        default=Path.home() / ".ai-workbench/vector-stores",
        description="Directory for vector store databases",
    )

    # RAG Configuration
    rag_top_k: int = Field(
        default=5,
        description="Number of documents to retrieve for RAG",
    )
    rag_score_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score for RAG retrieval",
    )
    rag_context_max_tokens: int = Field(
        default=4000,
        description="Maximum tokens for RAG context",
    )

    # MCP Configuration
    mcp_server_port: int = Field(
        default=8080,
        description="Default MCP server port",
    )
    mcp_enabled_servers: List[str] = Field(
        default_factory=list,
        description="List of enabled MCP servers",
    )

    # Chatbot Configuration
    chat_history_limit: int = Field(
        default=50,
        description="Maximum messages in chat history",
    )
    chat_temperature: float = Field(
        default=0.7,
        description="Default LLM temperature for chat",
    )
    chat_max_tokens: int = Field(
        default=4000,
        description="Maximum tokens in chat response",
    )

    model_config = SettingsConfigDict(
        env_prefix="WORKBENCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    def ensure_output_dir(self) -> Path:
        """Ensure the default output directory exists."""
        self.default_output_dir.mkdir(parents=True, exist_ok=True)
        return self.default_output_dir

    def ensure_vector_store_path(self) -> Path:
        """Ensure the vector store directory exists."""
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        return self.vector_store_path


# Global config instance
_config: Optional[WorkbenchConfig] = None


def get_config() -> WorkbenchConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = WorkbenchConfig()
    return _config


def reload_config() -> WorkbenchConfig:
    """Reload the configuration from environment/file."""
    global _config
    _config = WorkbenchConfig()
    return _config
