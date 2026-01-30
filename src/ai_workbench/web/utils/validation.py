"""
Input validation utilities for the web UI.
"""
from pathlib import Path
from typing import Optional, Tuple
import httpx


def validate_url(url: str) -> Tuple[bool, str]:
    """
    Validate a URL format.

    Args:
        url: URL to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, "URL is required"

    if not url.startswith(("http://", "https://")):
        return False, "URL must start with http:// or https://"

    return True, ""


def validate_api_key(api_key: str, provider: str = "generic") -> Tuple[bool, str]:
    """
    Validate an API key format.

    Args:
        api_key: API key to validate
        provider: Provider name (mistral, etc.)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not api_key:
        return False, f"{provider.capitalize()} API key is required"

    if len(api_key) < 10:
        return False, "API key seems too short"

    return True, ""


def validate_file_exists(file_path: str) -> Tuple[bool, str]:
    """
    Validate that a file exists.

    Args:
        file_path: Path to file

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path:
        return False, "File path is required"

    path = Path(file_path)
    if not path.exists():
        return False, f"File not found: {file_path}"

    if not path.is_file():
        return False, f"Path is not a file: {file_path}"

    return True, ""


def validate_directory_exists(dir_path: str, create_if_missing: bool = False) -> Tuple[bool, str]:
    """
    Validate that a directory exists.

    Args:
        dir_path: Path to directory
        create_if_missing: Create directory if it doesn't exist

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not dir_path:
        return False, "Directory path is required"

    path = Path(dir_path)

    if not path.exists():
        if create_if_missing:
            try:
                path.mkdir(parents=True, exist_ok=True)
                return True, ""
            except Exception as e:
                return False, f"Failed to create directory: {e}"
        else:
            return False, f"Directory not found: {dir_path}"

    if not path.is_dir():
        return False, f"Path is not a directory: {dir_path}"

    return True, ""


def check_ollama_connection(base_url: str = "http://localhost:11434") -> Tuple[bool, str]:
    """
    Check if Ollama is running and accessible.

    Args:
        base_url: Ollama API base URL

    Returns:
        Tuple of (is_running, error_message)
    """
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                return True, ""
            else:
                return False, f"Ollama returned status {response.status_code}"
    except httpx.ConnectError:
        return False, f"Cannot connect to Ollama at {base_url}"
    except httpx.TimeoutException:
        return False, f"Ollama connection timeout at {base_url}"
    except Exception as e:
        return False, f"Error connecting to Ollama: {e}"


def validate_vector_db(db_name: str, vector_store_path: Path) -> Tuple[bool, str]:
    """
    Validate that a vector database exists.

    Args:
        db_name: Name of the vector database
        vector_store_path: Path to vector stores directory

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not db_name:
        return False, "Vector database name is required"

    db_path = vector_store_path / db_name

    if not db_path.exists():
        return False, f"Vector database not found: {db_name}"

    if not db_path.is_dir():
        return False, f"Vector database path is not a directory: {db_name}"

    # Check if it contains ChromaDB files
    chroma_file = db_path / "chroma.sqlite3"
    if not chroma_file.exists():
        return False, f"Invalid vector database (missing chroma.sqlite3): {db_name}"

    return True, ""
