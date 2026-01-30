"""
Error handling utilities for the web UI.
"""
import gradio as gr
from typing import Tuple, Optional, Dict, Any
from functools import wraps


def safe_execute(error_message: str = "An error occurred"):
    """
    Decorator to safely execute functions and return Gradio-friendly error responses.

    Args:
        error_message: Custom error message prefix

    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_text = f"{error_message}: {str(e)}"
                # Return a Gradio warning update
                return gr.Warning(error_text)
        return wrapper
    return decorator


def validate_required_fields(**fields) -> Tuple[bool, Optional[str]]:
    """
    Validate that required fields are not empty.

    Args:
        **fields: Named fields to validate

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        is_valid, error = validate_required_fields(
            url=url_input,
            api_key=api_key_input
        )
    """
    for field_name, field_value in fields.items():
        if not field_value or (isinstance(field_value, str) and not field_value.strip()):
            return False, f"{field_name.replace('_', ' ').title()} is required"

    return True, None


def format_error_message(error: Exception, context: str = "") -> str:
    """
    Format an error message for user display.

    Args:
        error: Exception object
        context: Additional context about where the error occurred

    Returns:
        Formatted error message
    """
    error_type = type(error).__name__
    error_msg = str(error)

    # Shorten very long error messages
    if len(error_msg) > 200:
        error_msg = error_msg[:200] + "..."

    if context:
        return f"❌ {context}: {error_type} - {error_msg}"
    else:
        return f"❌ {error_type}: {error_msg}"


def create_success_message(message: str) -> Dict[str, Any]:
    """
    Create a success message update for Gradio components.

    Args:
        message: Success message

    Returns:
        Dictionary with Gradio update
    """
    return gr.update(value=f"✅ {message}", visible=True)


def create_error_message(message: str) -> Dict[str, Any]:
    """
    Create an error message update for Gradio components.

    Args:
        message: Error message

    Returns:
        Dictionary with Gradio update
    """
    return gr.update(value=f"❌ {message}", visible=True)


def create_warning_message(message: str) -> Dict[str, Any]:
    """
    Create a warning message update for Gradio components.

    Args:
        message: Warning message

    Returns:
        Dictionary with Gradio update
    """
    return gr.update(value=f"⚠️ {message}", visible=True)


def create_info_message(message: str) -> Dict[str, Any]:
    """
    Create an info message update for Gradio components.

    Args:
        message: Info message

    Returns:
        Dictionary with Gradio update
    """
    return gr.update(value=f"ℹ️ {message}", visible=True)


def handle_job_error(job_id: str, error: Exception) -> str:
    """
    Format error message for job failures.

    Args:
        job_id: Job ID that failed
        error: Exception that occurred

    Returns:
        Formatted error message
    """
    return f"Job {job_id[:8]}... failed: {str(error)}"


def validate_file_upload(file_path: Optional[str], allowed_extensions: list = None) -> Tuple[bool, Optional[str]]:
    """
    Validate file upload.

    Args:
        file_path: Path to uploaded file
        allowed_extensions: List of allowed extensions (e.g., ['.json', '.txt'])

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path:
        return False, "No file uploaded"

    if allowed_extensions:
        from pathlib import Path
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in allowed_extensions:
            return False, f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"

    try:
        from pathlib import Path
        path = Path(file_path)
        if not path.exists():
            return False, "File not found"
        if not path.is_file():
            return False, "Path is not a file"
    except Exception as e:
        return False, f"Error checking file: {str(e)}"

    return True, None


def confirm_action(action: str) -> str:
    """
    Generate confirmation message for destructive actions.

    Args:
        action: Action being performed

    Returns:
        Confirmation message
    """
    return f"⚠️ Are you sure you want to {action}? This action cannot be undone."


def get_empty_state_message(component: str) -> str:
    """
    Get empty state message for a component.

    Args:
        component: Component name

    Returns:
        Empty state message
    """
    messages = {
        "jobs": "No jobs yet. Start by running a crawler or scraper from the Data Collection tab.",
        "vector_dbs": "No vector databases found. Build an index from the Vector Index tab.",
        "chat_history": "No messages yet. Start a conversation by typing a message below.",
        "mcp_tools": "No MCP servers connected. Connect a server to see available tools.",
        "search_results": "No results found. Try a different query or adjust the search parameters.",
    }

    return messages.get(component, "No items found.")
