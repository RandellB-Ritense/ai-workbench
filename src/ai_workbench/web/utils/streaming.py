"""
Streaming utilities for Gradio chat interface.
"""
from typing import List, Optional, Generator, Dict, Any
from ai_workbench.llm.base import Message


def _content_to_text(content: Any) -> str:
    """Normalize Gradio message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", content.get("content", "")))
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join([p for p in parts if p])
    return str(content)


def build_messages_from_history(history: List[Dict[str, str]], system_prompt: Optional[str] = None) -> List[Message]:
    """
    Convert Gradio chat history to LLM message format.

    Args:
        history: Gradio chat history format [{"role": "...", "content": "..."}, ...]
        system_prompt: Optional system prompt to prepend

    Returns:
        List of Message objects for LLM
    """
    messages = []

    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))

    for msg in history:
        role = msg.get("role")
        content = _content_to_text(msg.get("content"))
        if role and content != "":
            messages.append(Message(role=role, content=content))

    return messages


def chat_stream_generator(
    message: str,
    history: List[Dict[str, str]],
    llm_client,
    retriever=None,
    context_builder=None,
    rag_enabled: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 4000,
) -> Generator[List[Dict[str, str]], None, None]:
    """
    Generator function for streaming chat responses in Gradio.

    Args:
        message: User's message
        history: Chat history
        llm_client: LLM client instance (MistralClient or OllamaClient)
        retriever: RAG retriever instance (optional)
        context_builder: Context builder instance (optional)
        rag_enabled: Whether to use RAG
        temperature: LLM temperature
        max_tokens: Maximum tokens in response

    Yields:
        Updated history with streaming response
    """
    # Add user message to history
    history = history + [{"role": "user", "content": message}]

    # Perform RAG retrieval if enabled
    rag_context = None
    retrieved_docs = None

    if rag_enabled and retriever and context_builder:
        try:
            # Retrieve relevant documents
            retrieved_docs = retriever.retrieve(message)

            if retrieved_docs:
                # Build context
                context_result = context_builder.build_context(retrieved_docs, message)
                rag_context = context_result['context']
        except Exception as e:
            # Continue without RAG if retrieval fails
            print(f"RAG retrieval error: {e}")

    # Build messages for LLM
    messages = []

    # Add RAG context as system message if available
    if rag_context:
        messages.append(Message(
            role="system",
            content="You are a helpful assistant. Use the following context to answer the user's question:\n\n" + rag_context
        ))

    # Convert history to messages (exclude current user message which is last)
    for msg in history[:-1]:
        role = msg.get("role")
        content = _content_to_text(msg.get("content"))
        if role and content != "":
            messages.append(Message(role=role, content=content))

    # Add current user message
    messages.append(Message(role="user", content=message))

    # Stream response from LLM
    assistant_response = ""

    try:
        for chunk in llm_client.generate_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        ):
            assistant_response += chunk
            # Append/update assistant message with accumulated response
            if history and history[-1].get("role") == "assistant":
                history[-1] = {"role": "assistant", "content": assistant_response}
            else:
                history = history + [{"role": "assistant", "content": assistant_response}]
            yield history

    except Exception as e:
        # Show error in chat
        error_msg = f"Error: {str(e)}"
        history = history + [{"role": "assistant", "content": error_msg}]
        yield history


def format_rag_sources(documents: List[Any]) -> str:
    """
    Format retrieved documents as a sources list.

    Args:
        documents: List of retrieved documents

    Returns:
        Formatted string with sources
    """
    if not documents:
        return ""

    sources = "\n\n---\n**Sources:**\n"
    for i, doc in enumerate(documents, 1):
        sources += f"{i}. [{doc.title or 'Untitled'}]({doc.source_url}) (score: {doc.score:.3f})\n"

    return sources


def create_chat_history_from_session(session_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Recreate Gradio chat history from saved session data.

    Args:
        session_data: Saved session dictionary

    Returns:
        Gradio chat history format
    """
    return [
        {"role": msg.get("role", ""), "content": msg.get("content", "")}
        for msg in session_data.get("messages", [])
        if msg.get("role") and "content" in msg
    ]


def save_chat_session(history: List[Dict[str, str]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save chat session to a dictionary format.

    Args:
        history: Gradio chat history
        metadata: Additional metadata (model, temperature, etc.)

    Returns:
        Session data dictionary
    """
    messages = []
    for msg in history:
        role = msg.get("role")
        content = _content_to_text(msg.get("content"))
        if role and content != "":
            messages.append({"role": role, "content": content})

    return {
        "messages": messages,
        "metadata": metadata,
        "timestamp": None  # Will be set by caller
    }
