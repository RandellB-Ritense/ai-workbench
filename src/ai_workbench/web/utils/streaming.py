"""
Streaming utilities for Gradio chat interface.
"""
from typing import List, Tuple, Optional, Generator, Dict, Any
from dataclasses import dataclass


@dataclass
class Message:
    """A message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str


def build_messages_from_history(history: List[List[str]], system_prompt: Optional[str] = None) -> List[Message]:
    """
    Convert Gradio chat history to LLM message format.

    Args:
        history: Gradio chat history format [[user_msg, assistant_msg], ...]
        system_prompt: Optional system prompt to prepend

    Returns:
        List of Message objects for LLM
    """
    messages = []

    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))

    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append(Message(role="user", content=user_msg))
        if assistant_msg:
            messages.append(Message(role="assistant", content=assistant_msg))

    return messages


def chat_stream_generator(
    message: str,
    history: List[List[Optional[str]]],
    llm_client,
    retriever=None,
    context_builder=None,
    rag_enabled: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 4000,
) -> Generator[List[List[Optional[str]]], None, None]:
    """
    Generator function for streaming chat responses in Gradio.

    Args:
        message: User's message
        history: Chat history
        llm_client: LLM client instance (AnthropicClient or OllamaClient)
        retriever: RAG retriever instance (optional)
        context_builder: Context builder instance (optional)
        rag_enabled: Whether to use RAG
        temperature: LLM temperature
        max_tokens: Maximum tokens in response

    Yields:
        Updated history with streaming response
    """
    # Add user message to history
    history = history + [[message, None]]

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

    # Convert history to messages
    for user_msg, assistant_msg in history[:-1]:  # Exclude the current message
        if user_msg:
            messages.append(Message(role="user", content=user_msg))
        if assistant_msg:
            messages.append(Message(role="assistant", content=assistant_msg))

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
            # Update the last message in history with accumulated response
            history[-1] = [message, assistant_response]
            yield history

    except Exception as e:
        # Show error in chat
        error_msg = f"Error: {str(e)}"
        history[-1] = [message, error_msg]
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


def create_chat_history_from_session(session_data: Dict[str, Any]) -> List[List[Optional[str]]]:
    """
    Recreate Gradio chat history from saved session data.

    Args:
        session_data: Saved session dictionary

    Returns:
        Gradio chat history format
    """
    history = []

    messages = session_data.get("messages", [])

    # Convert message pairs to Gradio format
    i = 0
    while i < len(messages):
        user_msg = None
        assistant_msg = None

        if i < len(messages) and messages[i]["role"] == "user":
            user_msg = messages[i]["content"]
            i += 1

        if i < len(messages) and messages[i]["role"] == "assistant":
            assistant_msg = messages[i]["content"]
            i += 1

        if user_msg or assistant_msg:
            history.append([user_msg, assistant_msg])

    return history


def save_chat_session(history: List[List[Optional[str]]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save chat session to a dictionary format.

    Args:
        history: Gradio chat history
        metadata: Additional metadata (model, temperature, etc.)

    Returns:
        Session data dictionary
    """
    messages = []

    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})

    return {
        "messages": messages,
        "metadata": metadata,
        "timestamp": None  # Will be set by caller
    }
