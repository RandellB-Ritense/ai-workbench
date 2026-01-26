"""
Conversation session management for chatbot.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class ChatMessage:
    """A message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class ChatSession:
    """
    Manages conversation history for a chat session.

    Handles message storage, history pruning, and session persistence.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        max_history: int = 50,
    ):
        """
        Initialize chat session.

        Args:
            session_id: Optional session identifier
            max_history: Maximum messages to keep in history
        """
        self.session_id = session_id or self._generate_session_id()
        self.max_history = max_history
        self.messages: List[ChatMessage] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "model": None,
            "rag_enabled": False,
        }

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a message to the conversation.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata (tokens, RAG docs, etc.)
        """
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        self.messages.append(message)

        # Prune history if needed
        if len(self.messages) > self.max_history:
            self._prune_history()

    def _prune_history(self):
        """Prune old messages, keeping system messages."""
        # Keep system messages and recent messages
        system_messages = [m for m in self.messages if m.role == "system"]
        recent_messages = [
            m for m in self.messages if m.role != "system"
        ][-self.max_history + len(system_messages):]

        self.messages = system_messages + recent_messages

    def get_messages(
        self,
        include_system: bool = True,
        last_n: Optional[int] = None,
    ) -> List[ChatMessage]:
        """
        Get conversation messages.

        Args:
            include_system: Whether to include system messages
            last_n: Optional limit to last N messages

        Returns:
            List of ChatMessage objects
        """
        messages = self.messages

        if not include_system:
            messages = [m for m in messages if m.role != "system"]

        if last_n is not None:
            messages = messages[-last_n:]

        return messages

    def clear_history(self, keep_system: bool = True):
        """
        Clear conversation history.

        Args:
            keep_system: Whether to keep system messages
        """
        if keep_system:
            self.messages = [m for m in self.messages if m.role == "system"]
        else:
            self.messages = []

    def save(self, file_path: Path):
        """
        Save session to a JSON file.

        Args:
            file_path: Path to save session
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)

        session_data = {
            "session_id": self.session_id,
            "metadata": self.metadata,
            "messages": [asdict(m) for m in self.messages],
            "saved_at": datetime.now().isoformat(),
        }

        with open(file_path, "w") as f:
            json.dump(session_data, f, indent=2)

    @classmethod
    def load(cls, file_path: Path) -> "ChatSession":
        """
        Load session from a JSON file.

        Args:
            file_path: Path to load session from

        Returns:
            ChatSession object
        """
        with open(file_path, "r") as f:
            session_data = json.load(f)

        session = cls(
            session_id=session_data["session_id"],
        )
        session.metadata = session_data.get("metadata", {})

        # Load messages
        for msg_data in session_data.get("messages", []):
            session.messages.append(
                ChatMessage(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    timestamp=msg_data["timestamp"],
                    metadata=msg_data.get("metadata"),
                )
            )

        return session

    def get_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.

        Returns:
            Dictionary with session stats
        """
        user_messages = [m for m in self.messages if m.role == "user"]
        assistant_messages = [m for m in self.messages if m.role == "assistant"]

        return {
            "session_id": self.session_id,
            "total_messages": len(self.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "created_at": self.metadata.get("created_at"),
            "model": self.metadata.get("model"),
            "rag_enabled": self.metadata.get("rag_enabled"),
        }

    def update_metadata(self, **kwargs):
        """Update session metadata."""
        self.metadata.update(kwargs)
