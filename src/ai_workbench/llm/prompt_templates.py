"""
Prompt templates for LLM interactions.
"""
from typing import Optional


class PromptTemplates:
    """Collection of prompt templates for different use cases."""

    @staticmethod
    def system_message(
        include_rag: bool = True,
        custom_instructions: Optional[str] = None,
    ) -> str:
        """
        Generate system message for the LLM.

        Args:
            include_rag: Whether RAG context will be provided
            custom_instructions: Optional custom instructions

        Returns:
            System message string
        """
        base = "You are a helpful AI assistant."

        if include_rag:
            base += (
                " You have access to documentation that will be provided in the user's messages. "
                "Use this documentation to answer questions accurately. "
                "If the answer is not in the documentation, say so clearly."
            )

        if custom_instructions:
            base += f"\n\n{custom_instructions}"

        return base

    @staticmethod
    def rag_context_message(
        context: str,
        query: str,
    ) -> str:
        """
        Format RAG context for inclusion in user message.

        Args:
            context: Retrieved documentation context
            query: User's query

        Returns:
            Formatted message with context and query
        """
        return f"""I have a question about the documentation.

<documentation>
{context}
</documentation>

Question: {query}

Please answer based on the documentation provided above."""

    @staticmethod
    def rag_system_message() -> str:
        """
        System message specifically for RAG mode.

        Returns:
            RAG-optimized system message
        """
        return """You are a helpful AI assistant with access to documentation.

Your task is to answer questions based on the documentation provided in each user message.

Guidelines:
- Base your answers on the documentation provided
- If the answer is not in the documentation, clearly state that
- Cite specific sections when relevant
- Be concise and accurate
- If you're unsure, say so rather than guessing"""

    @staticmethod
    def no_rag_system_message() -> str:
        """
        System message for non-RAG mode.

        Returns:
            Standard system message
        """
        return """You are a helpful AI assistant.

Answer questions to the best of your ability using your training data.
Be helpful, accurate, and concise."""

    @staticmethod
    def experimentation_system_message() -> str:
        """
        System message for experimentation playground.

        Returns:
            Experimentation-focused system message
        """
        return """You are a helpful AI assistant in an experimentation playground.

The user is testing different LLM configurations, RAG systems, and MCP tools.
Be helpful and provide clear, informative responses so the user can evaluate your performance."""

    @staticmethod
    def format_conversation_history(
        messages: list,
        max_messages: int = 10,
    ) -> str:
        """
        Format conversation history for display.

        Args:
            messages: List of message dictionaries
            max_messages: Maximum messages to include

        Returns:
            Formatted conversation history
        """
        if not messages:
            return "No conversation history."

        recent = messages[-max_messages:] if len(messages) > max_messages else messages

        formatted = []
        for msg in recent:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")

        return "\n\n".join(formatted)

    @staticmethod
    def build_rag_prompt(
        query: str,
        context: str,
        include_sources: bool = True,
    ) -> str:
        """
        Build a complete RAG prompt.

        Args:
            query: User's question
            context: Retrieved documentation
            include_sources: Whether to emphasize source citations

        Returns:
            Complete RAG prompt
        """
        prompt = f"""Based on the following documentation, please answer this question: {query}

Documentation:
{context}

"""
        if include_sources:
            prompt += "Please cite specific sections from the documentation in your answer."

        return prompt
