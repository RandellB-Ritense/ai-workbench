"""
Interactive REPL interface for chatbot.
"""
from pathlib import Path
from typing import Optional
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live

from ai_workbench.chatbot.session import ChatSession
from ai_workbench.llm.base import LLMProvider, Message
from ai_workbench.llm.prompt_templates import PromptTemplates
from ai_workbench.rag.retriever import RAGRetriever
from ai_workbench.rag.context_builder import ContextBuilder


class InteractiveChatbot:
    """
    Interactive chatbot with REPL interface.

    Provides a rich terminal interface for chatting with LLMs
    with optional RAG augmentation.
    """

    def __init__(
        self,
        llm: LLMProvider,
        retriever: Optional[RAGRetriever] = None,
        context_builder: Optional[ContextBuilder] = None,
        rag_enabled: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ):
        """
        Initialize interactive chatbot.

        Args:
            llm: LLM provider instance
            retriever: Optional RAG retriever
            context_builder: Optional context builder
            rag_enabled: Whether RAG is initially enabled
            temperature: LLM temperature
            max_tokens: Maximum tokens per response
        """
        self.llm = llm
        self.retriever = retriever
        self.context_builder = context_builder
        self.rag_enabled = rag_enabled and retriever is not None
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.console = Console()
        self.session = ChatSession()
        self.prompt_session = PromptSession(history=InMemoryHistory())

        # Track last RAG sources for /sources command
        self.last_rag_docs = []

        # Initialize session metadata
        self.session.update_metadata(
            model=self.llm.get_model_info().get("id"),
            rag_enabled=self.rag_enabled,
        )

        # Add system message
        if self.rag_enabled:
            system_msg = PromptTemplates.rag_system_message()
        else:
            system_msg = PromptTemplates.experimentation_system_message()

        self.session.add_message("system", system_msg)

    def run(self):
        """Start the interactive chat loop."""
        self._print_welcome()

        while True:
            try:
                # Get user input
                user_input = self.prompt_session.prompt("\n[You] > ")

                if not user_input.strip():
                    continue

                # Check for special commands
                if user_input.startswith("/"):
                    if not self._handle_command(user_input):
                        break  # Exit if command returns False
                    continue

                # Process user message
                self._process_user_message(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use /exit to quit[/yellow]")
                continue
            except EOFError:
                break

        self._print_goodbye()

    def _print_welcome(self):
        """Print welcome message."""
        model_info = self.llm.get_model_info()

        welcome = f"""[bold cyan]AI Workbench - Interactive Chat[/bold cyan]

[bold]Model:[/bold] {model_info.get('name', 'Unknown')} ({self.llm.get_provider_name()})
[bold]RAG:[/bold] {'[green]Enabled[/green]' if self.rag_enabled else '[yellow]Disabled[/yellow]'}
[bold]Temperature:[/bold] {self.temperature}

[dim]Type /help for available commands[/dim]
"""
        self.console.print(Panel(welcome, border_style="cyan"))

    def _print_goodbye(self):
        """Print goodbye message."""
        stats = self.session.get_stats()
        self.console.print(f"\n[cyan]Session ended. Messages: {stats['total_messages']}[/cyan]")
        self.console.print("[dim]Use /save <file> to save this conversation[/dim]\n")

    def _handle_command(self, command: str) -> bool:
        """
        Handle special commands.

        Args:
            command: Command string (starting with /)

        Returns:
            False to exit, True to continue
        """
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "/help":
            self._show_help()
        elif cmd == "/exit" or cmd == "/quit":
            return False
        elif cmd == "/clear":
            self.session.clear_history(keep_system=True)
            self.console.print("[green]✓[/green] Conversation history cleared")
        elif cmd == "/rag":
            self._toggle_rag(args)
        elif cmd == "/model":
            self.console.print(f"[cyan]Current model:[/cyan] {self.llm.get_model_info()}")
        elif cmd == "/sources":
            self._show_sources()
        elif cmd == "/save":
            self._save_session(args)
        elif cmd == "/load":
            self._load_session(args)
        elif cmd == "/stats":
            self._show_stats()
        else:
            self.console.print(f"[red]Unknown command:[/red] {cmd}")
            self.console.print("[dim]Type /help for available commands[/dim]")

        return True

    def _show_help(self):
        """Show available commands."""
        help_text = """[bold cyan]Available Commands:[/bold cyan]

[bold]/help[/bold]         - Show this help message
[bold]/exit[/bold]         - Exit the chat
[bold]/clear[/bold]        - Clear conversation history
[bold]/rag on|off[/bold]   - Toggle RAG on/off
[bold]/sources[/bold]      - Show last RAG sources used
[bold]/model[/bold]        - Show current model info
[bold]/stats[/bold]        - Show session statistics
[bold]/save <file>[/bold]  - Save conversation to file
[bold]/load <file>[/bold]  - Load conversation from file
"""
        self.console.print(Panel(help_text, border_style="cyan"))

    def _toggle_rag(self, arg: str):
        """Toggle RAG on/off."""
        if not self.retriever:
            self.console.print("[yellow]RAG not available (no vector database)[/yellow]")
            return

        arg = arg.lower().strip()
        if arg == "on":
            self.rag_enabled = True
            self.console.print("[green]✓[/green] RAG enabled")
        elif arg == "off":
            self.rag_enabled = False
            self.console.print("[yellow]RAG disabled[/yellow]")
        else:
            status = "[green]enabled[/green]" if self.rag_enabled else "[yellow]disabled[/yellow]"
            self.console.print(f"RAG is currently {status}")
            self.console.print("[dim]Use: /rag on or /rag off[/dim]")

        self.session.update_metadata(rag_enabled=self.rag_enabled)

    def _show_sources(self):
        """Show last RAG sources."""
        if not self.last_rag_docs:
            self.console.print("[yellow]No RAG sources in last response[/yellow]")
            return

        self.console.print(f"\n[bold cyan]RAG Sources ({len(self.last_rag_docs)} documents):[/bold cyan]\n")
        for i, doc in enumerate(self.last_rag_docs, 1):
            self.console.print(f"  {i}. [bold]{doc.get('title', 'Untitled')}[/bold]")
            self.console.print(f"     Source: {doc.get('source_url')}")
            self.console.print(f"     Score: {doc.get('score', 0):.3f}")
            self.console.print()

    def _save_session(self, file_path: str):
        """Save session to file."""
        if not file_path:
            self.console.print("[red]Usage: /save <file>[/red]")
            return

        try:
            self.session.save(Path(file_path))
            self.console.print(f"[green]✓[/green] Session saved to {file_path}")
        except Exception as e:
            self.console.print(f"[red]Error saving session:[/red] {e}")

    def _load_session(self, file_path: str):
        """Load session from file."""
        if not file_path:
            self.console.print("[red]Usage: /load <file>[/red]")
            return

        try:
            self.session = ChatSession.load(Path(file_path))
            self.console.print(f"[green]✓[/green] Session loaded from {file_path}")
        except Exception as e:
            self.console.print(f"[red]Error loading session:[/red] {e}")

    def _show_stats(self):
        """Show session statistics."""
        stats = self.session.get_stats()
        stats_text = f"""[bold cyan]Session Statistics:[/bold cyan]

[bold]Session ID:[/bold] {stats['session_id']}
[bold]Total Messages:[/bold] {stats['total_messages']}
[bold]User Messages:[/bold] {stats['user_messages']}
[bold]Assistant Messages:[/bold] {stats['assistant_messages']}
[bold]Model:[/bold] {stats['model']}
[bold]RAG:[/bold] {'Enabled' if stats['rag_enabled'] else 'Disabled'}
"""
        self.console.print(Panel(stats_text, border_style="cyan"))

    def _process_user_message(self, user_input: str):
        """Process and respond to user message."""
        # Add user message to session
        self.session.add_message("user", user_input)

        # Retrieve RAG context if enabled
        rag_context = None
        if self.rag_enabled and self.retriever:
            with self.console.status("[dim]Retrieving relevant documents...[/dim]"):
                docs = self.retriever.retrieve(user_input)
                if docs:
                    result = self.context_builder.build_context(docs, user_input)
                    rag_context = result["context"]
                    self.last_rag_docs = result["documents_used"]

        # Build messages for LLM
        messages = []

        # Add system message
        system_messages = [m for m in self.session.get_messages() if m.role == "system"]
        if system_messages:
            messages.append(Message(
                role="system",
                content=system_messages[0].content,
            ))

        # Add conversation history (excluding system messages)
        history = self.session.get_messages(include_system=False, last_n=10)
        for msg in history[:-1]:  # Exclude the current user message
            messages.append(Message(
                role=msg.role,
                content=msg.content,
            ))

        # Add current user message with RAG context if available
        if rag_context:
            user_message_with_context = PromptTemplates.rag_context_message(rag_context, user_input)
            messages.append(Message(role="user", content=user_message_with_context))
        else:
            messages.append(Message(role="user", content=user_input))

        # Generate response with streaming
        self.console.print("\n[bold cyan][Assistant][/bold cyan]")
        response_text = ""

        try:
            with Live("", console=self.console, refresh_per_second=10) as live:
                for chunk in self.llm.generate_stream(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ):
                    response_text += chunk
                    live.update(Markdown(response_text))

        except Exception as e:
            self.console.print(f"[red]Error:[/red] {e}")
            return

        # Add assistant response to session
        metadata = {
            "rag_docs_used": len(self.last_rag_docs) if self.rag_enabled else 0,
        }
        self.session.add_message("assistant", response_text, metadata=metadata)
