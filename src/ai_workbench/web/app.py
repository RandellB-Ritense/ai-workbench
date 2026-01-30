"""
Main Gradio web application for AI Workbench.
"""
import gradio as gr
from pathlib import Path

from .tabs.data_collection import create_data_collection_tab
from .tabs.vector_index import create_vector_index_tab
from .tabs.chat import create_chat_tab
from .tabs.config import create_config_tab
from ..config import get_config


def create_app() -> gr.Blocks:
    """
    Create the main Gradio application.

    Returns:
        Gradio Blocks application
    """
    config = get_config()

    # Create Gradio interface
    with gr.Blocks(title="AI Workbench") as app:
        # Header
        gr.Markdown(
            """
            # 🛠️ AI Workbench

            A comprehensive toolkit for web scraping, embeddings, RAG, and AI chat.
            """
        )

        # Tabs
        with gr.Tabs(elem_classes="tab-nav"):
            # Data Collection
            create_data_collection_tab()

            # Vector Index
            create_vector_index_tab()

            # Chat
            create_chat_tab()

            # Configuration
            create_config_tab()

        # Footer
        gr.Markdown(
            """
            ---
            *AI Workbench v0.1.0 - Built with Gradio*
            """
        )

    return app


def launch_app(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
):
    """
    Launch the Gradio web application.

    Args:
        host: Host address to bind to
        port: Port number
        share: Enable Gradio sharing
    """
    app = create_app()

    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
        quiet=False,
        theme=gr.themes.Soft(),
    )
