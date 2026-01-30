"""
Configuration tab - API keys and system settings management.
"""
import gradio as gr
import json
import platform
from pathlib import Path
from typing import Dict, Any

from ...config import get_config, reload_config
from ..utils.validation import check_ollama_connection


def create_config_tab() -> gr.Tab:
    """
    Create the Configuration tab.

    Returns:
        Gradio Tab component
    """
    config = get_config()

    with gr.Tab("Configuration") as tab:
        gr.Markdown("## Configuration & Settings")
        gr.Markdown("Manage API keys, default settings, and view system information.")

        with gr.Tabs():
            # API Keys Tab
            with gr.Tab("API Keys"):
                gr.Markdown("### API Keys")
                gr.Markdown("Configure API keys for external services. Keys are stored in environment variables or config file.")

                with gr.Column():
                    anthropic_key_input = gr.Textbox(
                        label="Anthropic API Key",
                        type="password",
                        placeholder="sk-ant-...",
                        value=config.anthropic_api_key or "",
                        info="For Claude models"
                    )

                    mistral_key_input = gr.Textbox(
                        label="Mistral API Key",
                        type="password",
                        placeholder="Enter Mistral API key",
                        value=config.mistral_api_key or "",
                        info="For embeddings"
                    )

                    with gr.Row():
                        save_keys_btn = gr.Button("💾 Save API Keys", variant="primary")
                        clear_keys_btn = gr.Button("🗑️ Clear Keys", variant="stop")

                    keys_status = gr.Textbox(
                        label="Status",
                        value="",
                        interactive=False,
                        visible=False
                    )

                gr.Markdown("---")
                gr.Markdown(
                    """
                    **Note:** API keys are saved to `~/.ai-workbench/.env` file.
                    You can also set them as environment variables:
                    ```bash
                    export WORKBENCH_ANTHROPIC_API_KEY=sk-ant-...
                    export WORKBENCH_MISTRAL_API_KEY=...
                    ```
                    """
                )

            # Default Settings Tab
            with gr.Tab("Default Settings"):
                gr.Markdown("### Default Settings")
                gr.Markdown("Configure default values for various operations.")

                with gr.Accordion("Crawler Defaults", open=True):
                    crawler_max_depth = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=config.crawler_max_depth,
                        step=1,
                        label="Max Depth"
                    )

                    crawler_max_pages = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=config.crawler_max_pages,
                        step=10,
                        label="Max Pages"
                    )

                with gr.Accordion("Embeddings Defaults", open=True):
                    chunk_size = gr.Slider(
                        minimum=100,
                        maximum=1000,
                        value=config.chunk_size,
                        step=50,
                        label="Chunk Size (tokens)"
                    )

                    chunk_overlap = gr.Slider(
                        minimum=0,
                        maximum=200,
                        value=config.chunk_overlap,
                        step=10,
                        label="Chunk Overlap (tokens)"
                    )

                with gr.Accordion("RAG Defaults", open=True):
                    rag_top_k = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=config.rag_top_k,
                        step=1,
                        label="Top K Results"
                    )

                    rag_score_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=config.rag_score_threshold,
                        step=0.05,
                        label="Score Threshold"
                    )

                with gr.Accordion("Chat Defaults", open=True):
                    chat_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=config.chat_temperature,
                        step=0.1,
                        label="Temperature"
                    )

                    chat_max_tokens = gr.Slider(
                        minimum=500,
                        maximum=8000,
                        value=config.chat_max_tokens,
                        step=500,
                        label="Max Tokens"
                    )

                with gr.Row():
                    save_settings_btn = gr.Button("💾 Save Settings", variant="primary")
                    reset_settings_btn = gr.Button("🔄 Reset to Defaults", variant="secondary")

                settings_status = gr.Textbox(
                    label="Status",
                    value="",
                    interactive=False,
                    visible=False
                )

            # System Info Tab
            with gr.Tab("System Info"):
                gr.Markdown("### System Information")

                with gr.Group():
                    gr.Markdown("**Platform**")

                    platform_info = gr.Textbox(
                        label="Platform",
                        value=platform.system(),
                        interactive=False
                    )

                    platform_version = gr.Textbox(
                        label="Platform Version",
                        value=platform.version(),
                        interactive=False
                    )

                    python_version = gr.Textbox(
                        label="Python Version",
                        value=platform.python_version(),
                        interactive=False
                    )

                gr.Markdown("---")
                with gr.Group():
                    gr.Markdown("**Directories**")

                    output_dir = gr.Textbox(
                        label="Output Directory",
                        value=str(config.default_output_dir),
                        interactive=False
                    )

                    vector_store_dir = gr.Textbox(
                        label="Vector Stores",
                        value=str(config.vector_store_path),
                        interactive=False
                    )

                gr.Markdown("---")
                with gr.Group():
                    gr.Markdown("**Storage Usage**")

                    def get_storage_info():
                        """Get storage usage information."""
                        try:
                            # Vector stores
                            vector_dbs = []
                            if config.vector_store_path.exists():
                                vector_dbs = [d.name for d in config.vector_store_path.iterdir() if d.is_dir()]

                            # Output files
                            output_files = 0
                            if config.default_output_dir.exists():
                                output_files = len(list(config.default_output_dir.glob("*")))

                            return (
                                f"{len(vector_dbs)} databases",
                                f"{output_files} files"
                            )
                        except Exception as e:
                            return "Error", "Error"

                    vector_db_count = gr.Textbox(
                        label="Vector Databases",
                        value=get_storage_info()[0],
                        interactive=False
                    )

                    output_files_count = gr.Textbox(
                        label="Output Files",
                        value=get_storage_info()[1],
                        interactive=False
                    )

                    refresh_storage_btn = gr.Button("🔄 Refresh", size="sm")

                gr.Markdown("---")
                with gr.Group():
                    gr.Markdown("**External Services**")

                    def check_services():
                        """Check status of external services."""
                        # Check Ollama
                        ollama_running, ollama_msg = check_ollama_connection(config.ollama_base_url)
                        ollama_status_text = "✓ Running" if ollama_running else f"✗ Not running: {ollama_msg}"

                        # Check Anthropic (just check if key is set)
                        anthropic_status_text = "✓ API key set" if config.anthropic_api_key else "✗ API key not set"

                        # Check Mistral
                        mistral_status_text = "✓ API key set" if config.mistral_api_key else "✗ API key not set"

                        return anthropic_status_text, mistral_status_text, ollama_status_text

                    anthropic_service_status = gr.Textbox(
                        label="Anthropic",
                        value=check_services()[0],
                        interactive=False
                    )

                    mistral_service_status = gr.Textbox(
                        label="Mistral",
                        value=check_services()[1],
                        interactive=False
                    )

                    ollama_service_status = gr.Textbox(
                        label="Ollama",
                        value=check_services()[2],
                        interactive=False
                    )

                    refresh_services_btn = gr.Button("🔄 Refresh", size="sm")

        # Event handlers
        def save_api_keys(anthropic_key, mistral_key):
            """Save API keys to config file."""
            try:
                config_dir = Path.home() / ".ai-workbench"
                config_dir.mkdir(parents=True, exist_ok=True)

                env_file = config_dir / ".env"

                # Read existing .env
                env_vars = {}
                if env_file.exists():
                    with open(env_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                env_vars[key.strip()] = value.strip()

                # Update keys
                if anthropic_key:
                    env_vars['WORKBENCH_ANTHROPIC_API_KEY'] = anthropic_key
                if mistral_key:
                    env_vars['WORKBENCH_MISTRAL_API_KEY'] = mistral_key

                # Write back
                with open(env_file, 'w') as f:
                    for key, value in env_vars.items():
                        f.write(f"{key}={value}\n")

                # Reload config
                reload_config()

                return {keys_status: gr.update(value="✓ API keys saved successfully", visible=True)}

            except Exception as e:
                return {keys_status: gr.update(value=f"✗ Error saving keys: {str(e)}", visible=True)}

        def clear_api_keys():
            """Clear API keys."""
            try:
                config_dir = Path.home() / ".ai-workbench"
                env_file = config_dir / ".env"

                if env_file.exists():
                    # Read and filter out API keys
                    env_vars = {}
                    with open(env_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                if 'API_KEY' not in key:
                                    env_vars[key.strip()] = value.strip()

                    # Write back
                    with open(env_file, 'w') as f:
                        for key, value in env_vars.items():
                            f.write(f"{key}={value}\n")

                # Reload config
                reload_config()

                return {
                    anthropic_key_input: "",
                    mistral_key_input: "",
                    keys_status: gr.update(value="✓ API keys cleared", visible=True)
                }

            except Exception as e:
                return {keys_status: gr.update(value=f"✗ Error: {str(e)}", visible=True)}

        def save_settings(crawler_depth, crawler_pages, chunk_sz, chunk_ovlp, rag_k, rag_threshold, temp, max_tok):
            """Save default settings."""
            try:
                config_dir = Path.home() / ".ai-workbench"
                config_dir.mkdir(parents=True, exist_ok=True)

                config_file = config_dir / "config.json"

                settings = {
                    "crawler_max_depth": int(crawler_depth),
                    "crawler_max_pages": int(crawler_pages),
                    "chunk_size": int(chunk_sz),
                    "chunk_overlap": int(chunk_ovlp),
                    "rag_top_k": int(rag_k),
                    "rag_score_threshold": float(rag_threshold),
                    "chat_temperature": float(temp),
                    "chat_max_tokens": int(max_tok)
                }

                with open(config_file, 'w') as f:
                    json.dump(settings, f, indent=2)

                # Reload config
                reload_config()

                return {settings_status: gr.update(value="✓ Settings saved successfully", visible=True)}

            except Exception as e:
                return {settings_status: gr.update(value=f"✗ Error: {str(e)}", visible=True)}

        def reset_to_defaults():
            """Reset settings to defaults."""
            return {
                crawler_max_depth: 2,
                crawler_max_pages: 100,
                chunk_size: 500,
                chunk_overlap: 50,
                rag_top_k: 5,
                rag_score_threshold: 0.7,
                chat_temperature: 0.7,
                chat_max_tokens: 4000,
                settings_status: gr.update(value="✓ Reset to defaults (not saved)", visible=True)
            }

        def refresh_storage_info():
            """Refresh storage usage information."""
            info = get_storage_info()
            return {
                vector_db_count: info[0],
                output_files_count: info[1]
            }

        def refresh_service_status():
            """Refresh service status."""
            status = check_services()
            return {
                anthropic_service_status: status[0],
                mistral_service_status: status[1],
                ollama_service_status: status[2]
            }

        # Connect events
        save_keys_btn.click(
            fn=save_api_keys,
            inputs=[anthropic_key_input, mistral_key_input],
            outputs=[keys_status]
        )

        clear_keys_btn.click(
            fn=clear_api_keys,
            outputs=[anthropic_key_input, mistral_key_input, keys_status]
        )

        save_settings_btn.click(
            fn=save_settings,
            inputs=[crawler_max_depth, crawler_max_pages, chunk_size, chunk_overlap, rag_top_k, rag_score_threshold, chat_temperature, chat_max_tokens],
            outputs=[settings_status]
        )

        reset_settings_btn.click(
            fn=reset_to_defaults,
            outputs=[crawler_max_depth, crawler_max_pages, chunk_size, chunk_overlap, rag_top_k, rag_score_threshold, chat_temperature, chat_max_tokens, settings_status]
        )

        refresh_storage_btn.click(
            fn=refresh_storage_info,
            outputs=[vector_db_count, output_files_count]
        )

        refresh_services_btn.click(
            fn=refresh_service_status,
            outputs=[anthropic_service_status, mistral_service_status, ollama_service_status]
        )

    return tab
