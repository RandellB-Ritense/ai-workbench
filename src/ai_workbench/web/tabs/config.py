"""
Configuration tab - API keys and system settings management.
"""
import gradio as gr
import json
import platform
from pathlib import Path
from typing import Dict, Any, Optional

from ...config import get_config, reload_config
from ...project import project_paths_from_state
from ..utils.validation import check_ollama_connection


def create_config_tab(project_state: gr.State) -> gr.Tab:
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
                    mistral_key_input = gr.Textbox(
                        label="Mistral API Key",
                        type="password",
                        placeholder="Set in .env",
                        value=config.mistral_api_key or "",
                        info="Read-only; set in .env",
                        interactive=False
                    )

                    refresh_keys_btn = gr.Button("🔄 Refresh Keys", size="sm")

                    keys_status = gr.Textbox(
                        label="Status",
                        value="",
                        interactive=False,
                        visible=False
                    )

                gr.Markdown("---")
                gr.Markdown(
                    """
                    **Note:** API keys are read from the `.env` file only.
                    Set them in your `.env` file (or environment variables) and click Refresh:
                    ```bash
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
                    save_settings_btn = gr.Button("💾 Save Settings", variant="primary", interactive=False)
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

                    project_root_dir = gr.Textbox(
                        label="Active Project Root",
                        value="None",
                        interactive=False
                    )

                    output_dir = gr.Textbox(
                        label="Output Directory",
                        value="Project required",
                        interactive=False
                    )

                    vector_store_dir = gr.Textbox(
                        label="Vector Stores",
                        value="Project required",
                        interactive=False
                    )

                    job_store_dir = gr.Textbox(
                        label="Job Storage",
                        value="Project required",
                        interactive=False
                    )

                gr.Markdown("---")
                with gr.Group():
                    gr.Markdown("**Storage Usage**")

                    def get_storage_info(project: Optional[Dict]):
                        """Get storage usage information."""
                        try:
                            if not project:
                                return "No project", "No project"
                            paths = project_paths_from_state(project)
                            # Vector stores
                            vector_dbs = []
                            if paths.vector_store_dir.exists():
                                vector_dbs = [d.name for d in paths.vector_store_dir.iterdir() if d.is_dir()]

                            # Output files
                            output_files = 0
                            if paths.output_dir.exists():
                                output_files = len(list(paths.output_dir.glob("*")))

                            return (
                                f"{len(vector_dbs)} databases",
                                f"{output_files} files"
                            )
                        except Exception as e:
                            return "Error", "Error"

                    vector_db_count = gr.Textbox(
                        label="Vector Databases",
                        value=get_storage_info(None)[0],
                        interactive=False
                    )

                    output_files_count = gr.Textbox(
                        label="Output Files",
                        value=get_storage_info(None)[1],
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

                        # Check Mistral
                        mistral_status_text = "✓ API key set" if config.mistral_api_key else "✗ API key not set"

                        return mistral_status_text, ollama_status_text

                    mistral_service_status = gr.Textbox(
                        label="Mistral",
                        value=check_services()[0],
                        interactive=False
                    )

                    ollama_service_status = gr.Textbox(
                        label="Ollama",
                        value=check_services()[1],
                        interactive=False
                    )

                    refresh_services_btn = gr.Button("🔄 Refresh", size="sm")

        # Event handlers
        def refresh_api_keys():
            """Reload API keys from .env."""
            try:
                cfg = reload_config()
                return {
                    mistral_key_input: cfg.mistral_api_key or "",
                    keys_status: gr.update(value="✓ API keys refreshed from .env", visible=True),
                }
            except Exception as e:
                return {keys_status: gr.update(value=f"✗ Error refreshing keys: {str(e)}", visible=True)}

        def save_settings(crawler_depth, crawler_pages, chunk_sz, chunk_ovlp, rag_k, rag_threshold, temp, max_tok, project: Optional[Dict]):
            """Save default settings."""
            try:
                if not project:
                    return {settings_status: gr.update(value="✗ Create a project before saving settings", visible=True)}
                config_dir = project_paths_from_state(project).config_dir
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

        def refresh_storage_info(project: Optional[Dict]):
            """Refresh storage usage information."""
            info = get_storage_info(project)
            return {
                vector_db_count: info[0],
                output_files_count: info[1]
            }

        def apply_project_paths(project: Optional[Dict]):
            if not project:
                return {
                    project_root_dir: "None",
                    output_dir: "Project required",
                    vector_store_dir: "Project required",
                    job_store_dir: "Project required",
                    vector_db_count: "No project",
                    output_files_count: "No project",
                    save_settings_btn: gr.update(interactive=False),
                }
            paths = project_paths_from_state(project)
            info = get_storage_info(project)
            return {
                project_root_dir: str(paths.root),
                output_dir: str(paths.output_dir),
                vector_store_dir: str(paths.vector_store_dir),
                job_store_dir: str(paths.job_dir),
                vector_db_count: info[0],
                output_files_count: info[1],
                save_settings_btn: gr.update(interactive=True),
            }

        def refresh_service_status():
            """Refresh service status."""
            status = check_services()
            return {
                mistral_service_status: status[0],
                ollama_service_status: status[1]
            }

        # Connect events
        refresh_keys_btn.click(
            fn=refresh_api_keys,
            outputs=[mistral_key_input, keys_status]
        )

        save_settings_btn.click(
            fn=save_settings,
            inputs=[crawler_max_depth, crawler_max_pages, chunk_size, chunk_overlap, rag_top_k, rag_score_threshold, chat_temperature, chat_max_tokens, project_state],
            outputs=[settings_status]
        )

        reset_settings_btn.click(
            fn=reset_to_defaults,
            outputs=[crawler_max_depth, crawler_max_pages, chunk_size, chunk_overlap, rag_top_k, rag_score_threshold, chat_temperature, chat_max_tokens, settings_status]
        )

        refresh_storage_btn.click(
            fn=refresh_storage_info,
            inputs=[project_state],
            outputs=[vector_db_count, output_files_count]
        )

        refresh_services_btn.click(
            fn=refresh_service_status,
            outputs=[mistral_service_status, ollama_service_status]
        )

        project_state.change(
            fn=apply_project_paths,
            inputs=[project_state],
            outputs=[project_root_dir, output_dir, vector_store_dir, job_store_dir, vector_db_count, output_files_count, save_settings_btn]
        )

    return tab
