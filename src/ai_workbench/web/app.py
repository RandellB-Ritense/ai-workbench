"""
Main Gradio web application for AI Workbench.
"""
import gradio as gr
from pathlib import Path
from typing import Dict

from .tabs.data_collection import create_data_collection_tab
from .tabs.vector_index import create_vector_index_tab
from .tabs.chat import create_chat_tab
from .tabs.config import create_config_tab
from ..config import get_config
from ..project import (
    ProjectInfo,
    build_project_paths,
    ensure_project_dirs,
    load_project_from_dir,
    serialize_project_state,
    write_project_metadata,
)


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

        project_state = gr.State(value=None)
        browser_project_state = gr.BrowserState(
            default_value=None,
            storage_key="ai_workbench_active_project",
        )
        recent_projects_state = gr.BrowserState(
            default_value=[],
            storage_key="ai_workbench_recent_projects",
        )

        def create_project(name: str, directory: str) -> Dict:
            """Create a new project and initialize directories."""
            if not name:
                return {
                    project_state: None,
                    browser_project_state: None,
                    project_status: "✗ Project name is required",
                }

            if not directory:
                return {
                    project_state: None,
                    browser_project_state: None,
                    project_status: "✗ Project directory is required",
                }

            try:
                root = Path(directory).expanduser()
                if root.exists() and not root.is_dir():
                    return {
                        project_state: None,
                        browser_project_state: None,
                        project_status: "✗ Project directory must be a folder",
                    }
                project = ProjectInfo(name=name.strip(), root=root)
                paths = build_project_paths(project.root)
                ensure_project_dirs(paths)
                write_project_metadata(project, paths)

                state = serialize_project_state(project)
                return {
                    project_state: state,
                    browser_project_state: state,
                    project_status: f"✓ Project created: {project.name}",
                    active_project_name: project.name,
                    active_project_dir: str(paths.root),
                }
            except Exception as e:
                return {
                    project_state: None,
                    browser_project_state: None,
                    project_status: f"✗ Error creating project: {str(e)}",
                }

        def load_project(directory) -> Dict:
            """Load an existing project from a directory."""
            if isinstance(directory, list):
                directory = directory[0] if directory else ""
            if not directory:
                return {
                    project_state: None,
                    browser_project_state: None,
                    project_status: "✗ Project directory is required",
                }

            try:
                root = Path(directory).expanduser()
                project = load_project_from_dir(root)
                paths = build_project_paths(project.root)
                ensure_project_dirs(paths)

                state = serialize_project_state(project)
                return {
                    project_state: state,
                    browser_project_state: state,
                    project_status: f"✓ Project loaded: {project.name}",
                    active_project_name: project.name,
                    active_project_dir: str(paths.root),
                }
            except Exception as e:
                return {
                    project_state: None,
                    browser_project_state: None,
                    project_status: f"✗ Error loading project: {str(e)}",
                }

        def hydrate_project(stored: Dict) -> Dict:
            """Restore project state from browser storage."""
            if not stored:
                return {
                    project_state: None,
                    project_status: "No active project",
                    active_project_name: "None",
                    active_project_dir: "None",
                }
            try:
                root = Path(stored.get("root", "")).expanduser()
                project = load_project_from_dir(root)
                paths = build_project_paths(project.root)
                ensure_project_dirs(paths)
                state = serialize_project_state(project)
                return {
                    project_state: state,
                    project_status: f"✓ Project loaded: {project.name}",
                    active_project_name: project.name,
                    active_project_dir: str(paths.root),
                }
            except Exception as e:
                return {
                    project_state: None,
                    browser_project_state: None,
                    project_status: f"✗ Error loading saved project: {str(e)}",
                    active_project_name: "None",
                    active_project_dir: "None",
                }

        def _dedupe_recent(recent, new_entry, limit=10):
            cleaned = []
            seen = set()
            for entry in [new_entry] + (recent or []):
                key = entry.get("root")
                if not key or key in seen:
                    continue
                seen.add(key)
                cleaned.append(entry)
                if len(cleaned) >= limit:
                    break
            return cleaned

        def update_recent_projects(project: Dict, recent: list) -> Dict:
            if not project:
                return {}
            new_entry = {"name": project.get("name", "Unknown"), "root": project.get("root", "")}
            updated = _dedupe_recent(recent, new_entry)
            return {
                recent_projects_state: updated,
                recent_projects_dropdown: gr.update(choices=_format_recent(updated)),
            }

        def _format_recent(recent: list) -> list:
            return [f"{item.get('name', 'Unknown')} — {item.get('root', '')}" for item in (recent or [])]

        def load_recent_project(selected: str, recent: list) -> Dict:
            if not selected or not recent:
                return {project_status: "✗ Select a recent project to load"}
            match = None
            for item in recent:
                label = f"{item.get('name', 'Unknown')} — {item.get('root', '')}"
                if label == selected:
                    match = item
                    break
            if not match:
                return {project_status: "✗ Selected project not found"}
            return load_project(match.get("root", ""))

        # Tabs
        with gr.Tabs(elem_classes="tab-nav"):
            # Project Setup
            with gr.Tab("Project"):
                gr.Markdown("## Project Setup (Required)")
                gr.Markdown(
                    "Create a project before using any tools. All files will be stored inside the project directory."
                )

                with gr.Row():
                    project_name_input = gr.Textbox(
                        label="Project Name",
                        placeholder="my-research-project",
                    )
                    project_dir_input = gr.Textbox(
                        label="Project Directory",
                        placeholder="/path/to/project-folder",
                    )

                with gr.Row():
                    create_project_btn = gr.Button("Create Project", variant="primary")

                gr.Markdown("### Load Existing Project")
                load_project_dir_input = gr.Textbox(
                    label="Project Directory",
                    placeholder="/path/to/existing-project",
                )
                load_project_btn = gr.Button("Load Project", variant="secondary")

                gr.Markdown("### Recent Projects")
                recent_projects_dropdown = gr.Dropdown(
                    label="Recent Projects",
                    choices=[],
                    value=None,
                )
                load_recent_btn = gr.Button("Load Recent", variant="secondary")

                project_status = gr.Textbox(
                    label="Status",
                    value="No active project",
                    interactive=False,
                )

                with gr.Row():
                    active_project_name = gr.Textbox(
                        label="Active Project Name",
                        value="None",
                        interactive=False,
                    )
                    active_project_dir = gr.Textbox(
                        label="Active Project Directory",
                        value="None",
                        interactive=False,
                    )

                create_project_btn.click(
                    fn=create_project,
                    inputs=[project_name_input, project_dir_input],
                    outputs=[project_state, browser_project_state, project_status, active_project_name, active_project_dir],
                )

                load_project_btn.click(
                    fn=load_project,
                    inputs=[load_project_dir_input],
                    outputs=[project_state, browser_project_state, project_status, active_project_name, active_project_dir],
                )

                load_recent_btn.click(
                    fn=load_recent_project,
                    inputs=[recent_projects_dropdown, recent_projects_state],
                    outputs=[project_state, browser_project_state, project_status, active_project_name, active_project_dir],
                )

                project_state.change(
                    fn=update_recent_projects,
                    inputs=[project_state, recent_projects_state],
                    outputs=[recent_projects_state, recent_projects_dropdown],
                )

            # Data Collection
            create_data_collection_tab(project_state)

            # Vector Index
            create_vector_index_tab(project_state)

            # Chat
            create_chat_tab(project_state)

            # Configuration
            create_config_tab(project_state)

        app.load(
            fn=hydrate_project,
            inputs=[browser_project_state],
            outputs=[project_state, project_status, active_project_name, active_project_dir],
        )

        app.load(
            fn=lambda recent: gr.update(choices=_format_recent(recent)),
            inputs=[recent_projects_state],
            outputs=[recent_projects_dropdown],
        )

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
