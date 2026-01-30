"""
Chat tab - Interactive chat interface with LLM and RAG.
"""
import gradio as gr
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import httpx

from ..utils.streaming import chat_stream_generator, save_chat_session, create_chat_history_from_session
from ..utils.validation import validate_api_key, check_ollama_connection, validate_vector_db
from ...llm.mistral_client import MistralClient
from ...llm.ollama_client import OllamaClient
from ...embedders.mistral_embedder import MistralEmbedder
from ...vector_stores.chroma_store import ChromaStore
from ...rag.retriever import RAGRetriever
from ...rag.context_builder import ContextBuilder
from ...mcp.client import MCPClientManager
from ...config import get_config
from ...project import project_paths_from_state


def create_chat_tab(project_state: gr.State) -> gr.Tab:
    """
    Create the Chat tab.

    Returns:
        Gradio Tab component
    """
    config = get_config()

    # State variables
    llm_client_state = gr.State(None)
    retriever_state = gr.State(None)
    context_builder_state = gr.State(None)
    mcp_client_state = gr.State(None)

    with gr.Tab("Chat") as tab:
        gr.Markdown("## Interactive Chat")
        gr.Markdown("Chat with LLMs using optional RAG context and streaming responses.")

        with gr.Row():
            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=600,
                    show_label=False
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here...",
                        scale=9,
                        show_label=False,
                        container=False,
                        interactive=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, interactive=False)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear History", size="sm")
                    save_btn = gr.Button("💾 Save Session", size="sm", interactive=False)

                session_status = gr.Textbox(
                    label="Status",
                    value="",
                    interactive=False,
                    visible=False
                )

            # Settings sidebar
            with gr.Column(scale=1):
                gr.Markdown("### Settings")

                # LLM Provider Selection
                with gr.Group():
                    gr.Markdown("**LLM Provider**")

                    llm_provider = gr.Radio(
                        choices=["Mistral", "Ollama"],
                        value="Mistral",
                        label="Provider",
                        show_label=False
                    )

                    # Mistral settings
                    with gr.Column(visible=True) as mistral_settings:
                        mistral_model = gr.Textbox(
                            label="Model",
                            value=config.default_llm_model,
                            placeholder="mistral-large-latest"
                        )

                    # Ollama settings
                    with gr.Column(visible=False) as ollama_settings:
                        ollama_base_url = gr.Textbox(
                            label="Base URL",
                            value=config.ollama_base_url,
                            placeholder="http://localhost:11434"
                        )

                        ollama_model = gr.Dropdown(
                            label="Model",
                            choices=[],
                            value=None,
                            allow_custom_value=True
                        )

                        refresh_ollama_btn = gr.Button("🔄 Refresh Models", size="sm")

                        ollama_status = gr.Textbox(
                            label="Status",
                            value="",
                            interactive=False,
                            max_lines=2
                        )

                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=config.chat_temperature,
                        step=0.1,
                        label="Temperature"
                    )

                    max_tokens = gr.Slider(
                        minimum=500,
                        maximum=8000,
                        value=config.chat_max_tokens,
                        step=500,
                        label="Max Tokens"
                    )

                    init_llm_btn = gr.Button("Initialize LLM", variant="primary", interactive=False)

                # RAG Settings
                gr.Markdown("---")
                with gr.Group():
                    gr.Markdown("**RAG Settings**")

                    rag_enabled = gr.Checkbox(
                        value=False,
                        label="Enable RAG",
                        info="Use vector database for context"
                    )

                    def get_vector_dbs(project: Optional[Dict]):
                        """List available vector databases."""
                        if not project:
                            return []
                        vector_store_path = project_paths_from_state(project).vector_store_dir
                        if not vector_store_path.exists():
                            return []
                        dbs = [d.name for d in vector_store_path.iterdir() if d.is_dir()]
                        return sorted(dbs)

                    vector_db_dropdown = gr.Dropdown(
                        label="Vector Database",
                        choices=get_vector_dbs(None),
                        value=None
                    )

                    refresh_dbs_btn = gr.Button("🔄 Refresh DBs", size="sm", interactive=False)

                    rag_top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=config.rag_top_k,
                        step=1,
                        label="Top K Results",
                        info="Number of documents to retrieve"
                    )

                    rag_score_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=config.rag_score_threshold,
                        step=0.05,
                        label="Similarity Threshold",
                        info="Minimum similarity score (0-1)"
                    )

                    init_rag_btn = gr.Button("Initialize RAG", variant="secondary", interactive=False)

                # MCP Settings
                gr.Markdown("---")
                with gr.Group():
                    gr.Markdown("**MCP Servers**")

                    with gr.Accordion("Connect MCP Server", open=False):
                        mcp_server_name = gr.Textbox(
                            label="Server Name",
                            placeholder="filesystem",
                            info="Unique name for this server"
                        )

                        mcp_command = gr.Textbox(
                            label="Command",
                            placeholder="npx",
                            info="Command to run the server"
                        )

                        mcp_args = gr.Textbox(
                            label="Arguments (comma-separated)",
                            placeholder="-y,@modelcontextprotocol/server-filesystem,/tmp",
                            info="Arguments for the command"
                        )

                        connect_mcp_btn = gr.Button("Connect Server", variant="secondary", interactive=False)

                    mcp_tools_display = gr.Dataframe(
                        headers=["Server", "Tool", "Description"],
                        label="Available Tools",
                        wrap=True,
                        interactive=False
                    )

                    refresh_mcp_tools_btn = gr.Button("🔄 Refresh Tools", size="sm")

                # Status indicators
                gr.Markdown("---")
                llm_status = gr.Textbox(
                    label="LLM Status",
                    value="Not initialized",
                    interactive=False,
                    max_lines=3
                )

                rag_status = gr.Textbox(
                    label="RAG Status",
                    value="Not initialized",
                    interactive=False,
                    max_lines=3
                )

                mcp_status = gr.Textbox(
                    label="MCP Status",
                    value="No servers connected",
                    interactive=False,
                    max_lines=3
                )

        # Event handlers
        def toggle_provider_settings(provider):
            """Toggle between Mistral and Ollama settings."""
            if provider == "Mistral":
                return {
                    mistral_settings: gr.update(visible=True),
                    ollama_settings: gr.update(visible=False)
                }
            else:
                return {
                    mistral_settings: gr.update(visible=False),
                    ollama_settings: gr.update(visible=True)
                }

        def refresh_ollama_models(base_url):
            """Refresh list of Ollama models."""
            is_running, error = check_ollama_connection(base_url)

            if not is_running:
                return {
                    ollama_model: gr.update(choices=[]),
                    ollama_status: f"Error: {error}"
                }

            try:
                url = base_url.rstrip("/") + "/api/tags"
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    data = response.json()

                models = data.get("models", [])
                model_names = [m.get("name", "") for m in models if m.get("name")]

                if not model_names:
                    return {
                        ollama_model: gr.update(choices=[]),
                        ollama_status: "No models installed. Run: ollama pull llama2"
                    }

                return {
                    ollama_model: gr.update(choices=model_names, value=model_names[0] if model_names else None),
                    ollama_status: f"✓ Connected ({len(model_names)} models)"
                }
            except Exception as e:
                return {
                    ollama_model: gr.update(choices=[]),
                    ollama_status: f"Error: {str(e)}"
                }

        def initialize_llm(provider, mistral_model_name, ollama_url, ollama_model_name, current_client, project: Optional[Dict]):
            """Initialize the LLM client."""
            try:
                if not project:
                    return None, "✗ Create a project before initializing tools"
                if provider == "Mistral":
                    mistral_key = get_config().mistral_api_key
                    # Validate API key
                    is_valid, error = validate_api_key(mistral_key, "Mistral")
                    if not is_valid:
                        return None, f"✗ {error}"

                    # Initialize Mistral client
                    client = MistralClient(api_key=mistral_key, model=mistral_model_name)
                    return client, f"✓ Mistral initialized\nModel: {mistral_model_name}"

                else:  # Ollama
                    # Check connection
                    is_running, error = check_ollama_connection(ollama_url)
                    if not is_running:
                        return None, f"✗ {error}"

                    if not ollama_model_name:
                        return None, "✗ Please select a model"

                    # Initialize Ollama client
                    client = OllamaClient(base_url=ollama_url, model=ollama_model_name)
                    return client, f"✓ Ollama initialized\nModel: {ollama_model_name}"

            except Exception as e:
                return None, f"✗ Error: {str(e)}"

        def initialize_rag(db_name, top_k, score_threshold, current_retriever, current_context_builder, project: Optional[Dict]):
            """Initialize RAG components."""
            try:
                if not project:
                    return None, None, False, "✗ Create a project before initializing tools"
                if not db_name:
                    return None, None, False, "✗ Please select a vector database"

                # Validate API key
                mistral_key = get_config().mistral_api_key
                is_valid, error = validate_api_key(mistral_key, "Mistral")
                if not is_valid:
                    return None, None, False, f"✗ {error}"

                # Validate vector DB
                vector_store_path = project_paths_from_state(project).vector_store_dir
                is_valid, error = validate_vector_db(db_name, vector_store_path)
                if not is_valid:
                    return None, None, False, f"✗ {error}"

                # Initialize components
                embedder = MistralEmbedder(api_key=mistral_key)

                vector_db_path = vector_store_path / db_name
                vector_store = ChromaStore(
                    persist_directory=vector_db_path,
                    embedding_dimension=embedder.get_embedding_dimension()
                )

                retriever = RAGRetriever(
                    embedder=embedder,
                    vector_store=vector_store,
                    top_k=int(top_k),
                    score_threshold=float(score_threshold)
                )

                context_builder = ContextBuilder(
                    max_tokens=config.rag_context_max_tokens,
                    include_sources=True
                )

                doc_count = vector_store.count()
                return retriever, context_builder, True, f"✓ RAG initialized and enabled\nDatabase: {db_name}\nDocuments: {doc_count}\nThreshold: {score_threshold:.2f}"

            except Exception as e:
                return None, None, False, f"✗ Error: {str(e)}"

        def chat_response(message, history, llm_client, retriever, context_builder, rag_enabled_val, temp, max_tok, project: Optional[Dict]):
            """Handle chat message and stream response."""
            if history is None:
                history = []
            if isinstance(message, list):
                # Gradio may pass rich content lists
                message = " ".join([str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in message]).strip()
            if not project:
                history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "Error: Create a project before using tools."},
                ]
                yield history
                return
            if not llm_client:
                history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "Error: LLM not initialized. Please initialize LLM first."},
                ]
                yield history
                return

            if rag_enabled_val and not retriever:
                history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "Error: RAG enabled but not initialized. Please initialize RAG first."},
                ]
                yield history
                return

            # Use the streaming generator
            yield from chat_stream_generator(
                message=message,
                history=history,
                llm_client=llm_client,
                retriever=retriever if rag_enabled_val else None,
                context_builder=context_builder if rag_enabled_val else None,
                rag_enabled=rag_enabled_val,
                temperature=temp,
                max_tokens=max_tok
            )

        def clear_history():
            """Clear chat history."""
            return [], {session_status: gr.update(value="History cleared", visible=True)}

        def save_session(history, provider, model_mistral, model_ollama, temp, rag_enabled_val, db_name, project: Optional[Dict]):
            """Save chat session to file."""
            if not project:
                return {session_status: gr.update(value="Create a project before saving sessions", visible=True)}
            if not history:
                return {session_status: gr.update(value="No history to save", visible=True)}

            try:
                # Determine model name
                model_name = model_mistral if provider == "Mistral" else model_ollama

                # Create session data
                metadata = {
                    "provider": provider,
                    "model": model_name,
                    "temperature": temp,
                    "rag_enabled": rag_enabled_val,
                    "vector_db": db_name if rag_enabled_val else None
                }

                session_data = save_chat_session(history, metadata)
                session_data["timestamp"] = datetime.now().isoformat()

                # Save to file
                output_dir = project_paths_from_state(project).chat_sessions_dir
                output_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                output_file = output_dir / f"chat-session-{timestamp}.json"

                with open(output_file, 'w') as f:
                    json.dump(session_data, f, indent=2)

                return {session_status: gr.update(value=f"✓ Session saved: {output_file.name}", visible=True)}

            except Exception as e:
                return {session_status: gr.update(value=f"✗ Error saving: {str(e)}", visible=True)}

        # Connect events
        llm_provider.change(
            fn=toggle_provider_settings,
            inputs=[llm_provider],
            outputs=[mistral_settings, ollama_settings]
        )

        refresh_ollama_btn.click(
            fn=refresh_ollama_models,
            inputs=[ollama_base_url],
            outputs=[ollama_model, ollama_status]
        )

        init_llm_btn.click(
            fn=initialize_llm,
            inputs=[llm_provider, mistral_model, ollama_base_url, ollama_model, llm_client_state, project_state],
            outputs=[llm_client_state, llm_status]
        )

        def refresh_vector_dbs(project: Optional[Dict]):
            return gr.update(choices=get_vector_dbs(project))

        project_state.change(
            fn=refresh_vector_dbs,
            inputs=[project_state],
            outputs=[vector_db_dropdown],
        )

        def apply_project_state(project: Optional[Dict]):
            enabled = bool(project)
            return {
                msg_input: gr.update(interactive=enabled),
                send_btn: gr.update(interactive=enabled),
                save_btn: gr.update(interactive=enabled),
                init_llm_btn: gr.update(interactive=enabled),
                init_rag_btn: gr.update(interactive=enabled),
                refresh_dbs_btn: gr.update(interactive=enabled),
                connect_mcp_btn: gr.update(interactive=enabled),
            }

        project_state.change(
            fn=apply_project_state,
            inputs=[project_state],
            outputs=[msg_input, send_btn, save_btn, init_llm_btn, init_rag_btn, refresh_dbs_btn, connect_mcp_btn],
        )

        refresh_dbs_btn.click(
            fn=refresh_vector_dbs,
            inputs=[project_state],
            outputs=[vector_db_dropdown]
        )

        init_rag_btn.click(
            fn=initialize_rag,
            inputs=[vector_db_dropdown, rag_top_k, rag_score_threshold, retriever_state, context_builder_state, project_state],
            outputs=[retriever_state, context_builder_state, rag_enabled, rag_status]
        )

        # Chat interaction
        msg_input.submit(
            fn=chat_response,
            inputs=[msg_input, chatbot, llm_client_state, retriever_state, context_builder_state, rag_enabled, temperature, max_tokens, project_state],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            outputs=[msg_input]
        )

        send_btn.click(
            fn=chat_response,
            inputs=[msg_input, chatbot, llm_client_state, retriever_state, context_builder_state, rag_enabled, temperature, max_tokens, project_state],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            outputs=[msg_input]
        )

        clear_btn.click(
            fn=clear_history,
            outputs=[chatbot, session_status]
        )

        save_btn.click(
            fn=save_session,
            inputs=[chatbot, llm_provider, mistral_model, ollama_model, temperature, rag_enabled, vector_db_dropdown, project_state],
            outputs=[session_status]
        )

        # MCP events
        def connect_mcp_server(server_name, command, args_str, current_mcp_client, project: Optional[Dict]):
            """Connect to an MCP server."""
            try:
                if not project:
                    return current_mcp_client, gr.update(), "✗ Create a project before using tools"
                if not server_name:
                    return current_mcp_client, gr.update(), f"✗ Server name required"

                if not command:
                    return current_mcp_client, gr.update(), f"✗ Command required"

                # Initialize MCP client if not exists
                if current_mcp_client is None:
                    current_mcp_client = MCPClientManager()

                # Parse arguments
                args = [arg.strip() for arg in args_str.split(",")] if args_str else []

                # Connect to server
                success = current_mcp_client.connect_server(
                    server_name=server_name,
                    command=command,
                    args=args
                )

                if success:
                    tools = current_mcp_client.list_tools()
                    server_tools = [t for t in tools if t.server_name == server_name]

                    # Update tools display
                    tools_data = []
                    for tool in current_mcp_client.list_tools():
                        tools_data.append([
                            tool.server_name,
                            tool.name,
                            tool.description[:80] + "..." if len(tool.description) > 80 else tool.description
                        ])

                    return (
                        current_mcp_client,
                        gr.update(value=tools_data),
                        f"✓ Connected to {server_name}\n{len(server_tools)} tools available"
                    )
                else:
                    return current_mcp_client, gr.update(), f"✗ Failed to connect to {server_name}"

            except Exception as e:
                return current_mcp_client, gr.update(), f"✗ Error: {str(e)}"

        def refresh_mcp_tools(mcp_client):
            """Refresh MCP tools display."""
            if not mcp_client:
                return gr.update(value=[])

            tools_data = []
            for tool in mcp_client.list_tools():
                tools_data.append([
                    tool.server_name,
                    tool.name,
                    tool.description[:80] + "..." if len(tool.description) > 80 else tool.description
                ])

            return gr.update(value=tools_data)

        connect_mcp_btn.click(
            fn=connect_mcp_server,
            inputs=[mcp_server_name, mcp_command, mcp_args, mcp_client_state, project_state],
            outputs=[mcp_client_state, mcp_tools_display, mcp_status]
        )

        refresh_mcp_tools_btn.click(
            fn=refresh_mcp_tools,
            inputs=[mcp_client_state],
            outputs=[mcp_tools_display]
        )

    return tab
