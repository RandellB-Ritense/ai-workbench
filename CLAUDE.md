# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Workbench is a modular playground for AI tools including web crawlers, scrapers, embedders, RAG systems, and LLM experimentation. It features both a CLI and a Gradio-based web UI with background job processing.

**Key Technologies:**
- **Package Manager:** UV (blazingly fast, replaces pip/virtualenv)
- **CLI Framework:** Typer with Rich for formatted output
- **Web UI:** Gradio with background job processing
- **LLM Providers:** Mistral (hosted) and Ollama (local)
- **Vector Store:** ChromaDB with Mistral embeddings
- **MCP:** Model Context Protocol for tool integration

## Essential Commands

### Setup & Development

```bash
# Install dependencies (creates venv automatically)
uv sync

# Run the web UI (primary interface)
uv run ai-workbench web

# Run CLI commands
uv run ai-workbench <command> --help

# Code quality
uv run ruff check        # Lint
uv run black .           # Format
```

### Testing Common Workflows

```bash
# CLI workflow: Crawl → Scrape → Build Index → Chat
uv run ai-workbench crawl --url https://docs.python.org/3/tutorial/ --output ./urls.json --max-depth 2
uv run ai-workbench scrape-batch --input ./urls.json --output ./scraped.json
uv run ai-workbench build-index --input ./scraped.json --output ./vector-db
uv run ai-workbench chat --llm mistral-large-latest --rag-source ./vector-db

# Test RAG retrieval without LLM
uv run ai-workbench test-rag --query "How to use lists?" --rag-source ./vector-db --show-context

# List available models
uv run ai-workbench llm-list --provider mistral
uv run ai-workbench llm-list --provider ollama

# Web UI (recommended)
uv run ai-workbench web --port 7860
```

### Environment Variables

API keys and settings can be configured via environment variables (prefix: `WORKBENCH_`) or `.env` file:

```bash
# Required for RAG (embeddings)
export WORKBENCH_MISTRAL_API_KEY=your-key

# For local models
export WORKBENCH_OLLAMA_BASE_URL=http://localhost:11434

# RAG tuning
export WORKBENCH_RAG_TOP_K=5
export WORKBENCH_RAG_SCORE_THRESHOLD=0.7
```

Web UI stores config in the project directory's `config/.env` file.

## Architecture & Design Patterns

### Project-First Architecture

**Critical Design:** The Web UI operates on a **project-first** model:
- Users must create a project before accessing any tools
- A project has a **single root directory** on the local filesystem
- All files (config, outputs, vector stores, jobs, chat sessions) are stored **inside** this project directory
- No file operations occur outside the project directory
- Project structure is managed by `src/ai_workbench/project.py`:
  - `ProjectInfo`: Name and root path
  - `ProjectPaths`: Standard subdirectory structure
  - `build_project_paths()`: Creates consistent path layout
  - `ensure_project_dirs()`: Creates all required subdirectories

**Standard Project Layout:**
```
<project_root>/
├── config/              # API keys (.env) and settings (config.json)
│   └── project.json     # Project metadata
├── outputs/             # Crawler and scraper results
├── vector-stores/       # Vector databases (one dir per index)
├── jobs/                # Job queue storage (SQLite)
├── chat-sessions/       # Saved chat sessions
└── logs/                # Application logs
```

**When modifying Web UI code:**
- Always use `project_paths_from_state()` to get paths from the active project
- Never use `~/.ai-workbench/` or other global paths
- UI components receive project state via Gradio state
- File paths in UI should be pre-populated using project paths

### Abstract Base Classes

The codebase uses ABC patterns for extensibility:

**`LLMProvider` (src/ai_workbench/llm/base.py):**
- Abstract interface for all LLM providers
- Key methods: `generate()`, `generate_stream()`, `list_models()`
- Implementations: `MistralClient`, `OllamaClient`
- Uses dataclasses: `Message` (role + content), `LLMResponse`

**Similar patterns exist for:**
- Embedders (Mistral embeddings)
- Vector stores (ChromaDB)
- Each component is swappable via the abstract interface

### Web UI Background Jobs

**Important:** The Web UI uses **synchronous** operations that block the UI:
- Located in: `src/ai_workbench/web/tabs/`
- Long operations (crawl, scrape, build-index) run synchronously
- Progress updates are shown in status messages
- Job state is NOT persisted (no SQLite job queue currently)
- Users must wait for operations to complete

**Key Web UI Tabs:**
- `data_collection.py`: Crawler and batch scraper with progress tracking
- `vector_index.py`: Build indexes and test RAG retrieval
- `chat.py`: LLM chat with streaming, RAG, and MCP
- `config.py`: API keys, settings, system info

**When adding new features to Web UI:**
- Follow existing synchronous patterns (no async/job queue)
- Use `gr.Progress()` for progress tracking
- Update status messages to keep users informed
- Store outputs in project directories using `ProjectPaths`

### RAG Pipeline

**Flow:** Query → Embed → Vector Search → Context Building → LLM

**Key Components:**
1. **Document Processing** (`embedders/document_processor.py`):
   - Chunks scraped content into token-sized pieces
   - Uses tiktoken for token counting
   - Maintains metadata (source URL, chunk index)

2. **Embeddings** (`embedders/mistral_embedder.py`):
   - Mistral API for generating embeddings
   - Batch processing for efficiency

3. **Vector Store** (`vector_stores/chroma_store.py`):
   - ChromaDB for persistent vector storage
   - Each index is a separate directory

4. **Retrieval** (`rag/retriever.py`):
   - Embeds query → searches ChromaDB
   - Filters by score threshold
   - Returns top-k results

5. **Context Builder** (`rag/context_builder.py`):
   - Formats retrieved documents for LLM
   - Respects token budget (default: 4000)
   - Includes source attribution

### MCP Integration

**Purpose:** Extend LLM capabilities with external tools (filesystem, GitHub, etc.)

**Architecture:**
- `mcp/client.py`: MCPClientManager handles multiple MCP servers
- Async operations using `mcp` Python library
- Each server provides tools with schemas
- Tools are called by the LLM or manually via chat commands

**Chat Commands:**
- `/mcp-tools`: List available tools
- `/mcp-call <tool> <json_args>`: Execute a tool
- Format: `server_name:command:comma,separated,args`

**Example MCP Server Spec:**
```bash
--mcp-server filesystem:npx:-y,@modelcontextprotocol/server-filesystem,/tmp
```

### CLI vs Web UI

**CLI (`cli.py`):**
- Single-command operations
- Good for scripting and automation
- No project requirement (uses paths directly)
- Blocking operations with Rich progress

**Web UI (`web/app.py`):**
- Interactive multi-step workflows
- Project-based file organization
- Gradio components with real-time progress
- Configuration persistence in project directory

**When adding features:**
- Implement core logic in domain modules (crawlers, scrapers, etc.)
- CLI commands are thin wrappers calling domain functions
- Web UI tabs orchestrate domain functions with Gradio components

## Code Patterns & Conventions

### Configuration Management

- All config is managed by `config.py` using pydantic-settings
- Single global instance via `get_config()`
- Environment variables prefixed with `WORKBENCH_`
- `.env` file support for local development
- **Web UI:** Config stored in project's `config/.env` and `config/config.json`

### Error Handling

- Use Rich console for user-facing errors in CLI
- Gradio info/warning messages for Web UI errors
- Validate inputs early (check API keys, file existence)
- Provide actionable error messages (e.g., "Set API key with: export ...")

### Streaming Responses

**LLM Streaming:**
- All LLM providers implement `generate_stream()` → `Iterator[str]`
- CLI uses Rich Live display for streaming
- Web UI uses `src/ai_workbench/web/utils/streaming.py` for Gradio streaming
- Chunk-by-chunk token streaming for better UX

### Data Formats

**Crawler Output (JSON):**
```json
{
  "start_url": "...",
  "pages": [{"url": "...", "depth": 0, "found_at": "..."}],
  "metadata": {...}
}
```

**Scraper Output (JSON):**
```json
{
  "pages": [
    {
      "url": "...",
      "title": "...",
      "content": "markdown content...",
      "word_count": 123,
      "scraped_at": "...",
      "error": null
    }
  ]
}
```

**Vector Store Metadata:**
- Each chunk includes: source_url, title, chunk_index, total_chunks
- Stored alongside vectors in ChromaDB

## Important Notes for Development

### When Modifying LLM Integration

**Current State:** The codebase has merge conflicts in `cli.py` between Anthropic and Mistral clients:
- Lines 343-528 show `<<<<<<< ours` (Anthropic) vs `>>>>>>> theirs` (Mistral)
- The Mistral client is the current implementation
- `anthropic_client.py` was deleted (see git status)

**If adding new LLM providers:**
1. Implement `LLMProvider` ABC from `llm/base.py`
2. Add to `llm-list` command
3. Update chat command to handle provider detection
4. Test both streaming and non-streaming modes

### When Adding New CLI Commands

1. Add `@app.command()` in `cli.py`
2. Use Typer options for arguments (see existing commands)
3. Import domain logic from module (keep CLI thin)
4. Use Rich console for output formatting
5. Handle errors gracefully with user-friendly messages

### When Adding Web UI Features

1. Create/modify tab in `src/ai_workbench/web/tabs/`
2. Use `gr.Progress()` for long operations
3. Access project paths via `project_paths_from_state(project_state)`
4. Store outputs in appropriate project subdirectory
5. Update status messages throughout operation
6. Follow synchronous patterns (no background jobs)

### Vector Store Management

- Each index is a separate ChromaDB directory
- Indexes are NOT automatically discovered globally
- Web UI lists indexes from `<project_root>/vector-stores/`
- Use refresh buttons to update index lists
- ChromaDB requires matching embedding dimensions

### Testing RAG Without Full Pipeline

Use `test-rag` command to validate retrieval without LLM:
```bash
uv run ai-workbench test-rag --query "test query" --rag-source ./vector-db --show-context
```

This shows retrieved documents, scores, and formatted context.

## Common Pitfalls

1. **Missing API Keys:** Many operations require `WORKBENCH_MISTRAL_API_KEY`. Check early and provide clear error messages.

2. **Project State in Web UI:** All Web UI operations must receive and use the active project state. Don't use hardcoded paths.

3. **Token Budgets:** RAG context must fit within LLM context window. Use `ContextBuilder` to respect token limits.

4. **Sync vs Async:** CLI is sync, MCP is async, Web UI is sync (uses `asyncio.run()` for MCP calls).

5. **Merge Conflicts:** The CLI file currently has unresolved merge conflicts. Resolve these before making changes.

6. **UV vs pip:** Always use `uv run` or `uv sync`. Don't use pip/virtualenv commands.

## Key Files Reference

- `cli.py`: All CLI commands and entry point
- `config.py`: Global configuration management
- `project.py`: Project directory structure and management
- `web/app.py`: Gradio app launcher and tab composition
- `llm/base.py`: LLM provider interface
- `rag/retriever.py`: Vector search and retrieval logic
- `rag/context_builder.py`: Format context for LLM
- `chatbot/interactive.py`: CLI chat REPL
- `web/utils/streaming.py`: Gradio streaming utilities

## Documentation

- `README.md`: User-facing docs, features, installation
- `WEB_UI_GUIDE.md`: Complete Web UI walkthrough
- `MCP_USAGE.md`: MCP integration guide
- `.env.example`: All available config options
