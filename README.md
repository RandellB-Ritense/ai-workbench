# AI Workbench

A modular playground for AI tools including web crawlers, scrapers, embedders, RAG systems, and LLM experimentation.

Built with [UV](https://github.com/astral-sh/uv) for blazingly fast dependency management.

## Features

### Data Collection
- **Web Crawler**: Discover and map website URLs with configurable depth
- **Web Scraper**: Extract and convert HTML content to clean markdown
- **Batch Processing**: Scrape multiple URLs efficiently

### AI/ML Capabilities
- **Embeddings**: Generate vector embeddings using Mistral API
- **Vector Store**: Persistent ChromaDB vector database
- **RAG System**: Retrieval-Augmented Generation for document Q&A
- **LLM Integration**: Claude (Anthropic) and Ollama (local models)
- **Interactive Chatbot**: Rich REPL interface with streaming responses
- **MCP Support**: Connect to Model Context Protocol servers for extended capabilities

### Architecture
- **Modular Design**: Each component is cleanly separated
- **Unified CLI**: Single command-line interface for all tools
- **Flexible I/O**: Configure input and output paths outside the project
- **Async Support**: Efficient async operations for LLM and MCP

## Prerequisites

- Python 3.10 or higher
- [UV](https://github.com/astral-sh/uv) - Install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Installation

```bash
uv sync
```

This will automatically create a virtual environment and install all dependencies.

**Why UV?**
- 10-100x faster than pip
- Deterministic dependency resolution with `uv.lock`
- Automatic virtual environment management
- Compatible with standard `pyproject.toml`

## Web UI (Recommended)

AI Workbench includes a comprehensive **Gradio-based web interface** that provides easy access to all functionality with background job processing and real-time progress tracking.

### Launch Web UI

```bash
# Start the web interface (opens at http://127.0.0.1:7860)
uv run ai-workbench web

# Custom port
uv run ai-workbench web --port 8080

# Enable public sharing
uv run ai-workbench web --share
```

### Web UI Features

The web interface includes five main tabs:

**1. Data Collection**
- Web crawler with real-time progress
- Batch scraper with per-URL tracking
- Background job processing
- Download results as JSON

**2. Vector Index**
- Build vector embeddings from scraped content
- Test RAG retrieval with live search
- Multi-stage progress tracking
- Vector database management

**3. Chat**
- Interactive AI chat with streaming responses
- Support for Anthropic Claude and Ollama
- Optional RAG integration with your indexed documents
- MCP server connections for tool use
- Session save/load functionality

**4. Jobs**
- Monitor all background tasks
- Real-time progress updates
- Job history with filtering
- Cancel running jobs
- View detailed job information

**5. Configuration**
- Manage API keys (saved to ~/.ai-workbench/.env)
- Configure default settings
- View system information
- Check service status

### Typical Web UI Workflow

```bash
# 1. Start the web UI
uv run ai-workbench web

# 2. In browser (http://127.0.0.1:7860):
#    - Data Collection tab: Crawl a website → Download URLs
#    - Data Collection tab: Upload URLs → Scrape content → Download JSON
#    - Vector Index tab: Upload scraped JSON → Build index (enter Mistral API key)
#    - Vector Index tab: Test RAG search with your query
#    - Chat tab: Initialize LLM → Enable RAG → Chat with your documents
#    - Jobs tab: Monitor all operations
#    - Configuration tab: Save API keys for future use
```

**Benefits of Web UI:**
- Visual progress tracking for all operations
- No need to remember CLI commands
- Background job processing (never blocks your browser)
- Job history and monitoring
- Easy configuration management
- Persistent settings across sessions

---

## CLI Quick Start

For advanced users who prefer the command line:

### 1. Data Collection

```bash
# Crawl a website
uv run ai-workbench crawl \
  --url https://docs.python.org/3/tutorial/ \
  --output ./urls.json \
  --max-depth 2

# Scrape content to markdown
uv run ai-workbench scrape-batch \
  --input ./urls.json \
  --output ./scraped.json
```

### 2. Build Vector Index

```bash
# Set API key
export WORKBENCH_MISTRAL_API_KEY=your-key-here

# Generate embeddings and build index
uv run ai-workbench build-index \
  --input ./scraped.json \
  --output ./vector-db
```

### 3. Interactive Chat with RAG

```bash
# Set API keys
export WORKBENCH_ANTHROPIC_API_KEY=your-key-here
export WORKBENCH_MISTRAL_API_KEY=your-key-here

# Start chat with RAG
uv run ai-workbench chat \
  --llm claude-3-5-sonnet-20241022 \
  --rag-source ./vector-db

# Or use local Ollama model
uv run ai-workbench chat \
  --llm ollama:llama2 \
  --rag-source ./vector-db
```

### 4. Add MCP Tools (Optional)

```bash
# Chat with RAG + MCP filesystem access
uv run ai-workbench chat \
  --llm claude-3-5-sonnet-20241022 \
  --rag-source ./vector-db \
  --mcp-server filesystem:npx:-y,@modelcontextprotocol/server-filesystem,/tmp
```

## Web UI vs CLI

| Feature | Web UI | CLI |
|---------|--------|-----|
| **Ease of Use** | ✅ Visual, no commands | ⚠️ Requires command knowledge |
| **Progress Tracking** | ✅ Real-time with progress bars | ⚠️ Text-based status |
| **Background Jobs** | ✅ Run in background, monitor later | ❌ Blocks terminal |
| **Job History** | ✅ Full history with details | ❌ Not available |
| **Configuration** | ✅ Visual settings management | ⚠️ Manual .env editing |
| **Multiple Operations** | ✅ Run multiple jobs concurrently | ⚠️ One at a time |
| **Session Management** | ✅ Save/load chat sessions | ✅ Save/load available |
| **Best For** | Interactive experimentation | Automation & scripting |

**Recommendation:** Use the **Web UI** for interactive work and the **CLI** for automation and scripts.

---

## CLI Commands

### Web Interface

```bash
# Launch web UI
uv run ai-workbench web [--port PORT] [--host HOST] [--share]
```

### Data Collection

```bash
# Crawl website
uv run ai-workbench crawl --url <URL> --output <FILE>

# Scrape single page
uv run ai-workbench scrape --url <URL> --output <FILE>

# Scrape multiple pages
uv run ai-workbench scrape-batch --input <FILE> --output <FILE>
```

### RAG Setup

```bash
# Build vector index
uv run ai-workbench build-index --input <SCRAPED_JSON> --output <VECTOR_DB>

# Test retrieval
uv run ai-workbench test-rag --query "Your question" --rag-source <VECTOR_DB>
```

### LLM & Chat

```bash
# List available models
uv run ai-workbench llm-list --provider anthropic
uv run ai-workbench llm-list --provider ollama

# Interactive chat
uv run ai-workbench chat --llm <MODEL> --rag-source <VECTOR_DB>
```

See [MCP_USAGE.md](MCP_USAGE.md) for detailed MCP integration guide.

## Project Structure

```
ai-workbench/
├── src/ai_workbench/
│   ├── web/               # Web UI (Gradio) (✓)
│   │   ├── app.py         # Main Gradio app
│   │   ├── jobs/          # Background job queue
│   │   │   ├── queue.py   # JobQueueManager
│   │   │   ├── models.py  # Job data models
│   │   │   └── storage.py # SQLite persistence
│   │   ├── tabs/          # UI tab components
│   │   │   ├── data_collection.py
│   │   │   ├── vector_index.py
│   │   │   ├── chat.py
│   │   │   ├── jobs.py
│   │   │   └── config.py
│   │   └── utils/         # UI utilities
│   │       ├── streaming.py
│   │       ├── validation.py
│   │       ├── error_handling.py
│   │       └── help_content.py
│   ├── crawlers/          # Web crawling (✓)
│   │   └── web_crawler.py
│   ├── scrapers/          # Web scraping (✓)
│   │   └── web_scraper.py
│   ├── embedders/         # Vector embeddings (✓)
│   │   ├── mistral_embedder.py
│   │   └── document_processor.py
│   ├── vector_stores/     # Vector databases (✓)
│   │   └── chroma_store.py
│   ├── rag/               # RAG retrieval (✓)
│   │   ├── retriever.py
│   │   └── context_builder.py
│   ├── llm/               # LLM clients (✓)
│   │   ├── anthropic_client.py
│   │   ├── ollama_client.py
│   │   └── prompt_templates.py
│   ├── chatbot/           # CLI chat interface (✓)
│   │   ├── session.py
│   │   └── interactive.py
│   ├── mcp/               # MCP client (✓)
│   │   └── client.py
│   ├── config.py          # Configuration
│   └── cli.py             # CLI entry point
├── examples/              # Usage examples
│   └── mcp_chat_example.md
├── MCP_USAGE.md          # MCP integration guide
└── tests/                # Test suite
```

## Configuration

### Using Web UI (Recommended)

Go to the **Configuration tab** in the web interface to:
- Enter and save API keys (stored in `~/.ai-workbench/.env`)
- Adjust default settings (stored in `~/.ai-workbench/config.json`)
- View system information and service status

### Using Environment Variables

Alternatively, configure via environment variables or `.env` file:

```bash
# Required for RAG (embeddings and indexing)
WORKBENCH_MISTRAL_API_KEY=your-mistral-key

# Required for Claude models
WORKBENCH_ANTHROPIC_API_KEY=your-anthropic-key

# Optional: Ollama configuration (for local models)
WORKBENCH_OLLAMA_BASE_URL=http://localhost:11434

# Optional: Customize RAG settings
WORKBENCH_RAG_TOP_K=5
WORKBENCH_RAG_SCORE_THRESHOLD=0.7
WORKBENCH_CHAT_TEMPERATURE=0.7
```

## Chat Commands

In the interactive chat, use these commands:

- `/help` - Show available commands
- `/exit` - Exit the chat
- `/clear` - Clear conversation history
- `/rag on|off` - Toggle RAG on/off
- `/sources` - Show last RAG sources used
- `/model` - Show current model info
- `/stats` - Show session statistics
- `/save <file>` - Save conversation
- `/load <file>` - Load conversation
- `/mcp-tools` - List available MCP tools
- `/mcp-call <tool> <args>` - Call an MCP tool

## Use Cases

### 1. Documentation Assistant

Build a chatbot that answers questions about any documentation:

1. Crawl and scrape documentation
2. Build vector index
3. Chat with RAG-enabled LLM

### 2. Local Model Experimentation

Test if cheap local models (Ollama) become useful with RAG:

```bash
# Without RAG
uv run ai-workbench chat --llm ollama:llama2 --no-rag

# With RAG
uv run ai-workbench chat --llm ollama:llama2 --rag-source ./vector-db
```

### 3. Extended Capabilities with MCP

Add tools like filesystem access, GitHub integration, database queries:

```bash
uv run ai-workbench chat \
  --llm claude-3-5-sonnet-20241022 \
  --rag-source ./docs-db \
  --mcp-server filesystem:npx:-y,@modelcontextprotocol/server-filesystem,/tmp \
  --mcp-server github:npx:-y,@modelcontextprotocol/server-github
```

## Examples

See the [examples](./examples/) directory for complete workflows:

- [MCP Chat Example](./examples/mcp_chat_example.md) - Complete RAG + MCP workflow
- [MCP_USAGE.md](./MCP_USAGE.md) - Detailed MCP integration guide

## Development

Development dependencies are automatically installed with:

```bash
uv sync
```

Run commands in the UV environment:

```bash
uv run ai-workbench --help
uv run pytest  # Run tests (when available)
uv run ruff check  # Lint code
uv run black .  # Format code
```

## Architecture Highlights

- **Modular Design**: Easy to add new LLM providers, embedders, or vector stores
- **Abstract Base Classes**: `LLMProvider`, `Embedder`, `VectorStore` for extensibility
- **Async-First**: Efficient streaming and concurrent operations
- **Token-Aware**: RAG context stays within token budgets
- **MCP Integration**: Connect to external tools via Model Context Protocol

## Web UI Tips

### Job Management
- All long-running operations (crawl, scrape, build index) run as background jobs
- Monitor progress in real-time from any tab
- Jobs persist across page refreshes
- Check the Jobs tab to cancel running jobs or view history

### API Keys
- Save API keys in the Configuration tab for persistence
- Keys are stored in `~/.ai-workbench/.env`
- Alternatively, set them as environment variables before launching

### Vector Databases
- Built indexes are saved to `~/.ai-workbench/vector-stores/`
- Each index is a separate directory with a unique name
- Use the refresh button to update the database list
- Test your indexes with the RAG search tool before using in chat

### Chat Sessions
- Save important conversations with the save button
- Sessions are stored in `~/ai-workbench-output/chat-sessions/`
- Sessions include conversation history and settings
- Load previous sessions (future feature)

### MCP Servers
- Connect multiple MCP servers simultaneously
- Each server provides its own set of tools
- Tools are namespaced as `server_name:tool_name`
- Check MCP status in the Configuration tab → System Info

### Performance
- Limit concurrent jobs to 2 (configurable in config)
- Large crawls (500+ pages) may take 10-15 minutes
- Building indexes for 100+ documents takes 5-10 minutes
- Use smaller test datasets first to verify settings

---

## Troubleshooting

### Web UI Issues

**Port Already in Use**
```bash
# Use a different port
uv run ai-workbench web --port 8080
```

**Can't Access Web UI**
```bash
# Check if it's running
ps aux | grep "ai-workbench web"

# Try binding to all interfaces
uv run ai-workbench web --host 0.0.0.0
```

**Jobs Not Running**
- Check the Jobs tab for error messages
- Verify API keys in Configuration tab
- Check System Info → External Services status

### CLI Issues

#### Ollama Not Found

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Pull a model
ollama pull llama2
```

### MCP Connection Issues

See [MCP_USAGE.md](./MCP_USAGE.md#troubleshooting) for detailed troubleshooting.
