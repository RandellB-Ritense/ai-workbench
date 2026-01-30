# AI Workbench Web UI Guide

Complete guide to using the AI Workbench web interface.

## Table of Contents

- [Getting Started](#getting-started)
- [Data Collection](#data-collection)
- [Vector Index](#vector-index)
- [Chat](#chat)
- [Configuration](#configuration)
- [Tips & Best Practices](#tips--best-practices)
- [Troubleshooting](#troubleshooting)

---

## Getting Started

### Launch the Web UI

```bash
# Start on default port (7860)
uv run ai-workbench web

# Custom port
uv run ai-workbench web --port 8080

# Enable public sharing (get a temporary public URL)
uv run ai-workbench web --share
```

The web UI will open in your default browser at `http://127.0.0.1:7860`

### First Time Setup

1. **Go to Configuration Tab**
   - Enter your Anthropic API key (for Claude models)
   - Enter your Mistral API key (for embeddings)
   - Click "Save API Keys"

2. **Verify Setup**
   - Go to Configuration → System Info
   - Click "Refresh" under External Services
   - Verify: "✓ API key set" for both services

You're now ready to use all features!

---

## Data Collection

### Web Crawler

**Purpose:** Discover and extract all URLs from a website by following links.

**Steps:**

1. **Enter Start URL**
   - Example: `https://docs.python.org/3/tutorial/`
   - Must be a valid HTTP(S) URL

2. **Configure Settings**
   - **Max Depth:** How many links deep to follow (2-3 recommended)
   - **Max Pages:** Total page limit (100 is a good start)
   - **Same Domain Only:** Check to stay on one website

3. **Start Crawl**
   - Click "Start Crawl"
   - Monitor progress in real-time
   - Progress shows: "Crawled X/Y pages (depth N): url..."

4. **Download Results**
   - When complete, download button appears
   - JSON file contains all discovered URLs
   - Use this for batch scraping

**Tips:**
- Start small (depth=2, pages=50) to test
- Higher depth = more pages but much slower
- Same domain only prevents going to external sites
- Operations block the UI while running, so plan accordingly

**Common Issues:**
- "Invalid URL" → Check URL starts with http:// or https://
- Operation takes too long → Reduce max pages or depth
- No results → Site may block crawlers or have no links

---

### Web Scraper

**Purpose:** Extract and convert HTML pages to clean markdown text.

**Steps:**

1. **Choose Input Method**
   - **Upload Crawler JSON:** Use output from crawler
   - **Paste URLs:** Enter URLs directly, one per line

2. **Configure Options**
   - **Ignore Links:** Remove links from output (cleaner text)
   - **Ignore Images:** Remove images from output (smaller files)
   - **Max URLs:** Limit how many to scrape (0 = all)

3. **Start Scraping**
   - Click "Start Scraping"
   - Progress shows: "Scraped X/Y pages ✓/✗ url..."
   - ✓ = success, ✗ = failed

4. **Download Results**
   - JSON file with markdown content for each page
   - Use this to build vector indexes

**Tips:**
- 100 URLs takes about 5-10 minutes
- Scraping can fail for some URLs (protected, 404, timeout)
- Use "Ignore Links" for cleaner text processing
- Failed pages are included in results with error messages

**Common Issues:**
- "No file uploaded" → Select a file or paste URLs
- "Invalid JSON file" → Upload must be from crawler
- Many failures → URLs may be invalid or protected
- Slow progress → Network speed dependent, be patient

---

## Vector Index

### Building an Index

**Purpose:** Convert scraped documents to searchable vector embeddings for RAG.

**Steps:**

1. **Upload Scraped Content**
   - Use JSON file from scraper
   - Must contain markdown content

2. **Configure Index**
   - **Index Name:** Unique identifier (e.g., "python-docs")
   - **Mistral API Key:** Required for embeddings
   - **Chunk Size:** Text chunk size in tokens (500 recommended)
   - **Chunk Overlap:** Overlap between chunks (50 recommended)

3. **Build Index**
   - Click "Build Index"
   - Three stages:
     - Chunking documents (10-30%)
     - Generating embeddings (35-80%)
     - Storing vectors (85-95%)
   - Time: 2-5 minutes for 50 documents

4. **Index Ready**
   - Saved to `~/.ai-workbench/vector-stores/{name}/`
   - Available for RAG search and chat

**Tips:**
- Smaller chunks = more precise search
- Larger chunks = better context
- 500 tokens is a good balance
- Overlap prevents splitting important content

**Common Issues:**
- "No API key" → Enter Mistral key in field or Config tab
- "No chunks generated" → File may be empty or invalid
- Slow progress → Embedding API calls are rate-limited
- Job fails → Check API key is valid and has credits

---

### Testing RAG Retrieval

**Purpose:** Test your index by searching for relevant documents.

**Steps:**

1. **Select Vector Database**
   - Dropdown shows all available indexes
   - Click "Refresh DBs" to update list

2. **Enter Query**
   - Type your search question
   - Example: "How to use Python lists?"

3. **Configure Search**
   - **Top K:** Number of results to retrieve (5 recommended)
   - **Score Threshold:** Minimum similarity (0.7 recommended)

4. **Search**
   - Click "Search"
   - Results show in table:
     - Title, Source URL, Score, Preview
   - Higher score = more relevant

**Tips:**
- Try multiple queries to test quality
- Adjust top-k if too few/many results
- Lower threshold to get more results
- Check scores to evaluate relevance

**Common Issues:**
- "No database selected" → Choose from dropdown or refresh
- "No documents found" → Lower score threshold or try different query
- "Database not found" → Build an index first
- "No API key" → Enter Mistral key for query embedding

---

## Chat

### Setting Up Chat

**1. Initialize LLM**

**Option A: Anthropic Claude**
1. Select "Anthropic" provider
2. Enter API key (or use saved from Config)
3. Choose model (claude-3-5-sonnet-20241022 recommended)
4. Set temperature (0.7 = balanced)
5. Click "Initialize LLM"
6. Wait for "✓ Anthropic initialized"

**Option B: Ollama (Local)**
1. Start Ollama: `ollama serve`
2. Select "Ollama" provider
3. Enter base URL (default: http://localhost:11434)
4. Click "Refresh Models"
5. Select a model from dropdown
6. Click "Initialize LLM"
7. Wait for "✓ Ollama initialized"

**2. Enable RAG (Optional)**

1. Check "Enable RAG"
2. Click "Refresh DBs" if needed
3. Select your vector database
4. Enter Mistral API key
5. Set Top K results (5 recommended)
6. Click "Initialize RAG"
7. Wait for "✓ RAG initialized"

**3. Connect MCP (Optional)**

1. Expand "Connect MCP Server" accordion
2. Enter server details:
   - **Server Name:** `filesystem`
   - **Command:** `npx`
   - **Arguments:** `-y,@modelcontextprotocol/server-filesystem,/tmp`
3. Click "Connect Server"
4. Tools appear in table

### Chatting

1. **Type Message**
   - Enter in message box
   - Press Enter or click Send

2. **Watch Response**
   - Streams in real-time (word by word)
   - RAG context included if enabled
   - MCP tools called if needed

3. **Continue Conversation**
   - Chat maintains history
   - Each message includes previous context

4. **Manage Session**
   - **Clear History:** Reset conversation
   - **Save Session:** Export to JSON file
   - Saved to: `~/ai-workbench-output/chat-sessions/`

**Tips:**
- Initialize LLM before chatting
- Test without RAG first, then enable
- RAG improves answers for domain-specific questions
- MCP tools extend capabilities (file access, etc.)
- Save important conversations

**Common Issues:**
- "LLM not initialized" → Click Initialize LLM first
- "RAG enabled but not initialized" → Click Initialize RAG
- No streaming → Check API key and network
- Slow responses → Normal for large contexts or Ollama
- MCP connection fails → Check command and arguments

---

## Configuration

### API Keys

**Save Keys:**
1. Go to Configuration → API Keys
2. Enter Anthropic API key
3. Enter Mistral API key
4. Click "Save API Keys"
5. Keys saved to `~/.ai-workbench/.env`

**Clear Keys:**
1. Click "Clear Keys"
2. All API keys removed from file
3. Re-enter when needed

### Default Settings

**Configure Defaults:**
- Crawler: max depth, max pages
- Embeddings: chunk size, overlap
- RAG: top-k, score threshold
- Chat: temperature, max tokens

**Save Settings:**
1. Adjust sliders to desired values
2. Click "Save Settings"
3. Saved to `~/.ai-workbench/config.json`
4. Applied to all future operations

**Reset to Defaults:**
- Click "Reset to Defaults"
- Sliders reset to factory values
- Must click Save to persist

### System Info

**Platform:**
- OS type and version
- Python version

**Directories:**
- Output directory path
- Vector stores location
- Job storage location

**Storage Usage:**
- Vector database count
- Jobs in storage
- Output files count
- Click "Refresh" to update

**External Services:**
- Anthropic status
- Mistral status
- Ollama status
- Click "Refresh" to check

---

## Tips & Best Practices

### General

- **Start Small:** Test with small datasets before large operations
- **Be Patient:** Operations run synchronously and will block the UI
- **Save API Keys:** Use Configuration tab for persistence
- **Use Web UI:** Easier than CLI for interactive work

### Crawling

- **Depth:** 2-3 is usually sufficient
- **Same Domain:** Enable to avoid crawling the entire web
- **Max Pages:** Set conservative limits (50-100 for testing)
- **Test First:** Try small crawls before large ones

### Scraping

- **Batch Size:** Start with 10-20 URLs for testing
- **Options:** Ignore links/images for cleaner text
- **Error Handling:** Some URLs will fail, that's normal
- **Be Patient:** 100 URLs takes 5-10 minutes

### Indexing

- **Chunk Size:** 500 tokens is a good default
- **Overlap:** 50 tokens maintains context
- **API Key:** Mistral API required, check you have credits
- **Time:** Plan for 2-5 minutes per 50 documents

### Chatting

- **Initialize First:** LLM and RAG must be initialized
- **Test RAG:** Use Test RAG before enabling in chat
- **Save Sessions:** Export important conversations
- **Try Both:** Compare Claude vs Ollama, RAG vs no-RAG

### Performance

- **Synchronous Operations:** All operations block the UI while running
- **Large Operations:** Be patient - they take time to complete
- **Refresh:** Use refresh buttons instead of reloading page
- **Clean Up:** Delete old output files periodically

---

## Troubleshooting

### Web UI Won't Start

```bash
# Check port availability
lsof -i :7860

# Try different port
uv run ai-workbench web --port 8080

# Check for errors
uv run ai-workbench web --help
```

### Operations Not Working

1. Check error messages displayed in the status field
2. Verify API keys in Configuration
3. Check System Info → External Services
4. Try running operation again
5. Check network connectivity

### Can't Find Vector Database

1. Go to Vector Index tab
2. Click "Refresh DBs"
3. Check: `ls ~/.ai-workbench/vector-stores/`
4. Build index if missing

### Chat Not Responding

1. Check LLM Status shows "initialized"
2. Verify API key in Configuration → System Info
3. Check network connection
4. Try different model
5. Check error messages in chat status

### MCP Connection Fails

1. Verify command and arguments are correct
2. Check command is installed (`npx`, `python`, etc.)
3. Test command in terminal first
4. Check MCP server is compatible
5. See MCP_USAGE.md for details

### Slow Performance

- **Crawling:** Higher depth exponentially increases time
- **Scraping:** Network dependent, be patient
- **Indexing:** API rate limits slow this down
- **Chat:** Large contexts take longer to process

**Optimize:**
- Reduce max pages/depth for crawls
- Limit URLs for scraping
- Use smaller chunk counts
- Lower RAG top-k value

### Out of Memory

- Reduce concurrent jobs (config.job_max_workers)
- Limit max pages in crawler
- Use smaller batches for scraping
- Close other applications

### API Errors

- **"API key not set"** → Add key in Configuration tab
- **"Invalid API key"** → Check key is correct and active
- **"Rate limit"** → Wait and try again, or upgrade plan
- **"Insufficient credits"** → Add credits to account

---

## Keyboard Shortcuts

- **Enter** in message box → Send message
- **Ctrl/Cmd + C** → Copy from chat
- **F5** → Refresh page (jobs persist)

---

## File Locations

All AI Workbench files are stored in standard locations:

```
~/.ai-workbench/              # Configuration and data
├── .env                      # API keys
├── config.json              # Default settings
└── vector-stores/           # Vector databases
    ├── python-docs/
    └── my-docs/

~/ai-workbench-output/       # Output files
├── crawl-xxxxx.json        # Crawler results
├── scraped-xxxxx.json      # Scraper results
└── chat-sessions/          # Saved chat sessions
    └── chat-session-xxxxx.json
```

---

## Getting Help

- **In-App Help:** Check tooltips by hovering over fields
- **Documentation:** See README.md and MCP_USAGE.md
- **System Info:** Configuration → System Info for diagnostics
- **Status Fields:** View error messages in operation status fields
- **GitHub Issues:** Report bugs at repository

---

## Next Steps

1. **Basic Workflow:**
   - Crawl a small documentation site
   - Scrape the results
   - Build a vector index
   - Chat with your documents

2. **Advanced Features:**
   - Try Ollama for local models
   - Connect MCP servers for tools
   - Compare RAG vs no-RAG quality
   - Experiment with different chunk sizes

3. **Automation:**
   - Save your settings in Configuration
   - Use saved configurations for consistent results
   - Build multiple indexes for different domains
   - Create specialized chatbots for specific topics

Enjoy using AI Workbench! 🚀
