# Complete Example: Interactive Chat with RAG and MCP

This example demonstrates the full workflow of using AI Workbench with RAG and MCP integration.

## Scenario

We'll crawl Python documentation, build a vector index, and then use an interactive chatbot with both RAG (for documentation context) and MCP (for filesystem access).

## Step 1: Crawl Documentation

```bash
# Crawl Python documentation
uv run ai-workbench crawl \
  --url https://docs.python.org/3/tutorial/ \
  --output ~/data/python-urls.json \
  --max-depth 2 \
  --max-pages 50
```

## Step 2: Scrape Content

```bash
# Convert pages to markdown
uv run ai-workbench scrape-batch \
  --input ~/data/python-urls.json \
  --output ~/data/python-scraped.json
```

## Step 3: Build Vector Index

```bash
# Create embeddings and vector database
export WORKBENCH_MISTRAL_API_KEY=your-mistral-key

uv run ai-workbench build-index \
  --input ~/data/python-scraped.json \
  --output ~/data/python-vector-db
```

## Step 4: Test RAG Retrieval

```bash
# Test retrieval before using in chat
uv run ai-workbench test-rag \
  --query "How do I open and read a file in Python?" \
  --rag-source ~/data/python-vector-db \
  --top-k 3 \
  --show-context
```

Expected output:
```
✓ Found 3 relevant documents:

Document 1:
  Title: 7. Input and Output — Python 3.12 documentation
  Source: https://docs.python.org/3/tutorial/inputoutput.html
  Score: 0.876
  Chunk: 1
  Preview: Reading and Writing Files...

[Formatted Context for LLM shows the combined context]
```

## Step 5: Interactive Chat with RAG + MCP

```bash
# Start chat with RAG and MCP filesystem access
export WORKBENCH_ANTHROPIC_API_KEY=your-anthropic-key
export WORKBENCH_MISTRAL_API_KEY=your-mistral-key

uv run ai-workbench chat \
  --llm claude-3-5-sonnet-20241022 \
  --rag-source ~/data/python-vector-db \
  --mcp-server filesystem:npx:-y,@modelcontextprotocol/server-filesystem,/tmp \
  --temperature 0.7
```

## Interactive Chat Session

```
AI Workbench - Interactive Chat
════════════════════════════════════════════════════════════

Model: Claude 3.5 Sonnet (anthropic)
RAG: Enabled
Temperature: 0.7

Type /help for available commands

[You] > How do I read a file in Python?

[Assistant]
Based on the documentation, here's how to read a file in Python:

```python
# Basic file reading
with open('filename.txt', 'r') as f:
    content = f.read()
    print(content)

# Read line by line
with open('filename.txt', 'r') as f:
    for line in f:
        print(line.strip())
```

The `with` statement ensures the file is properly closed after use.

[You] > /mcp-call filesystem:write_file {"path": "/tmp/example.py", "content": "# Example file\nprint('Hello, World!')"}

✓ filesystem:write_file succeeded:
File written successfully

[You] > /mcp-call filesystem:read_text_file {"path": "/tmp/example.py"}

✓ filesystem:read_text_file succeeded:
# Example file
print('Hello, World!')

[You] > /sources

RAG Sources (2 documents):

  1. 7. Input and Output
     Source: https://docs.python.org/3/tutorial/inputoutput.html
     Score: 0.876

  2. Reading and Writing Files
     Source: https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
     Score: 0.843

[You] > /rag off
RAG disabled

[You] > Now answer without RAG: What's the capital of France?

[Assistant]
The capital of France is Paris.

[You] > /stats

Session Statistics:
════════════════════════════════════════════════════════════

Session ID: abc123...
Total Messages: 10
User Messages: 5
Assistant Messages: 5
Model: claude-3-5-sonnet-20241022
RAG: Disabled

[You] > /exit

Session ended. Messages: 10
Use /save <file> to save this conversation
```

## Example Use Cases

### 1. Code Assistant with File Access

Ask questions about code and have the assistant read/write files:

```
[You] > Read the file at /tmp/app.py and suggest improvements

[Assistant] Let me read that file first...
[Calls filesystem:read_text_file internally]

Based on the code, here are suggestions:
1. Add error handling for file operations
2. Use context managers for database connections
3. ...
```

### 2. Documentation Search and Experimentation

Search docs via RAG and test code via MCP:

```
[You] > How do I use list comprehensions?

[Assistant] [Uses RAG to find relevant Python docs]
List comprehensions provide a concise way to create lists:
[squares = [x**2 for x in range(10)]]

[You] > /mcp-call filesystem:write_file {"path": "/tmp/test.py", "content": "squares = [x**2 for x in range(10)]\nprint(squares)"}

[You] > Now run it with Python
[User can manually run: python /tmp/test.py]
```

### 3. Multi-Source Context

Combine RAG documentation with real files:

```
[You] > According to Python docs, what's the best way to handle configuration files? Also check if I have a config.py file in /tmp

[Assistant]
[Uses RAG for Python docs]
[Calls filesystem:search_files to check for config.py]

Based on the docs, configparser is recommended. I found you have /tmp/config.py which uses JSON. Here's how to migrate...
```

## Switching Between Models

Test cheap local models with RAG augmentation:

```bash
# Try with Ollama (cheap local model) + RAG
uv run ai-workbench chat \
  --llm ollama:llama2 \
  --rag-source ~/data/python-vector-db \
  --mcp-server filesystem:npx:-y,@modelcontextprotocol/server-filesystem,/tmp

# Compare without RAG
uv run ai-workbench chat \
  --llm ollama:llama2 \
  --no-rag
```

Compare how well the local model performs with and without RAG augmentation.

## Tips

1. **RAG Quality**: Better scraped content = better RAG results
   - Use `--max-depth` and `--max-pages` to control crawler scope
   - Target documentation sites with good structure

2. **MCP Security**: Be careful with filesystem access
   - The filesystem server only accesses specified directories
   - In production, restrict access to safe directories only

3. **Model Selection**:
   - **Claude 3.5 Sonnet**: Best quality, fast, good for complex reasoning
   - **Ollama llama2**: Free local model, test with RAG augmentation
   - **Ollama mistral**: Good balance of speed and quality

4. **Temperature Settings**:
   - `0.0-0.3`: Deterministic, good for code generation
   - `0.5-0.7`: Balanced (default)
   - `0.8-1.0`: More creative, varied responses

5. **Save Important Sessions**:
   ```
   /save ~/sessions/python-tutorial-$(date +%Y%m%d).json
   ```

6. **Load Previous Sessions**:
   ```
   /load ~/sessions/python-tutorial-20260129.json
   ```

## Environment Variables

Create a `.env` file in your project:

```bash
# Required for chat
WORKBENCH_ANTHROPIC_API_KEY=sk-ant-...

# Required for RAG (embeddings + build-index)
WORKBENCH_MISTRAL_API_KEY=...

# Optional: Ollama base URL (default: http://localhost:11434)
WORKBENCH_OLLAMA_BASE_URL=http://localhost:11434

# Optional: Default settings
WORKBENCH_RAG_TOP_K=5
WORKBENCH_RAG_SCORE_THRESHOLD=0.7
WORKBENCH_CHAT_TEMPERATURE=0.7
```

## Next Steps

- Experiment with different MCP servers (GitHub, PostgreSQL, etc.)
- Build vector indices from your own documentation
- Compare model performance with/without RAG
- Create custom MCP servers for your specific needs
