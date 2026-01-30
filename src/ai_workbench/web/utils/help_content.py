"""
Help content and tooltips for the web UI.
"""

# Tab descriptions
TAB_DESCRIPTIONS = {
    "data_collection": """
        **Data Collection** allows you to crawl websites and scrape content for processing.

        1. **Crawler**: Extract URLs from websites by following links
        2. **Scraper**: Convert HTML pages to clean markdown format

        All operations run in the background with progress tracking.
    """,

    "vector_index": """
        **Vector Index** builds searchable embeddings from your scraped content.

        1. **Build Index**: Convert documents to vector embeddings
        2. **Test RAG**: Search your indexed documents by similarity

        Requires a Mistral API key for generating embeddings.
    """,

    "chat": """
        **Chat** provides an interactive AI assistant with optional RAG and MCP tools.

        - Choose your LLM provider (Anthropic or Ollama)
        - Optionally enable RAG to use your indexed documents
        - Connect MCP servers for additional capabilities

        All responses stream in real-time.
    """,

    "jobs": """
        **Jobs** shows all background tasks with their status and progress.

        - Monitor active jobs in real-time
        - Browse job history
        - Cancel running jobs
        - View detailed job information
    """,

    "config": """
        **Configuration** manages API keys, default settings, and system info.

        - Store API keys securely
        - Customize default parameters
        - View system diagnostics
        - Check service status
    """
}

# Field tooltips
FIELD_TOOLTIPS = {
    # Crawler
    "crawler_url": "The starting URL to begin crawling. Must be a valid HTTP(S) URL.",
    "crawler_max_depth": "How many links deep to follow from the start URL. Higher values find more pages but take longer.",
    "crawler_max_pages": "Maximum total pages to crawl. Acts as a safety limit to prevent runaway crawls.",
    "crawler_same_domain": "Only follow links within the same domain as the start URL. Recommended for focused crawls.",

    # Scraper
    "scraper_input_type": "Choose how to provide URLs: upload a crawler JSON file or paste URLs directly.",
    "scraper_ignore_links": "Remove all links from the markdown output for cleaner text processing.",
    "scraper_ignore_images": "Remove all images from the markdown output to reduce size.",
    "scraper_max_urls": "Limit how many URLs to scrape from the input. Set to 0 for unlimited.",

    # Vector Index
    "index_chunk_size": "Number of tokens per text chunk. Larger chunks provide more context but less precision.",
    "index_chunk_overlap": "Overlap between consecutive chunks to maintain context across boundaries.",
    "index_api_key": "Your Mistral API key for generating embeddings. Get one at https://console.mistral.ai/",

    # RAG Test
    "rag_top_k": "Number of most similar documents to retrieve for each query.",
    "rag_score_threshold": "Minimum similarity score (0-1) required to include a document in results.",

    # Chat
    "chat_temperature": "Controls randomness. Lower = more focused, higher = more creative.",
    "chat_max_tokens": "Maximum length of the assistant's response.",
    "chat_rag_enabled": "Use your vector database to provide context for responses.",

    # MCP
    "mcp_server_name": "Unique identifier for this MCP server (e.g., 'filesystem', 'github').",
    "mcp_command": "Command to launch the MCP server (e.g., 'npx', 'python').",
    "mcp_args": "Arguments for the command, separated by commas (e.g., '-y,@mcp/server-filesystem,/tmp').",
}

# Error messages
ERROR_MESSAGES = {
    "no_file": "Please upload a file before proceeding.",
    "invalid_url": "Invalid URL format. URL must start with http:// or https://",
    "no_api_key": "API key is required. Please enter your API key in the Configuration tab.",
    "connection_failed": "Connection failed. Please check your network connection and try again.",
    "server_not_running": "Server not running. Please ensure the service is started.",
    "invalid_json": "Invalid JSON file. Please upload a valid JSON file from the crawler or scraper.",
    "no_results": "No results found. Try adjusting your search parameters or query.",
    "job_failed": "Job failed to complete. Check the job details for error information.",
}

# Success messages
SUCCESS_MESSAGES = {
    "job_submitted": "Job submitted successfully! Monitor progress in the Jobs tab.",
    "api_key_saved": "API keys saved successfully to ~/.ai-workbench/.env",
    "settings_saved": "Settings saved successfully to ~/.ai-workbench/config.json",
    "connection_success": "Connected successfully!",
    "operation_complete": "Operation completed successfully!",
}

# Warning messages
WARNING_MESSAGES = {
    "large_crawl": "Warning: Large crawls may take a long time and consume significant resources.",
    "many_urls": "Warning: Scraping many URLs will take considerable time.",
    "no_rag": "RAG is not enabled. Responses will not use your indexed documents.",
    "no_mcp": "No MCP servers connected. Advanced tool use is not available.",
}

# Help sections for each tab
HELP_SECTIONS = {
    "crawler": {
        "title": "How to Use the Crawler",
        "content": """
            1. Enter a starting URL (e.g., https://docs.python.org)
            2. Set max depth (2-3 is usually sufficient)
            3. Set max pages to limit the crawl size
            4. Check "Same Domain Only" to stay on one site
            5. Click "Start Crawl" and monitor progress
            6. Download the results when complete

            **Tips:**
            - Start with small values (depth=2, pages=50) for testing
            - Higher depth finds more pages but takes much longer
            - Use the output JSON for batch scraping
        """
    },

    "scraper": {
        "title": "How to Use the Scraper",
        "content": """
            1. Choose input method:
               - Upload crawler JSON for batch processing
               - Or paste URLs directly (one per line)
            2. Configure options (links, images)
            3. Set max URLs to limit the batch
            4. Click "Start Scraping" and wait
            5. Download the scraped content

            **Tips:**
            - Scraping 100+ URLs can take 10-15 minutes
            - Ignore links/images for cleaner text
            - Use output for building vector indexes
        """
    },

    "vector_index": {
        "title": "How to Build an Index",
        "content": """
            1. Upload scraped content JSON
            2. Give your index a unique name
            3. Enter your Mistral API key
            4. Adjust chunk size/overlap if needed
            5. Click "Build Index" and wait
            6. Test your index with the search tool

            **Tips:**
            - Building an index for 50 docs takes 2-5 minutes
            - Smaller chunks = more precise search
            - Larger chunks = better context
            - 500 tokens is a good default
        """
    },

    "chat": {
        "title": "How to Chat with AI",
        "content": """
            **Setup:**
            1. Choose your LLM provider (Anthropic or Ollama)
            2. Enter API key or configure Ollama
            3. Select a model
            4. Click "Initialize LLM"

            **Optional - Enable RAG:**
            1. Check "Enable RAG"
            2. Select a vector database
            3. Enter Mistral API key
            4. Click "Initialize RAG"

            **Optional - Connect MCP:**
            1. Enter server details in accordion
            2. Click "Connect Server"
            3. Tools will appear in the table

            **Chat:**
            - Type your message and press Enter
            - Responses stream in real-time
            - Save sessions with the save button
        """
    },
}

# FAQ
FAQ = {
    "What is RAG?": """
        RAG (Retrieval-Augmented Generation) enhances AI responses by providing
        relevant context from your indexed documents. When you ask a question,
        the system finds related documents and includes them in the prompt.
    """,

    "What is MCP?": """
        MCP (Model Context Protocol) allows AI assistants to use external tools
        like reading files, accessing APIs, or running commands. MCP servers
        provide specific capabilities that the AI can invoke during conversations.
    """,

    "Why use Ollama?": """
        Ollama runs LLMs locally on your machine, providing:
        - Complete privacy (no data sent externally)
        - No API costs
        - Works offline
        - Faster for repeated use

        However, Anthropic's Claude models are generally more capable.
    """,

    "How long does indexing take?": """
        Indexing time depends on document count:
        - 10 documents: ~30 seconds
        - 50 documents: 2-5 minutes
        - 100 documents: 5-10 minutes

        The bottleneck is usually the embedding API calls.
    """,

    "Can I cancel a running job?": """
        Yes! Go to the Jobs tab, find your job in the active jobs table,
        copy the job ID, paste it in the "Job ID to Cancel" field, and
        click "Cancel Job". The job will stop gracefully.
    """,
}
