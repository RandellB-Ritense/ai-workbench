"""
Main CLI entry point for AI Workbench.
Provides a unified command-line interface for all tools.
"""
import typer
from rich.console import Console
from typing import Optional, List
from pathlib import Path

app = typer.Typer(
    name="ai-workbench",
    help="A modular workbench for AI tools",
    no_args_is_help=True,
)

console = Console()


@app.command()
def crawl(
    url: str = typer.Option(..., "--url", "-u", help="URL to crawl"),
    output: Path = typer.Option(..., "--output", "-o", help="Output file path (e.g., /path/to/urls.json)"),
    max_depth: int = typer.Option(2, "--max-depth", "-d", help="Maximum crawl depth"),
    max_pages: int = typer.Option(100, "--max-pages", "-m", help="Maximum pages to crawl"),
    same_domain: bool = typer.Option(True, "--same-domain/--all-domains", help="Only crawl same domain"),
):
    """
    Crawl a website and extract URLs.

    Example:
        uv run ai-workbench crawl --url https://example.com --output ./urls.json
    """
    from ai_workbench.crawlers.web_crawler import WebCrawler

    console.print(f"[bold blue]Starting crawl of:[/bold blue] {url}")
    console.print(f"[bold blue]Output:[/bold blue] {output}")

    crawler = WebCrawler(
        start_url=url,
        max_depth=max_depth,
        max_pages=max_pages,
        same_domain_only=same_domain,
    )

    with console.status("[bold green]Crawling..."):
        results = crawler.crawl()

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    crawler.save_results(results, output)

    console.print(f"[bold green]✓[/bold green] Crawled {len(results)} URLs")
    console.print(f"[bold green]✓[/bold green] Results saved to {output}")


@app.command()
def scrape(
    url: str = typer.Option(..., "--url", "-u", help="URL to scrape"),
    output: Path = typer.Option(..., "--output", "-o", help="Output file path (JSON)"),
    ignore_links: bool = typer.Option(False, "--ignore-links", help="Strip links from markdown"),
    ignore_images: bool = typer.Option(False, "--ignore-images", help="Strip images from markdown"),
):
    """
    Scrape content from a web page and convert to markdown.

    Example:
        uv run ai-workbench scrape --url https://example.com --output ./content.json
    """
    from ai_workbench.scrapers.web_scraper import WebScraper

    console.print(f"[bold blue]Scraping:[/bold blue] {url}")
    console.print(f"[bold blue]Output:[/bold blue] {output}")

    scraper = WebScraper(
        ignore_links=ignore_links,
        ignore_images=ignore_images,
    )

    with console.status("[bold green]Scraping..."):
        result = scraper.scrape(url)

    # Save result
    scraper.save_results([result], output)

    if result.error:
        console.print(f"[bold red]✗[/bold red] Error: {result.error}")
    else:
        console.print(f"[bold green]✓[/bold green] Scraped: {result.title or 'Untitled'}")
        console.print(f"[bold green]✓[/bold green] Words: {result.word_count}")
        console.print(f"[bold green]✓[/bold green] Saved to {output}")


@app.command()
def scrape_batch(
    input: Path = typer.Option(..., "--input", "-i", help="Input file (crawler JSON output)"),
    output: Path = typer.Option(..., "--output", "-o", help="Output file path (JSON)"),
    ignore_links: bool = typer.Option(False, "--ignore-links", help="Strip links from markdown"),
    ignore_images: bool = typer.Option(False, "--ignore-images", help="Strip images from markdown"),
    max_urls: int = typer.Option(0, "--max-urls", help="Maximum URLs to scrape (0 = unlimited)"),
):
    """
    Scrape multiple URLs from crawler output.

    Example:
        uv run ai-workbench scrape-batch --input ./crawl-results.json --output ./scraped-content.json
    """
    from ai_workbench.scrapers.web_scraper import WebScraper
    import json

    console.print(f"[bold blue]Input:[/bold blue] {input}")
    console.print(f"[bold blue]Output:[/bold blue] {output}")

    # Load crawler output
    try:
        with open(input, "r") as f:
            crawler_data = json.load(f)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Error reading input file: {e}")
        raise typer.Exit(1)

    # Extract URLs
    urls = [page["url"] for page in crawler_data.get("pages", [])]

    if max_urls > 0:
        urls = urls[:max_urls]

    console.print(f"[bold blue]URLs to scrape:[/bold blue] {len(urls)}")

    scraper = WebScraper(
        ignore_links=ignore_links,
        ignore_images=ignore_images,
    )

    results = []
    with console.status("[bold green]Scraping pages...") as status:
        for i, url in enumerate(urls, 1):
            status.update(f"[bold green]Scraping {i}/{len(urls)}: {url}")
            result = scraper.scrape(url)
            results.append(result)

    # Save results
    scraper.save_results(results, output)

    # Summary
    successful = sum(1 for r in results if not r.error)
    failed = len(results) - successful
    total_words = sum(r.word_count for r in results)

    console.print(f"[bold green]✓[/bold green] Scraped {successful} pages successfully")
    if failed > 0:
        console.print(f"[bold yellow]![/bold yellow] Failed: {failed} pages")
    console.print(f"[bold green]✓[/bold green] Total words: {total_words:,}")
    console.print(f"[bold green]✓[/bold green] Saved to {output}")


@app.command()
def build_index(
    input: Path = typer.Option(..., "--input", "-i", help="Input file (scraped JSON)"),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory for vector database"),
    embedder: str = typer.Option("mistral", "--embedder", "-e", help="Embedder to use (mistral)"),
    chunk_size: int = typer.Option(500, "--chunk-size", help="Chunk size in tokens"),
    chunk_overlap: int = typer.Option(50, "--chunk-overlap", help="Chunk overlap in tokens"),
):
    """
    Build vector index from scraped documents.

    Example:
        uv run ai-workbench build-index --input ./scraped.json --output ./vector-db
    """
    from ai_workbench.embedders.document_processor import DocumentProcessor
    from ai_workbench.embedders.mistral_embedder import MistralEmbedder
    from ai_workbench.vector_stores.chroma_store import ChromaStore
    from ai_workbench.config import get_config

    config = get_config()

    console.print(f"[bold blue]Input:[/bold blue] {input}")
    console.print(f"[bold blue]Output:[/bold blue] {output}")
    console.print(f"[bold blue]Embedder:[/bold blue] {embedder}")

    # Check API key
    if embedder == "mistral" and not config.mistral_api_key:
        console.print("[bold red]✗[/bold red] Error: WORKBENCH_MISTRAL_API_KEY not set in environment")
        console.print("Set your Mistral API key:")
        console.print("  export WORKBENCH_MISTRAL_API_KEY=your-key-here")
        raise typer.Exit(1)

    # Initialize components
    console.print("[bold blue]Initializing components...[/bold blue]")
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if embedder == "mistral":
        embedder_model = MistralEmbedder(api_key=config.mistral_api_key)
    else:
        console.print(f"[bold red]✗[/bold red] Unknown embedder: {embedder}")
        raise typer.Exit(1)

    vector_store = ChromaStore(
        persist_directory=output,
        embedding_dimension=embedder_model.get_embedding_dimension(),
    )

    # Process documents
    console.print("[bold blue]Processing documents...[/bold blue]")
    with console.status("[bold green]Chunking documents..."):
        chunks = processor.process_scraped_file(input)

    console.print(f"[bold green]✓[/bold green] Created {len(chunks)} chunks from documents")

    # Get statistics
    stats = processor.get_chunk_stats(chunks)
    console.print(f"[bold blue]Documents:[/bold blue] {stats['total_documents']}")
    console.print(f"[bold blue]Total tokens:[/bold blue] {stats['total_tokens']:,}")
    console.print(f"[bold blue]Avg tokens/chunk:[/bold blue] {stats['avg_tokens_per_chunk']:.1f}")

    # Generate embeddings
    console.print("[bold blue]Generating embeddings...[/bold blue]")
    texts = [chunk.chunk_text for chunk in chunks]

    with console.status(f"[bold green]Embedding {len(texts)} chunks..."):
        embeddings = embedder_model.embed_batch(texts)

    console.print(f"[bold green]✓[/bold green] Generated {len(embeddings)} embeddings")

    # Store in vector database
    console.print("[bold blue]Storing in vector database...[/bold blue]")
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # Add source_url to metadata for easier filtering
    for i, chunk in enumerate(chunks):
        metadatas[i]["source_url"] = chunk.source_url
        metadatas[i]["chunk_index"] = chunk.chunk_index
        metadatas[i]["total_chunks"] = chunk.total_chunks

    with console.status("[bold green]Adding to vector store..."):
        vector_store.add(chunk_ids, texts, embeddings, metadatas)

    console.print(f"[bold green]✓[/bold green] Stored {vector_store.count()} vectors")
    console.print(f"[bold green]✓[/bold green] Vector database saved to {output}")


@app.command()
def test_rag(
    query: str = typer.Option(..., "--query", "-q", help="Search query"),
    rag_source: Path = typer.Option(..., "--rag-source", "-r", help="Path to vector database"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to retrieve"),
    show_context: bool = typer.Option(False, "--show-context", help="Show formatted context for LLM"),
):
    """
    Test RAG retrieval without LLM.

    Example:
        uv run ai-workbench test-rag --query "How to use crawler?" --rag-source ./vector-db
    """
    from ai_workbench.embedders.mistral_embedder import MistralEmbedder
    from ai_workbench.vector_stores.chroma_store import ChromaStore
    from ai_workbench.rag.retriever import RAGRetriever
    from ai_workbench.rag.context_builder import ContextBuilder
    from ai_workbench.config import get_config

    config = get_config()

    console.print(f"[bold blue]Query:[/bold blue] {query}")
    console.print(f"[bold blue]RAG Source:[/bold blue] {rag_source}")
    console.print(f"[bold blue]Top-K:[/bold blue] {top_k}")

    # Check API key
    if not config.mistral_api_key:
        console.print("[bold red]✗[/bold red] Error: WORKBENCH_MISTRAL_API_KEY not set")
        console.print("Set your Mistral API key:")
        console.print("  export WORKBENCH_MISTRAL_API_KEY=your-key-here")
        raise typer.Exit(1)

    # Check if vector DB exists
    if not rag_source.exists():
        console.print(f"[bold red]✗[/bold red] Error: Vector database not found at {rag_source}")
        console.print("Build an index first:")
        console.print(f"  uv run ai-workbench build-index --input scraped.json --output {rag_source}")
        raise typer.Exit(1)

    # Initialize components
    console.print("[bold blue]Initializing components...[/bold blue]")
    embedder = MistralEmbedder(api_key=config.mistral_api_key)
    vector_store = ChromaStore(
        persist_directory=rag_source,
        embedding_dimension=embedder.get_embedding_dimension(),
    )

    console.print(f"[bold blue]Vector store:[/bold blue] {vector_store.count()} documents")

    # Initialize retriever
    retriever = RAGRetriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=top_k,
        score_threshold=config.rag_score_threshold,
    )

    # Retrieve documents
    console.print(f"[bold blue]Searching...[/bold blue]")
    with console.status("[bold green]Retrieving documents..."):
        documents = retriever.retrieve(query, top_k=top_k)

    # Display results
    if not documents:
        console.print("[bold yellow]No documents found matching the query.[/bold yellow]")
        return

    console.print(f"\n[bold green]✓[/bold green] Found {len(documents)} relevant documents:\n")

    for i, doc in enumerate(documents, 1):
        console.print(f"[bold cyan]Document {i}:[/bold cyan]")
        console.print(f"  [bold]Title:[/bold] {doc.title or 'Untitled'}")
        console.print(f"  [bold]Source:[/bold] {doc.source_url}")
        console.print(f"  [bold]Score:[/bold] {doc.score:.3f}")
        console.print(f"  [bold]Chunk:[/bold] {doc.chunk_index + 1}")
        console.print(f"  [bold]Preview:[/bold] {doc.text[:200]}...")
        console.print()

    # Optionally show formatted context
    if show_context:
        context_builder = ContextBuilder(
            max_tokens=config.rag_context_max_tokens,
            include_sources=True,
        )
        result = context_builder.build_context(documents, query)

        console.print("[bold blue]Formatted Context for LLM:[/bold blue]")
        console.print(f"  [bold]Tokens used:[/bold] {result['tokens_used']}")
        console.print(f"  [bold]Documents included:[/bold] {len(result['documents_used'])}/{result['total_available']}")
        console.print(f"  [bold]Truncated:[/bold] {result['truncated']}")
        console.print()
        console.print("[dim]" + "-" * 80 + "[/dim]")
        console.print(result['context'])
        console.print("[dim]" + "-" * 80 + "[/dim]")


@app.command()
def llm_list(
    provider: List[str] = typer.Option(["all"], "--provider", "-p", help="Provider(s) to list (anthropic, ollama, all)"),
):
    """
    List available LLM models.

    Example:
        uv run ai-workbench llm-list --provider anthropic
        uv run ai-workbench llm-list --provider ollama
    """
    from ai_workbench.llm.anthropic_client import AnthropicClient
    from ai_workbench.llm.ollama_client import OllamaClient
    from ai_workbench.config import get_config

    config = get_config()

    # Determine which providers to list
    providers_to_check = []
    if "all" in provider:
        providers_to_check = ["anthropic", "ollama"]
    else:
        providers_to_check = provider

    # List Anthropic models
    if "anthropic" in providers_to_check:
        console.print("\n[bold cyan]Anthropic (Claude) Models:[/bold cyan]")

        if config.anthropic_api_key:
            client = AnthropicClient(api_key=config.anthropic_api_key)
            models = client.list_models()

            for model in models:
                console.print(f"\n  [bold]{model['name']}[/bold]")
                console.print(f"    ID: {model['id']}")
                console.print(f"    Context Window: {model['context_window']:,} tokens")
                console.print(f"    Max Output: {model['output_tokens']:,} tokens")
        else:
            console.print("  [yellow]API key not set (WORKBENCH_ANTHROPIC_API_KEY)[/yellow]")
            console.print("  Available models:")
            client = AnthropicClient(api_key="dummy")  # Just to list models
            models = client.list_models()
            for model in models:
                console.print(f"    - {model['name']} ({model['id']})")

    # List Ollama models
    if "ollama" in providers_to_check:
        console.print("\n[bold cyan]Ollama (Local) Models:[/bold cyan]")

        client = OllamaClient(base_url=config.ollama_base_url)

        if not client.is_running():
            console.print(f"  [yellow]Ollama not running at {config.ollama_base_url}[/yellow]")
            console.print("  Start Ollama:")
            console.print("    ollama serve")
            console.print("\n  Popular models to pull:")
            console.print("    ollama pull llama2")
            console.print("    ollama pull mistral")
            console.print("    ollama pull phi")
        else:
            models = client.list_models()

            if not models:
                console.print("  [yellow]No models installed[/yellow]")
                console.print("\n  Pull a model:")
                console.print("    ollama pull llama2")
            else:
                for model in models:
                    console.print(f"\n  [bold]{model['name']}[/bold]")
                    console.print(f"    Size: {model['size']}")
                    console.print(f"    Modified: {model['modified']}")

    console.print()


@app.command()
def chat(
    llm: str = typer.Option("claude-3-5-sonnet-20241022", "--llm", "-l", help="LLM model (claude-3-5-sonnet-20241022, ollama:llama2, etc.)"),
    rag_source: Optional[Path] = typer.Option(None, "--rag-source", "-r", help="Path to vector database for RAG"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="LLM temperature (0.0-1.0)"),
    max_tokens: int = typer.Option(4000, "--max-tokens", help="Maximum tokens per response"),
    no_rag: bool = typer.Option(False, "--no-rag", help="Disable RAG initially"),
):
    """
    Start interactive chat with LLM.

    Example:
        uv run ai-workbench chat --llm claude-3-5-sonnet-20241022 --rag-source ./vector-db
        uv run ai-workbench chat --llm ollama:llama2 --rag-source ./vector-db
    """
    from ai_workbench.llm.anthropic_client import AnthropicClient
    from ai_workbench.llm.ollama_client import OllamaClient
    from ai_workbench.embedders.mistral_embedder import MistralEmbedder
    from ai_workbench.vector_stores.chroma_store import ChromaStore
    from ai_workbench.rag.retriever import RAGRetriever
    from ai_workbench.rag.context_builder import ContextBuilder
    from ai_workbench.chatbot.interactive import InteractiveChatbot
    from ai_workbench.config import get_config

    config = get_config()

    # Initialize LLM
    console.print(f"[bold blue]Initializing LLM:[/bold blue] {llm}")

    if llm.startswith("ollama:"):
        # Ollama model
        model_name = llm.split(":", 1)[1]
        llm_client = OllamaClient(model=model_name, base_url=config.ollama_base_url)

        # Check if Ollama is running
        if not llm_client.is_running():
            console.print(f"[bold red]✗[/bold red] Ollama not running at {config.ollama_base_url}")
            console.print("Start Ollama:")
            console.print("  ollama serve")
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] Connected to Ollama")
    else:
        # Anthropic Claude
        if not config.anthropic_api_key:
            console.print("[bold red]✗[/bold red] Error: WORKBENCH_ANTHROPIC_API_KEY not set")
            console.print("Set your Anthropic API key:")
            console.print("  export WORKBENCH_ANTHROPIC_API_KEY=your-key-here")
            raise typer.Exit(1)

        llm_client = AnthropicClient(api_key=config.anthropic_api_key, model=llm)
        console.print(f"[green]✓[/green] Connected to Anthropic")

    # Initialize RAG if requested
    retriever = None
    context_builder = None
    rag_enabled = not no_rag

    if rag_source and rag_enabled:
        console.print(f"[bold blue]Initializing RAG:[/bold blue] {rag_source}")

        # Check if vector DB exists
        if not rag_source.exists():
            console.print(f"[bold red]✗[/bold red] Vector database not found at {rag_source}")
            console.print("Build an index first:")
            console.print(f"  uv run ai-workbench build-index --input scraped.json --output {rag_source}")
            raise typer.Exit(1)

        # Check API key for embeddings
        if not config.mistral_api_key:
            console.print("[bold red]✗[/bold red] Error: WORKBENCH_MISTRAL_API_KEY not set")
            console.print("RAG requires Mistral API for embeddings:")
            console.print("  export WORKBENCH_MISTRAL_API_KEY=your-key-here")
            raise typer.Exit(1)

        # Initialize RAG components
        embedder = MistralEmbedder(api_key=config.mistral_api_key)
        vector_store = ChromaStore(
            persist_directory=rag_source,
            embedding_dimension=embedder.get_embedding_dimension(),
        )

        retriever = RAGRetriever(
            embedder=embedder,
            vector_store=vector_store,
            top_k=config.rag_top_k,
            score_threshold=config.rag_score_threshold,
        )

        context_builder = ContextBuilder(
            max_tokens=config.rag_context_max_tokens,
            include_sources=True,
        )

        console.print(f"[green]✓[/green] RAG initialized ({vector_store.count()} documents)")

    # Start interactive chatbot
    console.print("[bold blue]Starting interactive chat...[/bold blue]\n")

    chatbot = InteractiveChatbot(
        llm=llm_client,
        retriever=retriever,
        context_builder=context_builder,
        rag_enabled=rag_enabled,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    chatbot.run()


@app.command()
def embed(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input file path"),
    output: Path = typer.Option(..., "--output", "-o", help="Output file path"),
    model: str = typer.Option("sentence-transformers", "--model", "-m", help="Embedding model to use"),
):
    """
    Generate embeddings for text data.

    Example:
        uv run ai-workbench embed --input ./documents.txt --output ./embeddings.json
    """
    console.print("[yellow]Embedder module coming soon![/yellow]")


@app.command()
def version():
    """Show version information."""
    console.print("[bold]AI Workbench[/bold] v0.1.0")


if __name__ == "__main__":
    app()
