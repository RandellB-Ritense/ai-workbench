# AI Workbench

A modular playground for AI tools including web crawlers, scrapers, embedders, and more.

Built with [UV](https://github.com/astral-sh/uv) for blazingly fast dependency management.

## Features

- **Modular Architecture**: Each tool is cleanly separated into its own module
- **Unified CLI**: Single command-line interface for all tools
- **Flexible I/O**: Configure input and output paths outside the project
- **Web Crawler**: Discover and map website URLs with configurable depth
- **Web Scraper**: Extract and convert HTML content to clean markdown
- **Batch Processing**: Scrape multiple URLs from crawler output

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

## Usage

### Web Crawler

Crawl a website and extract URLs:

```bash
uv run ai-workbench crawl --url https://example.com --output /path/to/output/urls.json
```

### Web Scraper

Scrape a single page and convert to markdown:

```bash
uv run ai-workbench scrape --url https://example.com --output /path/to/content.json
```

Scrape multiple pages from crawler output:

```bash
# First crawl to discover URLs
uv run ai-workbench crawl --url https://docs.example.com --output ./urls.json

# Then scrape all discovered pages
uv run ai-workbench scrape-batch --input ./urls.json --output ./content.json
```

## Project Structure

```
ai-workbench/
├── src/
│   └── ai_workbench/
│       ├── crawlers/      # Web crawling tools (✓ Implemented)
│       │   └── web_crawler.py
│       ├── scrapers/      # Web scraping tools (✓ Implemented)
│       │   └── web_scraper.py
│       ├── embedders/     # Text embedding tools
│       └── utils/         # Shared utilities
│           └── io.py
└── tests/                 # Test suite
```

## Development

Development dependencies are automatically installed with:

```bash
uv sync
```

Run commands in the UV environment:

```bash
uv run ai-workbench --help
uv run pytest  # Run tests
uv run ruff check  # Lint code
uv run black .  # Format code
```
