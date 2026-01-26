# Quick Start Guide

## Prerequisites

If you don't have UV installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

Install using UV (fast Python package manager):

```bash
uv sync
```

This automatically creates a virtual environment and installs all dependencies.

UV is 10-100x faster than pip and provides deterministic dependency resolution.

## Usage Examples

### Web Crawler

**Basic crawl:**
```bash
uv run ai-workbench crawl \
  --url https://example.com \
  --output ~/my-data/crawl-results.json
```

**Advanced crawl with options:**
```bash
uv run ai-workbench crawl \
  --url https://example.com \
  --output ~/my-data/crawl-results.json \
  --max-depth 3 \
  --max-pages 200 \
  --all-domains  # Follow links to external domains
```

**Output format:**
```json
{
  "start_url": "https://example.com",
  "total_pages": 42,
  "pages": [
    {
      "url": "https://example.com",
      "status_code": 200,
      "depth": 0,
      "links_found": 15,
      "title": "Example Domain",
      "error": null
    },
    ...
  ]
}
```

### Configuration

**Option 1: Environment variables**
```bash
export WORKBENCH_DEFAULT_OUTPUT_DIR=~/my-ai-data
export WORKBENCH_CRAWLER_MAX_DEPTH=3
export WORKBENCH_CRAWLER_MAX_PAGES=500
```

**Option 2: .env file**
```bash
# Copy the example and customize
cp .env.example .env
# Edit .env with your settings
```

**Option 3: Command-line flags**
```bash
uv run ai-workbench crawl --url https://example.com --max-depth 3 --output ./results.json
```

## Web Scraper

### Basic Scraping

**Scrape a single page:**
```bash
uv run ai-workbench scrape \
  --url https://example.com \
  --output ~/data/content.json
```

**Scrape without links and images:**
```bash
uv run ai-workbench scrape \
  --url https://example.com \
  --output ~/data/content.json \
  --ignore-links \
  --ignore-images
```

### Batch Scraping

**Scrape multiple pages from crawler output:**
```bash
# Step 1: Crawl to discover URLs
uv run ai-workbench crawl \
  --url https://docs.example.com \
  --output ~/data/crawl-results.json \
  --max-depth 3

# Step 2: Scrape all discovered pages
uv run ai-workbench scrape-batch \
  --input ~/data/crawl-results.json \
  --output ~/data/scraped-content.json
```

**Limit number of pages to scrape:**
```bash
uv run ai-workbench scrape-batch \
  --input ~/data/crawl-results.json \
  --output ~/data/scraped-content.json \
  --max-urls 50
```

**Output format:**
```json
{
  "total_pages": 42,
  "total_words": 15234,
  "scraped_at": "2026-01-26T09:55:16.080222",
  "pages": [
    {
      "url": "https://example.com/page",
      "title": "Page Title",
      "markdown_content": "# Page Title\n\nContent here...",
      "word_count": 350,
      "status_code": 200,
      "metadata": {
        "description": "Page description",
        "keywords": "keyword1, keyword2"
      }
    }
  ]
}
```

## Common Workflows

### 1. Crawl and scrape documentation
```bash
# Discover all documentation pages
uv run ai-workbench crawl \
  --url https://docs.python.org \
  --output ~/data/python-docs-urls.json \
  --max-depth 3 \
  --max-pages 500

# Scrape all pages to markdown
uv run ai-workbench scrape-batch \
  --input ~/data/python-docs-urls.json \
  --output ~/data/python-docs-content.json
```

### 2. Crawl a documentation site
```bash
uv run ai-workbench crawl \
  --url https://docs.python.org \
  --output ~/data/python-docs-urls.json \
  --max-depth 3 \
  --same-domain
```

### 2. Crawl multiple starting points
```bash
# Create a simple script
for url in "https://site1.com" "https://site2.com"; do
  uv run ai-workbench crawl \
    --url "$url" \
    --output ~/data/crawl-$(echo $url | md5).json
done
```

### 3. Output to external directory
```bash
# All outputs go outside the project
uv run ai-workbench crawl \
  --url https://example.com \
  --output /Users/you/Documents/ai-projects/data/crawl-2024-01-26.json
```

## Next Steps

- **Add scrapers**: Implement content extraction in `src/ai_workbench/scrapers/`
- **Add embedders**: Add text embedding tools in `src/ai_workbench/embedders/`
- **Extend CLI**: Add new commands in `src/ai_workbench/cli.py`
- **Custom modules**: Create new tool categories as needed

## Troubleshooting

**Command not found after installation:**
```bash
# Use uv run to execute commands
uv run ai-workbench --help

# Or activate the virtual environment
source .venv/bin/activate
ai-workbench --help

# Reinstall dependencies
uv sync
```

**Output directory not writable:**
```bash
# Check permissions
mkdir -p ~/my-output-dir
chmod u+w ~/my-output-dir
```
