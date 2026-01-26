"""
Web scraper module for extracting and converting web page content to markdown.
"""
import json
import httpx
import html2text
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ScrapeResult:
    """Result of scraping a single page."""
    url: str
    title: Optional[str]
    markdown_content: str
    word_count: int
    scraped_at: str
    status_code: int
    error: Optional[str] = None
    metadata: Optional[Dict] = None


class WebScraper:
    """
    Web scraper for converting HTML pages to markdown.

    Attributes:
        ignore_links: Whether to strip links from markdown output
        ignore_images: Whether to strip images from markdown output
        body_width: Width for wrapping text (0 = no wrap)
    """

    def __init__(
        self,
        ignore_links: bool = False,
        ignore_images: bool = False,
        body_width: int = 0,
    ):
        self.ignore_links = ignore_links
        self.ignore_images = ignore_images
        self.body_width = body_width

        # Configure html2text converter
        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = ignore_links
        self.converter.ignore_images = ignore_images
        self.converter.body_width = body_width
        self.converter.ignore_emphasis = False
        self.converter.skip_internal_links = False
        self.converter.single_line_break = False

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract metadata from HTML."""
        metadata = {}

        # Extract meta tags
        meta_description = soup.find("meta", attrs={"name": "description"})
        if meta_description:
            metadata["description"] = meta_description.get("content", "")

        meta_keywords = soup.find("meta", attrs={"name": "keywords"})
        if meta_keywords:
            metadata["keywords"] = meta_keywords.get("content", "")

        # Extract Open Graph tags
        og_tags = soup.find_all("meta", property=lambda x: x and x.startswith("og:"))
        if og_tags:
            metadata["og"] = {
                tag.get("property", "").replace("og:", ""): tag.get("content", "")
                for tag in og_tags
            }

        # Extract canonical URL
        canonical = soup.find("link", rel="canonical")
        if canonical:
            metadata["canonical_url"] = canonical.get("href", "")

        return metadata

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title from HTML."""
        # Try <title> tag first
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()

        # Try h1 as fallback
        h1_tag = soup.find("h1")
        if h1_tag:
            return h1_tag.get_text().strip()

        # Try Open Graph title
        og_title = soup.find("meta", property="og:title")
        if og_title:
            return og_title.get("content", "").strip()

        return None

    def _clean_html(self, soup: BeautifulSoup) -> str:
        """Clean HTML by removing unwanted elements."""
        # Remove script and style elements
        for script in soup(["script", "style", "noscript"]):
            script.decompose()

        # Remove navigation, footer, header (optional - can be configured)
        for nav in soup(["nav", "footer", "header"]):
            nav.decompose()

        return str(soup)

    def scrape(self, url: str) -> ScrapeResult:
        """
        Scrape a single URL and convert to markdown.

        Args:
            url: URL to scrape

        Returns:
            ScrapeResult object
        """
        try:
            with httpx.Client(follow_redirects=True, timeout=30.0) as client:
                response = client.get(url)
                response.raise_for_status()

                # Parse HTML
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract metadata and title
                title = self._extract_title(soup)
                metadata = self._extract_metadata(soup)

                # Clean HTML before conversion
                cleaned_html = self._clean_html(soup)

                # Convert to markdown
                markdown_content = self.converter.handle(cleaned_html)

                # Count words
                word_count = len(markdown_content.split())

                return ScrapeResult(
                    url=url,
                    title=title,
                    markdown_content=markdown_content,
                    word_count=word_count,
                    scraped_at=datetime.now().isoformat(),
                    status_code=response.status_code,
                    metadata=metadata,
                )

        except Exception as e:
            return ScrapeResult(
                url=url,
                title=None,
                markdown_content="",
                word_count=0,
                scraped_at=datetime.now().isoformat(),
                status_code=0,
                error=str(e),
            )

    def scrape_batch(self, urls: List[str]) -> List[ScrapeResult]:
        """
        Scrape multiple URLs.

        Args:
            urls: List of URLs to scrape

        Returns:
            List of ScrapeResult objects
        """
        results = []
        for url in urls:
            result = self.scrape(url)
            results.append(result)
        return results

    def scrape_from_crawler_output(self, crawler_output_path: Path) -> List[ScrapeResult]:
        """
        Scrape URLs from crawler JSON output.

        Args:
            crawler_output_path: Path to crawler output JSON file

        Returns:
            List of ScrapeResult objects
        """
        with open(crawler_output_path, "r") as f:
            crawler_data = json.load(f)

        # Extract URLs from crawler output
        urls = [page["url"] for page in crawler_data.get("pages", [])]

        return self.scrape_batch(urls)

    def save_results(self, results: List[ScrapeResult], output_path: Path):
        """
        Save scrape results to a JSON file.

        Args:
            results: List of ScrapeResult objects
            output_path: Path to output file
        """
        # Convert results to dict format
        results_dict = {
            "total_pages": len(results),
            "total_words": sum(r.word_count for r in results),
            "scraped_at": datetime.now().isoformat(),
            "pages": [asdict(result) for result in results],
        }

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to JSON
        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)
