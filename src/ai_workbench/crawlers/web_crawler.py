"""
Web crawler module for extracting URLs from websites.
"""
import json
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Set, List, Dict, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class CrawlResult:
    """Result of a single page crawl."""
    url: str
    status_code: int
    depth: int
    links_found: int
    title: Optional[str] = None
    error: Optional[str] = None


class WebCrawler:
    """
    Web crawler for extracting URLs from websites.

    Attributes:
        start_url: Starting URL for the crawl
        max_depth: Maximum depth to crawl
        max_pages: Maximum number of pages to crawl
        same_domain_only: Only crawl URLs from the same domain
    """

    def __init__(
        self,
        start_url: str,
        max_depth: int = 2,
        max_pages: int = 100,
        same_domain_only: bool = True,
    ):
        self.start_url = start_url
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.same_domain_only = same_domain_only

        self.visited: Set[str] = set()
        self.to_visit: List[tuple[str, int]] = [(start_url, 0)]
        self.results: List[CrawlResult] = []

        # Parse the starting domain
        self.start_domain = urlparse(start_url).netloc

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL should be crawled."""
        parsed = urlparse(url)

        # Must have a scheme and netloc
        if not parsed.scheme or not parsed.netloc:
            return False

        # Check same domain constraint
        if self.same_domain_only and parsed.netloc != self.start_domain:
            return False

        # Skip common non-html resources
        skip_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.tar', '.gz'}
        if any(parsed.path.lower().endswith(ext) for ext in skip_extensions):
            return False

        return True

    def _extract_links(self, html: str, base_url: str) -> Set[str]:
        """Extract all links from HTML content."""
        soup = BeautifulSoup(html, 'html.parser')
        links = set()

        for tag in soup.find_all('a', href=True):
            href = tag['href']
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            # Remove fragments
            absolute_url = absolute_url.split('#')[0]

            if self._is_valid_url(absolute_url):
                links.add(absolute_url)

        return links

    def _extract_title(self, html: str) -> Optional[str]:
        """Extract page title from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        title_tag = soup.find('title')
        return title_tag.string.strip() if title_tag and title_tag.string else None

    def _crawl_page(self, url: str, depth: int) -> CrawlResult:
        """Crawl a single page."""
        try:
            with httpx.Client(follow_redirects=True, timeout=10.0) as client:
                response = client.get(url)
                response.raise_for_status()

                # Extract links and title
                links = self._extract_links(response.text, url)
                title = self._extract_title(response.text)

                # Add new links to queue if not at max depth
                if depth < self.max_depth:
                    for link in links:
                        if link not in self.visited and link not in [u for u, _ in self.to_visit]:
                            self.to_visit.append((link, depth + 1))

                return CrawlResult(
                    url=url,
                    status_code=response.status_code,
                    depth=depth,
                    links_found=len(links),
                    title=title,
                )

        except Exception as e:
            return CrawlResult(
                url=url,
                status_code=0,
                depth=depth,
                links_found=0,
                error=str(e),
            )

    def crawl(self, progress_callback: Optional[Callable[[float, str], None]] = None) -> List[CrawlResult]:
        """
        Start the crawl process.

        Args:
            progress_callback: Optional callback for progress updates.
                             Called with (progress: float 0-1, message: str)

        Returns:
            List of CrawlResult objects
        """
        while self.to_visit and len(self.visited) < self.max_pages:
            url, depth = self.to_visit.pop(0)

            if url in self.visited:
                continue

            self.visited.add(url)
            result = self._crawl_page(url, depth)
            self.results.append(result)

            # Update progress
            if progress_callback:
                progress = len(self.visited) / self.max_pages
                message = f"Crawled {len(self.visited)}/{self.max_pages} pages (depth {depth}): {url[:60]}..."
                progress_callback(progress, message)

        return self.results

    def save_results(self, results: List[CrawlResult], output_path: Path):
        """
        Save crawl results to a file.

        Args:
            results: List of CrawlResult objects
            output_path: Path to output file
        """
        # Convert results to dict format
        results_dict = {
            "start_url": self.start_url,
            "total_pages": len(results),
            "pages": [asdict(result) for result in results],
        }

        # Save based on file extension
        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
        else:
            # Default to JSON
            with open(output_path, 'w') as f:
                json.dump(results_dict, f, indent=2)

    def get_url_list(self) -> List[str]:
        """Get a simple list of all crawled URLs."""
        return [result.url for result in self.results]
