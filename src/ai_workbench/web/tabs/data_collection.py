"""
Data Collection tab - Web crawler and scraper interface.
"""
import gradio as gr
import json
from pathlib import Path
from typing import Optional

from ...crawlers.web_crawler import WebCrawler
from ...scrapers.web_scraper import WebScraper


def create_data_collection_tab() -> gr.Tab:
    """
    Create the Data Collection tab.

    Returns:
        Gradio Tab component
    """
    with gr.Tab("Data Collection") as tab:
        gr.Markdown("## Web Crawler")
        gr.Markdown("Crawl a website to extract URLs for later scraping.")

        with gr.Row():
            with gr.Column(scale=2):
                # Crawler inputs
                crawler_url = gr.Textbox(
                    label="Start URL",
                    placeholder="https://example.com",
                    info="The URL to start crawling from"
                )

                with gr.Row():
                    max_depth = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=2,
                        step=1,
                        label="Max Depth",
                        info="Maximum crawl depth from start URL"
                    )

                    max_pages = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10,
                        label="Max Pages",
                        info="Maximum number of pages to crawl"
                    )

                same_domain = gr.Checkbox(
                    value=True,
                    label="Same Domain Only",
                    info="Only crawl pages within the same domain"
                )

                # Output directory
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value=str(Path.home() / "ai-workbench-output"),
                    info="Directory to save crawler results"
                )

                crawl_button = gr.Button("Start Crawl", variant="primary", size="lg")

            with gr.Column(scale=1):
                # Status display
                crawl_status = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False,
                    lines=5
                )

        # Results section
        with gr.Row():
            with gr.Column():
                crawl_result_display = gr.JSON(
                    label="Crawl Results",
                    visible=False
                )

                crawl_download_button = gr.File(
                    label="Download Results",
                    visible=False
                )

        # Scraper Section
        gr.Markdown("---")
        gr.Markdown("## Web Scraper")
        gr.Markdown("Scrape content from URLs and convert to markdown format.")

        with gr.Row():
            with gr.Column(scale=2):
                # Scraper inputs
                scraper_input_type = gr.Radio(
                    choices=["Upload Crawler JSON", "Paste URLs"],
                    value="Upload Crawler JSON",
                    label="Input Method"
                )

                scraper_file_input = gr.File(
                    label="Crawler Output JSON - Upload the crawler output file",
                    file_types=[".json"],
                    visible=True
                )

                scraper_urls_input = gr.Textbox(
                    label="URLs (one per line)",
                    placeholder="https://example.com/page1\nhttps://example.com/page2",
                    lines=5,
                    visible=False
                )

                with gr.Row():
                    ignore_links = gr.Checkbox(
                        value=False,
                        label="Ignore Links",
                        info="Strip links from markdown output"
                    )

                    ignore_images = gr.Checkbox(
                        value=False,
                        label="Ignore Images",
                        info="Strip images from markdown output"
                    )

                max_urls = gr.Slider(
                    minimum=1,
                    maximum=500,
                    value=100,
                    step=1,
                    label="Max URLs",
                    info="Maximum URLs to scrape"
                )

                scraper_output_dir = gr.Textbox(
                    label="Output Directory",
                    value=str(Path.home() / "ai-workbench-output"),
                    info="Directory to save scraped content"
                )

                scrape_button = gr.Button("Start Scraping", variant="primary", size="lg")

            with gr.Column(scale=1):
                # Status display
                scrape_status = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False,
                    lines=5
                )

        # Results section
        with gr.Row():
            with gr.Column():
                scrape_result_display = gr.JSON(
                    label="Scrape Results Summary",
                    visible=False
                )

                scrape_download_button = gr.File(
                    label="Download Scraped Content",
                    visible=False
                )

        # Event handlers
        def start_crawl(url: str, depth: int, pages: int, same_dom: bool, output: str, progress=gr.Progress()):
            """Execute crawler synchronously with progress tracking."""
            if not url:
                return {
                    crawl_status: "❌ Error: URL is required",
                    crawl_result_display: gr.update(visible=False),
                    crawl_download_button: gr.update(visible=False)
                }

            try:
                progress(0, desc="Initializing crawler...")

                # Initialize crawler
                crawler = WebCrawler(
                    start_url=url,
                    max_depth=depth,
                    max_pages=pages,
                    same_domain_only=same_dom
                )

                progress(0.1, desc="Starting crawl...")

                # Crawl with progress callback
                def progress_callback(prog, message):
                    progress(0.1 + prog * 0.8, desc=message)

                results = crawler.crawl(progress_callback=progress_callback)

                progress(0.9, desc=f"Crawled {len(results)} URLs, saving results...")

                # Save results
                output_dir_path = Path(output)
                output_dir_path.mkdir(parents=True, exist_ok=True)

                import uuid
                output_file = output_dir_path / f"crawl-{str(uuid.uuid4())[:8]}.json"
                crawler.save_results(results, output_file)

                progress(1.0, desc="Complete!")

                result_summary = {
                    "url_count": len(results),
                    "max_depth_reached": crawler.max_depth,
                    "output_file": str(output_file),
                    "start_url": url
                }

                return {
                    crawl_status: f"✅ Complete! Crawled {len(results)} URLs\nSaved to: {output_file.name}",
                    crawl_result_display: gr.update(value=result_summary, visible=True),
                    crawl_download_button: gr.update(value=str(output_file), visible=True)
                }

            except Exception as e:
                return {
                    crawl_status: f"❌ Error: {str(e)}",
                    crawl_result_display: gr.update(visible=False),
                    crawl_download_button: gr.update(visible=False)
                }

        def toggle_scraper_input(input_type):
            """Toggle between file upload and URL paste."""
            if input_type == "Upload Crawler JSON":
                return {
                    scraper_file_input: gr.update(visible=True),
                    scraper_urls_input: gr.update(visible=False)
                }
            else:
                return {
                    scraper_file_input: gr.update(visible=False),
                    scraper_urls_input: gr.update(visible=True)
                }

        def start_scrape(input_type, file_input, urls_input, ignore_links_val, ignore_images_val, max_urls_val, output, progress=gr.Progress()):
            """Execute scraper synchronously with progress tracking."""
            try:
                # Get URLs based on input type
                urls = []
                if input_type == "Upload Crawler JSON":
                    if not file_input:
                        return {
                            scrape_status: "❌ Error: No file uploaded",
                            scrape_result_display: gr.update(visible=False),
                            scrape_download_button: gr.update(visible=False)
                        }

                    with open(file_input, 'r') as f:
                        crawler_data = json.load(f)
                    urls = [page["url"] for page in crawler_data.get("pages", [])]
                else:
                    if not urls_input:
                        return {
                            scrape_status: "❌ Error: No URLs provided",
                            scrape_result_display: gr.update(visible=False),
                            scrape_download_button: gr.update(visible=False)
                        }
                    urls = [url.strip() for url in urls_input.split('\n') if url.strip()]

                if not urls:
                    return {
                        scrape_status: "❌ Error: No valid URLs found",
                        scrape_result_display: gr.update(visible=False),
                        scrape_download_button: gr.update(visible=False)
                    }

                # Limit URLs
                if max_urls_val > 0:
                    urls = urls[:max_urls_val]

                progress(0, desc="Initializing scraper...")

                # Initialize scraper
                scraper = WebScraper(
                    ignore_links=ignore_links_val,
                    ignore_images=ignore_images_val
                )

                progress(0.05, desc=f"Starting to scrape {len(urls)} URLs...")

                # Scrape with progress callback
                def progress_callback(prog, message):
                    progress(0.05 + prog * 0.9, desc=message)

                results = scraper.scrape_batch(
                    urls=urls,
                    progress_callback=progress_callback
                )

                progress(0.95, desc=f"Scraped {len(results)} pages, saving results...")

                # Save results
                output_dir_path = Path(output)
                output_dir_path.mkdir(parents=True, exist_ok=True)

                import uuid
                output_file = output_dir_path / f"scraped-{str(uuid.uuid4())[:8]}.json"
                scraper.save_results(results, output_file)

                # Calculate stats
                successful = sum(1 for r in results if not r.error)
                failed = len(results) - successful
                total_words = sum(r.word_count for r in results)

                progress(1.0, desc="Complete!")

                result_summary = {
                    "total_pages": len(results),
                    "successful": successful,
                    "failed": failed,
                    "total_words": total_words,
                    "output_file": str(output_file)
                }

                return {
                    scrape_status: f"✅ Complete! Scraped {successful} pages ({total_words:,} words)\n{failed} failed\nSaved to: {output_file.name}",
                    scrape_result_display: gr.update(value=result_summary, visible=True),
                    scrape_download_button: gr.update(value=str(output_file), visible=True)
                }

            except Exception as e:
                return {
                    scrape_status: f"❌ Error: {str(e)}",
                    scrape_result_display: gr.update(visible=False),
                    scrape_download_button: gr.update(visible=False)
                }

        # Connect events
        crawl_button.click(
            fn=start_crawl,
            inputs=[crawler_url, max_depth, max_pages, same_domain, output_dir],
            outputs=[crawl_status, crawl_result_display, crawl_download_button]
        )

        scraper_input_type.change(
            fn=toggle_scraper_input,
            inputs=[scraper_input_type],
            outputs=[scraper_file_input, scraper_urls_input]
        )

        scrape_button.click(
            fn=start_scrape,
            inputs=[scraper_input_type, scraper_file_input, scraper_urls_input, ignore_links, ignore_images, max_urls, scraper_output_dir],
            outputs=[scrape_status, scrape_result_display, scrape_download_button]
        )

    return tab
