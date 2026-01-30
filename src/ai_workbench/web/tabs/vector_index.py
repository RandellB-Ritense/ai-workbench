"""
Vector Index tab - Build and test vector embeddings for RAG.
"""
import gradio as gr
from pathlib import Path
from typing import Optional, List
import json

from ...embedders.document_processor import DocumentProcessor
from ...embedders.mistral_embedder import MistralEmbedder
from ...vector_stores.chroma_store import ChromaStore
from ...rag.retriever import RAGRetriever
from ...config import get_config


def create_vector_index_tab() -> gr.Tab:
    """
    Create the Vector Index tab.

    Returns:
        Gradio Tab component
    """
    config = get_config()

    with gr.Tab("Vector Index") as tab:
        gr.Markdown("## Build Vector Index")
        gr.Markdown("Process scraped documents and create embeddings for RAG retrieval.")

        # Build Index Section
        with gr.Row():
            with gr.Column(scale=2):
                # Build index inputs
                index_scraped_file = gr.File(
                    label="Scraped Content JSON - Upload the scraped content file from the scraper",
                    file_types=[".json"]
                )

                index_output_name = gr.Textbox(
                    label="Index Name",
                    placeholder="my-docs-index",
                    info="Name for the vector database (will be saved in ~/.ai-workbench/vector-stores/)"
                )

                mistral_api_key = gr.Textbox(
                    label="Mistral API Key",
                    type="password",
                    placeholder="Enter your Mistral API key",
                    value=config.mistral_api_key or "",
                    info="Required for generating embeddings"
                )

                with gr.Row():
                    chunk_size = gr.Slider(
                        minimum=100,
                        maximum=1000,
                        value=config.chunk_size,
                        step=50,
                        label="Chunk Size (tokens)",
                        info="Size of text chunks for embedding"
                    )

                    chunk_overlap = gr.Slider(
                        minimum=0,
                        maximum=200,
                        value=config.chunk_overlap,
                        step=10,
                        label="Chunk Overlap (tokens)",
                        info="Overlap between chunks"
                    )

                build_button = gr.Button("Build Index", variant="primary", size="lg")

            with gr.Column(scale=1):
                # Status display
                build_status = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False,
                    lines=5
                )

        # Build results section
        with gr.Row():
            with gr.Column():
                build_result_display = gr.JSON(
                    label="Build Results",
                    visible=False
                )

        # Test RAG Section
        gr.Markdown("---")
        gr.Markdown("## Test RAG Retrieval")
        gr.Markdown("Search your vector databases to test retrieval quality.")

        with gr.Row():
            with gr.Column(scale=2):
                # Get list of available vector DBs
                def get_vector_dbs():
                    """List available vector databases."""
                    vector_store_path = config.vector_store_path
                    if not vector_store_path.exists():
                        return []

                    # List directories in vector store path
                    dbs = [d.name for d in vector_store_path.iterdir() if d.is_dir()]
                    return sorted(dbs)

                test_vector_db = gr.Dropdown(
                    label="Vector Database",
                    choices=get_vector_dbs(),
                    info="Select a vector database to search",
                    allow_custom_value=False
                )

                refresh_dbs_button = gr.Button("🔄 Refresh List", size="sm")

                test_query = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your search query...",
                    lines=2
                )

                with gr.Row():
                    test_top_k = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=config.rag_top_k,
                        step=1,
                        label="Top K Results",
                        info="Number of documents to retrieve"
                    )

                    test_score_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=config.rag_score_threshold,
                        step=0.05,
                        label="Score Threshold",
                        info="Minimum similarity score"
                    )

                test_mistral_key = gr.Textbox(
                    label="Mistral API Key",
                    type="password",
                    placeholder="Enter your Mistral API key",
                    value=config.mistral_api_key or "",
                    info="Required for query embedding"
                )

                search_button = gr.Button("Search", variant="primary", size="lg")

            with gr.Column(scale=1):
                test_status = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False
                )

        # Test results section
        with gr.Row():
            with gr.Column():
                test_results_table = gr.Dataframe(
                    headers=["Title", "Source URL", "Score", "Preview"],
                    label="Search Results",
                    visible=False,
                    wrap=True
                )

        # Event handlers
        def start_build_index(file_input, output_name, api_key, chunk_sz, chunk_ovlp, progress=gr.Progress()):
            """Execute build index synchronously with progress tracking."""
            if not file_input:
                return {
                    build_status: "❌ Error: No file uploaded",
                    build_result_display: gr.update(visible=False)
                }

            if not output_name:
                return {
                    build_status: "❌ Error: No index name provided",
                    build_result_display: gr.update(visible=False)
                }

            if not api_key:
                return {
                    build_status: "❌ Error: No API key provided",
                    build_result_display: gr.update(visible=False)
                }

            try:
                progress(0, desc="Initializing components...")

                # Initialize components
                processor = DocumentProcessor(
                    chunk_size=int(chunk_sz),
                    chunk_overlap=int(chunk_ovlp)
                )

                embedder = MistralEmbedder(api_key=api_key)

                # Create output directory
                output_path = config.vector_store_path / output_name
                output_path.mkdir(parents=True, exist_ok=True)

                vector_store = ChromaStore(
                    persist_directory=output_path,
                    embedding_dimension=embedder.get_embedding_dimension()
                )

                progress(0.1, desc="Processing documents and chunking...")

                # Process documents
                def chunking_progress(prog, message):
                    progress(0.1 + prog * 0.2, desc=message)

                chunks = processor.process_scraped_file(
                    Path(file_input),
                    progress_callback=chunking_progress
                )

                if not chunks:
                    return {
                        build_status: "❌ Error: No chunks generated from documents",
                        build_result_display: gr.update(visible=False)
                    }

                stats = processor.get_chunk_stats(chunks)
                progress(0.3, desc=f"Created {len(chunks)} chunks from {stats['total_documents']} documents")

                # Generate embeddings
                progress(0.35, desc="Generating embeddings...")

                texts = [chunk.chunk_text for chunk in chunks]

                def embedding_progress(prog, message):
                    progress(0.35 + prog * 0.45, desc=message)

                embeddings = embedder.embed_batch(texts, progress_callback=embedding_progress)

                progress(0.85, desc=f"Generated {len(embeddings)} embeddings")

                # Store in vector database
                progress(0.9, desc="Storing vectors in database...")

                chunk_ids = [chunk.chunk_id for chunk in chunks]
                metadatas = [chunk.metadata for chunk in chunks]

                # Add source_url to metadata
                for i, chunk in enumerate(chunks):
                    metadatas[i]["source_url"] = chunk.source_url
                    metadatas[i]["chunk_index"] = chunk.chunk_index
                    metadatas[i]["total_chunks"] = chunk.total_chunks

                vector_store.add(chunk_ids, texts, embeddings, metadatas)

                progress(0.95, desc=f"Stored {vector_store.count()} vectors")
                progress(1.0, desc="Complete!")

                result_summary = {
                    "index_name": output_name,
                    "index_path": str(output_path),
                    "total_chunks": len(chunks),
                    "total_documents": stats['total_documents'],
                    "total_tokens": stats['total_tokens'],
                    "avg_tokens_per_chunk": stats['avg_tokens_per_chunk'],
                    "vector_count": vector_store.count()
                }

                return {
                    build_status: f"✅ Complete! Built index with {len(chunks)} chunks from {stats['total_documents']} documents\nSaved to: {output_path}",
                    build_result_display: gr.update(value=result_summary, visible=True)
                }

            except Exception as e:
                return {
                    build_status: f"❌ Error: {str(e)}",
                    build_result_display: gr.update(visible=False)
                }

        def test_rag_search(db_name, query, top_k, score_threshold, api_key):
            """Execute RAG search."""
            if not db_name:
                return {
                    test_status: "Error: No database selected",
                    test_results_table: gr.update(visible=False)
                }

            if not query:
                return {
                    test_status: "Error: No query entered",
                    test_results_table: gr.update(visible=False)
                }

            if not api_key:
                return {
                    test_status: "Error: No API key",
                    test_results_table: gr.update(visible=False)
                }

            try:
                # Initialize components
                vector_db_path = config.vector_store_path / db_name

                if not vector_db_path.exists():
                    return {
                        test_status: f"Error: Database not found at {vector_db_path}",
                        test_results_table: gr.update(visible=False)
                    }

                embedder = MistralEmbedder(api_key=api_key)
                vector_store = ChromaStore(
                    persist_directory=vector_db_path,
                    embedding_dimension=embedder.get_embedding_dimension()
                )

                retriever = RAGRetriever(
                    embedder=embedder,
                    vector_store=vector_store,
                    top_k=int(top_k),
                    score_threshold=score_threshold
                )

                # Retrieve documents
                documents = retriever.retrieve(query, top_k=int(top_k))

                if not documents:
                    return {
                        test_status: "No documents found matching query",
                        test_results_table: gr.update(visible=False)
                    }

                # Format results as table
                results_data = []
                for doc in documents:
                    results_data.append([
                        doc.title or "Untitled",
                        doc.source_url,
                        f"{doc.score:.3f}",
                        doc.text[:150] + "..." if len(doc.text) > 150 else doc.text
                    ])

                return {
                    test_status: f"✓ Found {len(documents)} results",
                    test_results_table: gr.update(value=results_data, visible=True)
                }

            except Exception as e:
                return {
                    test_status: f"Error: {str(e)}",
                    test_results_table: gr.update(visible=False)
                }

        def refresh_vector_dbs():
            """Refresh the list of vector databases."""
            return gr.update(choices=get_vector_dbs())

        # Connect events
        build_button.click(
            fn=start_build_index,
            inputs=[index_scraped_file, index_output_name, mistral_api_key, chunk_size, chunk_overlap],
            outputs=[build_status, build_result_display]
        )

        # Test RAG events
        refresh_dbs_button.click(
            fn=refresh_vector_dbs,
            outputs=[test_vector_db]
        )

        search_button.click(
            fn=test_rag_search,
            inputs=[test_vector_db, test_query, test_top_k, test_score_threshold, test_mistral_key],
            outputs=[test_status, test_results_table]
        )

    return tab
