"""
Upload & Process Tab - Document upload and URL processing.

This module provides the UI for uploading documents and processing web URLs.
Extracted from unified_launcher.py as part of Phase 4 UI refactoring.
"""

import gradio as gr
from typing import Optional, Any


def create_upload_tab(system: Optional[Any] = None) -> None:
    """
    Create the Upload & Process tab.

    Args:
        system: AcademicRAGSystem instance

    Returns:
        None (modifies Gradio UI in-place)
    """
    gr.Markdown("### Upload Documents or Add Web Links")
    gr.Markdown("""
    **🎯 Purpose:**
    This feature allows you to upload research papers and build a knowledge graph from them. The system extracts entities (papers, authors, concepts), relationships (citations, authorship), and metadata automatically.

    **📋 Step-by-Step Instructions:**
    1. **Upload Files** (Optional):
       - Click the file upload box below
       - Select one or multiple files (PDF, DOCX, TXT, or MD formats)
       - You can drag and drop files directly into the box

    2. **OR Add Web URLs** (Optional):
       - Paste URLs in the text box (one URL per line)
       - Supported sources: arXiv papers, direct PDF links, research repositories

    3. **Configure Options**:
       - ✅ **Extract metadata**: Automatically extracts titles, authors, publication dates, citations
       - ✅ **Build knowledge graph**: Creates nodes and relationships in Neo4j/in-memory graph

    4. **Process**:
       - Click "🚀 Process All" button
       - Watch the status panel for progress updates
       - Review results in the "Detailed Results" JSON output

    **💡 Example Use Cases:**

    *Example 1: Upload arXiv Papers*
    ```
    URLs:
    https://arxiv.org/abs/1706.03762
    https://arxiv.org/abs/1810.04805
    https://arxiv.org/abs/2010.11929

    Options: ✓ Extract metadata, ✓ Build knowledge graph
    Result: Creates paper nodes with citations and author relationships
    ```

    *Example 2: Upload Local PDFs*
    ```
    Files: research_paper_1.pdf, thesis_chapter_2.pdf
    Options: ✓ Extract metadata, ✓ Build knowledge graph
    Result: Extracts text, builds graph, enables querying
    ```

    *Example 3: Mixed Upload (Files + URLs)*
    ```
    Files: local_paper.pdf
    URLs: https://arxiv.org/abs/1706.03762
    Options: ✓ Extract metadata, ✓ Build knowledge graph
    Result: Processes both and connects related papers
    ```

    **⚙️ Configuration Details:**
    - **Extract metadata ON**: Extracts title, authors, year, abstract, citations, keywords
    - **Extract metadata OFF**: Only processes raw text content
    - **Build knowledge graph ON**: Creates nodes (papers, authors, concepts) and edges (citations, authorship, similarity)
    - **Build knowledge graph OFF**: Only stores text for search, no graph relationships

    **⏱️ Processing Time:**
    - Small papers (5-10 pages): ~30 seconds
    - Large papers (20+ pages): ~60-90 seconds
    - Batch of 10 papers: ~3-10 seconds (with parallel processing)

    **✅ Success Indicators:**
    - Status shows "✅ Processed X/Y files successfully"
    - Detailed Results JSON shows "status": "success" for each file
    - No errors in the status panel

    **🔧 Troubleshooting:**
    - **Error: "Document processor not available"**: System not initialized properly, restart launcher
    - **Error: "Could not extract text from PDF"**: File may be scanned image, try OCR first
    - **Slow processing**: Large files or many citations take longer, this is normal
    - **URL download fails**: Check internet connection or try direct PDF link

    **📝 Tips:**
    - Upload related papers together to build connected knowledge graphs
    - Use arXiv URLs when possible (better metadata extraction)
    - Process papers in batches of 5-10 for optimal performance (parallel processing)
    - Wait for processing to complete before querying
    """)

    with gr.Row():
        with gr.Column():
            # Multiple file upload
            file_upload = gr.File(
                label="Upload Files (PDF, DOCX, TXT, MD)",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".doc", ".txt", ".md"]
            )

            # Web URL input
            web_urls = gr.Textbox(
                label="Web URLs (one per line)",
                placeholder="https://arxiv.org/abs/1706.03762\nhttps://example.com/paper.pdf",
                lines=5
            )

            extract_metadata = gr.Checkbox(
                label="Extract metadata",
                value=True
            )

            build_graph = gr.Checkbox(
                label="Build knowledge graph",
                value=True
            )

            process_btn = gr.Button("🚀 Process All", variant="primary")

        with gr.Column():
            processing_status = gr.Textbox(
                label="Processing Status",
                lines=15,
                interactive=False
            )

            processing_results = gr.JSON(label="Detailed Results")

    def handle_upload_and_urls(files, urls_text, extract_meta, build_kg):
        """Handle both file uploads and web URLs with parallel processing."""
        if not system:
            return "System not initialized", {}

        results = {
            'files_processed': 0,
            'urls_processed': 0,
            'errors': [],
            'details': []
        }

        status_lines = []

        # Process uploaded files (now with parallel processing!)
        if files:
            status_lines.append(f"📄 Processing {len(files)} file(s) in parallel...")
            file_paths = [f.name if hasattr(f, 'name') else f for f in files]

            try:
                # Check for both doc_processor and document_processor (backwards compatibility)
                doc_proc = getattr(system, 'doc_processor', None) or getattr(system, 'document_processor', None)
                if doc_proc:
                    # Parallel processing (2-4x faster)
                    file_results = doc_proc.process_multiple_files(
                        file_paths,
                        academic_graph_manager=system.academic if build_kg else None,
                        extract_metadata=extract_meta,
                        parallel=True  # Enable parallel processing
                    )
                    results['details'].extend(file_results)
                    results['files_processed'] = len([r for r in file_results if r.get('status') == 'success'])
                    status_lines.append(f"✅ Processed {results['files_processed']}/{len(files)} files successfully")
                else:
                    status_lines.append("⚠️ Document processor not available")
            except Exception as e:
                status_lines.append(f"❌ Error processing files: {e}")
                results['errors'].append(str(e))

        # Process URLs (now with parallel processing!)
        if urls_text and urls_text.strip():
            urls = [u.strip() for u in urls_text.split('\n') if u.strip()]
            status_lines.append(f"🌐 Fetching {len(urls)} URL(s) in parallel...")

            try:
                # Check for both doc_processor and document_processor (backwards compatibility)
                doc_proc = getattr(system, 'doc_processor', None) or getattr(system, 'document_processor', None)
                if doc_proc:
                    # Parallel processing (3-5x faster)
                    url_results = doc_proc.process_multiple_urls(
                        urls,
                        academic_graph_manager=system.academic if build_kg else None,
                        parallel=True  # Enable parallel processing
                    )
                    results['details'].extend(url_results)
                    results['urls_processed'] = len([r for r in url_results if r.get('status') == 'success'])
                    status_lines.append(f"✅ Processed {results['urls_processed']}/{len(urls)} URLs successfully")
                else:
                    status_lines.append("⚠️ Document processor not available")
            except Exception as e:
                status_lines.append(f"❌ Error processing URLs: {e}")
                results['errors'].append(str(e))

        if not files and not urls_text:
            return "No files or URLs provided", {}

        # Summary
        status_lines.append("")
        status_lines.append("📊 Summary:")
        status_lines.append(f"  Files: {results['files_processed']} processed")
        status_lines.append(f"  URLs: {results['urls_processed']} processed")
        if results['errors']:
            status_lines.append(f"  Errors: {len(results['errors'])}")

        return "\n".join(status_lines), results

    # Connect button to handler
    process_btn.click(
        handle_upload_and_urls,
        inputs=[file_upload, web_urls, extract_metadata, build_graph],
        outputs=[processing_status, processing_results]
    )
