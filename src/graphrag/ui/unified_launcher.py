#!/usr/bin/env python3
"""
Unified Gradio Launcher - Comprehensive UI for Research Compass.

This unified launcher combines all features from the previous launchers:
- Multiple file upload and web URL processing
- Streaming responses with intelligent caching
- Research Assistant with GNN-powered reasoning
- Temporal Analysis and citation metrics
- Personalized recommendations and collaborative filtering
- Citation network exploration
- Discovery and cross-disciplinary connections
- Advanced metrics and settings management
- Cache management

This consolidation eliminates redundancy and provides a single, complete interface.
"""

import os
import json
import logging
import time
import requests
from pathlib import Path
from typing import Optional, List, Generator, Dict, Any

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Install with: pip install gradio")
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Helper functions for LLM connection testing and model detection
def test_ollama_connection() -> Dict[str, Any]:
    """Test connection to Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return {"success": True, "message": "Connected to Ollama", "data": response.json()}
        return {"success": False, "message": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "message": "Cannot connect to Ollama. Is it running on localhost:11434?"}
    except requests.exceptions.Timeout:
        return {"success": False, "message": "Connection timeout (5s)"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


def test_lmstudio_connection() -> Dict[str, Any]:
    """Test connection to LM Studio."""
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            return {"success": True, "message": "Connected to LM Studio", "data": response.json()}
        return {"success": False, "message": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "message": "Cannot connect to LM Studio. Is it running on localhost:1234?"}
    except requests.exceptions.Timeout:
        return {"success": False, "message": "Connection timeout (5s)"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


def test_openrouter_connection(api_key: str) -> Dict[str, Any]:
    """Test connection to OpenRouter."""
    if not api_key or not api_key.strip():
        return {"success": False, "message": "API key is required for OpenRouter"}

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://research-compass.local",
            "X-Title": "Research Compass"
        }
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=5)
        if response.status_code == 200:
            return {"success": True, "message": "Connected to OpenRouter", "data": response.json()}
        elif response.status_code == 401:
            return {"success": False, "message": "Invalid API key"}
        return {"success": False, "message": f"HTTP {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"success": False, "message": "Connection timeout (5s)"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


def test_openai_connection(api_key: str) -> Dict[str, Any]:
    """Test connection to OpenAI."""
    if not api_key or not api_key.strip():
        return {"success": False, "message": "API key is required for OpenAI"}

    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=5)
        if response.status_code == 200:
            return {"success": True, "message": "Connected to OpenAI", "data": response.json()}
        elif response.status_code == 401:
            return {"success": False, "message": "Invalid API key"}
        return {"success": False, "message": f"HTTP {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"success": False, "message": "Connection timeout (5s)"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


def detect_ollama_models() -> List[str]:
    """Detect available models from Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            return [model.get("name", "") for model in models if model.get("name")]
        return []
    except Exception as e:
        logger.error(f"Error detecting Ollama models: {e}")
        return []


def detect_lmstudio_models() -> List[str]:
    """Detect available models from LM Studio."""
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            return [model.get("id", "") for model in models if model.get("id")]
        return []
    except Exception as e:
        logger.error(f"Error detecting LM Studio models: {e}")
        return []


def detect_openrouter_models(api_key: str) -> List[str]:
    """Detect available models from OpenRouter."""
    if not api_key or not api_key.strip():
        return []

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://research-compass.local",
            "X-Title": "Research Compass"
        }
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            # Return top popular models (limit to 50 for UI)
            return [model.get("id", "") for model in models[:50] if model.get("id")]
        return []
    except Exception as e:
        logger.error(f"Error detecting OpenRouter models: {e}")
        return []


def detect_openai_models(api_key: str) -> List[str]:
    """Detect available models from OpenAI."""
    if not api_key or not api_key.strip():
        return []

    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            # Filter for chat models
            return sorted([model.get("id", "") for model in models
                          if model.get("id") and ("gpt" in model.get("id", "").lower())])
        return []
    except Exception as e:
        logger.error(f"Error detecting OpenAI models: {e}")
        return []


def test_neo4j_connection(uri: str, username: str, password: str) -> Dict[str, Any]:
    """Test connection to Neo4j database."""
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            result.single()
        driver.close()

        return {"success": True, "message": "Successfully connected to Neo4j"}
    except ImportError:
        return {"success": False, "message": "neo4j package not installed. Install with: pip install neo4j"}
    except Exception as e:
        return {"success": False, "message": f"Connection failed: {str(e)}"}


def create_unified_ui(system=None):
    """
    Create unified Gradio UI with all Research Compass features.

    Args:
        system: AcademicRAGSystem or similar system instance

    Returns:
        Gradio Blocks interface
    """

    # Initialize system if not provided
    if system is None:
        try:
            from src.graphrag.core.academic_rag_system import AcademicRAGSystem
            logger.info("Initializing AcademicRAGSystem...")
            system = AcademicRAGSystem()
            logger.info("✓ AcademicRAGSystem initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize system: {e}")
            logger.info("Continuing with limited functionality...")
            system = None

    # Initialize cache
    try:
        from src.graphrag.core.cache_manager import get_cache
        cache = get_cache()
    except Exception as e:
        logger.warning(f"Cache not available: {e}")
        cache = None

    def stream_response(text: str, delay: float = 0.02) -> Generator[str, None, None]:
        """Stream text word by word for better UX."""
        words = text.split()
        current = ""
        for word in words:
            current += word + " "
            yield current
            time.sleep(delay)

    # Create Gradio interface
    with gr.Blocks(title="Research Compass - Unified Platform", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🧭 Research Compass - Unified Research Platform")
        gr.Markdown("Upload papers, explore with GNN-powered insights, discover research connections, and get personalized recommendations")

        with gr.Tabs():
            # ========== TAB 1: Upload & Process ==========
            with gr.TabItem("📤 Upload & Process"):
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
                - Batch of 10 papers: ~5-10 minutes

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
                - Process papers in batches of 5-10 for optimal performance
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
                    """Handle both file uploads and web URLs."""
                    if not system:
                        return "System not initialized", {}

                    results = {
                        'files_processed': 0,
                        'urls_processed': 0,
                        'errors': [],
                        'details': []
                    }

                    status_lines = []

                    # Process uploaded files
                    if files:
                        status_lines.append(f"📄 Processing {len(files)} file(s)...")
                        file_paths = [f.name if hasattr(f, 'name') else f for f in files]

                        try:
                            # Check for both doc_processor and document_processor (backwards compatibility)
                            doc_proc = getattr(system, 'doc_processor', None) or getattr(system, 'document_processor', None)
                            if doc_proc:
                                file_results = doc_proc.process_multiple_files(
                                    file_paths,
                                    academic_graph_manager=system.academic if build_kg else None,
                                    extract_metadata=extract_meta
                                )
                                results['details'].extend(file_results)
                                results['files_processed'] = len([r for r in file_results if r.get('status') == 'success'])
                                status_lines.append(f"✅ Processed {results['files_processed']}/{len(files)} files successfully")
                            else:
                                status_lines.append("⚠️ Document processor not available")
                        except Exception as e:
                            status_lines.append(f"❌ Error processing files: {e}")
                            results['errors'].append(str(e))

                    # Process URLs
                    if urls_text and urls_text.strip():
                        urls = [u.strip() for u in urls_text.split('\n') if u.strip()]
                        status_lines.append(f"🌐 Processing {len(urls)} URL(s)...")

                        try:
                            # Check for both doc_processor and document_processor (backwards compatibility)
                            doc_proc = getattr(system, 'doc_processor', None) or getattr(system, 'document_processor', None)
                            if doc_proc:
                                url_results = doc_proc.process_multiple_urls(
                                    urls,
                                    academic_graph_manager=system.academic if build_kg else None
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

                process_btn.click(
                    handle_upload_and_urls,
                    inputs=[file_upload, web_urls, extract_metadata, build_graph],
                    outputs=[processing_status, processing_results]
                )

            # ========== TAB 2: Graph & GNN Dashboard ==========
            with gr.TabItem("🕸️ Graph & GNN Dashboard"):
                gr.Markdown("### Knowledge Graph Visualization & GNN Model Management")
                gr.Markdown("""
                **🎯 Purpose:**
                This dashboard provides complete control over your knowledge graph and Graph Neural Network (GNN) models. View statistics, visualize connections, train AI models, and make intelligent predictions about research relationships.

                **💡 Overview:**
                - **Graph Statistics**: View node/edge counts, types, and academic metrics
                - **Visualize Graph**: Interactive network visualization with multiple layouts
                - **Train GNN Models**: Train AI models for link prediction, node classification, and embeddings
                - **GNN Predictions**: Use trained models to predict citations, classify papers, find similar research
                - **Export Graph**: Download your knowledge graph for external analysis
                """)

                with gr.Tabs():
                    # Sub-tab: Graph Statistics
                    with gr.Tab("📊 Graph Statistics"):
                        gr.Markdown("""
                        **🎯 Purpose:**
                        Get comprehensive statistics about your knowledge graph including size, node types, edge types, and academic-specific metrics.

                        **📋 Instructions:**
                        1. Click "🔄 Refresh Statistics" button
                        2. View the following metrics:
                           - **Graph Size**: Total nodes and edges
                           - **Node Types**: Breakdown by paper, author, concept, institution
                           - **Edge Types**: Citation counts, authorship links, co-authorship, concept relationships
                           - **GNN Status**: Whether models are trained and ready
                           - **Academic Statistics**: Papers by year, top authors, citation counts

                        **💡 Example Output:**
                        ```json
                        Graph Size:
                        {
                          "total_nodes": 150,
                          "total_edges": 420
                        }

                        Node Types:
                        {
                          "Paper": 100,
                          "Author": 35,
                          "Concept": 15
                        }

                        Edge Types:
                        {
                          "CITES": 200,
                          "AUTHORED_BY": 100,
                          "MENTIONS": 120
                        }
                        ```

                        **📝 Tips:**
                        - Refresh after uploading new papers to see updated stats
                        - Use these metrics to decide on GNN model training parameters
                        - Large graphs (>1000 nodes) benefit more from GNN analysis
                        """)

                        stats_refresh_btn = gr.Button("🔄 Refresh Statistics", variant="primary")

                        with gr.Row():
                            with gr.Column():
                                graph_size_display = gr.JSON(label="Graph Size")
                            with gr.Column():
                                node_types_display = gr.JSON(label="Node Types")

                        with gr.Row():
                            with gr.Column():
                                edge_types_display = gr.JSON(label="Edge Types")
                            with gr.Column():
                                gnn_status_display = gr.JSON(label="GNN Status")

                        academic_stats_display = gr.JSON(label="Academic Statistics")

                        def refresh_graph_stats():
                            """Refresh and display graph statistics."""
                            try:
                                from src.graphrag.ui.graph_gnn_dashboard import GraphGNNDashboard
                                dashboard = GraphGNNDashboard(system)
                                stats = dashboard.get_graph_statistics()

                                return (
                                    stats.get("graph_size", {}),
                                    stats.get("node_types", {}),
                                    stats.get("edge_types", {}),
                                    stats.get("gnn_status", {}),
                                    stats.get("academic", {})
                                )
                            except Exception as e:
                                error_msg = {"error": str(e)}
                                return error_msg, error_msg, error_msg, error_msg, error_msg

                        stats_refresh_btn.click(
                            refresh_graph_stats,
                            inputs=[],
                            outputs=[graph_size_display, node_types_display, edge_types_display, gnn_status_display, academic_stats_display]
                        )

                    # Sub-tab: Graph Visualization
                    with gr.Tab("🎨 Visualize Graph"):
                        gr.Markdown("""
                        **🎯 Purpose:**
                        Generate interactive visualizations of your knowledge graph to explore research connections, citation networks, and author collaborations.

                        **📋 Step-by-Step Instructions:**
                        1. **Set Maximum Nodes**: Use slider to limit displayed nodes (10-500)
                           - Start with 50-100 for initial exploration
                           - Increase for comprehensive view (slower rendering)

                        2. **Choose Layout Algorithm**:
                           - **Spring**: Natural clustering, best for most graphs
                           - **Circular**: Nodes arranged in circles, good for finding patterns
                           - **Hierarchical**: Top-down tree structure, best for citation chains

                        3. **Generate**: Click "🎨 Generate Visualization"

                        4. **Interact**: The visualization is interactive:
                           - Zoom in/out with mouse wheel
                           - Drag nodes to rearrange
                           - Click nodes to see details
                           - Hover over edges to see relationship types

                        **💡 Example Use Cases:**

                        *Example 1: Explore Citation Network*
                        ```
                        Max Nodes: 100
                        Layout: Spring
                        Result: See which papers cite each other, identify clusters of related work
                        ```

                        *Example 2: Author Collaboration Map*
                        ```
                        Max Nodes: 50
                        Layout: Circular
                        Result: Visualize co-authorship patterns and research communities
                        ```

                        *Example 3: Citation Hierarchy*
                        ```
                        Max Nodes: 75
                        Layout: Hierarchical
                        Result: See foundational papers at top, newer citations below
                        ```

                        **🎨 Layout Comparison:**
                        | Layout | Best For | Speed | Visual Style |
                        |--------|----------|-------|--------------|
                        | **Spring** | General exploration, finding clusters | Medium | Organic, natural grouping |
                        | **Circular** | Pattern detection, symmetry | Fast | Organized, circular arrangement |
                        | **Hierarchical** | Citation chains, paper evolution | Slow | Tree-like, top-down flow |

                        **⚡ Performance Tips:**
                        - Start with 50 nodes for fast loading
                        - Spring layout: O(n²) complexity, slower for large graphs
                        - Circular layout: O(n) complexity, fastest option
                        - Hierarchical layout: O(n log n), moderate speed

                        **📝 Tips:**
                        - Use smaller node counts for faster rendering
                        - Spring layout best reveals natural communities
                        - Export visualization as HTML for presentations
                        - Colors indicate node types: Papers (blue), Authors (green), Concepts (orange)
                        """)

                        with gr.Row():
                            max_nodes_slider = gr.Slider(
                                minimum=10,
                                maximum=500,
                                value=100,
                                step=10,
                                label="Maximum Nodes to Display"
                            )
                            layout_choice = gr.Dropdown(
                                choices=["spring", "circular", "hierarchical"],
                                value="spring",
                                label="Layout Algorithm"
                            )

                        visualize_btn = gr.Button("🎨 Generate Visualization", variant="primary")
                        graph_viz_output = gr.HTML(label="Graph Visualization")

                        def generate_graph_viz(max_nodes, layout):
                            """Generate full graph visualization."""
                            try:
                                from src.graphrag.ui.graph_gnn_dashboard import GraphGNNDashboard
                                dashboard = GraphGNNDashboard(system)
                                html = dashboard.visualize_full_graph(
                                    max_nodes=int(max_nodes),
                                    layout=layout
                                )
                                return html
                            except Exception as e:
                                return f"<html><body><h2>Error</h2><p>{str(e)}</p><p>Make sure you have uploaded documents first!</p></body></html>"

                        visualize_btn.click(
                            generate_graph_viz,
                            inputs=[max_nodes_slider, layout_choice],
                            outputs=[graph_viz_output]
                        )

                    # Sub-tab: GNN Training
                    with gr.Tab("🤖 Train GNN Models"):
                        gr.Markdown("""
                        **🎯 Purpose:**
                        Train Graph Neural Network (GNN) models to learn patterns in your research graph. These AI models can predict missing citations, classify papers, and generate semantic embeddings.

                        **📋 Step-by-Step Instructions:**
                        1. **Select Model Type**:
                           - **GCN** (Graph Convolutional Network): Fast, good baseline
                           - **GAT** (Graph Attention): Best quality, learns important connections
                           - **Transformer**: Cutting-edge, best for large graphs
                           - **Hetero**: For graphs with multiple node/edge types

                        2. **Choose Task Type**:
                           - **Link Prediction**: Predict missing citations between papers
                           - **Node Classification**: Classify papers by topic/field
                           - **Embedding**: Generate vector representations for similarity search

                        3. **Set Training Parameters**:
                           - **Epochs**: 50-100 for most tasks (more = better but slower)
                           - **Learning Rate**: 0.001-0.01 (lower = stable, higher = faster)

                        4. **Start Training**: Click "🚀 Start Training"

                        5. **Monitor Progress**: Watch training status and metrics

                        **💡 Example Training Scenarios:**

                        *Example 1: Find Missing Citations*
                        ```
                        Model Type: GAT (best accuracy)
                        Task: link_prediction
                        Epochs: 50
                        Learning Rate: 0.01

                        Result: Model learns to predict which papers should cite each other
                        Use Case: Discover papers you might have missed in literature review
                        ```

                        *Example 2: Classify Research Topics*
                        ```
                        Model Type: GCN (faster)
                        Task: node_classification
                        Epochs: 75
                        Learning Rate: 0.005

                        Result: Automatically categorize papers by research area
                        Use Case: Organize large paper collections by topic
                        ```

                        *Example 3: Semantic Search*
                        ```
                        Model Type: Transformer (best quality)
                        Task: embedding
                        Epochs: 100
                        Learning Rate: 0.001

                        Result: Generate embeddings for similarity-based search
                        Use Case: Find conceptually similar papers, even without citations
                        ```

                        **🤖 Model Comparison:**
                        | Model | Speed | Accuracy | Memory | Best For |
                        |-------|-------|----------|--------|----------|
                        | **GCN** | ⚡⚡⚡ Fast | Good | Low | Quick experiments, baselines |
                        | **GAT** | ⚡⚡ Medium | Best | Medium | Production, highest accuracy |
                        | **Transformer** | ⚡ Slow | Excellent | High | Large graphs, cutting-edge results |
                        | **Hetero** | ⚡⚡ Medium | Very Good | Medium | Multi-type graphs (papers+authors+concepts) |

                        **📊 Task Type Details:**

                        **Link Prediction:**
                        - Learns: Which papers should cite each other
                        - Output: Probability scores for potential citations
                        - Metrics: AUC-ROC, precision@k, recall@k
                        - Use: Literature gap analysis, recommendation

                        **Node Classification:**
                        - Learns: Paper categories, research fields
                        - Output: Class labels and probabilities
                        - Metrics: Accuracy, F1-score, confusion matrix
                        - Use: Automatic tagging, organization

                        **Embedding:**
                        - Learns: Dense vector representations
                        - Output: N-dimensional embeddings (typically 128-512d)
                        - Metrics: Reconstruction loss, clustering quality
                        - Use: Similarity search, visualization, clustering

                        **⚙️ Parameter Tuning Guide:**

                        **Epochs:**
                        - Small graphs (<100 nodes): 30-50 epochs
                        - Medium graphs (100-1000 nodes): 50-100 epochs
                        - Large graphs (>1000 nodes): 100-200 epochs
                        - Watch for: Training loss plateau = enough epochs

                        **Learning Rate:**
                        - Too high (>0.1): Model doesn't converge, loss oscillates
                        - Good range (0.001-0.01): Steady improvement
                        - Too low (<0.0001): Very slow learning
                        - Recommended: Start with 0.01, reduce if unstable

                        **⏱️ Training Time:**
                        - GCN (50 epochs, 100 nodes): ~30 seconds
                        - GAT (50 epochs, 100 nodes): ~1-2 minutes
                        - Transformer (100 epochs, 500 nodes): ~5-10 minutes

                        **✅ Success Indicators:**
                        - Training loss decreases steadily
                        - Validation metrics improve over epochs
                        - Status shows "✅ Training completed successfully!"
                        - Metrics JSON shows reasonable accuracy (>0.7 for most tasks)

                        **🔧 Troubleshooting:**
                        - **Error: "Not enough nodes"**: Need minimum 20 nodes, upload more papers
                        - **Error: "Graph not built"**: Go to Upload tab, enable "Build knowledge graph"
                        - **Loss = NaN**: Learning rate too high, reduce to 0.001
                        - **No improvement**: Try different model type or increase epochs
                        - **Out of memory**: Reduce graph size or use GCN instead of Transformer

                        **📝 Pro Tips:**
                        - Train link prediction first - most useful for research
                        - GAT gives best accuracy-to-speed tradeoff
                        - Save models automatically after training
                        - Retrain when you add significant new papers (>20% growth)
                        - Monitor training metrics - stop early if no improvement
                        """)

                        with gr.Row():
                            with gr.Column():
                                model_type_choice = gr.Dropdown(
                                    choices=["gcn", "gat", "transformer", "hetero"],
                                    value="gat",
                                    label="Model Type",
                                    info="GAT (Graph Attention) recommended for most tasks"
                                )

                                task_type_choice = gr.Dropdown(
                                    choices=["node_classification", "link_prediction", "embedding"],
                                    value="link_prediction",
                                    label="Task Type",
                                    info="Link prediction finds missing citations"
                                )

                                epochs_slider = gr.Slider(
                                    minimum=10,
                                    maximum=200,
                                    value=50,
                                    step=10,
                                    label="Training Epochs"
                                )

                                lr_slider = gr.Slider(
                                    minimum=0.0001,
                                    maximum=0.1,
                                    value=0.01,
                                    step=0.0001,
                                    label="Learning Rate"
                                )

                                train_gnn_btn = gr.Button("🚀 Start Training", variant="primary")

                            with gr.Column():
                                training_status = gr.Textbox(
                                    label="Training Status",
                                    lines=10,
                                    interactive=False
                                )
                                training_results = gr.JSON(label="Training Metrics")

                        def train_gnn(model_type, task_type, epochs, lr):
                            """Train GNN model."""
                            try:
                                from src.graphrag.ui.graph_gnn_dashboard import GraphGNNDashboard
                                dashboard = GraphGNNDashboard(system)

                                status = f"Training {model_type} model for {task_type}...\n"
                                status += f"Epochs: {epochs}, Learning Rate: {lr}\n\n"

                                results = dashboard.train_gnn_model(
                                    model_type=model_type,
                                    task=task_type,
                                    epochs=int(epochs),
                                    learning_rate=float(lr)
                                )

                                if results["status"] == "success":
                                    status += "✅ Training completed successfully!\n"
                                    status += f"Results: {results.get('message', '')}"
                                else:
                                    status += f"❌ Training failed: {results.get('message', '')}"

                                return status, results.get("metrics", {})
                            except Exception as e:
                                return f"❌ Error: {str(e)}", {"error": str(e)}

                        train_gnn_btn.click(
                            train_gnn,
                            inputs=[model_type_choice, task_type_choice, epochs_slider, lr_slider],
                            outputs=[training_status, training_results]
                        )

                    # Sub-tab: GNN Predictions
                    with gr.Tab("🔮 GNN Predictions"):
                        gr.Markdown("""
                        **🎯 Purpose:**
                        Use your trained GNN models to make intelligent predictions about research connections, find similar papers, and discover potential citations.

                        **⚠️ Prerequisites:**
                        You must train a GNN model first (use "Train GNN Models" tab)!

                        **📋 Step-by-Step Instructions:**
                        1. **Select Prediction Type**:
                           - **Link Prediction**: Find papers that should cite each other
                           - **Node Classification**: Classify a paper by topic/field
                           - **Similar Nodes**: Find papers similar to a given paper

                        2. **Enter Node ID**:
                           - Use paper title (e.g., "Attention Is All You Need")
                           - OR use internal node ID if known

                        3. **Set Top K**: Number of results to return (5-50)

                        4. **Get Predictions**: Click "🔮 Get Predictions"

                        **💡 Example Use Cases:**

                        *Example 1: Find Missing Citations*
                        ```
                        Prediction Type: link_prediction
                        Node ID: "Attention Is All You Need"
                        Top K: 10

                        Result:
                        [
                          {"paper": "BERT", "score": 0.95, "reason": "Both use transformers"},
                          {"paper": "GPT-2", "score": 0.92, "reason": "Transformer architecture"},
                          {"paper": "XLNet", "score": 0.89, "reason": "Attention mechanism"}
                        ]

                        Interpretation: These papers should cite the transformer paper but may not
                        ```

                        *Example 2: Classify Research Paper*
                        ```
                        Prediction Type: node_classification
                        Node ID: "My Research Paper"
                        Top K: 5

                        Result:
                        {
                          "predicted_class": "Natural Language Processing",
                          "confidence": 0.87,
                          "top_classes": [
                            {"class": "NLP", "prob": 0.87},
                            {"class": "Machine Learning", "prob": 0.08},
                            {"class": "Computer Vision", "prob": 0.03}
                          ]
                        }

                        Interpretation: Paper is most likely NLP-related (87% confidence)
                        ```

                        *Example 3: Find Similar Papers*
                        ```
                        Prediction Type: similar_nodes
                        Node ID: "Graph Neural Networks Survey"
                        Top K: 8

                        Result:
                        [
                          {"paper": "GCN Paper", "similarity": 0.94},
                          {"paper": "GAT Paper", "similarity": 0.91},
                          {"paper": "GraphSAGE", "similarity": 0.89},
                          {"paper": "Message Passing Networks", "similarity": 0.85}
                        ]

                        Interpretation: These papers cover similar topics/methods
                        ```

                        **🔍 Prediction Type Details:**

                        **Link Prediction:**
                        - **Use Case**: Literature review gap analysis
                        - **Output**: Ranked list of papers with citation probability scores
                        - **Interpretation**: High scores (>0.8) = strong citation relationship
                        - **Best For**: Finding papers you should have cited

                        **Node Classification:**
                        - **Use Case**: Automatic paper categorization
                        - **Output**: Predicted class label + confidence scores
                        - **Interpretation**: Confidence >0.7 = reliable classification
                        - **Best For**: Organizing paper collections, identifying research areas

                        **Similar Nodes:**
                        - **Use Case**: Exploratory research, finding related work
                        - **Output**: Papers ranked by similarity score
                        - **Interpretation**: Scores >0.85 = very similar content/structure
                        - **Best For**: Discovering papers on same topic

                        **📊 Score Interpretation:**
                        - **0.9-1.0**: Very strong relationship/similarity
                        - **0.8-0.9**: Strong relationship, high confidence
                        - **0.7-0.8**: Moderate relationship, worth investigating
                        - **0.6-0.7**: Weak relationship, may be relevant
                        - **<0.6**: Low confidence, probably not related

                        **⚙️ Parameter Tuning:**

                        **Top K:**
                        - Small (5-10): Most relevant results only
                        - Medium (10-20): Balanced exploration
                        - Large (20-50): Comprehensive analysis

                        **📝 Tips:**
                        - Start with Top K = 10 for most tasks
                        - Link prediction works best with >100 papers in graph
                        - Node classification requires trained classifier (not just embeddings)
                        - Similar nodes works even with embedding-only model
                        - Higher scores = more confident predictions

                        **🔧 Troubleshooting:**
                        - **Error: "Model not trained"**: Go to "Train GNN Models" tab first
                        - **Error: "Node not found"**: Check paper title spelling, try partial match
                        - **Empty results**: Graph too small, add more papers
                        - **Low scores (<0.5)**: May need more training epochs or better model
                        - **Prediction errors**: Retrain model with current graph data
                        """)

                        with gr.Row():
                            with gr.Column():
                                prediction_type_choice = gr.Dropdown(
                                    choices=["link_prediction", "node_classification", "similar_nodes"],
                                    value="link_prediction",
                                    label="Prediction Type"
                                )

                                node_id_input = gr.Textbox(
                                    label="Node ID (paper title or ID)",
                                    placeholder="Enter paper title or node ID"
                                )

                                top_k_slider = gr.Slider(
                                    minimum=5,
                                    maximum=50,
                                    value=10,
                                    step=5,
                                    label="Top K Results"
                                )

                                predict_btn = gr.Button("🔮 Get Predictions", variant="primary")

                            with gr.Column():
                                prediction_output = gr.JSON(label="Predictions")

                        def get_predictions(pred_type, node_id, top_k):
                            """Get GNN predictions."""
                            try:
                                from src.graphrag.ui.graph_gnn_dashboard import GraphGNNDashboard
                                dashboard = GraphGNNDashboard(system)

                                results = dashboard.get_gnn_predictions(
                                    prediction_type=pred_type,
                                    node_id=node_id,
                                    top_k=int(top_k)
                                )

                                return results
                            except Exception as e:
                                return {"error": str(e)}

                        predict_btn.click(
                            get_predictions,
                            inputs=[prediction_type_choice, node_id_input, top_k_slider],
                            outputs=[prediction_output]
                        )

                    # Sub-tab: Export Graph
                    with gr.Tab("💾 Export Graph"):
                        gr.Markdown("""
                        **🎯 Purpose:**
                        Export your knowledge graph data for external analysis, backup, sharing, or use in other tools (Gephi, Cytoscape, custom scripts).

                        **📋 Step-by-Step Instructions:**
                        1. **Select Export Format**:
                           - **JSON**: Full graph with all properties, best for programmatic access
                           - **CSV**: Spreadsheet format, good for Excel/data analysis

                        2. **Export**: Click "📥 Export Graph Data"

                        3. **Use Exported Data**:
                           - Copy from text box and save to file
                           - Import into other tools
                           - Use for backups or sharing

                        **💡 Example Use Cases:**

                        *Example 1: Backup Your Research Graph*
                        ```
                        Format: JSON
                        Action: Export → Save to "my_research_graph_backup_2024.json"
                        Use Case: Version control, disaster recovery
                        ```

                        *Example 2: Analyze in Excel/Python*
                        ```
                        Format: CSV
                        Action: Export → Import into pandas/Excel
                        Use Case: Statistical analysis, custom visualizations
                        ```

                        *Example 3: Share with Collaborators*
                        ```
                        Format: JSON
                        Action: Export → Send file to team
                        Use Case: Collaborative research, knowledge sharing
                        ```

                        **📊 Format Comparison:**

                        **JSON Format:**
                        ```json
                        {
                          "nodes": [
                            {
                              "id": "paper_1",
                              "type": "Paper",
                              "title": "Attention Is All You Need",
                              "authors": ["Vaswani", "Shazeer"],
                              "year": 2017,
                              "citations": 50000
                            },
                            {
                              "id": "author_1",
                              "type": "Author",
                              "name": "Ashish Vaswani"
                            }
                          ],
                          "edges": [
                            {
                              "source": "paper_2",
                              "target": "paper_1",
                              "type": "CITES",
                              "weight": 1.0
                            }
                          ]
                        }
                        ```

                        **Advantages:**
                        - Preserves all node/edge properties
                        - Easy to parse programmatically
                        - Standard format for graph tools
                        - Supports nested data structures

                        **CSV Format:**
                        ```csv
                        nodes.csv:
                        id,type,title,authors,year
                        paper_1,Paper,"Attention Is All You Need","Vaswani;Shazeer",2017
                        author_1,Author,,"Ashish Vaswani",

                        edges.csv:
                        source,target,type,weight
                        paper_2,paper_1,CITES,1.0
                        ```

                        **Advantages:**
                        - Easy to import into Excel/spreadsheets
                        - Simple text format
                        - Good for statistical analysis
                        - Compatible with many tools

                        **🔧 What Gets Exported:**

                        **Nodes:**
                        - Paper nodes: title, authors, year, abstract, keywords, citations
                        - Author nodes: name, affiliations, paper count
                        - Concept nodes: term, frequency, related papers
                        - Institution nodes: name, location

                        **Edges:**
                        - CITES: Citation relationships between papers
                        - AUTHORED_BY: Authorship connections
                        - CO_AUTHORED: Author collaboration
                        - MENTIONS: Paper-concept relationships
                        - AFFILIATED_WITH: Author-institution links

                        **📝 Usage Examples:**

                        **Python Analysis:**
                        ```python
                        import json
                        import pandas as pd

                        # Load JSON export
                        with open('graph_export.json') as f:
                            graph = json.load(f)

                        # Convert to DataFrame
                        papers = pd.DataFrame(graph['nodes'])
                        citations = pd.DataFrame(graph['edges'])

                        # Analyze
                        print(papers['year'].value_counts())
                        print(f"Total citations: {len(citations)}")
                        ```

                        **Import to Gephi:**
                        1. Export as JSON
                        2. Open Gephi
                        3. File → Import → JSON file
                        4. Visualize and analyze

                        **Import to Cytoscape:**
                        1. Export as CSV
                        2. Open Cytoscape
                        3. Import Network from File
                        4. Map columns to node/edge attributes

                        **📏 Export Size Estimates:**
                        - Small graph (50 nodes): ~5-10 KB JSON
                        - Medium graph (500 nodes): ~50-100 KB JSON
                        - Large graph (5000 nodes): ~500 KB - 1 MB JSON
                        - CSV is typically 50-70% of JSON size

                        **📝 Tips:**
                        - Use JSON for complete data preservation
                        - Use CSV for quick Excel analysis
                        - Export regularly as backups
                        - Large graphs may take 5-10 seconds to export
                        - Save exports with timestamps for version tracking
                        """)

                        export_format = gr.Dropdown(
                            choices=["json", "csv"],
                            value="json",
                            label="Export Format"
                        )

                        export_btn = gr.Button("📥 Export Graph Data", variant="primary")
                        export_output = gr.Textbox(
                            label="Exported Data",
                            lines=20,
                            max_lines=30
                        )

                        def export_graph(format_choice):
                            """Export graph data."""
                            try:
                                from src.graphrag.ui.graph_gnn_dashboard import GraphGNNDashboard
                                dashboard = GraphGNNDashboard(system)
                                data = dashboard.export_graph_data(format=format_choice)
                                return data
                            except Exception as e:
                                return f"Error: {str(e)}"

                        export_btn.click(
                            export_graph,
                            inputs=[export_format],
                            outputs=[export_output]
                        )

            # ========== TAB 3: Research Assistant (with Streaming & GNN) ==========
            with gr.TabItem("🔍 Research Assistant"):
                gr.Markdown("### Ask questions with streaming responses, caching, and GNN reasoning")
                gr.Markdown("""
                **🎯 Purpose:**
                An intelligent Q&A system that answers research questions using your uploaded papers, knowledge graph, and optional GNN-powered reasoning.

                **📋 Step-by-Step Instructions:**
                1. **Type Your Question**: Enter a research question (be specific for better answers)
                2. **Configure Options**:
                   - ✅ **Use GNN reasoning**: Graph-aware analysis (+30% accuracy)
                   - ✅ **Use cache**: 10-100x speedup for repeated queries
                   - ✅ **Stream response**: Real-time word-by-word display
                3. **Ask**: Click "Ask" button
                4. **Review**: Check Answer, Confidence, Sources, and Reasoning Visualization

                **💡 Example Questions:**

                *Comparisons:*
                - "How does BERT differ from GPT models?"
                - "What are the advantages of GAT over GCN?"

                *Summaries:*
                - "Summarize the key contributions of the Attention Is All You Need paper"
                - "What problem does the BERT paper solve?"

                *Concepts:*
                - "What are the main innovations in transformer architecture?"
                - "Explain how attention mechanisms work"
                - "What are recent advances in Graph Neural Networks?"

                **📊 Confidence Score Guide:**
                - **90-100%**: Very high confidence, well-supported by sources
                - **70-89%**: High confidence, good evidence
                - **50-69%**: Moderate confidence, some uncertainty
                - **30-49%**: Low confidence, limited information

                **📝 Tips:**
                - Upload relevant papers first for better answers
                - GNN reasoning best for citation/relationship questions
                - Cache saves time during literature review sessions
                - Check sources to verify answer accuracy
                - More papers in graph = better answers
                """)

                with gr.Row():
                    with gr.Column():
                        query_input = gr.Textbox(
                            label="Research Question",
                            placeholder="What are recent advances in transformer architectures?",
                            lines=3
                        )

                        with gr.Row():
                            use_gnn_checkbox = gr.Checkbox(label="Use GNN reasoning", value=True)
                            use_cache_checkbox = gr.Checkbox(label="Use cache", value=True)
                            stream_checkbox = gr.Checkbox(label="Stream response", value=True)

                        ask_btn = gr.Button("Ask", variant="primary")
                        clear_cache_btn = gr.Button("Clear Cache", variant="secondary")

                    with gr.Column():
                        answer_output = gr.Textbox(
                            label="Answer",
                            lines=12,
                            interactive=False
                        )

                        with gr.Row():
                            confidence_output = gr.Textbox(label="Confidence", lines=1)
                            cache_status = gr.Textbox(label="Cache Status", lines=1)

                sources_output = gr.JSON(label="Sources")
                reasoning_viz = gr.HTML(label="Reasoning Visualization")

                def handle_query_with_streaming(question, use_gnn, use_cache_flag, use_streaming):
                    """Handle query with optional caching and streaming."""
                    if not system:
                        return "System not initialized", "0%", "N/A", [], ""

                    cache_hit = False

                    # Try cache first
                    if use_cache_flag and cache:
                        # Correct cache API: cache.get(namespace, *args, **kwargs)
                        cached_result = cache.get('queries', question, use_gnn=use_gnn)
                        if cached_result:
                            cache_hit = True
                            if use_streaming:
                                # Stream from cache
                                for partial in stream_response(cached_result['answer']):
                                    yield partial, f"{cached_result.get('confidence', 0):.1%}", "✅ Cache Hit", cached_result.get('sources', []), cached_result.get('visualization', '')
                            else:
                                yield cached_result['answer'], f"{cached_result.get('confidence', 0):.1%}", "✅ Cache Hit", cached_result.get('sources', []), cached_result.get('visualization', '')
                            return

                    # Execute query
                    try:
                        if hasattr(system, 'gnn_query_engine') and use_gnn:
                            result = system.gnn_query_engine.answer_with_gnn_reasoning(question)
                        elif hasattr(system, 'query'):
                            result = system.query(question, use_graph=use_gnn)
                            if isinstance(result, str):
                                result = {'answer': result, 'confidence': 0.8, 'sources': [], 'visualization': ''}
                        else:
                            result = {'answer': 'Query engine not available', 'confidence': 0, 'sources': [], 'visualization': ''}

                        # Cache result - correct API: cache.set(namespace, value, *args, **kwargs)
                        if use_cache_flag and cache:
                            cache.set('queries', result, question, use_gnn=use_gnn, ttl_seconds=1800)

                        answer = result.get('answer', 'No answer')
                        conf = result.get('confidence', 0)
                        srcs = result.get('sources', [])
                        viz = result.get('visualization', '')

                        # Stream response
                        if use_streaming and isinstance(answer, str):
                            for partial in stream_response(answer):
                                yield partial, f"{conf:.1%}", "🔄 Fresh", srcs, viz
                        else:
                            yield answer, f"{conf:.1%}", "🔄 Fresh", srcs, viz

                    except Exception as e:
                        logger.error(f"Query error: {e}", exc_info=True)
                        yield f"Error: {e}", "0%", "❌ Error", [], ""

                def clear_query_cache():
                    """Clear query cache."""
                    if cache:
                        cache.invalidate_namespace('queries')
                        return "✅ Query cache cleared"
                    return "⚠️ Cache not available"

                ask_btn.click(
                    handle_query_with_streaming,
                    inputs=[query_input, use_gnn_checkbox, use_cache_checkbox, stream_checkbox],
                    outputs=[answer_output, confidence_output, cache_status, sources_output, reasoning_viz]
                )

                clear_cache_btn.click(
                    clear_query_cache,
                    outputs=[cache_status]
                )

            # ========== TAB 4: Temporal Analysis ==========
            with gr.TabItem("📊 Temporal Analysis"):
                gr.Markdown("### Analyze how research evolves over time")
                gr.Markdown("""
                **🎯 Purpose:**
                Track research trends, citation patterns, author impact, and emerging topics over time. Identify hot areas, measure paper influence, and discover new research directions.

                **📋 Available Analysis Types:**

                **1. Topic Evolution**
                - **What it does**: Tracks how many papers on a topic are published over time
                - **Use case**: Identify growing/declining research areas
                - **Example**: "Graph Neural Networks" → See exponential growth from 2015-2024

                **2. Citation Velocity**
                - **What it does**: Measures how quickly a paper gains citations
                - **Use case**: Predict future impact, identify seminal papers
                - **Example**: High-velocity paper = 50+ cites in first year

                **3. H-Index Timeline**
                - **What it does**: Calculates author's h-index progression over years
                - **Use case**: Track researcher career trajectory
                - **Example**: Author h-index grows from 5 (2010) to 45 (2024)

                **💡 Example Use Cases:**

                *Track Emerging Field:*
                ```
                Tab: Topic Evolution
                Topic: "Large Language Models"
                Time Window: Yearly
                Result: See explosion from 2020 onwards, exponential growth
                ```

                *Measure Paper Impact:*
                ```
                Tab: Citation Velocity
                Paper: "Attention Is All You Need"
                Result: 100 citations/month, identifies as breakthrough paper
                ```

                *Analyze Author Career:*
                ```
                Tab: H-Index Timeline
                Author: "Geoffrey Hinton"
                Result: H-index timeline shows consistent growth, peak influence periods
                ```

                **⚙️ Parameters:**

                **Time Window:**
                - **Monthly**: Fine-grained, good for very recent trends (2023-2024)
                - **Quarterly**: Balanced, smooths out noise
                - **Yearly**: Long-term trends, best for historical analysis

                **📊 Interpreting Results:**

                **Topic Evolution:**
                - Upward trend: Growing research area, more attention
                - Plateau: Mature field, stable interest
                - Decline: Possibly superseded by newer methods

                **Citation Velocity:**
                - High (>50 cites/year): Influential, seminal work
                - Medium (10-50 cites/year): Solid contribution
                - Low (<10 cites/year): Niche or too recent

                **H-Index:**
                - Steady growth: Consistent researcher
                - Rapid growth: Breakthrough period
                - Plateau: Established researcher, maintained impact

                **📝 Tips:**
                - Upload papers with year metadata for accurate analysis
                - Use yearly window for 5+ year trends
                - Monthly window best for very recent work (< 2 years)
                - Citation velocity requires citation metadata
                - More papers in database = more accurate trends
                """)

                with gr.Tab("Topic Evolution"):
                    topic_input = gr.Textbox(label="Topic", placeholder="e.g., neural networks")
                    time_window = gr.Dropdown(
                        choices=["monthly", "quarterly", "yearly"],
                        value="yearly",
                        label="Time Window"
                    )
                    analyze_topic_btn = gr.Button("Analyze Topic Evolution")

                    evolution_output = gr.JSON(label="Evolution Data")
                    evolution_chart = gr.Plot(label="Timeline Chart")

                    def analyze_topic_evolution(topic, window):
                        if not system or not hasattr(system, 'temporal_analytics'):
                            return {}, None

                        try:
                            result = system.temporal_analytics.analyze_topic_evolution(topic, window)

                            # Create simple chart
                            timeline = result.get('timeline', [])
                            if timeline:
                                import matplotlib.pyplot as plt
                                fig, ax = plt.subplots()
                                years = [t[0] for t in timeline]
                                papers = [t[1] for t in timeline]
                                ax.plot(years, papers, marker='o')
                                ax.set_xlabel('Year')
                                ax.set_ylabel('Paper Count')
                                ax.set_title(f'Evolution of "{topic}"')
                                plt.tight_layout()
                                return result, fig

                            return result, None
                        except Exception as e:
                            logger.exception("Topic evolution failed")
                            return {"error": str(e)}, None

                    analyze_topic_btn.click(
                        analyze_topic_evolution,
                        inputs=[topic_input, time_window],
                        outputs=[evolution_output, evolution_chart]
                    )

                with gr.Tab("Citation Velocity"):
                    paper_id_input = gr.Textbox(label="Paper ID")
                    analyze_velocity_btn = gr.Button("Analyze Citation Velocity")
                    velocity_output = gr.JSON(label="Velocity Data")

                    def analyze_velocity(paper_id):
                        if not system or not hasattr(system, 'temporal_analytics'):
                            return {}

                        try:
                            return system.temporal_analytics.analyze_citation_velocity(paper_id)
                        except Exception as e:
                            return {"error": str(e)}

                    analyze_velocity_btn.click(
                        analyze_velocity,
                        inputs=[paper_id_input],
                        outputs=[velocity_output]
                    )

                with gr.Tab("H-Index Timeline"):
                    author_id_input = gr.Textbox(label="Author ID")
                    h_index_btn = gr.Button("Compute H-Index Timeline")
                    h_index_output = gr.JSON(label="H-Index Data")

                    def compute_h_index(author_id):
                        if not system or not hasattr(system, 'temporal_analytics'):
                            return {}

                        try:
                            return system.temporal_analytics.compute_h_index_timeline(author_id)
                        except Exception as e:
                            return {"error": str(e)}

                    h_index_btn.click(
                        compute_h_index,
                        inputs=[author_id_input],
                        outputs=[h_index_output]
                    )

            # ========== TAB 5: Personalized Recommendations ==========
            with gr.TabItem("💡 Recommendations"):
                gr.Markdown("### Get personalized paper recommendations")
                gr.Markdown("""
                **🎯 Purpose:**
                Get personalized paper recommendations based on your reading history, interests, and preferences. Uses collaborative filtering + content-based algorithms to suggest relevant papers you haven't read yet.

                **📋 Step-by-Step Instructions:**

                **Part 1: Create User Profile**
                1. **User ID**: Choose a unique identifier (e.g., "researcher123", your name)
                2. **Papers Read**: List papers you've already read (comma-separated titles from your database)
                3. **Papers Liked**: List papers you particularly enjoyed or found useful
                4. **Research Interests**: Keywords describing your research focus (comma-separated)
                5. **Create/Update Profile**: Click button to save your profile

                **Part 2: Get Recommendations**
                6. **Number of Recommendations**: How many papers to suggest (5-20)
                7. **Diversity Weight**: Slider from 0 (similar) to 1 (exploratory)
                   - 0.0-0.2: Very similar to what you've read (safe choices)
                   - 0.3-0.5: Balanced mix (recommended)
                   - 0.6-1.0: Exploratory, outside your usual topics (discovery mode)
                8. **Get Recommendations**: Click to receive personalized suggestions

                **💡 Example Profiles:**

                *Focused Researcher (Low Diversity):*
                ```
                User ID: "alice_nlp"
                Papers Read: "BERT, GPT-2, RoBERTa, ALBERT"
                Papers Liked: "BERT, GPT-2"
                Interests: "transformers, natural language processing, pretraining"
                Diversity: 0.2

                Result: Suggests GPT-3, ELECTRA, T5 (all transformer-based NLP)
                Use Case: Deep dive into specific area
                ```

                *Exploratory Researcher (High Diversity):*
                ```
                User ID: "bob_explorer"
                Papers Read: "Attention Is All You Need, GCN"
                Papers Liked: "Attention Is All You Need"
                Interests: "neural networks, deep learning"
                Diversity: 0.7

                Result: Suggests papers from NLP, CV, GNNs, RL (varied topics)
                Use Case: Discover cross-disciplinary connections
                ```

                *Balanced Researcher (Medium Diversity):*
                ```
                User ID: "carol_balanced"
                Papers Read: "GCN, GAT, GraphSAGE"
                Papers Liked: "GAT"
                Interests: "graph neural networks, representation learning"
                Diversity: 0.3

                Result: Mix of GNN papers + related areas (attention, message passing)
                Use Case: Stay in field but explore adjacent topics
                ```

                **⚙️ How It Works:**

                **Algorithm Combination:**
                1. **Content-Based**: Finds papers similar to ones you liked (based on text, concepts)
                2. **Collaborative Filtering**: "Users who read X also read Y"
                3. **Graph-Based**: Papers cited by/citing papers you liked
                4. **Diversity Injection**: Adds varied papers based on diversity weight

                **Recommendation Score:**
                - Relevance Score (0-1): How well paper matches your profile
                - Higher score = stronger match
                - Score considers: reading history, likes, interests, citations

                **📊 Diversity Weight Explained:**

                | Weight | Behavior | Example Result |
                |--------|----------|----------------|
                | **0.0** | Pure exploitation | Papers almost identical to what you've read |
                | **0.2** | Low diversity | Variations within same sub-field |
                | **0.3-0.5** | Balanced | Main field + adjacent areas |
                | **0.7** | High diversity | Cross-disciplinary suggestions |
                | **1.0** | Pure exploration | Potentially unrelated but interesting papers |

                **📝 Tips for Best Results:**

                **Profile Creation:**
                - Include 3-5 papers you've read for minimum viable profile
                - List 1-3 "liked" papers (most influential to you)
                - Use 3-5 interest keywords (not too broad, not too narrow)
                - Update profile regularly as you read more

                **Getting Recommendations:**
                - Start with diversity 0.3 for balanced results
                - Increase diversity when feeling stuck or want new ideas
                - Decrease diversity when doing deep literature review
                - Request 10-15 recommendations for good variety

                **Profile Management:**
                - Create multiple profiles for different research projects
                - Update "Papers Read" list after each session
                - Refresh recommendations as your interests evolve
                - Liked papers have 3x more influence than just read papers

                **💡 Use Cases:**

                *Literature Review:*
                ```
                Diversity: 0.2
                Goal: Find all relevant papers in narrow area
                Strategy: Low diversity ensures comprehensive coverage
                ```

                *Research Inspiration:*
                ```
                Diversity: 0.7
                Goal: Find novel connections and ideas
                Strategy: High diversity reveals unexpected connections
                ```

                *Stay Updated:*
                ```
                Diversity: 0.4
                Goal: Keep current with field developments
                Strategy: Medium diversity balances core + adjacent work
                ```

                **🔧 Troubleshooting:**
                - **No recommendations**: Need more papers in database, try lower diversity
                - **Recommendations not relevant**: Update profile with more specific interests
                - **Too similar recommendations**: Increase diversity weight
                - **Too random recommendations**: Decrease diversity weight
                - **Error: User not found**: Create profile first before getting recommendations
                """)

                with gr.Row():
                    user_id_input = gr.Textbox(label="User ID", value="user1")

                with gr.Row():
                    read_papers_input = gr.Textbox(
                        label="Papers Read (comma-separated IDs)",
                        placeholder="paper1, paper2, paper3"
                    )
                    liked_papers_input = gr.Textbox(
                        label="Papers Liked (comma-separated IDs)",
                        placeholder="paper1"
                    )

                interests_input = gr.Textbox(
                    label="Research Interests (comma-separated)",
                    placeholder="machine learning, neural networks"
                )

                create_profile_btn = gr.Button("Create/Update Profile")
                profile_status = gr.Textbox(label="Status", interactive=False)

                num_recs = gr.Slider(
                    minimum=5,
                    maximum=20,
                    value=10,
                    step=1,
                    label="Number of Recommendations"
                )
                diversity_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    label="Diversity Weight"
                )

                get_recs_btn = gr.Button("Get Recommendations", variant="primary")
                recommendations_output = gr.JSON(label="Recommendations")

                def create_profile(user_id, read_papers, liked_papers, interests):
                    if not system or not hasattr(system, 'recommendation_engine'):
                        return "Recommendation system not available"

                    try:
                        read_list = [p.strip() for p in read_papers.split(',') if p.strip()]
                        liked_list = [p.strip() for p in liked_papers.split(',') if p.strip()]
                        interest_list = [i.strip() for i in interests.split(',') if i.strip()]

                        profile = system.recommendation_engine.create_user_profile(
                            user_id, read_list, liked_list, interest_list
                        )

                        return profile.get('profile_summary', 'Profile created')
                    except Exception as e:
                        logger.exception("Profile creation failed")
                        return f"Error: {e}"

                def get_recommendations(user_id, n, diversity):
                    if not system or not hasattr(system, 'recommendation_engine'):
                        return []

                    try:
                        return system.recommendation_engine.recommend_papers(
                            user_id, n=int(n), diversity_weight=diversity
                        )
                    except Exception as e:
                        logger.exception("Recommendation failed")
                        return [{"error": str(e)}]

                create_profile_btn.click(
                    create_profile,
                    inputs=[user_id_input, read_papers_input, liked_papers_input, interests_input],
                    outputs=[profile_status]
                )

                get_recs_btn.click(
                    get_recommendations,
                    inputs=[user_id_input, num_recs, diversity_slider],
                    outputs=[recommendations_output]
                )

            # ========== TAB 6: Citation Explorer ==========
            with gr.TabItem("🕸️ Citation Explorer"):
                gr.Markdown("### Explore citation networks interactively")
                gr.Markdown("""
                **🎯 Purpose:**
                Visualize citation chains starting from a specific paper. Explore forward citations (who cites this paper), backward citations (what this paper cites), or both. Discover citation cascades and research impact.

                **📋 Step-by-Step Instructions:**
                1. **Paper ID**: Enter the title or ID of a paper from your database
                2. **Max Depth**: How many citation hops to explore (1-5)
                   - Depth 1: Direct citations only
                   - Depth 2: Citations + their citations (recommended)
                   - Depth 3+: Extended network (can get large)
                3. **Direction**: Choose exploration direction
                   - **Forward**: Papers that cite this paper (impact)
                   - **Backward**: Papers this paper cites (foundations)
                   - **Both**: Complete citation context (recommended)
                4. **Explore**: Click button to generate interactive visualization

                **💡 Example Use Cases:**

                *Measure Paper Impact:*
                ```
                Paper: "Attention Is All You Need"
                Max Depth: 2
                Direction: Forward

                Result: See BERT, GPT-2, GPT-3, and hundreds of papers citing it
                Interpretation: Foundational paper with massive impact
                ```

                *Trace Research Origins:*
                ```
                Paper: "BERT"
                Max Depth: 2
                Direction: Backward

                Result: See transformer paper, ELMo, Word2Vec foundations
                Interpretation: Builds on attention + word embeddings
                ```

                *Full Citation Context:*
                ```
                Paper: "Graph Attention Networks"
                Max Depth: 2
                Direction: Both

                Result: See GCN foundation + GAT applications
                Interpretation: Understand paper's place in research evolution
                ```

                **⚙️ Parameter Guide:**

                **Max Depth:**
                | Depth | Nodes | Best For | Example |
                |-------|-------|----------|---------|
                | **1** | 5-20 | Quick overview | Direct citations only |
                | **2** | 20-100 | Standard analysis | Recommended default |
                | **3** | 100-500 | Deep dive | Extended network |
                | **4-5** | 500+ | Research history | May be overwhelming |

                **Direction:**
                - **Forward (Impact)**: Who built on this work? How influential is it?
                - **Backward (Context)**: What does this work build on? Research foundations?
                - **Both (Complete)**: Full citation context, bidirectional view

                **📊 Interpreting Visualizations:**

                **Network Patterns:**
                - **Hub (many forward citations)**: Influential/seminal paper
                - **Bridge (connects clusters)**: Cross-disciplinary work
                - **Chain (linear citations)**: Incremental improvements
                - **Star (many backward citations)**: Survey/review paper

                **Citation Density:**
                - Dense network: Active, well-connected research area
                - Sparse network: Niche topic or emerging field
                - Clustering: Research communities/sub-fields

                **📝 Tips:**
                - Start with depth 2, direction "both" for balanced view
                - Forward citations show impact and influence
                - Backward citations reveal theoretical foundations
                - Click nodes in visualization to see paper details
                - Zoom and pan to navigate large networks
                - Look for bridge papers connecting different areas

                **🔧 Troubleshooting:**
                - **No citations found**: Paper not in database or has no citations
                - **Visualization too large**: Reduce max depth
                - **Slow loading**: Depth 4+ can take 10-30 seconds
                - **Empty result**: Check paper ID spelling, try partial title match
                """)

                paper_id_cite = gr.Textbox(label="Paper ID")
                max_depth = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Max Depth")
                direction = gr.Dropdown(
                    choices=["forward", "backward", "both"],
                    value="both",
                    label="Direction"
                )

                explore_btn = gr.Button("Explore Citation Network")
                citation_viz = gr.HTML(label="Interactive Citation Network")

                def explore_citations(paper_id, depth, dir):
                    if not system or not hasattr(system, 'citation_explorer'):
                        return "<p>Citation explorer not available</p>"

                    try:
                        return system.citation_explorer.create_citation_chain_viz(
                            paper_id, max_depth=int(depth), direction=dir
                        )
                    except Exception as e:
                        logger.exception("Citation exploration failed")
                        return f"<p>Error: {e}</p>"

                explore_btn.click(
                    explore_citations,
                    inputs=[paper_id_cite, max_depth, direction],
                    outputs=[citation_viz]
                )

            # ========== TAB 7: Discovery ==========
            with gr.TabItem("🔬 Discovery"):
                gr.Markdown("### Discover unexpected connections")
                gr.Markdown("""
                **🎯 Purpose:**
                Find hidden connections between research papers. Discover similar work based on semantic content (not just citations), and identify cross-disciplinary opportunities where different fields tackle similar problems.

                **📋 Two Discovery Modes:**

                **1. Similar Papers**
                - Finds papers with similar content/concepts using embeddings
                - Works even without citation relationships
                - Great for finding related work you might have missed

                **2. Cross-Disciplinary Connections**
                - Discovers papers from different fields with similar ideas
                - Identifies potential knowledge transfer opportunities
                - Reveals unexpected methodological parallels

                **💡 When to Use Each:**

                **Similar Papers:**
                - Literature review (find everything on your topic)
                - Validate your approach (see what others did)
                - Find contemporary work (recent papers without citations yet)
                - Discover papers you should have cited

                **Cross-Disciplinary:**
                - Get inspiration from other fields
                - Find novel applications of your methods
                - Identify research gaps at field intersections
                - Discover transferable techniques
                """)

                with gr.Tab("Similar Papers"):
                    gr.Markdown("""
                    **📋 Instructions:**
                    1. **Paper ID**: Enter title or ID of a paper from your database
                    2. **Similarity Method**: Choose distance metric
                       - **Cosine**: Standard, works well for most cases (recommended)
                       - **Euclidean**: Geometric distance, good for dense embeddings
                       - **Dot Product**: Fast, good for normalized embeddings
                    3. **Find Similar Papers**: Click to get results

                    **💡 Example Use Cases:**

                    *Find Related Work:*
                    ```
                    Paper ID: "Graph Attention Networks"
                    Method: Cosine

                    Result:
                    1. "Graph Convolutional Networks" (similarity: 0.92)
                    2. "Message Passing Neural Networks" (0.89)
                    3. "Attention Is All You Need" (0.87)

                    Interpretation: GAT is semantically close to GCN + attention mechanisms
                    ```

                    *Validate Approach:*
                    ```
                    Paper ID: "Your New Paper"
                    Method: Cosine

                    Result: Shows papers with similar approaches
                    Use: Check if your method is novel or incremental
                    ```

                    **⚙️ Similarity Method Comparison:**

                    | Method | Range | Best For | Speed |
                    |--------|-------|----------|-------|
                    | **Cosine** | 0-1 | General purpose, sparse data | Fast |
                    | **Euclidean** | 0-∞ | Dense embeddings, magnitude matters | Fast |
                    | **Dot Product** | -∞-∞ | Normalized embeddings | Fastest |

                    **📊 Similarity Score Interpretation:**
                    - **0.9-1.0**: Nearly identical content/topic
                    - **0.8-0.9**: Very similar, likely same sub-field
                    - **0.7-0.8**: Similar topic, related work
                    - **0.6-0.7**: Loosely related
                    - **<0.6**: Different topics (filter out)

                    **📝 Tips:**
                    - Cosine similarity recommended for most use cases
                    - High scores (>0.85) may indicate duplicate or very close work
                    - Medium scores (0.7-0.85) are sweet spot for related work
                    - Works great for finding papers without citation connections
                    """)
                    paper_id_sim = gr.Textbox(label="Paper ID")
                    similarity_method = gr.Dropdown(
                        choices=["cosine", "euclidean", "dot_product"],
                        value="cosine",
                        label="Similarity Method"
                    )
                    find_similar_btn = gr.Button("Find Similar Papers")
                    similar_output = gr.JSON(label="Similar Papers")

                    def find_similar(paper_id, method):
                        if not system or not hasattr(system, 'discovery_engine'):
                            return []

                        try:
                            return system.discovery_engine.find_similar_by_embedding(
                                paper_id, n=10, method=method
                            )
                        except Exception as e:
                            return [{"error": str(e)}]

                    find_similar_btn.click(
                        find_similar,
                        inputs=[paper_id_sim, similarity_method],
                        outputs=[similar_output]
                    )

                with gr.Tab("Cross-Disciplinary"):
                    gr.Markdown("""
                    **📋 Instructions:**
                    1. **Paper ID**: Enter title or ID of a paper from your database
                    2. **Find Cross-Disciplinary Connections**: Click to discover papers from other fields

                    **💡 Example Use Cases:**

                    *Find Inspiration from Other Fields:*
                    ```
                    Paper ID: "Graph Neural Networks for Molecule Property Prediction"

                    Result:
                    1. "CNNs for Image Recognition" (Computer Vision)
                    2. "RNNs for Time Series" (Signal Processing)
                    3. "Attention for NLP" (Natural Language Processing)

                    Interpretation: GNNs share architectural patterns with other domains
                    Discovery: Techniques from CV/NLP may transfer to graph domain
                    ```

                    *Identify Knowledge Transfer:*
                    ```
                    Paper ID: "Attention Mechanisms in NLP"

                    Result: Graph Attention Networks (GNN field)
                    Discovery: Attention concept successfully transferred from NLP to graphs
                    ```

                    *Find Novel Applications:*
                    ```
                    Paper ID: "Your Method for Problem X"

                    Result: Papers solving similar problems in different domains
                    Opportunity: Apply your method to their domain for novel contribution
                    ```

                    **🔍 How It Works:**
                    The system identifies cross-disciplinary connections by:
                    1. Finding papers with similar methodologies but different keywords/venues
                    2. Detecting shared problem structures across fields
                    3. Identifying papers that bridge multiple research communities
                    4. Looking for common mathematical foundations

                    **📊 Interpreting Results:**
                    - **Same methods, different data**: Direct application opportunity
                    - **Different methods, same problem**: Alternative approaches to learn from
                    - **Similar math, different context**: Fundamental connection
                    - **Bridge papers**: Explicitly connect multiple fields

                    **💡 What to Do with Cross-Disciplinary Findings:**
                    1. **Import Techniques**: Adapt methods from other fields to your problem
                    2. **Export Solutions**: Apply your methods to other domains
                    3. **Collaborate**: Reach out to researchers in connected fields
                    4. **Novel Research**: Identify unexplored intersections

                    **📝 Tips:**
                    - Look for papers with 20-40% keyword overlap (too much = same field)
                    - Best discoveries often come from seemingly unrelated fields
                    - Check if method assumptions transfer to new domain
                    - Cross-disciplinary work often leads to high-impact papers
                    - Consider joint papers with experts from discovered fields
                    """)
                    paper_id_cross = gr.Textbox(label="Paper ID")
                    find_cross_btn = gr.Button("Find Cross-Disciplinary Connections")
                    cross_output = gr.JSON(label="Cross-Disciplinary Papers")

                    def find_cross_disciplinary(paper_id):
                        if not system or not hasattr(system, 'discovery_engine'):
                            return []

                        try:
                            return system.discovery_engine.discover_cross_disciplinary_connections(
                                paper_id, n=5
                            )
                        except Exception as e:
                            return [{"error": str(e)}]

                    find_cross_btn.click(
                        find_cross_disciplinary,
                        inputs=[paper_id_cross],
                        outputs=[cross_output]
                    )

            # ========== TAB 7: Advanced Metrics ========== 
            # DISABLED for GNN-focused research project
            # This tab uses traditional graph algorithms (PageRank, Disruption Index)
            # rather than GNN-based methods. For GNN-powered metrics, see Graph & GNN Dashboard.
            # 
            # To re-enable, uncomment the section below:
            # with gr.TabItem("📈 Advanced Metrics"):
            #     gr.Markdown("### Sophisticated citation analysis")
            #     paper_id_metrics = gr.Textbox(label="Paper ID")
            #     with gr.Row():
            #         disruption_btn = gr.Button("Disruption Index")
            #         sleeping_beauty_btn = gr.Button("Sleeping Beauty Score")
            #         cascades_btn = gr.Button("Citation Cascades")
            #     metrics_output = gr.JSON(label="Metrics Data")
            #     def compute_disruption(paper_id):
            #         if not system or not hasattr(system, 'citation_metrics'):
            #             return {}
            #         try:
            #             index = system.citation_metrics.calculate_disruption_index(paper_id)
            #             return {"disruption_index": index}
            #         except Exception as e:
            #             return {"error": str(e)}
            #     def compute_sleeping_beauty(paper_id):
            #         if not system or not hasattr(system, 'citation_metrics'):
            #             return {}
            #         try:
            #             return system.citation_metrics.compute_sleeping_beauty_score(paper_id)
            #         except Exception as e:
            #             return {"error": str(e)}
            #     def analyze_cascades(paper_id):
            #         if not system or not hasattr(system, 'citation_metrics'):
            #             return {}
            #         try:
            #             return system.citation_metrics.analyze_citation_cascades(paper_id)
            #         except Exception as e:
            #             return {"error": str(e)}
            #     disruption_btn.click(compute_disruption, inputs=[paper_id_metrics], outputs=[metrics_output])
            #     sleeping_beauty_btn.click(compute_sleeping_beauty, inputs=[paper_id_metrics], outputs=[metrics_output])
            #     cascades_btn.click(analyze_cascades, inputs=[paper_id_metrics], outputs=[metrics_output])

            # ========== TAB 8: Cache Management ==========
            with gr.TabItem("💾 Cache Management"):
                gr.Markdown("### View and manage cache statistics")
                gr.Markdown("""
                **🎯 Purpose:**
                Monitor and manage the intelligent cache system that speeds up queries by 10-100x. View cache statistics, hit rates, memory usage, and manually clear cache when needed.

                **📋 Available Actions:**

                **1. Refresh Stats** (🔄)
                - View current cache statistics
                - See cache hit rates, entry counts, memory usage
                - Check which namespaces are cached

                **2. Clear All Cache** (🗑️)
                - Removes all cached data
                - Use when: System behavior seems stale, after major updates
                - Warning: Next queries will be slower until cache rebuilds

                **3. Clear Expired** (🧹)
                - Removes only expired cache entries (past TTL)
                - Safe operation, recommended for routine maintenance
                - Frees memory without affecting active cache

                **💡 Understanding Cache Statistics:**

                **Example Output:**
                ```json
                {
                  "total_entries": 150,
                  "namespaces": {
                    "queries": 80,
                    "embeddings": 50,
                    "graph_data": 20
                  },
                  "hit_rate": 0.75,
                  "memory_usage_mb": 45,
                  "expired_entries": 12
                }
                ```

                **Metrics Explained:**
                - **total_entries**: Number of cached items across all namespaces
                - **namespaces**: Breakdown by cache type (queries, embeddings, etc.)
                - **hit_rate**: 0.75 = 75% of requests served from cache (higher is better)
                - **memory_usage_mb**: RAM used by cache (in megabytes)
                - **expired_entries**: Items past TTL but not yet cleared

                **📊 Cache Namespaces:**

                | Namespace | What It Caches | TTL | Impact if Cleared |
                |-----------|----------------|-----|-------------------|
                | **queries** | Research Assistant answers | 30 min | Queries re-run (2-5s each) |
                | **embeddings** | Vector representations | 24h | Papers re-embedded (~1s each) |
                | **graph_data** | Graph statistics, viz | 1h | Graph re-analyzed (~5-10s) |
                | **predictions** | GNN predictions | 2h | Models re-run (~1-3s) |
                | **recommendations** | User recommendations | 30 min | Recs recalculated (~2-5s) |

                **⚡ Cache Hit Rate Interpretation:**
                - **80-100%**: Excellent, most requests cached
                - **60-79%**: Good, cache is working well
                - **40-59%**: Fair, queries are varied
                - **<40%**: Low, consider increasing TTL or cache size

                **💡 When to Clear Cache:**

                **Clear All (🗑️):**
                - After uploading many new papers (>20% of database)
                - After retraining GNN models
                - System giving stale/outdated answers
                - After changing LLM settings
                - Debugging: Want fresh results to test
                - Memory usage too high (>500MB)

                **Clear Expired Only (🧹):**
                - Routine maintenance (daily/weekly)
                - Free memory without disrupting active cache
                - After extending sessions (remove old entries)
                - Before long compute tasks (free RAM)

                **Keep Cache (No Action):**
                - During active research sessions
                - When hit rate is high (>60%)
                - System working well, no issues
                - Memory usage acceptable

                **📝 Best Practices:**

                **Daily Use:**
                1. Check cache stats at start of session
                2. Clear expired entries if many accumulated
                3. Monitor hit rate - should improve during session
                4. Clear all only when necessary

                **After Major Changes:**
                1. Upload new papers → Clear "embeddings" namespace
                2. Retrain GNN → Clear "predictions" namespace
                3. Update LLM settings → Clear "queries" namespace
                4. When in doubt, clear all and rebuild

                **Memory Management:**
                - Cache grows to ~1GB for large databases (500+ papers)
                - Each query cached uses ~100-500KB
                - Clear cache if memory usage exceeds 1GB
                - Expired entries still use memory until cleared

                **🎯 Optimization Tips:**

                **Maximize Cache Benefits:**
                - Ask similar questions during session (higher hit rate)
                - Use same papers for multiple analyses
                - Enable cache in Research Assistant (default)
                - Let cache warm up over first few queries

                **When Cache Hurts:**
                - Rapidly changing data (papers added/removed frequently)
                - Each query unique (no repeat patterns)
                - Memory constrained systems (<4GB RAM)
                - In these cases: Disable cache or use short TTL

                **📊 Example Scenarios:**

                *Scenario 1: Active Research Session*
                ```
                Stats: 120 entries, 85% hit rate, 60MB memory
                Action: None - cache working great!
                Result: Queries 10-100x faster
                ```

                *Scenario 2: After Adding 50 New Papers*
                ```
                Stats: 200 entries, 45% hit rate, 150MB memory
                Action: Clear All Cache
                Result: Next queries slower, but will use new papers
                ```

                *Scenario 3: End of Day*
                ```
                Stats: 180 entries, 70% hit rate, 120MB memory, 45 expired
                Action: Clear Expired
                Result: Freed 35MB, keep active cache
                ```

                **🔧 Troubleshooting:**
                - **Cache stats empty**: Cache not initialized, restart application
                - **Hit rate 0%**: Cache disabled or very first queries
                - **High memory (>1GB)**: Clear all cache, consider reducing TTL
                - **Stale answers**: Clear queries namespace or clear all
                - **"Cache not available" error**: Cache system not running, check logs
                """)

                with gr.Row():
                    refresh_stats_btn = gr.Button("🔄 Refresh Stats")
                    clear_all_cache_btn = gr.Button("🗑️ Clear All Cache", variant="stop")
                    clear_expired_btn = gr.Button("🧹 Clear Expired")

                cache_stats_output = gr.JSON(label="Cache Statistics")
                cache_action_status = gr.Textbox(label="Status", lines=2)

                def get_cache_stats():
                    """Get current cache statistics."""
                    if not cache:
                        return {"error": "Cache not available"}, "⚠️ Cache not initialized"

                    try:
                        stats = cache.get_stats()
                        return stats, f"✅ Cache stats retrieved at {time.strftime('%H:%M:%S')}"
                    except Exception as e:
                        return {"error": str(e)}, f"❌ Error: {e}"

                def clear_all_cache():
                    """Clear entire cache."""
                    if not cache:
                        return {}, "⚠️ Cache not available"

                    try:
                        cache.clear_all()
                        stats = cache.get_stats()
                        return stats, "✅ All cache cleared successfully"
                    except Exception as e:
                        return {}, f"❌ Error: {e}"

                def clear_expired_cache():
                    """Clear only expired entries."""
                    if not cache:
                        return {}, "⚠️ Cache not available"

                    try:
                        cache.clear_expired()
                        stats = cache.get_stats()
                        return stats, "✅ Expired entries cleared"
                    except Exception as e:
                        return {}, f"❌ Error: {e}"

                refresh_stats_btn.click(
                    get_cache_stats,
                    outputs=[cache_stats_output, cache_action_status]
                )

                clear_all_cache_btn.click(
                    clear_all_cache,
                    outputs=[cache_stats_output, cache_action_status]
                )

                clear_expired_btn.click(
                    clear_expired_cache,
                    outputs=[cache_stats_output, cache_action_status]
                )

            # ========== TAB 9: Settings ==========
            with gr.TabItem("⚙️ Settings"):
                gr.Markdown("### Configure Models and System Settings")
                gr.Markdown("""
                **🎯 Purpose:**
                Configure the AI models and system settings that power Research Compass. Set up LLM (language model), embedding model, cache behavior, and database connections.

                **⚙️ Available Settings:**
                1. **🤖 LLM Model**: Language model for question answering (local or cloud)
                2. **🔢 Embedding Model**: Vector embeddings for similarity search
                3. **💾 Cache Settings**: Cache TTL and memory limits
                4. **🗄️ Database Connection**: Neo4j or in-memory graph database

                **💡 Quick Start:**
                - **Local Setup** (Free): Use Ollama (LLM) + HuggingFace (embeddings)
                - **Cloud Setup** (Paid): Use OpenAI/OpenRouter (LLM) + HuggingFace (embeddings)
                - **Hybrid** (Recommended): Local embeddings + Cloud LLM for best quality
                """)

                with gr.Tabs():
                    # LLM Settings (Enhanced)
                    with gr.TabItem("🤖 LLM Model"):
                        gr.Markdown("#### Select Language Model")
                        gr.Markdown("""
                        **🎯 Purpose:**
                        Choose the language model that answers research questions. Supports local (free, private) and cloud (paid, higher quality) options.

                        **📋 Supported Providers:**

                        **Local (Free):**
                        - **Ollama** (localhost:11434): Run models like llama3.2, mistral locally
                        - **LM Studio** (localhost:1234): GUI for local models, easy setup

                        **Cloud (Paid):**
                        - **OpenRouter**: Access 100+ models (GPT-4, Claude, etc.) with single API key
                        - **OpenAI**: Direct access to GPT-3.5, GPT-4, GPT-4 Turbo

                        **📋 Setup Instructions:**

                        **For Ollama (Recommended for Local):**
                        1. Install: `brew install ollama` (Mac) or download from ollama.ai
                        2. Start: `ollama serve` in terminal
                        3. Pull model: `ollama pull llama3.2`
                        4. In UI: Select "ollama", detect models, choose llama3.2
                        5. Save settings

                        **For LM Studio:**
                        1. Download LM Studio from lmstudio.ai
                        2. Launch app, download a model (e.g., Mistral 7B)
                        3. Start local server in LM Studio (port 1234)
                        4. In UI: Select "lmstudio", detect models, choose model
                        5. Save settings

                        **For OpenRouter (Best Cloud Option):**
                        1. Get API key from openrouter.ai (requires payment)
                        2. In UI: Select "openrouter"
                        3. Enter API key when prompted
                        4. Test connection, detect models
                        5. Choose model (e.g., claude-3.5-sonnet, gpt-4)
                        6. Save settings

                        **For OpenAI:**
                        1. Get API key from platform.openai.com (requires payment)
                        2. In UI: Select "openai"
                        3. Enter API key
                        4. Test connection, choose model (gpt-4 recommended)
                        5. Save settings

                        **⚙️ Model Parameters:**

                        **Temperature** (0.0-1.0):
                        - **0.0-0.3**: Focused, deterministic (recommended for research)
                        - **0.4-0.7**: Balanced creativity and consistency
                        - **0.8-1.0**: Creative, varied responses

                        **Max Tokens** (100-4000):
                        - **100-500**: Short answers
                        - **500-1000**: Standard answers (recommended)
                        - **1000-4000**: Long, detailed responses

                        **💡 Provider Comparison:**

                        | Provider | Cost | Quality | Speed | Privacy | Setup |
                        |----------|------|---------|-------|---------|-------|
                        | **Ollama** | Free | Good | Medium | Private | Easy |
                        | **LM Studio** | Free | Good | Medium | Private | Very Easy |
                        | **OpenRouter** | $0.01-0.10/1K tokens | Excellent | Fast | Shared | Easy |
                        | **OpenAI** | $0.002-0.06/1K tokens | Excellent | Fast | Shared | Easy |

                        **📝 Recommendations:**
                        - **Budget**: Ollama + llama3.2 (free, decent quality)
                        - **Best Quality**: OpenRouter + claude-3.5-sonnet or GPT-4
                        - **Balance**: Ollama + mistral (free, good performance)
                        - **Privacy**: Always use local (Ollama/LM Studio)

                        **🔧 Troubleshooting:**
                        - **"Cannot connect to Ollama"**: Run `ollama serve` in terminal
                        - **"Cannot connect to LM Studio"**: Start local server in LM Studio settings
                        - **"Invalid API key"**: Check key, ensure billing enabled for cloud providers
                        - **Slow responses**: Local models depend on hardware, cloud is faster
                        - **"Model not found"**: Pull model first with `ollama pull <model>`
                        """)

                        with gr.Row():
                            with gr.Column():
                                llm_provider = gr.Dropdown(
                                    label="LLM Provider",
                                    choices=["ollama", "lmstudio", "openrouter", "openai"],
                                    value=os.getenv("LLM_PROVIDER", "ollama"),
                                    info="Select your LLM provider (local or cloud)"
                                )

                                # API Key input (hidden for local providers)
                                llm_api_key = gr.Textbox(
                                    label="API Key (for OpenRouter/OpenAI)",
                                    type="password",
                                    value=os.getenv("OPENROUTER_API_KEY", "") if os.getenv("LLM_PROVIDER") == "openrouter" else os.getenv("OPENAI_API_KEY", ""),
                                    placeholder="Enter API key if using OpenRouter or OpenAI",
                                    visible=os.getenv("LLM_PROVIDER") in ["openrouter", "openai"]
                                )

                                with gr.Row():
                                    test_connection_btn = gr.Button("🔌 Test Connection", variant="secondary")
                                    detect_models_btn = gr.Button("🔍 Detect Models", variant="secondary")

                                connection_status = gr.Textbox(
                                    label="Connection Status",
                                    value="Not tested",
                                    interactive=False,
                                    lines=2
                                )

                                llm_model = gr.Dropdown(
                                    label="Model Name",
                                    choices=["llama3.2"],
                                    value="llama3.2",
                                    allow_custom_value=True,
                                    info="Select or enter model name"
                                )

                                available_models_display = gr.Textbox(
                                    label="Available Models",
                                    lines=5,
                                    interactive=False,
                                    placeholder="Click 'Detect Models' to see available models"
                                )

                                llm_temperature = gr.Slider(
                                    label="Temperature",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.3,
                                    step=0.1,
                                    info="0=focused, 1=creative"
                                )

                                llm_max_tokens = gr.Slider(
                                    label="Max Tokens",
                                    minimum=100,
                                    maximum=4000,
                                    value=1000,
                                    step=100,
                                    info="Maximum response length"
                                )

                                llm_save_btn = gr.Button("💾 Save LLM Settings", variant="primary")

                            with gr.Column():
                                llm_status = gr.Textbox(
                                    label="Status",
                                    lines=15,
                                    interactive=False
                                )

                        # Show/hide API key based on provider
                        def update_api_key_visibility(provider):
                            """Show API key field for cloud providers."""
                            needs_api_key = provider in ["openrouter", "openai"]
                            return gr.update(visible=needs_api_key)

                        llm_provider.change(
                            update_api_key_visibility,
                            inputs=[llm_provider],
                            outputs=[llm_api_key]
                        )

                        def test_llm_connection(provider, api_key):
                            """Test connection to selected LLM provider."""
                            if provider == "ollama":
                                result = test_ollama_connection()
                            elif provider == "lmstudio":
                                result = test_lmstudio_connection()
                            elif provider == "openrouter":
                                result = test_openrouter_connection(api_key)
                            elif provider == "openai":
                                result = test_openai_connection(api_key)
                            else:
                                result = {"success": False, "message": "Unknown provider"}

                            if result["success"]:
                                return f"✓ Connected\n{result['message']}"
                            else:
                                return f"✗ Not connected\n{result['message']}"

                        def detect_available_models(provider, api_key):
                            """Detect and populate available models."""
                            if provider == "ollama":
                                models = detect_ollama_models()
                                test_result = test_ollama_connection()
                            elif provider == "lmstudio":
                                models = detect_lmstudio_models()
                                test_result = test_lmstudio_connection()
                            elif provider == "openrouter":
                                models = detect_openrouter_models(api_key)
                                test_result = test_openrouter_connection(api_key)
                            elif provider == "openai":
                                models = detect_openai_models(api_key)
                                test_result = test_openai_connection(api_key)
                            else:
                                return gr.update(), "Unknown provider", f"✗ Not connected\nUnknown provider"

                            if not test_result["success"]:
                                return (
                                    gr.update(),
                                    "No models detected",
                                    f"✗ Not connected\n{test_result['message']}"
                                )

                            if models:
                                models_text = "\n".join([f"- {m}" for m in models])
                                return (
                                    gr.update(choices=models, value=models[0] if models else None),
                                    f"Found {len(models)} models:\n\n{models_text}",
                                    f"✓ Connected\n{len(models)} models available"
                                )
                            else:
                                return (
                                    gr.update(),
                                    "No models found. Ensure provider is running.",
                                    f"✓ Connected\nBut no models detected"
                                )

                        test_connection_btn.click(
                            test_llm_connection,
                            inputs=[llm_provider, llm_api_key],
                            outputs=[connection_status]
                        )

                        detect_models_btn.click(
                            detect_available_models,
                            inputs=[llm_provider, llm_api_key],
                            outputs=[llm_model, available_models_display, connection_status]
                        )

                        def save_llm_settings(provider, model, temp, max_tok, api_key):
                            """Save LLM configuration."""
                            try:
                                import os
                                from pathlib import Path

                                # Update environment variables
                                os.environ['LLM_PROVIDER'] = provider
                                os.environ['LLM_MODEL'] = model
                                os.environ['LLM_TEMPERATURE'] = str(temp)
                                os.environ['LLM_MAX_TOKENS'] = str(max_tok)

                                # Update .env file
                                env_path = Path.cwd() / '.env'
                                env_lines = []

                                if env_path.exists():
                                    with open(env_path, 'r') as f:
                                        for line in f:
                                            if not any(line.startswith(k) for k in [
                                                'LLM_PROVIDER=', 'LLM_MODEL=', 'LLM_TEMPERATURE=',
                                                'LLM_MAX_TOKENS=', 'OPENROUTER_API_KEY=', 'OPENAI_API_KEY='
                                            ]):
                                                env_lines.append(line.rstrip())

                                # Add updated values
                                env_lines.extend([
                                    f'LLM_PROVIDER={provider}',
                                    f'LLM_MODEL={model}',
                                    f'LLM_TEMPERATURE={temp}',
                                    f'LLM_MAX_TOKENS={max_tok}'
                                ])

                                # Add API keys if provided
                                if provider == "openrouter" and api_key:
                                    env_lines.append(f'OPENROUTER_API_KEY={api_key}')
                                    os.environ['OPENROUTER_API_KEY'] = api_key
                                elif provider == "openai" and api_key:
                                    env_lines.append(f'OPENAI_API_KEY={api_key}')
                                    os.environ['OPENAI_API_KEY'] = api_key

                                with open(env_path, 'w') as f:
                                    f.write('\n'.join(env_lines) + '\n')

                                # Update system in real-time without restart
                                update_msg = ""
                                if system:
                                    try:
                                        # Get the LLM manager from container
                                        llm_mgr = None

                                        if hasattr(system, 'container'):
                                            try:
                                                llm_mgr = system.container.resolve('llm_manager')
                                            except Exception as e:
                                                logger.debug(f"Could not resolve llm_manager from container: {e}")

                                        # Fallback: try to access llm_manager directly
                                        if not llm_mgr and hasattr(system, 'llm_manager'):
                                            llm_mgr = system.llm_manager

                                        # Update the LLM manager configuration dynamically
                                        if llm_mgr and hasattr(llm_mgr, 'update_config'):
                                            # Determine base_url based on provider
                                            if provider == "ollama":
                                                base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
                                            elif provider == "lmstudio":
                                                base_url = os.environ.get('LMSTUDIO_BASE_URL', 'http://localhost:1234')
                                            else:
                                                base_url = None

                                            llm_mgr.update_config(
                                                provider=provider,
                                                model=model,
                                                base_url=base_url,
                                                api_key=api_key if api_key else None,
                                                temperature=temp,
                                                max_tokens=max_tok
                                            )
                                            update_msg = "\n\n✅ Configuration applied immediately - no restart needed!"
                                        else:
                                            update_msg = "\n\n⚠️ LLM manager not available in current session.\nSettings saved to .env - they will apply on next launch."
                                    except Exception as e:
                                        logger.error(f"Error updating LLM config: {e}")
                                        update_msg = f"\n\n⚠️ Could not update runtime config: {str(e)}\nSettings saved to .env - they will apply on next launch."
                                else:
                                    update_msg = "\n\n📝 Settings saved successfully!\nSettings will apply when you launch the application.\n\n💡 Tip: If you're already running, restart the application to use the new settings."

                                status = f"""✅ LLM Settings Saved Successfully!

Provider: {provider}
Model: {model}
Temperature: {temp}
Max Tokens: {max_tok}
API Key: {"Set" if api_key else "Not provided"}

Settings saved to .env file.{update_msg}"""

                                return status

                            except Exception as e:
                                return f"❌ Error saving settings: {str(e)}"

                        llm_save_btn.click(
                            save_llm_settings,
                            inputs=[llm_provider, llm_model, llm_temperature, llm_max_tokens, llm_api_key],
                            outputs=llm_status
                        )

                    # Embedding Settings
                    with gr.TabItem("🔢 Embedding Model"):
                        gr.Markdown("#### Select Embedding Provider & Model")

                        with gr.Row():
                            with gr.Column():
                                embedding_provider = gr.Radio(
                                    label="Embedding Provider",
                                    choices=["huggingface", "ollama"],
                                    value="huggingface",
                                    info="Choose between HuggingFace (cloud/download) or Ollama (local)"
                                )
                                
                                # HuggingFace models
                                hf_models = gr.Dropdown(
                                    label="HuggingFace Model",
                                    choices=[
                                        "all-MiniLM-L6-v2",
                                        "all-mpnet-base-v2",
                                        "paraphrase-multilingual-MiniLM-L12-v2",
                                        "all-distilroberta-v1",
                                        "multi-qa-mpnet-base-dot-v1"
                                    ],
                                    value="all-MiniLM-L6-v2",
                                    visible=True,
                                    info="Sentence transformer models (auto-download)"
                                )
                                
                                # Ollama models
                                ollama_embedding_model = gr.Textbox(
                                    label="Ollama Model",
                                    value="nomic-embed-text",
                                    visible=False,
                                    info="Local Ollama embedding model (e.g., nomic-embed-text, mxbai-embed-large)"
                                )
                                
                                ollama_base_url = gr.Textbox(
                                    label="Ollama Base URL",
                                    value="http://localhost:11434",
                                    visible=False,
                                    info="URL where Ollama is running"
                                )

                                gr.Markdown("""
                                **HuggingFace Models:**

                                | Model | Speed | Quality | Dimensions | Use Case |
                                |-------|-------|---------|------------|----------|
                                | **all-MiniLM-L6-v2** | ⚡⚡⚡ Fast | Good | 384 | General, fast |
                                | **all-mpnet-base-v2** | ⚡⚡ Medium | Best | 768 | High quality |
                                | **paraphrase-multilingual** | ⚡ Slow | Good | 384 | Multi-language |
                                | **all-distilroberta-v1** | ⚡⚡ Medium | Very Good | 768 | Balanced |
                                | **multi-qa-mpnet** | ⚡⚡ Medium | Very Good | 768 | Q&A focused |
                                
                                **Ollama Models:**
                                
                                | Model | Dimensions | Use Case |
                                |-------|------------|----------|
                                | **nomic-embed-text** | 768 | General text (recommended) |
                                | **mxbai-embed-large** | 1024 | High quality |
                                | **all-minilm** | 384 | Fast, lightweight |
                                
                                Install: `ollama pull nomic-embed-text`
                                """)

                                embedding_save_btn = gr.Button("💾 Save Embedding Settings", variant="primary")

                            with gr.Column():
                                embedding_status = gr.Textbox(
                                    label="Status",
                                    lines=12,
                                    interactive=False
                                )
                        
                        # Toggle model inputs based on provider
                        def toggle_embedding_inputs(provider):
                            if provider == "huggingface":
                                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
                            else:  # ollama
                                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
                        
                        embedding_provider.change(
                            toggle_embedding_inputs,
                            inputs=embedding_provider,
                            outputs=[hf_models, ollama_embedding_model, ollama_base_url]
                        )

                        def save_embedding_settings(provider, hf_model, ollama_model, ollama_url):
                            """Save embedding model configuration."""
                            try:
                                import os
                                from pathlib import Path

                                # Determine model name based on provider
                                model_name = hf_model if provider == "huggingface" else ollama_model
                                
                                # Update environment variables
                                os.environ['EMBEDDING_PROVIDER'] = provider
                                os.environ['EMBEDDING_MODEL_NAME'] = model_name
                                if provider == "ollama":
                                    os.environ['OLLAMA_BASE_URL'] = ollama_url

                                # Update .env file
                                env_path = Path.cwd() / '.env'
                                env_lines = []

                                if env_path.exists():
                                    with open(env_path, 'r') as f:
                                        for line in f:
                                            if not line.startswith(('EMBEDDING_PROVIDER=', 'EMBEDDING_MODEL_NAME=')):
                                                env_lines.append(line.rstrip())

                                env_lines.append(f'EMBEDDING_PROVIDER={provider}')
                                env_lines.append(f'EMBEDDING_MODEL_NAME={model_name}')
                                if provider == "ollama":
                                    # Add or update OLLAMA_BASE_URL
                                    env_lines = [l for l in env_lines if not l.startswith('OLLAMA_BASE_URL=')]
                                    env_lines.append(f'OLLAMA_BASE_URL={ollama_url}')

                                with open(env_path, 'w') as f:
                                    f.write('\n'.join(env_lines) + '\n')

                                # Update vector search in real-time
                                update_msg = ""
                                if system:
                                    try:
                                        # Get vector search from container
                                        vector_search = None
                                        
                                        if hasattr(system, 'container'):
                                            try:
                                                vector_search = system.container.resolve('vector_search')
                                            except Exception as e:
                                                logger.debug(f"Could not resolve vector_search from container: {e}")
                                        
                                        # Fallback: direct access
                                        if not vector_search and hasattr(system, 'vector_search'):
                                            vector_search = system.vector_search

                                        # Update the embedding model
                                        if vector_search:
                                            vector_search.provider = provider
                                            vector_search.model_name = model_name

                                            if provider == "huggingface":
                                                from sentence_transformers import SentenceTransformer
                                                vector_search.model = SentenceTransformer(model_name)
                                                update_msg = "\n\n✅ Configuration applied immediately - no restart needed!\nHuggingFace model loaded and ready to use."
                                            elif provider == "ollama":
                                                vector_search.base_url = ollama_url
                                                vector_search.model = None  # Ollama doesn't need pre-loaded model
                                                # Validate Ollama connection
                                                try:
                                                    import requests
                                                    response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                                                    response.raise_for_status()
                                                    update_msg = f"\n\n✅ Configuration applied immediately - no restart needed!\nConnected to Ollama at {ollama_url}\nModel: {model_name}"
                                                except Exception as e:
                                                    update_msg = f"\n\n⚠️ Ollama connection failed: {str(e)}\nSettings saved to .env. Make sure Ollama is running: ollama serve"
                                        else:
                                            update_msg = "\n\n📝 Settings saved successfully!\nEmbedding configuration will apply on next launch."
                                    except Exception as e:
                                        logger.error(f"Error updating embedding model: {e}")
                                        update_msg = f"\n\n⚠️ Could not update runtime: {str(e)}\nSettings saved to .env - restart to apply."
                                else:
                                    update_msg = "\n\n📝 Settings saved successfully!\nEmbedding configuration will apply when you launch the application.\n\n💡 Tip: If you're already running, restart to use the new settings."

                                status = f"""✅ Embedding Settings Saved!

Provider: {provider.upper()}
Model: {model_name}
{f'Base URL: {ollama_url}' if provider == 'ollama' else ''}

Settings saved to .env file.{update_msg}"""

                                return status

                            except Exception as e:
                                return f"❌ Error saving settings: {str(e)}"

                        embedding_save_btn.click(
                            save_embedding_settings,
                            inputs=[embedding_provider, hf_models, ollama_embedding_model, ollama_base_url],
                            outputs=embedding_status
                        )

                    # Cache Settings
                    with gr.TabItem("💾 Cache Settings"):
                        gr.Markdown("#### Configure Caching Behavior")

                        with gr.Row():
                            with gr.Column():
                                cache_ttl = gr.Slider(
                                    label="Default Cache TTL (seconds)",
                                    minimum=300,
                                    maximum=86400,
                                    value=3600,
                                    step=300,
                                    info="How long to keep cached items (1 hour = 3600)"
                                )

                                max_memory_items = gr.Slider(
                                    label="Max Memory Cache Items",
                                    minimum=100,
                                    maximum=5000,
                                    value=1000,
                                    step=100,
                                    info="Maximum items in memory cache"
                                )

                                cache_save_btn = gr.Button("💾 Save Cache Settings", variant="primary")

                            with gr.Column():
                                cache_settings_status = gr.Textbox(
                                    label="Status",
                                    lines=8,
                                    interactive=False
                                )

                        def save_cache_settings(ttl, max_items):
                            """Save cache configuration."""
                            try:
                                import os
                                from pathlib import Path

                                os.environ['DEFAULT_CACHE_TTL'] = str(ttl)
                                os.environ['MAX_CACHE_ITEMS'] = str(max_items)

                                # Update .env file
                                env_path = Path.cwd() / '.env'
                                env_lines = []

                                if env_path.exists():
                                    with open(env_path, 'r') as f:
                                        for line in f:
                                            if not any(line.startswith(k) for k in ['DEFAULT_CACHE_TTL=', 'MAX_CACHE_ITEMS=']):
                                                env_lines.append(line.rstrip())

                                env_lines.extend([
                                    f'DEFAULT_CACHE_TTL={ttl}',
                                    f'MAX_CACHE_ITEMS={max_items}'
                                ])

                                with open(env_path, 'w') as f:
                                    f.write('\n'.join(env_lines) + '\n')

                                # Update cache if available
                                if cache:
                                    cache.default_ttl_seconds = ttl
                                    cache.max_memory_items = max_items

                                return f"""✅ Cache Settings Saved!

TTL: {ttl} seconds ({ttl//3600}h {(ttl%3600)//60}m)
Max Memory Items: {max_items}

Settings applied immediately."""

                            except Exception as e:
                                return f"❌ Error saving settings: {str(e)}"

                        cache_save_btn.click(
                            save_cache_settings,
                            inputs=[cache_ttl, max_memory_items],
                            outputs=cache_settings_status
                        )

                    # Database Connection Settings
                    with gr.TabItem("🗄️ Database Connection"):
                        gr.Markdown("#### Configure Database Connection")

                        with gr.Row():
                            with gr.Column():
                                # Database type selection
                                db_type = gr.Radio(
                                    label="Database Type",
                                    choices=["Neo4j", "In-Memory Graph"],
                                    value="Neo4j",
                                    info="Choose between Neo4j database or in-memory graph"
                                )

                                gr.Markdown("##### Neo4j Connection Settings")
                                gr.Markdown("""
                                **Supports both Local and Cloud Neo4j:**
                                - **Local**: `neo4j://127.0.0.1:7687` or `bolt://localhost:7687`
                                - **Cloud (Aura)**: `neo4j+s://xxxxx.databases.neo4j.io`
                                """)

                                neo4j_uri = gr.Textbox(
                                    label="Neo4j URI",
                                    value=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
                                    placeholder="neo4j://127.0.0.1:7687 or neo4j+s://xxxxx.databases.neo4j.io",
                                    info="Neo4j database URI (supports local and cloud)"
                                )

                                neo4j_username = gr.Textbox(
                                    label="Username",
                                    value=os.getenv("NEO4J_USER", "neo4j"),
                                    placeholder="neo4j",
                                    info="Neo4j username"
                                )

                                neo4j_password = gr.Textbox(
                                    label="Password",
                                    type="password",
                                    value=os.getenv("NEO4J_PASSWORD", ""),
                                    placeholder="Enter your Neo4j password",
                                    info="Neo4j password"
                                )

                                with gr.Row():
                                    test_neo4j_btn = gr.Button("🔌 Test Connection", variant="secondary")
                                    save_db_btn = gr.Button("💾 Save Database Settings", variant="primary")

                            with gr.Column():
                                db_connection_status = gr.Textbox(
                                    label="Connection Status",
                                    value="Not tested",
                                    interactive=False,
                                    lines=4
                                )

                                db_settings_status = gr.Textbox(
                                    label="Settings Status",
                                    lines=10,
                                    interactive=False
                                )

                                db_current_config = gr.Textbox(
                                    label="Current Configuration",
                                    lines=6,
                                    interactive=False,
                                    value="Database type: Not configured"
                                )

                        def test_neo4j_conn(uri, username, password):
                            """Test Neo4j connection."""
                            if not uri or not username or not password:
                                return "✗ Not connected\nPlease provide URI, username, and password"

                            result = test_neo4j_connection(uri, username, password)

                            if result["success"]:
                                return f"✓ Connected\n{result['message']}\n\nURI: {uri}\nUsername: {username}"
                            else:
                                return f"✗ Not connected\n{result['message']}"

                        def save_database_settings(db_type_val, uri, username, password):
                            """Save database configuration."""
                            try:
                                import os
                                from pathlib import Path

                                # Update environment variables
                                os.environ['GRAPH_DB_TYPE'] = db_type_val.lower().replace(" ", "_")

                                if db_type_val == "Neo4j":
                                    os.environ['NEO4J_URI'] = uri
                                    os.environ['NEO4J_USERNAME'] = username
                                    os.environ['NEO4J_PASSWORD'] = password

                                # Update .env file
                                env_path = Path.cwd() / '.env'
                                env_lines = []

                                if env_path.exists():
                                    with open(env_path, 'r') as f:
                                        for line in f:
                                            if not any(line.startswith(k) for k in [
                                                'GRAPH_DB_TYPE=', 'NEO4J_URI=',
                                                'NEO4J_USERNAME=', 'NEO4J_PASSWORD='
                                            ]):
                                                env_lines.append(line.rstrip())

                                # Add updated values
                                env_lines.append(f'GRAPH_DB_TYPE={db_type_val.lower().replace(" ", "_")}')

                                if db_type_val == "Neo4j":
                                    env_lines.extend([
                                        f'NEO4J_URI={uri}',
                                        f'NEO4J_USERNAME={username}',
                                        f'NEO4J_PASSWORD={password}'
                                    ])

                                with open(env_path, 'w') as f:
                                    f.write('\n'.join(env_lines) + '\n')

                                # Update runtime configuration
                                update_msg = ""
                                if db_type_val == "Neo4j":
                                    # Test connection before confirming
                                    test_result = test_neo4j_connection(uri, username, password)

                                    if not test_result["success"]:
                                        status = f"""⚠️ Settings Saved but Connection Failed

Database Type: {db_type_val}
URI: {uri}
Username: {username}

Settings saved to .env file, but connection test failed:
{test_result['message']}

Please verify your Neo4j instance is running and credentials are correct.
Restart may be required for changes to take effect."""
                                        return status
                                    
                                    # Try to update graph manager in real-time
                                    update_msg = ""
                                    if system:
                                        try:
                                            graph_manager = None

                                            # Try to get from container
                                            if hasattr(system, 'container'):
                                                try:
                                                    graph_manager = system.container.resolve('graph_manager')
                                                except Exception as e:
                                                    logger.debug(f"Could not resolve from container: {e}")

                                            # Fallback: direct access
                                            if not graph_manager and hasattr(system, 'graph_manager'):
                                                graph_manager = system.graph_manager

                                            if graph_manager:
                                                # Close existing connection
                                                if hasattr(graph_manager, 'close'):
                                                    try:
                                                        graph_manager.close()
                                                    except Exception:
                                                        pass

                                                # Update connection parameters
                                                if hasattr(graph_manager, 'driver'):
                                                    from neo4j import GraphDatabase
                                                    graph_manager.uri = uri
                                                    graph_manager.user = username
                                                    graph_manager.password = password
                                                    graph_manager.driver = GraphDatabase.driver(uri, auth=(username, password))

                                                    # Test new connection
                                                    try:
                                                        with graph_manager.driver.session() as session:
                                                            session.run("RETURN 1")
                                                        update_msg = "\n\n✅ Configuration applied immediately - no restart needed!\nDatabase connection updated and verified."
                                                    except Exception as e:
                                                        update_msg = f"\n\n⚠️ Connection updated but verification failed: {str(e)}\nSettings saved to .env - restart to apply."
                                                else:
                                                    update_msg = "\n\n⚠️ Could not update runtime connection.\nSettings saved to .env - they will apply on next launch."
                                            else:
                                                update_msg = "\n\n📝 Settings saved successfully!\nDatabase connection will be established on next launch."
                                        except Exception as e:
                                            logger.debug(f"Could not update graph manager: {e}")
                                            update_msg = "\n\n📝 Settings saved to .env successfully!\nRestart the application to use the new database connection."
                                    else:
                                        update_msg = "\n\n📝 Settings saved successfully!\nDatabase connection will be established when you launch the application.\n\n💡 Tip: If you're already running, restart to use the new settings."
                                    
                                    status = f"""✅ Database Settings Saved Successfully!

Database Type: {db_type_val}
URI: {uri}
Username: {username}
Connection: ✓ Verified

Settings saved to .env file.{update_msg}"""
                                else:
                                    status = f"""✅ Database Settings Saved Successfully!

Database Type: {db_type_val} (In-Memory)

Settings saved to .env file.
In-memory graph will be used instead of Neo4j.
Restart may be required for changes to take effect."""

                                return status

                            except Exception as e:
                                return f"❌ Error saving settings: {str(e)}"

                        def update_db_config_display(db_type_val, uri, username):
                            """Update current configuration display."""
                            if db_type_val == "Neo4j":
                                return f"""Current Configuration:

Database Type: {db_type_val}
URI: {uri or 'Not set'}
Username: {username or 'Not set'}
Password: {'Set' if username else 'Not set'}"""
                            else:
                                return f"""Current Configuration:

Database Type: {db_type_val}
Status: Using in-memory graph
No external database required"""

                        test_neo4j_btn.click(
                            test_neo4j_conn,
                            inputs=[neo4j_uri, neo4j_username, neo4j_password],
                            outputs=[db_connection_status]
                        )

                        save_db_btn.click(
                            save_database_settings,
                            inputs=[db_type, neo4j_uri, neo4j_username, neo4j_password],
                            outputs=[db_settings_status]
                        )

                        # Update config display when settings change
                        for component in [db_type, neo4j_uri, neo4j_username, neo4j_password]:
                            component.change(
                                update_db_config_display,
                                inputs=[db_type, neo4j_uri, neo4j_username],
                                outputs=[db_current_config]
                            )

    return app


def launch_unified_ui(system=None, port: int = 7860, share: bool = False):
    """
    Launch the unified UI.

    Args:
        system: System instance (optional)
        port: Port number (default: 7860)
        share: Whether to create public link
    """
    app = create_unified_ui(system)

    # Try multiple ports if the specified one is busy
    max_attempts = 10
    for i in range(max_attempts):
        try:
            current_port = port + i
            logger.info(f"Attempting to launch on port {current_port}...")
            app.launch(
                server_name="127.0.0.1",
                server_port=current_port,
                share=share,
                show_error=True,
                quiet=False
            )
            break
        except OSError as e:
            if i < max_attempts - 1:
                logger.warning(f"Port {current_port} in use, trying next port...")
                continue
            else:
                logger.error(f"Could not find available port after {max_attempts} attempts")
                raise


if __name__ == '__main__':
    launch_unified_ui()
