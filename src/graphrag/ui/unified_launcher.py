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
                **Instructions:**
                1. Upload PDF, DOCX, TXT, or MD files using the file uploader
                2. OR paste web URLs (one per line) - supports arXiv, research papers, etc.
                3. Check "Extract metadata" to get authors, titles, citations
                4. Check "Build knowledge graph" to create nodes and relationships
                5. Click "Process All" to start processing
                
                **Examples:**
                - arXiv URL: `https://arxiv.org/abs/1706.03762` (Attention Is All You Need)
                - arXiv URL: `https://arxiv.org/abs/1810.04805` (BERT paper)
                - Direct PDF: `https://example.com/research-paper.pdf`
                
                **Note:** Processing may take 30-60 seconds per document depending on size.
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

                with gr.Tabs():
                    # Sub-tab: Graph Statistics
                    with gr.Tab("📊 Graph Statistics"):
                        gr.Markdown("View comprehensive statistics about your knowledge graph")

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
                        gr.Markdown("Explore your knowledge graph interactively")

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
                        gr.Markdown("Train Graph Neural Network models on your knowledge graph")

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
                        gr.Markdown("Get predictions from trained GNN models")

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
                        gr.Markdown("Export your knowledge graph for external analysis")

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
                **Instructions:**
                1. Type your research question in the text box
                2. Enable "Use GNN reasoning" for graph-aware analysis
                3. Enable "Use cache" for faster repeated queries (10-100x speedup)
                4. Enable "Stream response" to see answers word-by-word in real-time
                5. Click "Ask" to get your answer
                
                **Example Questions:**
                - "What are the main innovations in transformer architecture?"
                - "How does BERT differ from GPT models?"
                - "What are recent advances in Graph Neural Networks?"
                - "Summarize the key contributions of the Attention Is All You Need paper"
                - "What are the applications of Graph Attention Networks?"
                
                **Tips:**
                - GNN reasoning provides more accurate answers by using graph structure
                - Cache saves time for similar questions
                - More specific questions get better answers
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

            # ========== TAB 3: Temporal Analysis ==========
            with gr.TabItem("📊 Temporal Analysis"):
                gr.Markdown("### Analyze how research evolves over time")
                gr.Markdown("""
                **Instructions:**
                Use these tools to track research trends, citation patterns, and emerging topics over time.
                
                **Available Analysis Types:**
                1. **Topic Evolution** - Track how a research topic grows over time
                2. **Citation Velocity** - Measure how quickly a paper accumulates citations
                3. **H-Index Timeline** - Track an author's research impact over years
                4. **Emerging Topics** - Discover new and rapidly growing research areas
                
                **Examples:**
                - Topic Evolution: "Graph Neural Networks" with yearly window
                - Citation Velocity: Enter a paper title from your uploaded documents
                - H-Index Timeline: Enter author name (e.g., "Yoshua Bengio")
                - Emerging Topics: Set year to 2020 and threshold to 0.5
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

            # ========== TAB 4: Personalized Recommendations ==========
            with gr.TabItem("💡 Recommendations"):
                gr.Markdown("### Get personalized paper recommendations")
                gr.Markdown("""
                **Instructions:**
                1. Create your user profile with research interests and reading history
                2. List papers you've read (use titles from uploaded documents)
                3. List papers you liked (helps improve recommendations)
                4. Specify your research interests (e.g., "machine learning, GNN, transformers")
                5. Click "Create/Update Profile" to save your profile
                6. Adjust diversity slider (0 = similar papers, 1 = exploratory/diverse)
                7. Click "Get Recommendations" to get personalized suggestions
                
                **Example Profile:**
                - User ID: "researcher123"
                - Papers Read: "Attention Is All You Need, BERT, GPT-3"
                - Papers Liked: "Attention Is All You Need"
                - Interests: "transformers, natural language processing, deep learning"
                - Diversity: 0.3 (balanced between similar and exploratory)
                
                **Tips:**
                - Higher diversity shows more varied papers outside your usual topics
                - Lower diversity focuses on papers very similar to your interests
                - Update your profile as you read more papers for better recommendations
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

            # ========== TAB 5: Citation Explorer ==========
            with gr.TabItem("🕸️ Citation Explorer"):
                gr.Markdown("### Explore citation networks interactively")

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

            # ========== TAB 6: Discovery ==========
            with gr.TabItem("🔬 Discovery"):
                gr.Markdown("### Discover unexpected connections")

                with gr.Tab("Similar Papers"):
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

            # ========== TAB 7: Cache Management ==========
            with gr.TabItem("💾 Cache Management"):
                gr.Markdown("### View and manage cache statistics")

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

                with gr.Tabs():
                    # LLM Settings (Enhanced)
                    with gr.TabItem("🤖 LLM Model"):
                        gr.Markdown("#### Select Language Model")

                        with gr.Row():
                            with gr.Column():
                                llm_provider = gr.Dropdown(
                                    label="LLM Provider",
                                    choices=["ollama", "lmstudio", "openrouter", "openai"],
                                    value="ollama",
                                    info="Select your LLM provider"
                                )

                                # API Key input (hidden for local providers)
                                llm_api_key = gr.Textbox(
                                    label="API Key (for OpenRouter/OpenAI)",
                                    type="password",
                                    placeholder="Enter API key if using OpenRouter or OpenAI",
                                    visible=False
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
                                        
                                        # Update the LLM manager configuration dynamically
                                        if llm_mgr and hasattr(llm_mgr, 'update_config'):
                                            # Determine base_url based on provider
                                            if provider == "ollama":
                                                base_url = "http://localhost:11434"
                                            elif provider == "lmstudio":
                                                base_url = "http://localhost:1234"
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
                                            update_msg = "\n✅ Configuration applied immediately - no restart needed!"
                                        else:
                                            update_msg = "\n⚠️ LLM manager not found. Restart recommended for changes to take effect."
                                    except Exception as e:
                                        logger.error(f"Error updating LLM config: {e}")
                                        update_msg = f"\n⚠️ Could not update runtime config: {str(e)}\nRestart recommended."
                                else:
                                    update_msg = "\n⚠️ System not initialized. Restart required to apply changes."

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
                                        
                                        # Update the embedding model
                                        if vector_search:
                                            vector_search.provider = provider
                                            vector_search.model_name = model_name
                                            
                                            if provider == "huggingface":
                                                from sentence_transformers import SentenceTransformer
                                                vector_search.model = SentenceTransformer(model_name)
                                                update_msg = "\n✅ Configuration applied immediately - no restart needed!\nHuggingFace model loaded and ready to use."
                                            elif provider == "ollama":
                                                vector_search.base_url = ollama_url
                                                vector_search.model = None  # Ollama doesn't need pre-loaded model
                                                # Validate Ollama connection
                                                try:
                                                    import requests
                                                    response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                                                    response.raise_for_status()
                                                    update_msg = f"\n✅ Configuration applied immediately - no restart needed!\nConnected to Ollama at {ollama_url}\nModel: {model_name}"
                                                except Exception as e:
                                                    update_msg = f"\n⚠️ Ollama connection failed: {str(e)}\nMake sure Ollama is running: ollama serve"
                                        else:
                                            update_msg = "\n⚠️ Vector search not found. Restart recommended for changes to take effect."
                                    except Exception as e:
                                        logger.error(f"Error updating embedding model: {e}")
                                        update_msg = f"\n⚠️ Could not update runtime: {str(e)}\nRestart recommended."
                                else:
                                    update_msg = "\n⚠️ System not initialized. Restart required to apply changes."

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

                                neo4j_uri = gr.Textbox(
                                    label="Neo4j URI",
                                    value="neo4j://127.0.0.1:7687",
                                    placeholder="neo4j://127.0.0.1:7687",
                                    info="Neo4j database URI"
                                )

                                neo4j_username = gr.Textbox(
                                    label="Username",
                                    value="neo4j",
                                    placeholder="neo4j",
                                    info="Neo4j username"
                                )

                                neo4j_password = gr.Textbox(
                                    label="Password",
                                    type="password",
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
                                    if system and hasattr(system, 'container'):
                                        try:
                                            graph_manager = system.container.resolve('graph_manager')
                                            
                                            # Close existing connection
                                            if hasattr(graph_manager, 'close'):
                                                try:
                                                    graph_manager.close()
                                                except:
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
                                                    update_msg = "\n✅ Configuration applied immediately - no restart needed!\nDatabase connection updated and verified."
                                                except Exception as e:
                                                    update_msg = f"\n⚠️ Connection updated but verification failed: {str(e)}"
                                            else:
                                                update_msg = "\n⚠️ Could not update runtime connection. Restart recommended."
                                        except Exception as e:
                                            logger.debug(f"Could not update graph manager: {e}")
                                            update_msg = "\n⚠️ Restart recommended for changes to take full effect."
                                    else:
                                        update_msg = "\n⚠️ System not initialized. Restart required to apply changes."
                                    
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
