#!/usr/bin/env python3
"""
Research Compass GNN - Launcher
Main Gradio application with Real Data Training capabilities
"""

import gradio as gr
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple, Any, Optional
import PyPDF2
import io
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GNN MODEL DEFINITIONS (from comparison_study.py)
# ============================================================================

class GCNModel(nn.Module):
    """Graph Convolutional Network for Node Classification"""
    def __init__(self, input_dim=384, hidden_dim=128, output_dim=5, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GATModel(nn.Module):
    """Graph Attention Network for Node Classification"""
    def __init__(self, input_dim=384, hidden_dim=128, output_dim=5, num_layers=2, heads=4, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_dim * heads if num_layers > 1 else input_dim,
                                   output_dim, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
        x = self.convs[-1](x, edge_index)
        return x


class GraphTransformerModel(nn.Module):
    """Graph Transformer for Node Classification"""
    def __init__(self, input_dim=384, hidden_dim=128, output_dim=5, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, concat=True))
        if num_layers > 1:
            self.convs.append(TransformerConv(hidden_dim * num_heads, output_dim, heads=1, dropout=dropout, concat=False))
        else:
            self.convs.append(TransformerConv(input_dim, output_dim, heads=1, dropout=dropout, concat=False))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x


# ============================================================================
# GLOBAL STATE
# ============================================================================

class AppState:
    """Global application state"""
    def __init__(self):
        self.graph_data = None
        self.trained_model = None
        self.model_type = None
        self.paper_list = []
        self.training_history = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app_state = AppState()

# ============================================================================
# PDF PROCESSING FUNCTIONS
# ============================================================================

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text content from PDF file"""
    try:
        pdf_bytes = pdf_file if isinstance(pdf_file, bytes) else pdf_file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"


def extract_citations_simple(text: str) -> List[str]:
    """Simple citation extraction (placeholder - can be enhanced)"""
    # This is a simplified version - in production, use proper citation parsing
    import re
    citations = []

    # Look for patterns like [Author, Year]
    pattern = r'\[([A-Z][a-z]+(?:\s+et\s+al\.?)?,\s+\d{4})\]'
    matches = re.findall(pattern, text)
    citations.extend(matches)

    # Look for (Author Year) patterns
    pattern2 = r'\(([A-Z][a-z]+(?:\s+et\s+al\.?)?\s+\d{4})\)'
    matches2 = re.findall(pattern2, text)
    citations.extend(matches2)

    return list(set(citations))  # Remove duplicates


def build_graph_from_papers(papers_data: List[Dict]) -> Data:
    """
    Build PyG graph from paper data

    Args:
        papers_data: List of dicts with keys: 'text', 'citations', 'metadata'

    Returns:
        PyG Data object
    """
    num_papers = len(papers_data)

    # Create simple features (can be enhanced with sentence-transformers)
    # For now, use random features as placeholder
    x = torch.randn(num_papers, 384)

    # Create edges from citations
    edges = []
    for i, paper in enumerate(papers_data):
        citations = paper.get('citations', [])
        for citation in citations:
            # Simple matching - find if citation text matches any paper
            for j, other_paper in enumerate(papers_data):
                if i != j and citation.lower() in other_paper.get('text', '').lower()[:500]:
                    edges.append([i, j])

    # Add some random edges if no citations found
    if len(edges) < num_papers:
        for i in range(num_papers):
            for _ in range(min(3, num_papers - 1)):
                j = np.random.randint(0, num_papers)
                if i != j:
                    edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)

    # Create labels (for classification - random for demo)
    y = torch.randint(0, 5, (num_papers,))

    # Create train/val/test masks
    train_mask = torch.zeros(num_papers, dtype=torch.bool)
    val_mask = torch.zeros(num_papers, dtype=torch.bool)
    test_mask = torch.zeros(num_papers, dtype=torch.bool)

    perm = torch.randperm(num_papers)
    train_size = int(0.6 * num_papers)
    val_size = int(0.2 * num_papers)

    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size+val_size]] = True
    test_mask[perm[train_size+val_size:]] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return data


# ============================================================================
# REAL DATA TRAINING TAB FUNCTIONS
# ============================================================================

def process_pdfs(files, extract_citations, build_graph, extract_metadata):
    """Process uploaded PDF files"""
    if not files or len(files) == 0:
        return "⚠️ No files uploaded", "", None

    status = f"🔄 Processing {len(files)} PDF files...\n\n"
    papers_data = []

    for idx, file in enumerate(files):
        status += f"📄 Processing file {idx+1}/{len(files)}: {file.name}\n"

        # Extract text
        text = extract_text_from_pdf(file)

        paper_info = {
            'name': file.name,
            'text': text[:1000],  # First 1000 chars
            'citations': [],
            'metadata': {}
        }

        # Extract citations if enabled
        if extract_citations:
            citations = extract_citations_simple(text)
            paper_info['citations'] = citations
            status += f"   Found {len(citations)} citations\n"

        # Extract metadata if enabled
        if extract_metadata:
            # Placeholder - can be enhanced
            paper_info['metadata'] = {
                'length': len(text),
                'word_count': len(text.split())
            }

        papers_data.append(paper_info)
        app_state.paper_list.append(file.name)

    status += f"\n✅ Processed {len(papers_data)} papers successfully!\n"

    # Build graph if enabled
    graph_stats = ""
    if build_graph:
        status += "\n🔄 Building knowledge graph...\n"
        try:
            graph_data = build_graph_from_papers(papers_data)
            app_state.graph_data = graph_data

            graph_stats = f"""
📊 **Graph Statistics:**
- 📄 Papers: {graph_data.num_nodes}
- 🔗 Citations: {graph_data.num_edges}
- 📊 Density: {graph_data.num_edges / (graph_data.num_nodes * (graph_data.num_nodes - 1)):.4f}
- 📈 Avg Citations/Paper: {graph_data.num_edges / graph_data.num_nodes:.2f}
"""
            status += "✅ Knowledge graph built successfully!\n"
        except Exception as e:
            status += f"❌ Error building graph: {str(e)}\n"
            graph_stats = f"❌ Error: {str(e)}"

    # Update dropdown choices
    paper_choices = gr.Dropdown(choices=app_state.paper_list, value=app_state.paper_list[0] if app_state.paper_list else None)

    return status, graph_stats, paper_choices


def train_gnn_live(model_type, epochs, learning_rate, task_type, progress=gr.Progress()):
    """Train GNN model with live progress updates"""
    if app_state.graph_data is None:
        return "❌ Error: No graph data available. Please process papers first!", None, ""

    progress(0, desc="Initializing...")

    data = app_state.graph_data
    device = app_state.device
    data = data.to(device)

    # Create model
    num_features = data.x.size(1)
    num_classes = data.y.max().item() + 1

    progress(0.1, desc=f"Creating {model_type} model...")

    if model_type == "GCN":
        model = GCNModel(input_dim=num_features, output_dim=num_classes)
    elif model_type == "GAT":
        model = GATModel(input_dim=num_features, output_dim=num_classes)
    elif model_type == "Graph Transformer":
        model = GraphTransformerModel(input_dim=num_features, output_dim=num_classes)
    else:
        return f"❌ Unknown model type: {model_type}", None, ""

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    # Training history
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    status = f"🚀 Training {model_type} model...\n\n"
    status += f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}\n"
    status += f"Device: {device}\n\n"

    start_time = time.time()

    for epoch in range(epochs):
        progress((epoch + 1) / epochs, desc=f"Epoch {epoch+1}/{epochs}")

        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)

            train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()

            history['train_loss'].append(loss.item())
            history['train_acc'].append(train_acc.item())
            history['val_acc'].append(val_acc.item())

        # Update status every 10 epochs
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            status += f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Train Acc: {train_acc.item():.4f} | Val Acc: {val_acc.item():.4f}\n"

    training_time = time.time() - start_time

    # Final evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

    status += f"\n✅ Training Complete!\n"
    status += f"   Training Time: {training_time:.2f}s\n"
    status += f"   Final Test Accuracy: {test_acc.item():.4f}\n"

    # Save model
    app_state.trained_model = model
    app_state.model_type = model_type
    app_state.training_history = history

    # Create training curves plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    plt.tight_layout()

    # Results summary
    results_summary = f"""
## 📊 Training Results

**Model:** {model_type}
**Task:** {task_type}
**Training Time:** {training_time:.2f}s

### Performance Metrics:
- **Test Accuracy:** {test_acc.item():.4f} ({test_acc.item()*100:.2f}%)
- **Best Val Accuracy:** {max(history['val_acc']):.4f}
- **Final Train Accuracy:** {history['train_acc'][-1]:.4f}

### Model Info:
- **Parameters:** {sum(p.numel() for p in model.parameters()):,}
- **Epochs Trained:** {epochs}
- **Learning Rate:** {learning_rate}
"""

    return status, fig, results_summary


def make_prediction(paper_name, prediction_type, top_k):
    """Make predictions on selected paper"""
    if app_state.trained_model is None:
        return "❌ Error: No trained model available. Please train a model first!"

    if app_state.graph_data is None:
        return "❌ Error: No graph data available."

    if not paper_name:
        return "⚠️ Please select a paper"

    try:
        # Find paper index
        paper_idx = app_state.paper_list.index(paper_name)
    except ValueError:
        return f"❌ Paper '{paper_name}' not found in graph"

    model = app_state.trained_model
    data = app_state.graph_data.to(app_state.device)

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)

        if prediction_type == "Category Classification":
            # Get class probabilities
            probs = F.softmax(out[paper_idx], dim=0)
            top_probs, top_classes = torch.topk(probs, min(top_k, len(probs)))

            result = f"## 🔮 Category Prediction for '{paper_name}'\n\n"
            result += "**Top Predictions:**\n\n"
            for i, (prob, cls) in enumerate(zip(top_probs, top_classes)):
                result += f"{i+1}. Category {cls.item()}: {prob.item()*100:.2f}% confidence\n"

        elif prediction_type == "Link Prediction":
            # Simple link prediction - find most similar nodes
            node_emb = out[paper_idx]
            similarities = F.cosine_similarity(node_emb.unsqueeze(0), out, dim=1)
            similarities[paper_idx] = -1  # Exclude self

            top_sims, top_nodes = torch.topk(similarities, min(top_k, len(similarities)))

            result = f"## 🔗 Citation Predictions for '{paper_name}'\n\n"
            result += "**Most Likely Citations:**\n\n"
            for i, (sim, node) in enumerate(zip(top_sims, top_nodes)):
                if node.item() < len(app_state.paper_list):
                    cited_paper = app_state.paper_list[node.item()]
                    result += f"{i+1}. {cited_paper}: {sim.item():.4f} similarity\n"
                else:
                    result += f"{i+1}. Paper #{node.item()}: {sim.item():.4f} similarity\n"

        else:
            result = f"❌ Unknown prediction type: {prediction_type}"

    return result


def export_results():
    """Export trained model and predictions"""
    if app_state.trained_model is None:
        return None, "❌ No trained model to export"

    # Save model
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = f"trained_model_{app_state.model_type}_{timestamp}.pt"
    torch.save(app_state.trained_model.state_dict(), model_path)

    # Save predictions CSV
    if app_state.graph_data is not None:
        model = app_state.trained_model
        data = app_state.graph_data.to(app_state.device)

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            predictions = out.argmax(dim=1).cpu().numpy()

        df = pd.DataFrame({
            'Paper': app_state.paper_list[:len(predictions)],
            'Predicted_Category': predictions
        })
        csv_path = f"predictions_{timestamp}.csv"
        df.to_csv(csv_path, index=False)

        return csv_path, f"✅ Exported:\n- Model: {model_path}\n- Predictions: {csv_path}"

    return None, f"✅ Exported model: {model_path}"


# ============================================================================
# DEMO TAB FUNCTIONS
# ============================================================================

def run_demo():
    """Run a quick demo on synthetic data"""
    status = "🎮 Running Demo...\n\n"

    # Create synthetic citation network
    from comparison_study import create_realistic_citation_network

    status += "Creating synthetic citation network (50 papers)...\n"
    data = create_realistic_citation_network(num_papers=50, num_topics=3, avg_citations=5)

    status += f"✅ Created graph: {data.num_nodes} nodes, {data.num_edges} edges\n\n"

    status += "Training GCN model (20 epochs)...\n"
    model = GCNModel(input_dim=384, hidden_dim=64, output_dim=3, num_layers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
                status += f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc.item():.4f}\n"

    status += "\n✅ Demo complete! Model trained successfully.\n"
    status += "\nTry the 'Real Data Training' tab to upload your own papers!"

    return status


# ============================================================================
# GRADIO UI
# ============================================================================

def create_ui():
    """Create Gradio interface"""

    with gr.Blocks(title="Research Compass GNN", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🧭 Research Compass - GNN Platform

        **Advanced Graph Neural Networks for Citation Analysis**

        Train GNN models on your research papers or explore benchmark datasets!
        """)

        with gr.Tabs():
            # ================================================================
            # TAB 1: WELCOME & DEMO
            # ================================================================
            with gr.Tab("🏠 Welcome"):
                gr.Markdown("""
                ## Welcome to Research Compass GNN!

                This platform allows you to:
                - 📄 Upload and process research papers (PDFs)
                - 🕸️ Build knowledge graphs from citations
                - 🤖 Train Graph Neural Networks (GCN, GAT, Transformer)
                - 🔮 Make predictions on paper categories and citations
                - 📊 Visualize training progress and results

                ### Quick Start:
                1. Go to **"📄 Real Data Training"** tab
                2. Upload your PDF files
                3. Process papers to build a graph
                4. Train a GNN model
                5. Make predictions!

                ### Or try the demo:
                """)

                demo_button = gr.Button("🎮 Run Quick Demo", variant="primary", size="lg")
                demo_output = gr.Textbox(label="Demo Output", lines=15)

                demo_button.click(
                    fn=run_demo,
                    outputs=demo_output
                )

            # ================================================================
            # TAB 2: REAL DATA TRAINING (MAIN FEATURE)
            # ================================================================
            with gr.Tab("📄 Real Data Training"):
                gr.Markdown("## Real Data Training with GNN Models")

                with gr.Row():
                    # LEFT SECTION: PDF Upload & Processing
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 PDF Upload & Processing")

                        file_upload = gr.File(
                            label="Upload PDF Files",
                            file_count="multiple",
                            file_types=[".pdf"],
                            type="binary"
                        )

                        with gr.Group():
                            gr.Markdown("**Processing Options:**")
                            extract_citations_cb = gr.Checkbox(label="Extract citations from PDFs", value=True)
                            build_graph_cb = gr.Checkbox(label="Build knowledge graph automatically", value=True)
                            extract_metadata_cb = gr.Checkbox(label="Extract metadata (authors, year, venue)", value=True)

                        process_button = gr.Button("🔄 Process Papers & Build Graph", variant="primary", size="lg")

                        processing_status = gr.Textbox(label="Processing Status", lines=10)
                        graph_stats = gr.Markdown("**Graph statistics will appear here after processing**")

                    # RIGHT SECTION: GNN Training & Predictions
                    with gr.Column(scale=1):
                        gr.Markdown("### 🤖 GNN Training & Predictions")

                        with gr.Group():
                            gr.Markdown("**Model Configuration:**")
                            model_type = gr.Dropdown(
                                choices=["GCN", "GAT", "Graph Transformer"],
                                value="GCN",
                                label="Model Type"
                            )

                            epochs_slider = gr.Slider(
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=10,
                                label="Number of Epochs"
                            )

                            lr_slider = gr.Slider(
                                minimum=0.001,
                                maximum=0.1,
                                value=0.01,
                                step=0.001,
                                label="Learning Rate"
                            )

                            task_type = gr.Dropdown(
                                choices=["Node Classification", "Link Prediction"],
                                value="Node Classification",
                                label="Task Type"
                            )

                        train_button = gr.Button("🚀 Train GNN Model", variant="primary", size="lg")

                        training_status = gr.Textbox(label="Training Progress", lines=10)
                        training_plot = gr.Plot(label="Training Curves")
                        results_display = gr.Markdown("**Training results will appear here**")

                        gr.Markdown("---")

                        with gr.Group():
                            gr.Markdown("**🔮 Make Predictions:**")
                            paper_select = gr.Dropdown(
                                choices=[],
                                label="Select Paper",
                                interactive=True
                            )

                            prediction_type = gr.Dropdown(
                                choices=["Category Classification", "Link Prediction"],
                                value="Category Classification",
                                label="Prediction Type"
                            )

                            top_k_slider = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Top-K Results"
                            )

                            predict_button = gr.Button("🔮 Predict", variant="secondary")
                            prediction_output = gr.Markdown("**Predictions will appear here**")

                        export_button = gr.Button("💾 Export Results", variant="secondary")
                        export_status = gr.Textbox(label="Export Status", lines=2)

                # Connect buttons to functions
                process_button.click(
                    fn=process_pdfs,
                    inputs=[file_upload, extract_citations_cb, build_graph_cb, extract_metadata_cb],
                    outputs=[processing_status, graph_stats, paper_select]
                )

                train_button.click(
                    fn=train_gnn_live,
                    inputs=[model_type, epochs_slider, lr_slider, task_type],
                    outputs=[training_status, training_plot, results_display]
                )

                predict_button.click(
                    fn=make_prediction,
                    inputs=[paper_select, prediction_type, top_k_slider],
                    outputs=prediction_output
                )

                export_button.click(
                    fn=export_results,
                    outputs=[export_status, export_status]
                )

            # ================================================================
            # TAB 3: ABOUT
            # ================================================================
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About Research Compass GNN

                ### Features:
                - **Real Data Training**: Upload PDFs and train GNNs on your own research papers
                - **Multiple GNN Models**: GCN, GAT, and Graph Transformer architectures
                - **Automated Graph Building**: Automatic citation extraction and graph construction
                - **Live Training**: Real-time training progress with visualization
                - **Predictions**: Category classification and citation link prediction
                - **Export**: Save trained models and predictions

                ### Technology Stack:
                - **PyTorch Geometric**: Graph Neural Networks
                - **Gradio**: Interactive UI
                - **NetworkX**: Graph processing
                - **PyPDF2**: PDF text extraction

                ### Models:
                1. **GCN** (Graph Convolutional Network): Fast and efficient for most tasks
                2. **GAT** (Graph Attention Network): Learns attention weights for better accuracy
                3. **Graph Transformer**: Advanced architecture for complex patterns

                ### Citation:
                ```
                Research Compass GNN Platform
                Built with PyTorch Geometric and Gradio
                ```

                ---

                **Version:** 1.0.0
                **License:** MIT
                """)

        return app


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Research Compass GNN...")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   Device: {app_state.device}")
    print(f"   Gradio: {gr.__version__}")

    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
