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
import plotly.graph_objects as go
import networkx as nx
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple, Any, Optional
import PyPDF2
import io
import warnings
warnings.filterwarnings('ignore')

# Import enhanced analysis tools
from evaluation.metrics import NodeClassificationMetrics, LinkPredictionMetrics
from evaluation.visualizations import MetricsVisualizer
from visualization.attention_viz import AttentionVisualizer, analyze_attention_patterns
from analysis.temporal_analysis import TemporalAnalyzer

# ============================================================================
# GNN MODEL DEFINITIONS (Using enhanced models from models/ directory)
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
# INTERACTIVE GRAPH VISUALIZATION FUNCTIONS
# ============================================================================

def create_interactive_graph(graph_data, predictions=None, attention_weights=None,
                            show_predictions=False, paper_titles=None):
    """
    Create interactive Plotly graph visualization with real-time updates

    Args:
        graph_data: PyTorch Geometric Data object
        predictions: Model predictions (optional, for coloring nodes)
        attention_weights: GAT attention weights (optional, for edge thickness)
        show_predictions: If True, color nodes by predictions
        paper_titles: List of paper titles for hover text

    Returns:
        Plotly Figure object
    """
    try:
        # Convert to NetworkX
        G = nx.DiGraph()

        num_nodes = graph_data.x.shape[0]
        edge_index = graph_data.edge_index.cpu().numpy()

        # Add nodes
        for i in range(num_nodes):
            G.add_node(i)

        # Add edges
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            G.add_edge(int(src), int(dst))

        # Layout - use spring layout for better visualization
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Create edge traces
        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgba(150, 150, 150, 0.4)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )

        # Create node traces
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        node_sizes = []

        category_names = ['NLP', 'Computer Vision', 'GNN', 'RL', 'Theory']
        category_colors = {
            0: '#FF6B6B',  # NLP - Red
            1: '#4ECDC4',  # CV - Teal
            2: '#45B7D1',  # GNN - Blue
            3: '#FFA07A',  # RL - Orange
            4: '#98D8C8'   # Theory - Green
        }

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Node size based on degree (number of citations)
            degree = G.degree(node)
            node_sizes.append(20 + degree * 5)

            # Get paper title
            paper_title = paper_titles[node] if paper_titles and node < len(paper_titles) else f"Paper {node}"

            # Node color and text
            if show_predictions and predictions is not None:
                # Color by prediction
                pred = predictions[node].item()
                node_colors.append(category_colors.get(pred, '#CCCCCC'))

                # Check if correct
                true_label = graph_data.y[node].item()
                is_correct = (pred == true_label)
                marker = "✓" if is_correct else "✗"

                node_text.append(
                    f"<b>{paper_title}</b><br>"
                    f"Predicted: <b>{category_names[pred]}</b><br>"
                    f"Actual: {category_names[true_label]}<br>"
                    f"{marker} <b>{'Correct' if is_correct else 'Incorrect'}</b><br>"
                    f"Citations: {degree}"
                )
            else:
                # Color by true label
                true_label = graph_data.y[node].item()
                node_colors.append(category_colors.get(true_label, '#CCCCCC'))
                node_text.append(
                    f"<b>{paper_title}</b><br>"
                    f"Category: <b>{category_names[true_label]}</b><br>"
                    f"Citations: {degree}"
                )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                line=dict(width=2, color='white'),
                opacity=0.9
            ),
            showlegend=False
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])

        # Update layout
        title = "📊 Citation Network Visualization"
        if show_predictions:
            title += " - Model Predictions (✓ = Correct, ✗ = Incorrect)"

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color='#333'),
                x=0.5,
                xanchor='center'
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(245, 245, 245, 0.8)',
            paper_bgcolor='white',
            height=600
        )

        return fig

    except Exception as e:
        # Return empty figure on error
        fig = go.Figure()
        fig.update_layout(
            title=f"Error creating visualization: {str(e)}",
            height=600
        )
        return fig


# ============================================================================
# REAL DATA TRAINING TAB FUNCTIONS
# ============================================================================

def process_pdfs(files, extract_citations, build_graph, extract_metadata):
    """Process uploaded PDF files and create initial graph visualization"""
    if not files or len(files) == 0:
        return "⚠️ No files uploaded", "", None, None

    status = f"🔄 Processing {len(files)} PDF files...\n\n"
    papers_data = []

    for idx, file in enumerate(files):
        # Safely resolve filename from different upload types
        if hasattr(file, 'name'):
            filename = file.name
        elif isinstance(file, dict) and 'name' in file:
            filename = file['name']
        else:
            filename = f"uploaded_{idx}.pdf"

        status += f"📄 Processing file {idx+1}/{len(files)}: {filename}\n"

        # Prepare file for extract_text_from_pdf
        # Handle bytes, bytearray, file-like objects, and dict-shaped uploads
        if isinstance(file, (bytes, bytearray)):
            pdf_input = file
        elif isinstance(file, dict):
            # Gradio sometimes sends dict with 'data' or 'path' keys
            if 'data' in file:
                pdf_input = file['data']
            elif 'path' in file:
                with open(file['path'], 'rb') as f:
                    pdf_input = f.read()
            else:
                pdf_input = file
        elif hasattr(file, 'read'):
            # File-like object
            pdf_input = file
        else:
            pdf_input = file

        # Extract text
        text = extract_text_from_pdf(pdf_input)

        paper_info = {
            'name': filename,
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
        app_state.paper_list.append(filename)

    status += f"\n✅ Processed {len(papers_data)} papers successfully!\n"

    # Build graph if enabled
    graph_stats = ""
    initial_graph = None
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
            status += "👉 Click 'Train Model' to see live visualization during training!\n"

            # Create initial graph visualization
            initial_graph = create_interactive_graph(
                graph_data,
                predictions=None,
                show_predictions=False,
                paper_titles=app_state.paper_list
            )

        except Exception as e:
            status += f"❌ Error building graph: {str(e)}\n"
            graph_stats = f"❌ Error: {str(e)}"

    # Update dropdown choices
    paper_choices = gr.Dropdown(choices=app_state.paper_list, value=app_state.paper_list[0] if app_state.paper_list else None)

    return status, graph_stats, paper_choices, initial_graph


def train_gnn_live(model_type, epochs, learning_rate, task_type, progress=gr.Progress()):
    """Train GNN model with live progress updates AND interactive graph visualization"""
    if app_state.graph_data is None:
        yield "❌ Error: No graph data available. Please process papers first!", None, "", None
        return

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
        yield f"❌ Unknown model type: {model_type}", None, "", None
        return

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    # Training history
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    status = f"🚀 Training {model_type} model with LIVE graph visualization...\n\n"
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

        # Update status and visualization every 10 epochs
        update_interval = max(1, epochs // 10)
        if epoch % update_interval == 0 or epoch == epochs - 1:
            status += f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Train Acc: {train_acc.item():.4f} | Val Acc: {val_acc.item():.4f}\n"

            # Create live graph visualization showing current predictions
            live_graph = create_interactive_graph(
                data,
                predictions=pred,
                show_predictions=True,
                paper_titles=app_state.paper_list
            )

            # Yield intermediate results
            yield status, live_graph, "", None

    training_time = time.time() - start_time

    # Final evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        # Calculate per-class metrics
        correct = (pred == data.y).cpu().numpy()
        num_correct = correct.sum()
        num_total = len(correct)

    status += f"\n✅ Training Complete!\n"
    status += f"   Training Time: {training_time:.2f}s\n"
    status += f"   Final Test Accuracy: {test_acc.item():.4f} ({test_acc.item()*100:.2f}%)\n"
    status += f"   Correct Predictions: {num_correct}/{num_total}\n"

    # Save model
    app_state.trained_model = model
    app_state.model_type = model_type
    app_state.training_history = history

    # Create training curves plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2, color='#FF6B6B')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2, color='#4ECDC4')
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2, color='#45B7D1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    plt.tight_layout()

    # Create final graph visualization
    final_graph = create_interactive_graph(
        data,
        predictions=pred,
        show_predictions=True,
        paper_titles=app_state.paper_list
    )

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
- **Correct Predictions:** {num_correct}/{num_total}

### Model Info:
- **Parameters:** {sum(p.numel() for p in model.parameters()):,}
- **Epochs Trained:** {epochs}
- **Learning Rate:** {learning_rate}

### Legend:
- ✓ = Correct prediction
- ✗ = Incorrect prediction
- Node size = number of citations
- Node color = predicted category
"""

    # Final yield with all results
    yield status, final_graph, results_summary, fig


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
    from data.dataset_utils import create_synthetic_citation_network

    status += "Creating synthetic citation network (50 papers)...\n"
    data = create_synthetic_citation_network(num_papers=50, num_topics=3, avg_citations=5)

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
            # TAB 2: REAL DATA TRAINING WITH LIVE VISUALIZATION (MAIN FEATURE)
            # ================================================================
            with gr.Tab("📄 Real Data Training + Live Visualization"):
                gr.Markdown("""
                ## 🎨 Real Data Training with LIVE Interactive Graph Visualization

                Upload PDFs → Build Citation Network → Train GNN → Watch Live Updates!
                """)

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

                        train_button = gr.Button("🚀 Train with Live Visualization", variant="primary", size="lg")

                        training_status = gr.Textbox(label="Training Progress", lines=8)
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

                # Full-width visualization section
                gr.Markdown("---")
                gr.Markdown("## 🎨 Interactive Citation Network Visualization")

                with gr.Row():
                    with gr.Column(scale=2):
                        graph_viz = gr.Plot(
                            label="Live Graph Visualization - Watch nodes change color as model learns!",
                            value=None
                        )
                        viz_info = gr.Markdown("""
**Visualization Guide:**
- 🔵 **Node Colors**: Research categories (NLP=Red, CV=Teal, GNN=Blue, RL=Orange, Theory=Green)
- 📏 **Node Size**: Number of citations (bigger = more cited)
- ✓ **Green Checkmark**: Correct prediction
- ✗ **Red X**: Incorrect prediction
- 🔍 **Hover**: See paper details, predictions, and accuracy
- 🎬 **During Training**: Graph updates every 10 epochs showing live predictions
                        """)

                    with gr.Column(scale=1):
                        training_plot = gr.Plot(label="Training Curves (Loss & Accuracy)")

                # Connect buttons to functions
                process_button.click(
                    fn=process_pdfs,
                    inputs=[file_upload, extract_citations_cb, build_graph_cb, extract_metadata_cb],
                    outputs=[processing_status, graph_stats, paper_select, graph_viz]
                )

                train_button.click(
                    fn=train_gnn_live,
                    inputs=[model_type, epochs_slider, lr_slider, task_type],
                    outputs=[training_status, graph_viz, results_display, training_plot]
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
            # TAB 3: EVALUATION METRICS ANALYSIS
            # ================================================================
            with gr.Tab("📊 Evaluation Metrics"):
                gr.Markdown("""
                ## 📊 Comprehensive Evaluation Metrics

                Analyze model performance with professional ML metrics:
                - **Node Classification**: Precision, Recall, F1 (macro/weighted), Confusion Matrix
                - **ROC Analysis**: ROC curves, AUC scores, per-class performance
                - **Statistical Tests**: Paired t-tests, confidence intervals

                Upload predictions or use trained model results for detailed analysis.
                """)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📈 Input Data")
                        eval_predictions = gr.Textbox(
                            label="Predictions (comma-separated)",
                            placeholder="0,1,2,1,0,2...",
                            lines=3
                        )
                        eval_ground_truth = gr.Textbox(
                            label="Ground Truth (comma-separated)",
                            placeholder="0,1,2,1,1,2...",
                            lines=3
                        )
                        eval_button = gr.Button("📊 Analyze Metrics", variant="primary")

                    with gr.Column():
                        gr.Markdown("### 📊 Metrics Results")
                        eval_output = gr.Markdown("**Results will appear here**")

                gr.Markdown("---")
                gr.Markdown("### 📈 Visualizations")

                with gr.Row():
                    eval_confusion_plot = gr.Plot(label="Confusion Matrix")
                    eval_metrics_plot = gr.Plot(label="Performance Metrics")

                def analyze_metrics(pred_str, truth_str):
                    try:
                        import numpy as np
                        from sklearn.metrics import confusion_matrix

                        # Parse inputs
                        predictions = np.array([int(x.strip()) for x in pred_str.split(',') if x.strip()])
                        ground_truth = np.array([int(x.strip()) for x in truth_str.split(',') if x.strip()])

                        if len(predictions) != len(ground_truth):
                            return "❌ Error: Predictions and ground truth must have same length", None, None

                        # Compute metrics
                        metrics = NodeClassificationMetrics()
                        results = metrics.compute(ground_truth, predictions, None)

                        # Format output
                        output = f"""
## 📊 Classification Metrics

**Overall Performance:**
- **Accuracy:** {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)
- **Precision (Macro):** {results['precision_macro']:.4f}
- **Recall (Macro):** {results['recall_macro']:.4f}
- **F1 Score (Macro):** {results['f1_macro']:.4f}

**Weighted Metrics:**
- **Precision (Weighted):** {results['precision_weighted']:.4f}
- **Recall (Weighted):** {results['recall_weighted']:.4f}
- **F1 Score (Weighted):** {results['f1_weighted']:.4f}
"""

                        # Create confusion matrix plot
                        cm = confusion_matrix(ground_truth, predictions)
                        fig_cm, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(cm, cmap='Blues')
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('True')
                        ax.set_title('Confusion Matrix')
                        plt.colorbar(im, ax=ax)

                        # Create metrics plot
                        fig_metrics, ax2 = plt.subplots(figsize=(8, 6))
                        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
                        metrics_values = [
                            results['accuracy'],
                            results['precision_macro'],
                            results['recall_macro'],
                            results['f1_macro']
                        ]
                        ax2.bar(metrics_names, metrics_values, color=['#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
                        ax2.set_ylabel('Score')
                        ax2.set_title('Performance Metrics (Macro)')
                        ax2.set_ylim([0, 1])
                        ax2.grid(True, alpha=0.3)

                        return output, fig_cm, fig_metrics

                    except Exception as e:
                        return f"❌ Error: {str(e)}", None, None

                eval_button.click(
                    fn=analyze_metrics,
                    inputs=[eval_predictions, eval_ground_truth],
                    outputs=[eval_output, eval_confusion_plot, eval_metrics_plot]
                )

            # ================================================================
            # TAB 4: ATTENTION VISUALIZATION
            # ================================================================
            with gr.Tab("🔍 Attention Visualization"):
                gr.Markdown("""
                ## 🔍 GAT Attention Visualization

                Visualize and analyze attention weights from Graph Attention Networks:
                - **Attention Heatmaps**: See which nodes attend to which
                - **Top-K Analysis**: Find most important edges
                - **Distribution Analysis**: Understand attention patterns
                - **Gini Coefficient**: Measure attention concentration

                Train a GAT model first to enable attention analysis.
                """)

                gr.Markdown("### ⚠️ Note")
                gr.Markdown("""
                To use attention visualization:
                1. Train a GAT model in the "Real Data Training" tab
                2. Attention weights are automatically extracted during training
                3. Use the attention analysis tools below to visualize patterns

                **Coming Soon:** Direct attention visualization from trained models!
                """)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎨 Attention Pattern Demo")
                        attn_demo_button = gr.Button("🎮 Run Attention Demo", variant="primary")
                        attn_demo_output = gr.Markdown("**Demo results will appear here**")

                    with gr.Column():
                        gr.Markdown("### 📊 Attention Statistics")
                        attn_stats_output = gr.Markdown("**Statistics will appear here**")

                gr.Markdown("---")
                attn_plot = gr.Plot(label="Attention Heatmap")

                def run_attention_demo():
                    try:
                        # Create synthetic attention weights for demo
                        num_nodes = 20
                        attention_weights = torch.rand(num_nodes, num_nodes)
                        attention_weights = F.softmax(attention_weights, dim=1)

                        # Analyze patterns
                        patterns = analyze_attention_patterns(attention_weights, None)

                        stats_md = f"""
## 📊 Attention Statistics

**Distribution:**
- **Mean Attention:** {patterns['mean_attention']:.4f}
- **Median Attention:** {patterns['median_attention']:.4f}
- **Std Deviation:** {patterns['std_attention']:.4f}

**Concentration:**
- **Gini Coefficient:** {patterns['gini_coefficient']:.4f}
- **Max Attention:** {patterns['max_attention']:.4f}
- **Min Attention:** {patterns['min_attention']:.4f}

**Interpretation:**
- Gini > 0.5: Highly concentrated attention
- Gini < 0.3: Distributed attention
"""

                        # Create heatmap
                        fig, ax = plt.subplots(figsize=(10, 8))
                        im = ax.imshow(attention_weights.numpy(), cmap='YlOrRd', aspect='auto')
                        ax.set_xlabel('Target Node')
                        ax.set_ylabel('Source Node')
                        ax.set_title('GAT Attention Weights (Demo)')
                        plt.colorbar(im, ax=ax, label='Attention Weight')

                        return "✅ Attention demo completed! See heatmap and statistics.", stats_md, fig

                    except Exception as e:
                        return f"❌ Error: {str(e)}", "", None

                attn_demo_button.click(
                    fn=run_attention_demo,
                    outputs=[attn_demo_output, attn_stats_output, attn_plot]
                )

            # ================================================================
            # TAB 5: TEMPORAL ANALYSIS
            # ================================================================
            with gr.Tab("⏱️ Temporal Analysis"):
                gr.Markdown("""
                ## ⏱️ Temporal Citation Analysis

                Analyze temporal patterns in citation networks:
                - **Citation Velocity**: Track citation growth over time
                - **Topic Evolution**: See how research topics evolve
                - **Emerging Topics**: Detect new research areas with high acceleration
                - **Temporal Heatmaps**: Visualize citation patterns over time

                Generate synthetic temporal data or analyze real citation networks.
                """)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎮 Generate Demo Data")
                        temporal_num_papers = gr.Slider(
                            minimum=50,
                            maximum=500,
                            value=200,
                            step=50,
                            label="Number of Papers"
                        )
                        temporal_num_topics = gr.Slider(
                            minimum=3,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of Topics"
                        )
                        temporal_demo_button = gr.Button("📊 Run Temporal Analysis", variant="primary")

                    with gr.Column():
                        gr.Markdown("### 📈 Analysis Results")
                        temporal_output = gr.Markdown("**Results will appear here**")

                gr.Markdown("---")
                gr.Markdown("### 📊 Temporal Visualizations")

                with gr.Row():
                    temporal_plot1 = gr.Plot(label="Citation Growth Over Time")
                    temporal_plot2 = gr.Plot(label="Topic Evolution")

                def run_temporal_analysis(num_papers, num_topics):
                    try:
                        from data.dataset_utils import create_synthetic_citation_network

                        # Create temporal graph
                        data = create_synthetic_citation_network(
                            num_papers=int(num_papers),
                            num_topics=int(num_topics),
                            temporal=True
                        )

                        # Create temporal analyzer
                        analyzer = TemporalAnalyzer(data)

                        # Identify emerging topics
                        emerging = analyzer.identify_emerging_topics(lookback_years=3)

                        output_md = f"""
## 📊 Temporal Analysis Results

**Dataset:**
- **Papers:** {data.num_nodes}
- **Citations:** {data.num_edges}
- **Topics:** {num_topics}
- **Time Span:** 10 years

**Emerging Topics (Last 3 years):**
"""

                        if emerging:
                            for topic_id, accel in emerging[:3]:
                                output_md += f"\n- **Topic {topic_id}**: {accel:.2f}x acceleration"
                        else:
                            output_md += "\n- No strongly emerging topics detected"

                        # Create plots
                        fig1 = analyzer.plot_citation_growth()
                        fig2 = analyzer.plot_topic_evolution()

                        return output_md, fig1, fig2

                    except Exception as e:
                        return f"❌ Error: {str(e)}", None, None

                temporal_demo_button.click(
                    fn=run_temporal_analysis,
                    inputs=[temporal_num_papers, temporal_num_topics],
                    outputs=[temporal_output, temporal_plot1, temporal_plot2]
                )

            # ================================================================
            # TAB 6: ABOUT
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

def find_available_port(start_port=7860, max_tries=10):
    """Find an available port, starting from start_port"""
    import socket
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return start_port  # Fallback to start_port and let Gradio handle the error

if __name__ == "__main__":
    print("🚀 Starting Research Compass GNN...")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   Device: {app_state.device}")
    print(f"   Gradio: {gr.__version__}")

    # Find available port
    port = find_available_port(start_port=7860, max_tries=10)
    print(f"   Port: {port}")

    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
