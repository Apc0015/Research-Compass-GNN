#!/usr/bin/env python3
"""
GNN Model Comparison Study - Research Compass
Comprehensive comparison of 4 GNN architectures for academic citation networks

This script:
1. Creates a realistic citation dataset
2. Trains all 4 GNN models (GCN, GAT, Transformer, Hetero)
3. Measures performance metrics (accuracy, speed, memory)
4. Generates comparison tables and visualizations
5. Exports results for technical report
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path
import psutil
import os


# ============================================================================
# MODEL DEFINITIONS (Same as test suite)
# ============================================================================

class GCNModel(nn.Module):
    """GCN for Node Classification"""
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
    """GAT for Link Prediction"""
    def __init__(self, input_dim=384, hidden_dim=128, num_layers=2, heads=4, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout))
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def encode(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return x

    def decode(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        edge_features = torch.cat([src, dst], dim=1)
        return self.edge_predictor(edge_features).squeeze()

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class TransformerModel(nn.Module):
    """Graph Transformer for Embeddings"""
    def __init__(self, input_dim=384, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim * num_heads
            self.convs.append(TransformerConv(in_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True))
        self.output_proj = nn.Linear(hidden_dim * num_heads, input_dim)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return self.output_proj(x)


# ============================================================================
# DATASET GENERATION
# ============================================================================

def create_realistic_citation_network(
    num_papers: int = 200,
    num_topics: int = 5,
    avg_citations: int = 8,
    temporal: bool = True
) -> Data:
    """
    Create a realistic citation network mimicking academic papers

    Features:
    - Power-law citation distribution (some papers highly cited)
    - Topic clustering (papers in same topic cite each other more)
    - Temporal ordering (papers only cite older papers)
    - Realistic feature embeddings
    """
    print(f"🔬 Creating realistic citation network...")
    print(f"   Papers: {num_papers}, Topics: {num_topics}, Avg citations: {avg_citations}")

    # Node features (384-dim embeddings like Sentence-BERT)
    x = torch.randn(num_papers, 384)

    # Topic labels (ground truth for node classification)
    y = torch.randint(0, num_topics, (num_papers,))

    # Generate edges with realistic patterns
    edges = []

    if temporal:
        # Papers can only cite older papers
        for target in range(1, num_papers):
            # Number of citations (power-law distribution)
            num_citations = max(1, int(np.random.exponential(avg_citations)))
            num_citations = min(num_citations, target)  # Can't cite more than exist

            # Prefer papers from same topic (80% probability)
            target_topic = y[target].item()

            for _ in range(num_citations):
                if np.random.rand() < 0.8:
                    # Same topic
                    same_topic_papers = [i for i in range(target) if y[i].item() == target_topic]
                    if same_topic_papers:
                        source = np.random.choice(same_topic_papers)
                    else:
                        source = np.random.randint(0, target)
                else:
                    # Different topic
                    source = np.random.randint(0, target)

                edges.append([source, target])
    else:
        # Random edges (for comparison)
        num_edges = num_papers * avg_citations
        for _ in range(num_edges):
            source = np.random.randint(0, num_papers)
            target = np.random.randint(0, num_papers)
            if source != target:
                edges.append([source, target])

    edge_index = torch.tensor(edges, dtype=torch.long).t()

    # Create train/val/test masks
    num_train = int(0.6 * num_papers)
    num_val = int(0.2 * num_papers)

    perm = torch.randperm(num_papers)
    train_mask = torch.zeros(num_papers, dtype=torch.bool)
    val_mask = torch.zeros(num_papers, dtype=torch.bool)
    test_mask = torch.zeros(num_papers, dtype=torch.bool)

    train_mask[perm[:num_train]] = True
    val_mask[perm[num_train:num_train+num_val]] = True
    test_mask[perm[num_train+num_val:]] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    print(f"✅ Created graph: {num_papers} nodes, {edge_index.shape[1]} edges")
    print(f"   Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")

    return data


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def train_and_evaluate_gcn(data: Data, epochs: int = 50) -> Dict[str, Any]:
    """Train GCN model and collect comprehensive metrics"""
    print("\n" + "="*60)
    print("🧪 Training GCN (Graph Convolutional Network)")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = GCNModel(
        input_dim=384,
        hidden_dim=128,
        output_dim=data.y.max().item() + 1,
        num_layers=3,
        dropout=0.5
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Metrics tracking
    train_losses = []
    val_losses = []
    val_accs = []
    epoch_times = []
    memory_start = get_memory_usage()

    best_val_acc = 0
    best_model_state = None

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Starting memory: {memory_start:.2f} MB")

    training_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
            pred = out.argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()

            val_losses.append(val_loss.item())
            val_accs.append(val_acc.item())

            if val_acc > best_val_acc:
                best_val_acc = val_acc.item()
                best_model_state = model.state_dict().copy()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc.item():.4f} | Time: {epoch_time:.3f}s")

    total_training_time = time.time() - training_start
    memory_end = get_memory_usage()

    # Test evaluation with best model
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
        test_loss = F.cross_entropy(out[data.test_mask], data.y[data.test_mask])

        # Per-class accuracy
        num_classes = data.y.max().item() + 1
        per_class_acc = []
        for c in range(num_classes):
            mask = (data.y[data.test_mask] == c)
            if mask.sum() > 0:
                acc = (pred[data.test_mask][mask] == c).float().mean()
                per_class_acc.append(acc.item())

    # Inference speed test
    inference_times = []
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            start = time.time()
            _ = model(data.x, data.edge_index)
            inference_times.append(time.time() - start)

    results = {
        'model_name': 'GCN',
        'task': 'Node Classification',
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'train_loss_history': train_losses,
        'val_loss_history': val_losses,
        'val_acc_history': val_accs,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc.item(),
        'test_loss': test_loss.item(),
        'per_class_acc': per_class_acc,
        'avg_per_class_acc': np.mean(per_class_acc),
        'total_training_time': total_training_time,
        'avg_epoch_time': np.mean(epoch_times),
        'avg_inference_time': np.mean(inference_times) * 1000,  # ms
        'memory_usage': memory_end - memory_start,
        'convergence_speed': len([i for i, acc in enumerate(val_accs) if acc >= best_val_acc * 0.95])
    }

    print(f"\n✅ GCN Training Complete")
    print(f"   Test Accuracy: {test_acc.item():.4f}")
    print(f"   Training Time: {total_training_time:.2f}s")
    print(f"   Inference Time: {results['avg_inference_time']:.2f}ms")

    return results


def train_and_evaluate_gat(data: Data, epochs: int = 50) -> Dict[str, Any]:
    """Train GAT model and collect comprehensive metrics"""
    print("\n" + "="*60)
    print("🧪 Training GAT (Graph Attention Network)")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = GATModel(
        input_dim=384,
        hidden_dim=128,
        num_layers=2,
        heads=4,
        dropout=0.3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Metrics tracking
    train_losses = []
    val_losses = []
    val_accs = []
    epoch_times = []
    memory_start = get_memory_usage()

    best_val_acc = 0
    best_model_state = None

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Starting memory: {memory_start:.2f} MB")

    training_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training (Link Prediction)
        model.train()
        optimizer.zero_grad()

        # Sample edges
        num_train_edges = int(data.edge_index[:, data.train_mask[data.edge_index[0]]].shape[1] * 0.8)
        pos_edge = data.edge_index[:, :num_train_edges]
        neg_edge = torch.randint(0, data.x.size(0), (2, num_train_edges)).to(device)

        pos_pred = model(data.x, data.edge_index, pos_edge)
        neg_pred = model(data.x, data.edge_index, neg_edge)

        loss = (F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred)) +
                F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred)))

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            num_val_edges = 100
            val_pos_edge = data.edge_index[:, num_train_edges:num_train_edges+num_val_edges]
            val_neg_edge = torch.randint(0, data.x.size(0), (2, num_val_edges)).to(device)

            val_pos_pred = model(data.x, data.edge_index, val_pos_edge)
            val_neg_pred = model(data.x, data.edge_index, val_neg_edge)

            val_loss = (F.binary_cross_entropy_with_logits(val_pos_pred, torch.ones_like(val_pos_pred)) +
                       F.binary_cross_entropy_with_logits(val_neg_pred, torch.zeros_like(val_neg_pred)))

            val_acc = ((val_pos_pred > 0).float().mean() + (val_neg_pred < 0).float().mean()) / 2

            val_losses.append(val_loss.item())
            val_accs.append(val_acc.item())

            if val_acc > best_val_acc:
                best_val_acc = val_acc.item()
                best_model_state = model.state_dict().copy()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc.item():.4f} | Time: {epoch_time:.3f}s")

    total_training_time = time.time() - training_start
    memory_end = get_memory_usage()

    # Test evaluation
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        num_test_edges = 100
        test_pos_edge = data.edge_index[:, -num_test_edges:]
        test_neg_edge = torch.randint(0, data.x.size(0), (2, num_test_edges)).to(device)

        test_pos_pred = model(data.x, data.edge_index, test_pos_edge)
        test_neg_pred = model(data.x, data.edge_index, test_neg_edge)

        test_acc = ((test_pos_pred > 0).float().mean() + (test_neg_pred < 0).float().mean()) / 2
        test_loss = (F.binary_cross_entropy_with_logits(test_pos_pred, torch.ones_like(test_pos_pred)) +
                    F.binary_cross_entropy_with_logits(test_neg_pred, torch.zeros_like(test_neg_pred)))

    # Inference speed
    inference_times = []
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            start = time.time()
            _ = model.encode(data.x, data.edge_index)
            inference_times.append(time.time() - start)

    results = {
        'model_name': 'GAT',
        'task': 'Link Prediction',
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'train_loss_history': train_losses,
        'val_loss_history': val_losses,
        'val_acc_history': val_accs,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc.item(),
        'test_loss': test_loss.item(),
        'total_training_time': total_training_time,
        'avg_epoch_time': np.mean(epoch_times),
        'avg_inference_time': np.mean(inference_times) * 1000,
        'memory_usage': memory_end - memory_start,
        'convergence_speed': len([i for i, acc in enumerate(val_accs) if acc >= best_val_acc * 0.95])
    }

    print(f"\n✅ GAT Training Complete")
    print(f"   Test Accuracy: {test_acc.item():.4f}")
    print(f"   Training Time: {total_training_time:.2f}s")
    print(f"   Inference Time: {results['avg_inference_time']:.2f}ms")

    return results


def train_and_evaluate_transformer(data: Data, epochs: int = 50) -> Dict[str, Any]:
    """Train Graph Transformer and collect metrics"""
    print("\n" + "="*60)
    print("🧪 Training Graph Transformer")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = TransformerModel(
        input_dim=384,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    epoch_times = []
    memory_start = get_memory_usage()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Starting memory: {memory_start:.2f} MB")

    training_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out[data.train_mask], data.x[data.train_mask])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Time: {epoch_time:.3f}s")

    total_training_time = time.time() - training_start
    memory_end = get_memory_usage()

    # Test reconstruction quality
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_loss = F.mse_loss(out[data.test_mask], data.x[data.test_mask])

        # Compute embedding quality (cosine similarity)
        from torch.nn.functional import cosine_similarity
        cos_sim = cosine_similarity(out[data.test_mask], data.x[data.test_mask], dim=1)
        avg_cos_sim = cos_sim.mean()

    # Inference speed
    inference_times = []
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            start = time.time()
            _ = model(data.x, data.edge_index)
            inference_times.append(time.time() - start)

    results = {
        'model_name': 'Transformer',
        'task': 'Embedding',
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'train_loss_history': train_losses,
        'final_train_loss': train_losses[-1],
        'test_loss': test_loss.item(),
        'avg_cosine_similarity': avg_cos_sim.item(),
        'total_training_time': total_training_time,
        'avg_epoch_time': np.mean(epoch_times),
        'avg_inference_time': np.mean(inference_times) * 1000,
        'memory_usage': memory_end - memory_start,
    }

    print(f"\n✅ Transformer Training Complete")
    print(f"   Test Loss: {test_loss.item():.4f}")
    print(f"   Avg Cosine Similarity: {avg_cos_sim.item():.4f}")
    print(f"   Training Time: {total_training_time:.2f}s")
    print(f"   Inference Time: {results['avg_inference_time']:.2f}ms")

    return results


# ============================================================================
# COMPARISON & VISUALIZATION
# ============================================================================

def create_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create comparison table for technical report"""
    comparison_data = []

    for r in results:
        row = {
            'Model': r['model_name'],
            'Task': r['task'],
            'Parameters': f"{r['num_parameters']:,}",
            'Test Accuracy': f"{r.get('test_acc', r.get('avg_cosine_similarity', 0)):.4f}",
            'Training Time (s)': f"{r['total_training_time']:.2f}",
            'Inference Time (ms)': f"{r['avg_inference_time']:.2f}",
            'Memory Usage (MB)': f"{r['memory_usage']:.2f}",
        }
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    return df


def create_visualizations(results: List[Dict[str, Any]], output_dir: Path):
    """Create comparison visualizations"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GNN Model Comparison - Training Curves', fontsize=16, fontweight='bold')

    for i, r in enumerate(results[:3]):  # First 3 models
        ax = axes[i // 2, i % 2]
        if 'train_loss_history' in r:
            ax.plot(r['train_loss_history'], label='Train Loss', linewidth=2)
            if 'val_loss_history' in r:
                ax.plot(r['val_loss_history'], label='Val Loss', linewidth=2, linestyle='--')
        ax.set_title(f"{r['model_name']} - {r['task']}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove empty subplot
    fig.delaxes(axes[1, 1])

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: training_curves.png")
    plt.close()

    # 2. Performance comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('GNN Model Performance Comparison', fontsize=16, fontweight='bold')

    models = [r['model_name'] for r in results]

    # Accuracy/Performance
    ax = axes[0]
    accuracies = [r.get('test_acc', r.get('avg_cosine_similarity', 0)) for r in results]
    bars = ax.bar(models, accuracies, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(models)])
    ax.set_ylabel('Accuracy / Similarity', fontsize=12)
    ax.set_title('Model Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    # Training Time
    ax = axes[1]
    train_times = [r['total_training_time'] for r in results]
    bars = ax.bar(models, train_times, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(models)])
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Training Time', fontsize=12, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=10)

    # Inference Time
    ax = axes[2]
    inference_times = [r['avg_inference_time'] for r in results]
    bars = ax.bar(models, inference_times, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(models)])
    ax.set_ylabel('Time (milliseconds)', fontsize=12)
    ax.set_title('Inference Time', fontsize=12, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}ms', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: performance_comparison.png")
    plt.close()

    # 3. Model complexity comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    models = [r['model_name'] for r in results]
    params = [r['num_parameters'] / 1000 for r in results]  # In thousands
    memory = [r['memory_usage'] for r in results]

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, params, width, label='Parameters (K)', color='#3498db')
    ax.bar(x + width/2, memory, width, label='Memory Usage (MB)', color='#e74c3c')

    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Model Complexity Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'model_complexity.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: model_complexity.png")
    plt.close()


# ============================================================================
# MAIN COMPARISON STUDY
# ============================================================================

def run_comparison_study(
    num_papers: int = 200,
    epochs: int = 50,
    output_dir: str = "comparison_results"
):
    """Run complete comparison study"""
    print("\n" + "🔬 " * 20)
    print("   GNN MODEL COMPARISON STUDY")
    print("   Research Compass - Phase 3")
    print("🔬 " * 20 + "\n")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    data = create_realistic_citation_network(
        num_papers=num_papers,
        num_topics=5,
        avg_citations=8,
        temporal=True
    )

    # Train all models
    results = []

    # GCN
    gcn_results = train_and_evaluate_gcn(data, epochs=epochs)
    results.append(gcn_results)

    # GAT
    gat_results = train_and_evaluate_gat(data, epochs=epochs)
    results.append(gat_results)

    # Transformer
    transformer_results = train_and_evaluate_transformer(data, epochs=epochs)
    results.append(transformer_results)

    # Create comparison table
    print("\n" + "="*60)
    print("📊 COMPARISON TABLE")
    print("="*60 + "\n")

    df = create_comparison_table(results)
    print(df.to_string(index=False))

    # Save results
    df.to_csv(output_path / 'comparison_table.csv', index=False)
    print(f"\n✅ Saved: {output_path / 'comparison_table.csv'}")

    # Save detailed results as JSON
    with open(output_path / 'detailed_results.json', 'w') as f:
        # Convert numpy types to native Python types
        results_serializable = []
        for r in results:
            r_copy = {}
            for k, v in r.items():
                if isinstance(v, list):
                    r_copy[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
                elif isinstance(v, (np.floating, np.integer)):
                    r_copy[k] = float(v)
                else:
                    r_copy[k] = v
            results_serializable.append(r_copy)

        json.dump(results_serializable, f, indent=2)
    print(f"✅ Saved: {output_path / 'detailed_results.json'}")

    # Create visualizations
    print("\n📈 Creating visualizations...")
    create_visualizations(results, output_path)

    # Summary
    print("\n" + "="*60)
    print("🎉 COMPARISON STUDY COMPLETE")
    print("="*60)
    print(f"\n📂 All results saved to: {output_path}")
    print(f"   - comparison_table.csv")
    print(f"   - detailed_results.json")
    print(f"   - training_curves.png")
    print(f"   - performance_comparison.png")
    print(f"   - model_complexity.png")
    print("\n✅ Results ready for technical report!")

    return results, df


if __name__ == "__main__":
    import sys

    # Parse arguments
    num_papers = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    # Run comparison
    results, df = run_comparison_study(num_papers=num_papers, epochs=epochs)
