# Usage Guide - Research Compass GNN v3.0

This guide covers how to use all features of the enhanced Research Compass GNN platform, including **NEW: HAN and R-GCN models**.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Training Enhanced Models](#training-enhanced-models)
3. [Model Comparison](#model-comparison)
4. [Attention Visualization](#attention-visualization)
5. [Temporal Analysis](#temporal-analysis)
6. **[Advanced GNN Models (HAN & R-GCN)](#advanced-gnn-models-han--r-gcn)** ⭐ NEW
7. [Ablation Studies](#ablation-studies)
8. [Using Notebooks](#using-notebooks)
9. [Gradio UI](#gradio-ui)

---

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Research-Compass-GNN

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (CPU)
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# For GPU support (CUDA 11.8)
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Verify Installation

```python
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"
```

---

## Training Enhanced Models

The `train_enhanced.py` script provides comprehensive training with all Phase 1 enhancements.

### Basic Training

```bash
# Train GCN on Cora with LR scheduling
python train_enhanced.py --model GCN --dataset Cora --epochs 100

# Train GAT with multi-task learning
python train_enhanced.py --model GAT --multitask --dataset Cora --epochs 50

# Train GraphSAGE on synthetic data
python train_enhanced.py --model GraphSAGE --dataset synthetic --size 500 --epochs 100
```

### Advanced Options

```bash
# Force mini-batch training
python train_enhanced.py --model GCN --dataset synthetic --size 5000 --minibatch --batch-size 32

# Custom learning rate and scheduler
python train_enhanced.py \
    --model GCN \
    --dataset Cora \
    --lr 0.01 \
    --scheduler-patience 10 \
    --scheduler-factor 0.5 \
    --min-lr 1e-6

# Save trained model
python train_enhanced.py --model GCN --dataset Cora --save-model
```

### Output

Training generates:
- `results/<model>_<dataset>_<timestamp>/`
  - `evaluation_report.md` - Comprehensive metrics
  - `results.json` - Machine-readable results
  - `visualizations/` - Plots and figures
  - `<model>_best.pt` - Best model checkpoint (if --save-model)

---

## Model Comparison

Compare all models (4 GNNs + 6 baselines) on a dataset.

### Basic Comparison

```bash
# Compare on Cora
python compare_all_models.py --dataset Cora

# Compare on synthetic dataset
python compare_all_models.py --dataset synthetic --size 500
```

### Custom Configuration

```bash
# Longer training for better results
python compare_all_models.py --dataset Cora --epochs 200

# Custom output directory
python compare_all_models.py --dataset Cora --output-dir my_comparison

# Set random seed for reproducibility
python compare_all_models.py --dataset Cora --seed 42
```

### Output

Comparison generates:
- `results/comparison/<dataset>_<timestamp>/`
  - `comparison_results.csv` - Results table
  - `comparison_report.md` - Analysis and key findings
  - `comparison_plot.png` - Visualization
  - `results.json` - Detailed results

### Reading Results

```python
import pandas as pd

# Load comparison results
df = pd.read_csv('results/comparison/.../comparison_results.csv')
print(df.sort_values('Test Accuracy', ascending=False))

# Top model
best = df.iloc[0]
print(f"Best: {best['Model']} - {best['Test Accuracy']:.4f}")
```

---

## Attention Visualization

Visualize what GAT learns through attention weights.

### Extract Attention Weights

```python
from models import GATModel
from visualization import AttentionVisualizer
import torch

# Load trained GAT model
model = GATModel(input_dim=384, output_dim=5)
model.load_state_dict(torch.load('path/to/model.pt'))
model.eval()

# Forward pass with attention extraction
with torch.no_grad():
    out = model(data.x, data.edge_index, return_attention_weights=True)

# Get attention weights
attention_weights = model.get_attention_weights()
# Returns: (edge_index, attention_values)
```

### Visualize Attention

```python
# Create visualizer
viz = AttentionVisualizer()

# Heatmap of top-K nodes
viz.plot_attention_heatmap(
    attention_weights,
    node_names=paper_titles,
    top_k=20
)

# Top-K attention edges
viz.plot_top_k_attention(
    attention_weights,
    node_names=paper_titles,
    k=10
)

# Attention distribution
viz.plot_attention_distribution(attention_weights)

# Interactive graph
fig = viz.create_interactive_attention_graph(
    data,
    attention_weights,
    node_names=paper_titles,
    top_k_nodes=50
)
fig.show()

# Save all figures
viz.save_all('results/attention/')
```

### Analyze Attention Patterns

```python
from visualization import analyze_attention_patterns

analysis = analyze_attention_patterns(
    attention_weights,
    data,
    node_names=paper_titles
)

print(f"Mean attention: {analysis['mean_attention']:.4f}")
print(f"Attention Gini: {analysis['attention_gini']:.4f}")
print("\nTop 10 nodes by attention:")
for node in analysis['top_10_nodes']:
    print(f"  {node['node_name']}: {node['avg_attention']:.4f}")
```

---

## Temporal Analysis

Analyze research evolution over time.

### Basic Temporal Analysis

```python
from analysis import TemporalAnalyzer

# Create analyzer
analyzer = TemporalAnalyzer()

# Add temporal data
analyzer.add_temporal_data(
    data,
    years=publication_years,  # Optional, generates if None
    paper_titles=paper_titles
)

# Generate visualizations
analyzer.plot_citation_growth()
analyzer.plot_topic_evolution()
analyzer.plot_topic_heatmap()
analyzer.plot_citation_velocity_distribution(top_k=20)

# Save all plots
analyzer.save_all('results/temporal/')
```

### Analyze Citation Velocity

```python
# Get velocity for specific paper
velocity = analyzer.analyze_citation_velocity(node_idx=42)

print(f"Paper: {velocity['node_idx']}")
print(f"Citations: {velocity['total_citations']}")
print(f"Velocity: {velocity['citation_velocity']:.2f} cites/year")
```

### Identify Emerging Topics

```python
# Find emerging research areas
emerging = analyzer.identify_emerging_topics(
    lookback_years=3,
    min_papers=5
)

print("Emerging Topics:")
for topic in emerging[:5]:
    if topic['is_emerging']:
        print(f"  Topic {topic['topic']}: "
              f"acceleration={topic['acceleration']:.2f}")
```

### Generate Comprehensive Report

```python
from analysis import generate_temporal_report

report_path = generate_temporal_report(
    data,
    years=years,
    paper_titles=titles,
    output_dir='results/temporal'
)

print(f"Report: {report_path}")
```

---

## Advanced GNN Models (HAN & R-GCN)

### HAN (Heterogeneous Attention Network)

Train on multi-relational graphs with multiple node and edge types.

#### Convert to Heterogeneous Graph

```python
from torch_geometric.datasets import Planetoid
from data import convert_to_heterogeneous

# Load dataset
data = Planetoid(root='/tmp/Cora', name='Cora')[0]

# Convert to heterogeneous graph
hetero_data = convert_to_heterogeneous(
    data,
    num_venues=15,               # Number of publication venues
    num_authors_per_paper=(2, 4), # Average authors per paper
    author_collaboration_prob=0.3, # Collaboration probability
    venue_topic_correlation=0.7    # Topic-venue correlation
)

# View statistics
print(f"Node types: {hetero_data.node_types}")
print(f"Edge types: {len(hetero_data.edge_types)}")
# Node types: ['paper', 'author', 'venue', 'topic']
# Edge types: 7 (cites, written_by, published_in, belongs_to + reverse)
```

#### Train HAN Model

```python
from models import create_han_model
from training.trainer import HANTrainer
import torch.optim as optim

# Create HAN model
model = create_han_model(
    hetero_data,
    hidden_dim=128,
    num_heads=8,
    task='classification',
    num_classes=7
)

print(f"Parameters: {model.count_parameters():,}")

# Setup training
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
trainer = HANTrainer(
    model, optimizer,
    target_node_type='paper',
    scheduler_config={'mode': 'max', 'factor': 0.5, 'patience': 5}
)

# Training loop
for epoch in range(100):
    train_metrics = trainer.train_epoch(hetero_data)
    val_metrics = trainer.validate(hetero_data, return_attention=(epoch % 10 == 0))

    trainer.step_scheduler(val_metrics['accuracy'])
    is_best = trainer.save_best_model(val_metrics['accuracy'], epoch)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"LR: {train_metrics['lr']:.6f} "
              f"{'🌟' if is_best else ''}")

# Load best model
trainer.load_best_model()
print(f"Best epoch: {trainer.best_epoch}")

# Get attention weights
attention = trainer.get_attention_weights()
if attention:
    for node_type, attn in attention.items():
        if attn is not None:
            print(f"Attention for {node_type}: {attn.shape}")
```

### R-GCN (Relational GCN)

Train on citation networks with typed relationships.

#### Classify Citation Types

```python
from torch_geometric.datasets import Planetoid
from data import classify_citation_types

# Load dataset
data = Planetoid(root='/tmp/Cora', name='Cora')[0]

# Classify citations into 4 types
edge_types, typed_edges = classify_citation_types(data)

# View distribution
# Citation Type Distribution:
# EXTENDS       : 1234 (25.3%)
# METHODOLOGY   :  856 (17.5%)
# BACKGROUND    : 1789 (36.6%)
# COMPARISON    : 1002 (20.5%)

# Access typed edges
extends_edges = typed_edges['extends']
method_edges = typed_edges['methodology']
background_edges = typed_edges['background']
comparison_edges = typed_edges['comparison']
```

#### Train R-GCN Model

```python
from models import create_rgcn_model
import torch.optim as optim
import torch.nn.functional as F

# Create R-GCN model
model = create_rgcn_model(
    data,
    num_relations=4,      # 4 citation types
    hidden_dim=128,
    num_bases=30,         # Basis-decomposition for efficiency
    task='classification'
)

print(f"Parameters: {model.count_parameters():,}")

# Setup training
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
data = data.to(device)
edge_types = edge_types.to(device)

# Training loop
for epoch in range(100):
    # Train
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, edge_types)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # Validate
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, edge_types)
        pred = out[data.val_mask].argmax(dim=1)
        val_acc = (pred == data.y[data.val_mask]).float().mean().item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")

# Test
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index, edge_types)
    test_pred = out[data.test_mask].argmax(dim=1)
    test_acc = (test_pred == data.y[data.test_mask]).float().mean().item()

print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
```

#### Verify Implementations

```bash
# Run verification tests
python verify_han.py    # Test HAN implementation
python verify_rgcn.py   # Test R-GCN implementation

# Expected output:
# ======================================================================
# ✅ ALL VERIFICATIONS PASSED - HAN IMPLEMENTATION COMPLETE
# ======================================================================
```

---

## Ablation Studies

Systematically analyze which components contribute to performance.

### Run Comprehensive Ablation

```python
from experiments import run_comprehensive_ablation
from data import create_synthetic_citation_network

# Create dataset
data = create_synthetic_citation_network(num_papers=500)

# Run all ablations (takes time!)
results = run_comprehensive_ablation(
    data,
    output_dir='results/ablation'
)

# Outputs:
# - ablation_results.json
# - ablation_report.md
```

### Individual Ablation Studies

```python
from experiments import AblationStudy

# Create study
study = AblationStudy(data, epochs=100, num_runs=3)

# Feature ablation
feature_results = study.feature_ablation()
# Compares: full model vs graph-only vs features-only

# Architecture ablation
arch_results = study.architecture_ablation()
# Varies: layers, hidden_dim, dropout

# Training ablation
train_results = study.training_ablation()
# Compares: with/without LR scheduler

# Save results
study.save_results('results/my_ablation/')
```

### Read Ablation Results

```python
import json

with open('results/ablation/ablation_results.json') as f:
    results = json.load(f)

# Feature ablation results
fa = results['feature_ablation']
print(f"Full model: {fa['full_model']['mean']:.4f}")
print(f"Graph only: {fa['graph_only']['mean']:.4f}")
print(f"Features only: {fa['features_only']['mean']:.4f}")
```

---

## Using Notebooks

Interactive exploration with Jupyter notebooks.

### Available Notebooks

```bash
cd notebooks/

# 1. Benchmark on real datasets
jupyter notebook real_dataset_benchmark.ipynb

# 2. Original comparison study
jupyter notebook comparison_study.ipynb

# 3. Demo for presentations
jupyter notebook demo_for_professors.ipynb
```

### Import from Main Codebase

```python
# In any notebook
import sys
sys.path.append('..')  # Add parent to path

from models import GCNModel, GATModel, GraphSAGEModel
from data import load_citation_dataset, create_synthetic_citation_network
from evaluation import EvaluationReportGenerator
from visualization import AttentionVisualizer
from analysis import TemporalAnalyzer
```

---

## Gradio UI

Interactive web interface for training GNNs.

### Launch UI

```bash
python launcher.py

# Custom port
python launcher.py --port 8080

# Public sharing (creates public URL)
python launcher.py --share
```

### Access

Open browser to `http://localhost:7860`

### UI Features

**Tab 1: Welcome & Demo**
- Pre-trained models on synthetic data
- Quick predictions
- Visualization examples

**Tab 2: Real Data Training**
- Upload PDFs
- Extract citations
- Build knowledge graph
- Train GNN models
- Make predictions

**Tab 3: About**
- Project information
- Model descriptions
- Usage instructions

---

## Common Workflows

### 1. Quick Experiment

```bash
# Train GCN on Cora, compare with baselines
python train_enhanced.py --model GCN --dataset Cora
python compare_all_models.py --dataset Cora
```

### 2. Research Paper Workflow

```bash
# 1. Train all models
python compare_all_models.py --dataset Cora --epochs 200

# 2. Run ablation studies
python -c "from experiments import run_comprehensive_ablation; \
           from data import load_citation_dataset; \
           data, _ = load_citation_dataset('Cora'); \
           run_comprehensive_ablation(data)"

# 3. Visualize attention
# (Use notebook or Python script)

# 4. Generate temporal analysis
# (Use notebook or Python script)

# Results ready for paper!
```

### 3. New Dataset Evaluation

```python
from data import load_citation_dataset
from training.trainer import GCNTrainer
from models import GCNModel
import torch.optim as optim

# Load new dataset
data, info = load_citation_dataset('YourDataset')

# Train model
model = GCNModel(input_dim=info['num_features'], output_dim=info['num_classes'])
optimizer = optim.Adam(model.parameters(), lr=0.01)
trainer = GCNTrainer(model, optimizer)

# Training loop...
```

---

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Use CPU
python train_enhanced.py --model GCN --dataset Cora

# Or use mini-batch
python train_enhanced.py --model GCN --dataset Cora --minibatch --batch-size 16
```

**2. PyTorch Geometric not found**
```bash
pip install torch torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**3. Import errors**
```bash
# Make sure you're in the project root
cd Research-Compass-GNN

# Add to Python path if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Tips & Best Practices

### Performance
- Use mini-batch for graphs > 1000 nodes
- Enable CUDA if available
- Reduce epochs for quick experiments

### Reproducibility
- Always set `--seed` for experiments
- Save configuration files
- Version control your changes

### Experimentation
- Start with Cora (small, fast)
- Use notebooks for exploration
- Use scripts for production runs

### Documentation
- Check docstrings: `help(GCNModel)`
- See examples in each module
- Read ARCHITECTURE.md for design

---

## Additional Resources

- **ARCHITECTURE.md** - System design and architecture
- **ENHANCEMENTS.md** - v2.0 features and improvements
- **README.md** - Project overview and features
- **Inline Docs** - Comprehensive docstrings in all modules

---

## Getting Help

1. Check this guide first
2. Read module docstrings
3. Check examples in notebooks
4. Review ARCHITECTURE.md for design details

---

*Last Updated: 2025-01-XX*
*Version: 2.0*
