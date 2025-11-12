# 🧭 Research Compass GNN

Advanced Graph Neural Network platform for citation network analysis and node classification research.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)

---

## 📋 Overview

Research Compass GNN is a comprehensive platform for graph neural network research on citation networks. It implements **6 state-of-the-art GNN architectures** with advanced features for training, evaluation, and analysis.

### 🎯 Key Features

- **6 GNN Models:** GCN, GAT, GraphSAGE, Graph Transformer, HAN, R-GCN
- **Advanced Models:** Heterogeneous graphs (HAN) and relational graphs (R-GCN)
- **Multi-Format Upload:** PDF, DOCX, TXT, HTML, XML, TAR, ZIP archives
- **Web URL Support:** Download papers from arXiv, DOI, and other academic sources
- **Comprehensive Evaluation:** Node classification, link prediction metrics
- **Visualization Tools:** Attention weights, training curves, confusion matrices
- **Temporal Analysis:** Citation trends and research evolution tracking
- **Baseline Comparisons:** Traditional ML baselines (Logistic, RF, MLP, Label Propagation, Node2Vec)
- **Configuration Management:** YAML-based centralized configuration
- **Professional Infrastructure:** Logging, checkpointing, and experiment tracking

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Apc0015/Research-Compass-GNN.git
cd Research-Compass-GNN

# Create conda environment
conda create -n research_compass python=3.11 -y
conda activate research_compass

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (CPU version)
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# For GPU (CUDA 11.8)
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Basic Usage

**1. Launch Gradio UI (Interactive Training)**
```bash
python scripts/launcher.py
# Access at http://localhost:7860
```

**Supported Upload Formats:**
- **Documents:** PDF, DOCX, TXT, Markdown, HTML, XML
- **Archives:** TAR, TAR.GZ, ZIP (batch upload multiple papers)
- **URLs:** arXiv links, DOI URLs, direct PDF downloads

**Example URLs:**
```
https://arxiv.org/abs/1706.03762
https://arxiv.org/pdf/2010.11929.pdf
10.1145/3292500.3330989
```

**2. Train Single Model**
```bash
# Train GCN on Cora dataset
python scripts/train_enhanced.py --model GCN --dataset Cora --epochs 100

# Train HAN (heterogeneous) on synthetic data
python scripts/train_enhanced.py --model HAN --dataset synthetic --epochs 50

# Train R-GCN (relational) with mini-batch
python scripts/train_enhanced.py --model RGCN --dataset Cora --minibatch
```

**3. Compare All Models**
```bash
# Compare on Cora dataset
python scripts/compare_all_models.py --dataset Cora

# Compare on synthetic citation network
python scripts/compare_all_models.py --dataset synthetic --size 1000
```

**4. Run Verification Tests**
```bash
# Verify HAN implementation
python tests/verify_han.py

# Verify R-GCN implementation
python tests/verify_rgcn.py
```

---

## 🏗️ Architecture

### GNN Models Implemented

| Model | Type | Key Features |
|-------|------|--------------|
| **GCN** | Homogeneous | Graph convolution with spectral filtering |
| **GAT** | Homogeneous | Multi-head attention mechanism |
| **GraphSAGE** | Homogeneous | Inductive learning with sampling |
| **Graph Transformer** | Homogeneous | Full attention over graph structure |
| **HAN** | Heterogeneous | Hierarchical attention for multi-relational graphs |
| **R-GCN** | Relational | Relation-specific transformations with basis decomposition |

### Model Details

**Heterogeneous Attention Network (HAN)**
- 4 node types: paper, author, venue, topic
- 7 edge types: cites, written_by, published_in, belongs_to, writes, publishes, contains
- Node-level + semantic-level attention
- Ideal for: Multi-relational citation networks

**Relational GCN (R-GCN)**
- 4 citation types: EXTENDS, METHODOLOGY, BACKGROUND, COMPARISON
- Basis-decomposition for parameter efficiency
- Heuristic-based citation type classification
- Ideal for: Citation type-aware analysis

---

## 📂 Project Structure

```
Research-Compass-GNN/
├── scripts/                 # Executable scripts
│   ├── launcher.py         # Gradio UI application
│   ├── train_enhanced.py   # Enhanced training script
│   └── compare_all_models.py  # Model comparison
│
├── models/                  # GNN model implementations
│   ├── gcn.py              # Graph Convolutional Network
│   ├── gat.py              # Graph Attention Network
│   ├── graphsage.py        # GraphSAGE
│   ├── graph_transformer.py # Graph Transformer
│   ├── han.py              # Heterogeneous Attention Network
│   └── rgcn.py             # Relational GCN
│
├── data/                    # Data processing utilities
│   ├── dataset_utils.py    # Dataset loading and preprocessing
│   ├── heterogeneous_graph_builder.py  # HAN graph construction
│   ├── citation_type_classifier.py     # R-GCN edge typing
│   └── multi_format_processor.py       # Multi-format document processing
│
├── training/                # Training infrastructure
│   ├── trainer.py          # Base and specialized trainers
│   └── batch_training.py   # Mini-batch training
│
├── evaluation/              # Evaluation metrics
│   ├── metrics.py          # Node classification, link prediction
│   └── visualizations.py   # Performance plots
│
├── visualization/           # Visualization tools
│   └── attention_viz.py    # Attention weight analysis
│
├── analysis/                # Advanced analysis
│   └── temporal_analysis.py  # Citation trend analysis
│
├── baselines/               # Baseline models
│   ├── traditional_ml.py   # Logistic, RF, MLP
│   └── graph_baselines.py  # Label Propagation, Node2Vec
│
├── experiments/             # Research experiments
│   └── ablation_studies.py # Ablation study framework
│
├── config/                  # Configuration management
│   └── settings.py         # YAML config loader
│
├── utils/                   # Utility functions
│   ├── logger.py           # Training logger
│   └── checkpoint.py       # Model checkpointing
│
├── tests/                   # Test suite
│   ├── verify_han.py       # HAN verification
│   └── verify_rgcn.py      # R-GCN verification
│
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md     # System architecture
│   ├── USAGE_GUIDE.md      # Usage instructions
│   ├── ENHANCEMENTS.md     # Feature changelog
│   ├── MULTI_FORMAT_UPLOAD.md  # Multi-format upload guide
│   └── QUICK_REFERENCE_UPLOAD.md  # Quick reference guide
│
├── examples/                # Example scripts
│   └── demo_multi_format_upload.py  # Multi-format upload demo
│
├── notebooks/               # Jupyter notebooks
│   └── real_dataset_benchmark.ipynb  # Benchmark notebook
│
├── config.yaml              # Main configuration file
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🎓 Supported Datasets

### Standard Benchmarks (PyTorch Geometric)
- **Cora:** 2,708 papers, 7 classes, 5,429 citations
- **CiteSeer:** 3,327 papers, 6 classes, 4,732 citations
- **PubMed:** 19,717 papers, 3 classes, 44,338 citations

### Synthetic Citation Networks
- Configurable size (100-10,000+ papers)
- Temporal constraints (papers cite older papers)
- Topic-based communities
- Realistic citation patterns

### arXiv Papers Collection
- **10 foundational GNN papers** (2017-2020)
- Includes: GCN, GAT, GraphSAGE, HAN, R-GCN, GIN, Graph Transformers, and more
- All papers from arXiv.org (open access)
- Location: `datasets/arxiv_papers/`
- Total size: 11M
- See `datasets/DATASET_COLLECTION_REPORT.md` for details
- Upload instructions: `datasets/HOW_TO_UPLOAD.txt`

---

## 📊 Usage Examples

### Example 1: Train and Evaluate GCN

```python
from torch_geometric.datasets import Planetoid
from models import GCNModel
from training.trainer import GCNTrainer
import torch.optim as optim

# Load dataset
data = Planetoid(root='/tmp/Cora', name='Cora')[0]

# Create model
model = GCNModel(input_dim=data.x.shape[1], output_dim=7)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
trainer = GCNTrainer(model, optimizer)
for epoch in range(100):
    metrics = trainer.train_epoch(data)
    print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}")

# Evaluate
val_metrics = trainer.validate(data)
print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
```

### Example 2: Train HAN on Heterogeneous Graph

```python
from data import convert_to_heterogeneous
from models import create_han_model
from training.trainer import HANTrainer

# Convert to heterogeneous graph
hetero_data = convert_to_heterogeneous(data, num_venues=15)

# Create HAN model
model = create_han_model(hetero_data, hidden_dim=128, num_heads=8)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
trainer = HANTrainer(model, optimizer, target_node_type='paper')
metrics = trainer.train_epoch(hetero_data)
```

### Example 3: Compare All Models

```python
# From command line
python scripts/compare_all_models.py --dataset Cora --epochs 100

# Generates comprehensive report:
# - Performance comparison table
# - Training curves for all models
# - Confusion matrices
# - Per-class accuracy breakdown
```

### Example 4: Configuration Management

```python
from config import load_config, get_model_config, get_training_config

# Load configuration
load_config()

# Get model-specific config
han_config = get_model_config('han')
print(f"Hidden dim: {han_config['hidden_dim']}")  # 128

# Get training config
training = get_training_config()
print(f"Epochs: {training['epochs']}")  # 100
print(f"Learning rate: {training['learning_rate']}")  # 0.01
```

### Example 5: Logging and Checkpointing

```python
from utils import TrainingLogger, ModelCheckpoint

# Setup logging
logger = TrainingLogger('logs/experiment.log')
logger.log_training_start('GCN', num_params=50000, device='cuda')

# Setup checkpointing
checkpoint = ModelCheckpoint(
    checkpoint_dir='./checkpoints',
    monitor='val_acc',
    mode='max',
    keep_top_k=3
)

# Training loop
for epoch in range(100):
    # Train
    train_metrics = trainer.train_epoch(data)
    val_metrics = trainer.validate(data)

    # Log
    logger.log_epoch(epoch, train_metrics['loss'],
                    train_metrics['accuracy'], val_metrics['accuracy'])

    # Checkpoint
    metrics = {'val_acc': val_metrics['accuracy']}
    checkpoint.save(model, epoch, metrics, optimizer)

logger.log_training_complete(total_time, best_val_acc)
```

---

## 🔬 Advanced Features

### 1. Temporal Analysis

```python
from analysis import TemporalAnalyzer

# Create analyzer
analyzer = TemporalAnalyzer()
analyzer.add_temporal_data(data, years=publication_years)

# Identify emerging topics
emerging = analyzer.identify_emerging_topics(lookback_years=3)

# Analyze citation velocity
velocity = analyzer.analyze_citation_velocity(node_idx=42)
```

### 2. Attention Visualization

```python
from visualization import AttentionVisualizer

# Create visualizer
viz = AttentionVisualizer(model, data)

# Get attention weights
attention_weights = viz.get_attention_weights(layer=0, head=0)

# Create heatmap
fig = viz.create_attention_heatmap(attention_weights, top_k=20)
fig.savefig('attention_heatmap.png')
```

### 3. Ablation Studies

```python
from experiments import AblationStudy

# Run ablation study
study = AblationStudy(model='GAT', dataset='Cora')
results = study.run_ablation(
    components=['attention', 'normalization', 'residual']
)

# Results show impact of each component
study.plot_results()
```

### 4. Baseline Comparison

```python
from baselines import compare_with_baselines

# Compare GNN with traditional ML
results = compare_with_baselines(
    data=data,
    gnn_model='GCN',
    baselines=['logistic', 'rf', 'mlp']
)

# Print comparison table
print_comparison_table(results)
```

---

## 🎨 Gradio UI Features

Launch the interactive interface:
```bash
python scripts/launcher.py
```

**Available Tabs:**
1. **Real Data Training** - Train models on real citation networks with multi-format upload
2. **Evaluation Metrics** - Comprehensive metrics analysis
3. **Attention Visualization** - Attention weight heatmaps
4. **Temporal Analysis** - Citation trends and evolution
5. **About** - Project information

**Upload Features:**
- **Multiple Formats:** PDF, DOCX, TXT, Markdown, HTML, XML
- **Batch Upload:** TAR, TAR.GZ, ZIP archives containing multiple papers
- **Web URLs:** Direct download from arXiv, DOI resolvers, and academic repositories
- **Mixed Input:** Combine uploaded files and URLs in a single session

**Training Features:**
- Interactive model selection (GCN, GAT, GraphTransformer, HAN, R-GCN)
- Real-time training progress
- Live accuracy curves
- Confusion matrix visualization
- Attention pattern analysis
- Temporal trend charts

**Supported URL Sources:**
- arXiv.org (abstracts and PDFs)
- DOI resolvers (doi.org, dx.doi.org)
- ACL Anthology
- OpenReview.net
- NeurIPS/ICML proceedings
- Direct PDF links

---

## ⚙️ Configuration

All settings centralized in `config.yaml`:

```yaml
# Upload configuration
upload:
  formats:
    pdf: ['.pdf']
    text: ['.txt', '.md', '.text']
    docx: ['.docx', '.doc']
    html: ['.html', '.htm']
    xml: ['.xml']
    archive: ['.tar', '.tar.gz', '.tgz', '.zip']
  
  max_file_size: 104857600  # 100MB
  max_archive_files: 200
  
  allowed_domains:
    - 'arxiv.org'
    - 'doi.org'
    - 'aclanthology.org'
    - 'openreview.net'

# Model configuration
models:
  gcn:
    hidden_dim: 128
    num_layers: 3
    dropout: 0.5

  han:
    hidden_dim: 128
    num_heads: 8
    dropout: 0.3

training:
  epochs: 100
  learning_rate: 0.01
  weight_decay: 5e-4
  patience: 10

paths:
  data: "./data"
  models: "./checkpoints"
  results: "./results"
  logs: "./logs"
```

Modify `config.yaml` to customize behavior without code changes.

---

## 📈 Performance Benchmarks

### Node Classification (Test Accuracy)

| Model | Cora | CiteSeer | PubMed |
|-------|------|----------|--------|
| **GCN** | 81.5% | 70.3% | 79.0% |
| **GAT** | 83.0% | 72.5% | 79.0% |
| **GraphSAGE** | 80.3% | 69.8% | 78.5% |
| **Graph Transformer** | 82.1% | 71.0% | 78.2% |
| **HAN*** | 82.5% | 71.8% | - |
| **R-GCN*** | 81.0% | 70.5% | - |

*HAN and R-GCN results on converted heterogeneous/relational graphs

---

## 📤 Multi-Format Upload Guide

### Supported File Formats

| Format | Extensions | Description |
|--------|------------|-------------|
| **PDF** | `.pdf` | Research papers in PDF format |
| **Text** | `.txt`, `.md`, `.text` | Plain text and Markdown documents |
| **Word** | `.docx`, `.doc` | Microsoft Word documents |
| **Web** | `.html`, `.htm` | HTML documents |
| **XML** | `.xml` | XML formatted documents |
| **Archives** | `.tar`, `.tar.gz`, `.tgz`, `.zip` | Compressed archives (batch upload) |

### URL Download Support

**Supported Sources:**
- **arXiv:** `https://arxiv.org/abs/1706.03762` or `https://arxiv.org/pdf/2010.11929.pdf`
- **DOI:** `https://doi.org/10.1145/3292500.3330989` or `10.1145/3292500.3330989`
- **Direct PDFs:** Any direct link to PDF files from trusted domains
- **Academic Sites:** ACL Anthology, OpenReview, NeurIPS/ICML proceedings

**URL Input Formats:**
```
# One per line
https://arxiv.org/abs/1706.03762
https://arxiv.org/pdf/2010.11929.pdf

# Comma-separated
https://arxiv.org/abs/1706.03762, https://arxiv.org/pdf/2010.11929.pdf

# DOI format
10.1145/3292500.3330989
doi.org/10.1145/3292500.3330989
```

### Archive Upload (Batch Processing)

Upload TAR, TAR.GZ, or ZIP archives containing multiple papers:

**Example Archive Structure:**
```
papers.tar.gz
├── paper1.pdf
├── paper2.pdf
├── paper3.docx
├── subfolder/
│   ├── paper4.pdf
│   └── paper5.txt
└── paper6.html
```

**Features:**
- Automatically extracts all supported files
- Maintains folder structure metadata
- Limits: 200 files per archive, 100MB max file size
- Supports nested archives

### Usage Examples

**Example 1: Upload Mixed Formats**
1. Upload: `paper1.pdf`, `paper2.docx`, `notes.txt`
2. Click "Process Papers & Build Graph"
3. System automatically detects formats and extracts text

**Example 2: Download from arXiv**
1. Enter URLs in text box:
   ```
   https://arxiv.org/abs/1706.03762
   https://arxiv.org/abs/2010.11929
   ```
2. Click "Process Papers & Build Graph"
3. Papers are downloaded, metadata extracted, and graph built

**Example 3: Batch Upload with Archive**
1. Create `papers.zip` with 10 PDF files
2. Upload `papers.zip`
3. System extracts all PDFs and processes them

**Example 4: Combined Upload**
1. Upload files: `local_paper.pdf`, `papers.tar.gz`
2. Add URLs: `https://arxiv.org/abs/1706.03762`
3. Process all together in one graph

### Configuration

Customize upload settings in `config.yaml`:

```yaml
upload:
  max_file_size: 104857600      # 100MB per file
  max_archive_files: 200        # Max files per archive
  download_timeout: 30          # URL download timeout (seconds)
  
  allowed_domains:
    - 'arxiv.org'
    - 'doi.org'
    - 'aclanthology.org'
    - 'openreview.net'
```

---

## 🧪 Testing

```bash
# Run all verification tests
python tests/verify_han.py
python tests/verify_rgcn.py

# Expected output:
# ✅ HAN model creation: PASSED
# ✅ Forward pass: PASSED
# ✅ Training step: PASSED
# ✅ Validation: PASSED
# ✅ Attention weights: PASSED

# Run multi-format upload demo
python examples/demo_multi_format_upload.py

# Test multi-format processing
# Shows: Text extraction, URL detection, supported formats
```

---

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design and model architectures
- **[USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** - Detailed usage instructions
- **[ENHANCEMENTS.md](docs/ENHANCEMENTS.md)** - Feature changelog
- **[MULTI_FORMAT_UPLOAD.md](docs/MULTI_FORMAT_UPLOAD.md)** - Complete multi-format upload guide
- **[QUICK_REFERENCE_UPLOAD.md](docs/QUICK_REFERENCE_UPLOAD.md)** - Quick reference for uploads

---

## 🐛 Troubleshooting

### Common Issues

**1. PyTorch Geometric Installation**
```bash
# If pip install fails, install from source
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**2. CUDA Out of Memory**
```python
# Use smaller batch size or enable mini-batch training
python scripts/train_enhanced.py --model GCN --minibatch --batch_size 32
```

**3. Config Not Found**
```python
# Ensure config.yaml is in project root
from config import load_config
load_config('path/to/config.yaml')
```

**4. URL Download Fails**
```bash
# Check internet connection
# Verify URL is from allowed domain (see config.yaml)
# Try direct arXiv PDF link instead of abstract page
```

**5. Archive Extraction Fails**
```bash
# Check archive contains < 200 files
# Verify archive is not corrupted
# Check individual files are < 100MB
```

**6. DOCX Support Missing**
```bash
# Install python-docx if not already installed
pip install python-docx
```

---

## 🛣️ Roadmap

- [ ] Add Graph Isomorphism Network (GIN)
- [ ] Implement GraphGPS (GPS layer)
- [ ] Add temporal GNN models (TGAT, DySAT)
- [ ] Support for knowledge graph embeddings
- [ ] Integration with Weights & Biases
- [ ] Distributed training support
- [ ] Explainability features (GNNExplainer)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- PyTorch Geometric team for the excellent graph learning library
- Research papers that inspired the implementations:
  - **GCN:** Kipf & Welling (2017)
  - **GAT:** Veličković et al. (2018)
  - **GraphSAGE:** Hamilton et al. (2017)
  - **HAN:** Wang et al. (2019)
  - **R-GCN:** Schlichtkrull et al. (2018)

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Built with ❤️ for Graph Neural Network Research**
