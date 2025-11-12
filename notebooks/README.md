# Notebooks

This directory contains Jupyter notebooks for interactive exploration and demonstration.

## Available Notebooks

### 1. `real_dataset_benchmark.ipynb`
Comprehensive benchmark of GNN models on standard citation network datasets.

**Contents:**
- Dataset loading (Cora, CiteSeer, PubMed)
- Model training (GCN, GAT)
- Performance comparison with published benchmarks
- Visualization and analysis

**Usage:**
```bash
jupyter notebook notebooks/real_dataset_benchmark.ipynb
```

### 2. `comparison_study.ipynb`
Original comparison study of multiple GNN architectures.

**Contents:**
- Synthetic dataset generation
- Model training and evaluation
- Performance comparison
- Visualization

**Note:** This has been superseded by `compare_all_models.py` for production use, but kept for educational purposes.

### 3. `demo_for_professors.ipynb`
Interactive demonstration of GNN capabilities.

**Contents:**
- Quick model demonstrations
- Example predictions
- Visualization examples

**Note:** Kept for presentation/demo purposes.

## Running Notebooks

```bash
# Install Jupyter
pip install jupyter

# Launch Jupyter
jupyter notebook

# Navigate to notebooks/ directory in browser
```

## Integration with Main Codebase

All notebooks can import from the main codebase:

```python
import sys
sys.path.append('..')  # Add parent directory to path

from models import GCNModel, GATModel
from data import load_citation_dataset
from evaluation import EvaluationReportGenerator
```

## Best Practices

1. Use notebooks for **exploration and visualization**
2. Use scripts (`train_enhanced.py`, `compare_all_models.py`) for **production runs**
3. Keep notebooks **up to date** with main codebase
4. **Clear outputs** before committing to reduce file size

## Maintenance

These notebooks are maintained for:
- Educational demonstrations
- Interactive exploration
- Quick prototyping
- Result visualization

For reproducible experiments, use the Python scripts in the root directory.
