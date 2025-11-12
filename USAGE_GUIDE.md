# Usage Guide - Research Compass GNN

## Quick Start Guide

### 1. Running the Real Dataset Benchmark Notebook

The `real_dataset_benchmark.ipynb` notebook allows you to evaluate GNN models on standard citation network benchmarks.

**Steps:**

1. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

2. **Launch Jupyter:**
```bash
jupyter notebook real_dataset_benchmark.ipynb
```

3. **Run all cells** (Runtime: ~5-10 minutes)
   - Section 1: Setup & Imports
   - Section 2: Load datasets (Cora, CiteSeer, PubMed)
   - Section 3: Define GNN models
   - Section 4: Training functions
   - Section 5: Train on all benchmarks
   - Section 6: Visualizations
   - Section 7: Key findings

**Expected Output:**
- Training curves for all datasets
- Confusion matrices
- Performance comparison with published results
- Test accuracy: 70-82% depending on dataset
- PNG files: `benchmark_training_curves.png`, `benchmark_confusion_matrices.png`, etc.

---

### 2. Using the Gradio UI for Real Data Training

The `launcher.py` provides an interactive web interface for training GNNs on your own research papers.

**Steps:**

1. **Launch the application:**
```bash
python launcher.py
```

2. **Open your browser:**
   - Navigate to `http://localhost:7860`

3. **Upload and Process PDFs:**
   - Go to "📄 Real Data Training" tab
   - Click "Upload PDF Files"
   - Select multiple PDF files (research papers)
   - Enable options:
     - ✅ Extract citations from PDFs
     - ✅ Build knowledge graph automatically
     - ✅ Extract metadata
   - Click "🔄 Process Papers & Build Graph"
   - Wait for processing to complete

4. **Train GNN Model:**
   - Select model type: **GCN** (recommended for beginners)
   - Set epochs: **50** (default)
   - Set learning rate: **0.01** (default)
   - Select task: **Node Classification**
   - Click "🚀 Train GNN Model"
   - Watch live training progress

5. **Make Predictions:**
   - Select a paper from dropdown
   - Choose prediction type: **Category Classification** or **Link Prediction**
   - Set Top-K: **5** (default)
   - Click "🔮 Predict"
   - View results with confidence scores

6. **Export Results:**
   - Click "💾 Export Results"
   - Download trained model (.pt file)
   - Download predictions (CSV file)

---

### 3. Running the Comparison Study

The `comparison_study.py` script trains GNN models on a synthetic citation network.

**Steps:**

1. **Run the script:**
```bash
python comparison_study.py 200 50
# Arguments: [num_papers] [epochs]
```

2. **Output:**
   - `comparison_results/comparison_table.csv`
   - `comparison_results/detailed_results.json`
   - `comparison_results/training_curves.png`
   - `comparison_results/performance_comparison.png`
   - `comparison_results/model_complexity.png`

3. **Expected Results:**
   - GCN: ~87.5% test accuracy
   - GAT: ~85% test accuracy
   - Transformer: ~0.95 cosine similarity

---

## Model Selection Guide

### When to use each model:

#### GCN (Graph Convolutional Network)
- ✅ **Best for:** Fast training, large graphs, general-purpose
- ✅ **Pros:** Fast, efficient, interpretable
- ⚠️ **Cons:** May miss complex patterns
- 📊 **Accuracy:** 70-87% on benchmarks
- ⏱️ **Speed:** Fastest (1-2 minutes for 200 papers)

#### GAT (Graph Attention Network)
- ✅ **Best for:** When you need to understand which citations matter most
- ✅ **Pros:** Learns attention weights, better accuracy
- ⚠️ **Cons:** Slower training, more memory
- 📊 **Accuracy:** 72-83% on benchmarks
- ⏱️ **Speed:** Medium (2-3 minutes for 200 papers)

#### Graph Transformer
- ✅ **Best for:** Long-range dependencies, complex patterns
- ✅ **Pros:** State-of-the-art architecture, global context
- ⚠️ **Cons:** Slowest, most memory-intensive
- 📊 **Accuracy:** 69-82% on benchmarks
- ⏱️ **Speed:** Slowest (3-5 minutes for 200 papers)

---

## Hyperparameter Tuning Guide

### Epochs
- **Small datasets (<100 papers):** 50-100 epochs
- **Medium datasets (100-1000 papers):** 100-200 epochs
- **Large datasets (>1000 papers):** 50-100 epochs (converges faster)

### Learning Rate
- **Default:** 0.01 (good for most cases)
- **If model doesn't converge:** Try 0.001 (slower but more stable)
- **If training is slow:** Try 0.05 (faster but may be unstable)

### Hidden Dimensions
- **GCN:** 128 (default), increase to 256 for large datasets
- **GAT:** 128 (default), fewer parameters due to multi-head attention
- **Transformer:** 128 (default), increase to 256 for complex patterns

### Dropout
- **Default:** 0.5 for GCN, 0.3 for GAT, 0.1 for Transformer
- **If overfitting:** Increase dropout to 0.6-0.7
- **If underfitting:** Decrease dropout to 0.2-0.3

---

## Troubleshooting

### "PyTorch Geometric not installed"
```bash
pip install torch torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### "CUDA out of memory"
- Reduce batch size
- Use CPU instead: Set `device = torch.device('cpu')`
- Reduce hidden dimensions to 64

### "No graph data available"
- Make sure to click "🔄 Process Papers & Build Graph" first
- Check that PDFs were uploaded successfully
- Verify at least 2 PDFs are uploaded

### "Training accuracy not improving"
- Increase epochs to 100-200
- Try different learning rate (0.001 or 0.05)
- Check if graph has enough edges (citations)
- Try a different model (GAT often works better)

### "Predictions are random"
- Make sure model is trained (test accuracy > 0.5)
- Increase training epochs
- Check that graph has meaningful structure

---

## Example Workflows

### Workflow 1: Evaluating Models on Benchmarks
1. Open `real_dataset_benchmark.ipynb`
2. Run all cells
3. Compare your results with published benchmarks
4. Read key findings in Section 7

### Workflow 2: Training on Your Papers
1. Launch `python launcher.py`
2. Go to "📄 Real Data Training" tab
3. Upload 10-20 research PDFs from your field
4. Process papers with all options enabled
5. Train GCN model (50 epochs)
6. Make predictions on papers
7. Export results

### Workflow 3: Quick Demo
1. Launch `python launcher.py`
2. Go to "🏠 Welcome" tab
3. Click "🎮 Run Quick Demo"
4. See GNN training on synthetic data

---

## Performance Benchmarks

### Real Dataset Benchmark Results

| Dataset | Papers | Citations | Our GCN | Published GCN | Difference |
|---------|--------|-----------|---------|---------------|------------|
| Cora | 2,708 | 5,429 | ~0.815 | 0.815 | ±0.010 |
| CiteSeer | 3,327 | 4,732 | ~0.703 | 0.703 | ±0.015 |
| PubMed | 19,717 | 44,338 | ~0.790 | 0.790 | ±0.010 |

### Synthetic Dataset Results

| Model | Test Accuracy | Training Time | Parameters |
|-------|---------------|---------------|------------|
| GCN | 87.5% | 15s | 98,565 |
| GAT | 85.0% | 25s | 265,221 |
| Transformer | 0.95 (similarity) | 20s | 328,832 |

---

## Tips for Best Results

### For Better Accuracy:
- Upload papers from the same research area
- Ensure PDFs have clear text (not scanned images)
- Train for more epochs (100-200)
- Try GAT model for attention-based learning

### For Faster Training:
- Use GCN model
- Reduce epochs to 20-30
- Use smaller hidden dimensions (64)
- Process fewer papers at once

### For Better Predictions:
- Train on at least 20-30 papers
- Ensure good citation coverage between papers
- Use higher Top-K values (10-20)
- Check that graph has good connectivity

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{research_compass_gnn,
  title = {Research Compass GNN: Graph Neural Networks for Citation Analysis},
  author = {Research Compass Team},
  year = {2024},
  url = {https://github.com/Apc0015/Research-Compass-GNN}
}
```

---

## Support

For issues or questions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the notebook examples

**Happy researching!** 🚀
