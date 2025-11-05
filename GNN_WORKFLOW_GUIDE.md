# 🧠 Research Compass - Complete GNN Workflow Guide

## 📋 Overview

This guide explains how to use Research Compass from a **Graph Neural Networks (GNN) perspective**, showing you the complete workflow from data upload to GNN predictions.

---

## 🚀 Quick Start Workflow

### Step 1: Upload Documents (Tab 1: Upload & Process)

1. **Launch the application:**
   ```bash
   cd "Research Compass"
   python launcher.py
   # Open: http://localhost:7860
   ```

2. **Go to Tab 1: "📤 Upload & Process"**

3. **Upload research papers:**
   - Click "Upload Files" → Select PDF/DOCX files
   - Or paste URLs (one per line):
     ```
     https://arxiv.org/abs/1706.03762
     https://arxiv.org/abs/1810.04805
     ```

4. **Enable graph building:**
   - ✅ Check "Extract metadata"
   - ✅ Check "Build knowledge graph"

5. **Click "🚀 Process All"**

**What happens behind the scenes:**
```
Your files → Extract text → Extract entities (papers, authors, topics)
           → Build knowledge graph (Neo4j/NetworkX)
           → Create embeddings (FAISS)
           → Ready for GNN processing!
```

---

### Step 2: Explore Your Knowledge Graph (Tab 2: Graph & GNN Dashboard) ⭐ NEW!

#### 2.1 View Graph Statistics

1. **Go to Tab 2: "🕸️ Graph & GNN Dashboard"**
2. **Select sub-tab: "📊 Graph Statistics"**
3. **Click "🔄 Refresh Statistics"**

**You'll see:**
- **Graph Size:** Total nodes and edges
- **Node Types:** Papers, Authors, Topics, Venues
- **Edge Types:** CITES, AUTHORED_BY, DISCUSSES, etc.
- **GNN Status:** Are GNN models available?
- **Academic Statistics:** Papers by year, author count

**Example output:**
```json
{
  "total_nodes": 150,
  "total_edges": 380,
  "node_types": {
    "Paper": 50,
    "Author": 75,
    "Topic": 20,
    "Venue": 5
  },
  "edge_types": {
    "CITES": 120,
    "AUTHORED_BY": 200,
    "DISCUSSES": 60
  }
}
```

---

#### 2.2 Visualize the Graph

1. **Select sub-tab: "🎨 Visualize Graph"**
2. **Adjust settings:**
   - **Maximum Nodes:** 100 (start small, increase later)
   - **Layout:** "spring" (most common) or "circular"
3. **Click "🎨 Generate Visualization"**

**You'll see an interactive graph where:**
- **Nodes** = Papers, Authors, Topics (different colors)
- **Edges** = Citations, Authorship, Topics
- **Click nodes** to see details
- **Drag nodes** to rearrange
- **Zoom** in/out with mouse wheel

**Colors:**
- 🔵 Blue = Papers
- 🟢 Green = Authors
- 🟡 Yellow = Topics
- 🟠 Orange = Venues

---

#### 2.3 Train GNN Models

1. **Select sub-tab: "🤖 Train GNN Models"**

2. **Choose your model:**
   - **GCN** (Graph Convolutional Network) - Fast, basic
   - **GAT** (Graph Attention) - **Recommended!** Learns importance
   - **Transformer** - Most advanced, best for large graphs
   - **Hetero** - For mixed node types (papers + authors)

3. **Choose your task:**
   - **Link Prediction** - Find missing citations (most useful!)
   - **Node Classification** - Categorize papers by topic
   - **Embedding** - Generate paper representations

4. **Set training parameters:**
   - **Epochs:** 50 (default) - More = better but slower
   - **Learning Rate:** 0.01 (default)

5. **Click "🚀 Start Training"**

**Training time:**
- 50 papers: ~1-2 minutes (CPU)
- 500 papers: ~5-10 minutes (CPU)
- With GPU: 5-10x faster

**What the model learns:**
- Citation patterns (which papers cite which)
- Author collaboration networks
- Topic relationships
- Paper importance scores

---

#### 2.4 Get GNN Predictions

After training, you can get predictions:

1. **Select sub-tab: "🔮 GNN Predictions"**

2. **Choose prediction type:**

   **A) Link Prediction** (Find missing citations):
   - Enter paper title: "Attention Is All You Need"
   - Top K: 10
   - Click "🔮 Get Predictions"
   - **Result:** Papers that should cite this one (or should be cited by it)

   **B) Similar Nodes** (Find related papers):
   - Enter paper title
   - **Result:** Papers with similar GNN embeddings (structurally similar in the graph)

   **C) Node Classification** (Categorize papers):
   - Enter paper title
   - **Result:** Predicted research area/topic

**Example predictions:**
```json
{
  "status": "success",
  "predictions": [
    {
      "paper": "BERT: Pre-training of Deep Bidirectional Transformers",
      "score": 0.92,
      "reason": "Similar citation patterns and shared authors"
    },
    {
      "paper": "GPT-2: Language Models are Unsupervised Multitask Learners",
      "score": 0.87,
      "reason": "Strong structural similarity in graph"
    }
  ]
}
```

---

#### 2.5 Export Your Graph

1. **Select sub-tab: "💾 Export Graph"**
2. **Choose format:**
   - **JSON** - For programming (Python, JavaScript)
   - **CSV** - For Excel/spreadsheets
3. **Click "📥 Export Graph Data"**

**Use cases:**
- Backup your knowledge graph
- Analyze in external tools (Gephi, Cytoscape)
- Share with collaborators

---

### Step 3: Use GNN for Research Questions (Tab 3: Research Assistant)

1. **Go to Tab 3: "🔍 Research Assistant"**

2. **Ask a question:**
   ```
   What are the main innovations in transformer architecture?
   ```

3. **Enable GNN features:**
   - ✅ **Use Knowledge Graph** - Adds citation context
   - ✅ **Use GNN Reasoning** - Uses trained GNN models
   - ✅ **Stream Response** - See answer word-by-word
   - ✅ **Use Cache** - Faster repeated queries

4. **Adjust Top-K Sources:** 5 (default)

5. **Click "Ask Question"**

**What GNN adds:**
- **Without GNN:** Finds papers by keyword similarity only
- **With GNN:** Also considers graph structure, citation patterns, and learned relationships
- **Result:** More accurate, context-aware answers

**Example reasoning trace:**
```
📚 Vector Search found: 3 papers (semantic match)
🕸️ Graph Search found: 5 papers (citation network)
🧠 GNN Predictions added: 2 papers (structural similarity)
---
Total sources: 10 papers
Confidence: 0.89
```

---

### Step 4: Advanced GNN Features

#### 4.1 Citation Explorer (Tab 6)

1. **Go to Tab 6: "🕸️ Citation Explorer"**
2. **Enter paper title:** "Attention Is All You Need"
3. **Set max depth:** 2 hops
4. **Click "Explore Citations"**

**Shows:**
- Papers that cite this paper
- Papers cited by this paper
- Citation chains (A → B → C)
- Click nodes to expand further

---

#### 4.2 Temporal Analysis (Tab 4)

See how GNN-identified research areas evolve:

1. **Go to Tab 4: "📊 Temporal Analysis"**
2. **Topic Evolution:**
   - Enter topic: "graph neural networks"
   - **See:** Paper count by year, growth trends
3. **H-Index Timeline:**
   - Enter author name
   - **See:** Career trajectory, impact over time

---

#### 4.3 Recommendations (Tab 5)

Get GNN-powered paper recommendations:

1. **Go to Tab 5: "💡 Recommendations"**
2. **Enter your interests:**
   ```
   deep learning, computer vision, transformers
   ```
3. **Add papers you've read:**
   ```
   Attention Is All You Need
   BERT
   ```
4. **Adjust diversity slider:** 0.5 (balanced)
5. **Click "Get Recommendations"**

**GNN advantage:**
- Traditional: Only looks at paper content
- **GNN:** Considers graph structure, citation patterns, and similar readers

---

## 🎯 Complete Example Workflow

### Scenario: You're researching Graph Neural Networks

#### Step 1: Upload papers (5 minutes)
```bash
# Upload these papers:
1. "Semi-Supervised Classification with GCN" (Kipf & Welling, 2016)
2. "Graph Attention Networks" (Veličković et al., 2017)
3. "How Powerful are Graph Neural Networks?" (Xu et al., 2018)
4. "DeeperGCN" (Li et al., 2020)
5. "Graph Transformer Networks" (Yun et al., 2019)
```

✅ **Result:** Knowledge graph built with 5 papers + extracted authors/topics

---

#### Step 2: Visualize the graph (1 minute)
- Go to Tab 2 → Visualize Graph
- Set max nodes: 50
- **See:** Citation network connecting these papers

---

#### Step 3: Train GNN model (2 minutes)
- Go to Tab 2 → Train GNN Models
- Model: GAT (Graph Attention)
- Task: Link Prediction
- Epochs: 50
- **Click:** Start Training

✅ **Result:** GNN model trained, ready for predictions

---

#### Step 4: Get missing citation predictions (30 seconds)
- Go to Tab 2 → GNN Predictions
- Type: Link Prediction
- Node ID: "Semi-Supervised Classification with GCN"
- **Result:** Suggests papers that should cite this foundational work

---

#### Step 5: Ask research question (1 minute)
- Go to Tab 3 → Research Assistant
- Question: "What are the key differences between GCN and GAT?"
- Enable: GNN Reasoning
- **Result:** Answer with citations, informed by graph structure

---

#### Step 6: Get recommendations (30 seconds)
- Go to Tab 5 → Recommendations
- Interests: "graph neural networks, attention mechanisms"
- **Result:** 10 recommended papers, ranked by GNN

---

## 📊 Understanding GNN vs Traditional Methods

| Feature | Traditional | With GNN |
|---------|------------|----------|
| **Paper Search** | Keyword match | Keyword + graph structure + learned patterns |
| **Citations** | Count only | Patterns + importance + context |
| **Recommendations** | Content similarity | Content + citations + collaborations + structure |
| **New Papers** | Low rank (0 citations) | Fair rank (based on structure) |
| **Cross-field** | Rarely found | Discovered via graph connections |

---

## 🔧 Troubleshooting

### "GNN manager not available"

**Problem:** PyTorch Geometric not installed

**Solution:**
```bash
pip install torch torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

---

### "Graph is empty"

**Problem:** No documents uploaded yet

**Solution:**
1. Go to Tab 1
2. Upload at least 3-5 papers
3. Enable "Build knowledge graph"
4. Process files

---

### "Training failed"

**Problem:** Not enough data

**Solution:**
- Upload at least 10 papers for meaningful GNN training
- More papers = better GNN performance
- Aim for 50+ papers for best results

---

### "Visualization shows nothing"

**Problem:** Max nodes set too low or graph not built

**Solution:**
1. Check Tab 2 → Graph Statistics first
2. If nodes = 0, upload documents
3. If nodes > 0, increase "Max Nodes" slider

---

## 💡 Best Practices

### 1. Start Small, Scale Up
- **First time:** Upload 5-10 papers
- **Experiment:** Try visualization, train a small GNN
- **Scale:** Once comfortable, upload 50-100 papers

### 2. Build Domain-Specific Graphs
- Upload papers from **one research area** for best GNN performance
- Example: All papers about "transformers" or all papers about "drug discovery"
- Mixed topics work, but domain-specific is better

### 3. Train Multiple Models
- Try different GNN types (GAT, Transformer, Hetero)
- Compare predictions
- GAT usually best for link prediction
- Transformer best for large graphs (100+ nodes)

### 4. Use GNN Predictions to Guide Research
- **Link Prediction:** Find papers you should read next
- **Similar Nodes:** Discover related work
- **Node Classification:** Understand research landscape

### 5. Combine Features
- Use **Graph Visualization** to understand structure
- Use **GNN Training** to learn patterns
- Use **Research Assistant** with GNN reasoning for questions
- Use **Recommendations** for reading list

---

## 🎓 Understanding GNN Output

### Link Prediction Scores

```json
{
  "paper": "Paper Title",
  "score": 0.87,  // 0-1, higher = more likely
  "reason": "..."
}
```

- **0.9-1.0:** Very strong connection predicted
- **0.7-0.9:** Strong connection
- **0.5-0.7:** Moderate connection
- **<0.5:** Weak connection

### Node Classification Confidence

```json
{
  "category": "Natural Language Processing",
  "confidence": 0.92
}
```

- **>0.8:** High confidence
- **0.6-0.8:** Medium confidence
- **<0.6:** Low confidence (model uncertain)

---

## 📈 Performance Tips

### Speed Up GNN Training
1. **Use GPU:** 5-10x faster
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Reduce epochs:** 20-30 epochs often sufficient

3. **Train overnight:** For large graphs (500+ papers), train with 100+ epochs

### Improve GNN Accuracy
1. **More data:** 50+ papers minimum
2. **Complete metadata:** Ensure papers have authors, years, abstracts
3. **Domain-specific:** Papers in same field
4. **Balanced data:** Mix of old and new papers

---

## 🚀 Next Steps

### Beginner
- ✅ Upload 5 papers
- ✅ Visualize graph
- ✅ View statistics
- ✅ Ask questions (without GNN first)

### Intermediate
- ✅ Train your first GNN model
- ✅ Get link predictions
- ✅ Use GNN reasoning in queries
- ✅ Export graph data

### Advanced
- ✅ Train multiple model types
- ✅ Compare GNN performance
- ✅ Integrate GNN into research workflow
- ✅ Build large domain-specific graphs (100+ papers)

---

## 📚 Additional Resources

- **Technical Documentation:** `/TECHNICAL_DOCUMENTATION.md`
- **Quick Start Guide:** `/QUICK_START_GUIDE.md`
- **Demo Script:** `/DEMO_SCRIPT.py`
- **Presentation Materials:** `/presentations/`

---

## 🆘 Getting Help

### Common Questions

**Q: How many papers do I need for GNN?**
A: Minimum 10, recommended 50+, optimal 100+

**Q: Which GNN model should I use?**
A: Start with GAT (Graph Attention), then try Transformer for large graphs

**Q: How long does training take?**
A: 50 papers, 50 epochs: ~2-5 minutes CPU, ~30 seconds GPU

**Q: Can I use GNN without Neo4j?**
A: Yes! System automatically falls back to NetworkX (in-memory graph)

**Q: Do I need GPU?**
A: No, but recommended for graphs >100 papers

---

**Happy Research! 🎉**

If you have questions, check the documentation or experiment with the UI - it's designed to be intuitive!
