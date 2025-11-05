# 🚀 Quick Reference: GNN Features in Research Compass

## 🎯 Your Question: "I can only upload data, can't figure out the rest"

## ✅ **Answer: Here's exactly what you can do now!**

---

## 📍 **NEW: Tab 2 - Graph & GNN Dashboard** ⭐

After uploading papers in Tab 1, go to **Tab 2** for full GNN capabilities:

---

### 1️⃣ **See Your Graph** (30 seconds)

```
Tab 2 → "📊 Graph Statistics" → Click "🔄 Refresh Statistics"
```

**You'll see:**
- How many papers you have
- How many citations
- How many authors
- Graph structure

---

### 2️⃣ **Visualize Your Graph** (1 minute)

```
Tab 2 → "🎨 Visualize Graph" → Click "🎨 Generate Visualization"
```

**You'll see:**
- Interactive network graph
- Papers connected by citations
- Click nodes to explore
- Drag to rearrange

**Tip:** Start with 50-100 nodes max for best performance

---

### 3️⃣ **Train a GNN Model** (2 minutes)

```
Tab 2 → "🤖 Train GNN Models"
```

**Settings (use defaults first):**
- Model Type: **GAT** (recommended)
- Task: **Link Prediction** (finds missing citations)
- Epochs: **50**
- Learning Rate: **0.01**

**Click:** "🚀 Start Training"

**Wait:** 1-3 minutes

**Result:** ✅ GNN model trained!

---

### 4️⃣ **Get GNN Predictions** (30 seconds)

```
Tab 2 → "🔮 GNN Predictions"
```

**Try this:**
- Prediction Type: **Link Prediction**
- Node ID: `[enter any paper title from your graph]`
- Top K: **10**

**Click:** "🔮 Get Predictions"

**Result:** Papers that should cite (or be cited by) your paper!

---

### 5️⃣ **Export Your Graph** (30 seconds)

```
Tab 2 → "💾 Export Graph"
```

- Format: **JSON** (or CSV)
- **Click:** "📥 Export Graph Data"

**Use for:** Backup, external analysis, sharing

---

## 🔥 **Complete 5-Minute Workflow**

### Minute 1-2: Upload
```
Tab 1 → Upload 5-10 PDFs → ✅ Build knowledge graph → Process All
```

### Minute 3: Visualize
```
Tab 2 → Visualize Graph → Generate Visualization
```

### Minute 4-5: Train GNN
```
Tab 2 → Train GNN Models → GAT + Link Prediction → Start Training
```

### Minute 6: Get Predictions
```
Tab 2 → GNN Predictions → Enter paper title → Get Predictions
```

**Done! You now have:**
- ✅ Knowledge graph built
- ✅ Graph visualized
- ✅ GNN model trained
- ✅ Predictions ready

---

## 🎓 **What Each Tab Does** (Simple Explanation)

| Tab | What It Does | Why Use It |
|-----|-------------|------------|
| **1: Upload** | Add papers to system | Start here! |
| **2: Graph & GNN** ⭐ | See graph, train GNN, get predictions | **NEW! Main GNN features** |
| **3: Research Assistant** | Ask questions about papers | Get answers with GNN reasoning |
| **4: Temporal** | See research trends over time | Track topic evolution |
| **5: Recommendations** | Get paper suggestions | Find what to read next |
| **6: Citation Explorer** | See citation networks | Visualize how papers connect |
| **7: Discovery** | Find related papers | Cross-field connections |
| **8: Metrics** | Advanced citation analysis | Paper impact metrics |
| **9: Cache** | Performance management | Speed up queries |
| **10: Settings** | Configure LLM, database | System configuration |

---

## 💡 **Pro Tips**

### Tip 1: Start with at least 10 papers
- Fewer papers = Limited graph
- 10-50 papers = Good for learning
- 50+ papers = Best GNN performance

### Tip 2: Use domain-specific papers
- All papers about "transformers" = Better
- Mixed topics = Works, but less accurate
- Same research area = Best GNN predictions

### Tip 3: Visualize before training
- Check your graph structure first
- Make sure papers are connected
- If no connections = Upload more related papers

### Tip 4: Enable GNN reasoning in queries
```
Tab 3 (Research Assistant) → ✅ Use GNN Reasoning
```
This makes answers more accurate using graph structure!

### Tip 5: Try different GNN models
- **GAT:** Best for most tasks
- **Transformer:** Best for large graphs (100+ papers)
- **Hetero:** Best if you care about authors + papers + topics
- **GCN:** Fastest, simplest

---

## 🐛 **Troubleshooting**

### "Graph is empty"
**Problem:** No papers uploaded yet
**Fix:** Go to Tab 1, upload papers, enable "Build knowledge graph"

### "GNN not available"
**Problem:** PyTorch Geometric not installed
**Fix:**
```bash
pip install torch torch-geometric
```

### "Visualization shows nothing"
**Problem:** Max nodes too low or no graph
**Fix:**
1. Check Tab 2 → Statistics first
2. If nodes > 0, increase "Max Nodes" slider
3. If nodes = 0, upload papers first

### "Training failed"
**Problem:** Not enough data
**Fix:** Upload at least 10 papers minimum

---

## 📊 **Visual Guide**

### The GNN Workflow:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. UPLOAD PAPERS (Tab 1)                                    │
│    Upload PDFs or URLs → Build knowledge graph              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. VISUALIZE GRAPH (Tab 2 → Visualize)                      │
│    See papers, authors, citations as network                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. TRAIN GNN MODEL (Tab 2 → Train GNN)                      │
│    Model learns patterns in your graph                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. GET PREDICTIONS (Tab 2 → GNN Predictions)                │
│    Find missing citations, similar papers                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌──────────────────────────────────────────────────────────────┐
│ 5. USE IN QUERIES (Tab 3 → Research Assistant)              │
│    Enable GNN Reasoning for better answers                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎯 **Your First Session Script**

Copy and follow this:

```bash
# 1. Start the app
python launcher.py
# Open: http://localhost:7860

# 2. Upload papers (Tab 1)
# - Upload 5-10 PDFs
# - ✅ Check "Build knowledge graph"
# - Click "Process All"
# - Wait 1-2 minutes

# 3. See your graph (Tab 2)
# - Sub-tab: "Graph Statistics"
# - Click "Refresh Statistics"
# - Note: How many papers and citations?

# 4. Visualize it (Tab 2)
# - Sub-tab: "Visualize Graph"
# - Max Nodes: 100
# - Click "Generate Visualization"
# - Explore: Click and drag nodes!

# 5. Train GNN (Tab 2)
# - Sub-tab: "Train GNN Models"
# - Model: GAT
# - Task: Link Prediction
# - Epochs: 50
# - Click "Start Training"
# - Wait 2-3 minutes

# 6. Get predictions (Tab 2)
# - Sub-tab: "GNN Predictions"
# - Type: Link Prediction
# - Enter a paper title from your graph
# - Click "Get Predictions"
# - See: Suggested related papers!

# 7. Ask a question (Tab 3)
# - Enter: "What are the main themes in these papers?"
# - ✅ Enable "Use GNN Reasoning"
# - Click "Ask Question"
# - See: Answer using graph structure!
```

---

## 📚 **Next Steps**

After your first session:

1. **Read full guide:** `GNN_WORKFLOW_GUIDE.md`
2. **Try presentations:** `presentations/GNN_Presentation.md`
3. **Explore advanced features:** Temporal analysis, recommendations
4. **Scale up:** Upload 50-100 papers for better GNN performance

---

## 🆘 **Still Stuck?**

### Quick help:

1. **Read this again** - Follow the 5-minute workflow
2. **Check full guide** - `GNN_WORKFLOW_GUIDE.md`
3. **View presentations** - `presentations/` folder
4. **Technical docs** - `TECHNICAL_DOCUMENTATION.md`

### Remember:
- **Tab 2** is your new GNN dashboard - everything is there!
- **Start simple** - Upload, visualize, train, predict
- **Experiment** - Try different models and settings
- **Scale up** - More papers = better results

---

**You're ready to use GNN features! 🚀**

**Start with Tab 2 and follow the 5-minute workflow above!**
