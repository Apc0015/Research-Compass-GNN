# 🧭 Research Compass

> **Advanced AI-powered research exploration platform with Graph Neural Networks**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GraphRAG](https://img.shields.io/badge/type-GraphRAG-green.svg)]()

**Research Compass** is a cutting-edge research platform that combines knowledge graphs, vector embeddings, and Graph Neural Networks (GNNs) to provide intelligent paper recommendations, citation analysis, and research insights.

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd "Research Compass"
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Setup LLM (Ollama recommended)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2
ollama serve

# 4. Configure
cp .env.example .env
# Edit .env with your settings

# 5. Launch
python launcher.py
```

Access at **http://localhost:7860**

## ⚙️ Configuration

Create `.env` file in project root:

```bash
# LLM Configuration
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=1000

# Local LLM URLs
OLLAMA_BASE_URL=http://localhost:11434
LMSTUDIO_BASE_URL=http://localhost:1234

# Cloud LLM API Keys (optional)
OPENROUTER_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Neo4j Configuration (optional)
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Embedding Model
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

# Cache Settings
CACHE_DIR=data/cache
MAX_CACHE_ITEMS=1000
DEFAULT_CACHE_TTL=3600

# Server Settings
GRADIO_PORT=7860
```

## 🎯 Key Features

### 🧠 **GNN-Powered Core**
- **Graph Neural Networks**: Advanced GNN models for node classification, link prediction
- **Temporal Analysis**: Research evolution tracking and trend prediction
- **Neural Recommendations**: GNN-based personalized paper suggestions
- **Graph Search**: Semantic + structural search with graph context

### 📚 **Document Processing**
- **Multi-Format Support**: PDF, DOCX, TXT, Markdown processing
- **Knowledge Graph Construction**: Automated entity and relationship extraction
- **Batch Processing**: Handle multiple files simultaneously
- **Web URL Import**: Direct processing from arXiv and other sources

### 🎨 **Visualization & Analytics**
- **Interactive Networks**: Clickable citation and collaboration graphs
- **Attention Visualization**: See how GNN models make decisions
- **Citation Analysis**: Disruption index, sleeping beauty detection
- **Temporal Charts**: Research trends over time

### 🤖 **User Experience**
- **Streaming Responses**: Real-time word-by-word answer display
- **Intelligent Caching**: 10-100x performance optimization
- **Settings Panel**: User-friendly configuration with connection testing
- **Multi-Provider LLM**: Ollama, LM Studio, OpenRouter, OpenAI

## 🏗️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.11+ |
| **Graph Database** | Neo4j 5.0+ / NetworkX |
| **Vector Search** | FAISS |
| **NLP/NER** | spaCy 3.7+ |
| **Embeddings** | Sentence Transformers |
| **Web UI** | Gradio 4.0+ |
| **GNN Framework** | PyTorch Geometric |
| **Visualization** | Pyvis, Plotly, NetworkX |

## 📂 Project Structure

```
Research Compass/
├── launcher.py                    # Main application launcher
├── requirements.txt               # Python dependencies
├── config/                      # Configuration files
│   ├── academic_config.yaml
│   └── settings.py
├── src/graphrag/                 # Core GNN system
│   ├── core/                     # System core
│   ├── analytics/                 # Analytics modules
│   ├── ml/                        # Machine learning
│   ├── visualization/             # Graph visualization
│   └── ui/                        # User interface
├── data/                         # Data storage
│   ├── docs/                     # Research papers
│   ├── indices/                   # Search indices
│   └── cache/                     # Application cache
└── output/                        # Generated outputs
```

## 🔧 How to Use Research Compass

### Step 1: Launch the Application

```bash
# Start the application
python launcher.py

# The app will open at http://localhost:7860
```

### Step 2: Configure Your Settings

1. **Open the Settings tab** in the web interface
2. **Configure LLM**:
   - Select your provider (Ollama recommended)
   - Enter base URL or API key
   - Click "Refresh Available Models"
   - Select your model (e.g., llama3.2)
   - Click "Test Connection"
   - Click "Save Configuration"
3. **Configure Neo4j** (optional):
   - Enter URI, username, password
   - Click "Test Connection"
   - Click "Save Configuration"

### Step 3: Upload and Process Documents

1. **Go to Upload & Process tab**
2. **Upload Files**:
   - Click "Upload Files"
   - Select PDF, DOCX, TXT, or MD files
   - Enable "Build Knowledge Graph"
   - Enable "Extract Metadata"
   - Click "Process All"
3. **Or Add Web URLs**:
   - Enter URLs (one per line):
     ```
     https://arxiv.org/abs/1706.03762
     https://arxiv.org/abs/1810.04805
     ```
   - Click "Process URLs"

### Step 4: Ask Questions (Research Assistant)

1. **Go to Research Assistant tab**
2. **Enter your question**: "What are the main innovations in transformer architecture?"
3. **Enable options**:
   - ✅ Use Knowledge Graph (better context)
   - ✅ Use GNN Reasoning (advanced analysis)
   - ✅ Stream Response (real-time display)
   - ✅ Use Cache (faster repeated queries)
4. **Adjust Top-K sources** (default: 5)
5. **Click "Ask Question"**

### Step 5: Explore Advanced Features

#### 📊 Temporal Analysis
- **Topic Evolution**: Enter topic name, select time granularity
- **Citation Velocity**: Enter paper title to see citation growth
- **H-Index Timeline**: Enter author name to track impact
- **Emerging Topics**: Set year and acceleration threshold

#### 💡 Personalized Recommendations
- Enter your research interests (comma-separated)
- Add papers you've read/liked
- Adjust diversity slider (0 = similar, 1 = exploratory)
- Get tailored paper and author suggestions

#### 🕸️ Citation Explorer
- Enter paper title
- Interactive network appears - click nodes to expand
- Trace idea propagation between papers
- Export visualization as HTML

#### 🔬 Discovery Engine
- Enter paper title
- Find similar papers across disciplines
- Explore cross-disciplinary connections
- Toggle exploration mode for surprise papers

#### 📈 Advanced Metrics
- **Disruption Index**: Measure research impact
- **Sleeping Beauty**: Find delayed-recognition papers
- **Citation Cascades**: Track multi-generation influence
- **Citation Patterns**: Analyze diversity and concentration

### Step 6: Manage Performance

#### Cache Management
- Go to Cache Management tab
- View statistics (hit rate, memory usage)
- Clear expired entries or entire cache
- Monitor performance improvements

### Pro Tips for Best Results

🎯 **For Better Answers**:
- Use specific questions about your uploaded papers
- Enable "Use Knowledge Graph" for context-aware responses
- Enable "Use GNN Reasoning" for advanced analysis

🚀 **For Faster Performance**:
- Enable caching for repeated queries
- Reduce Top-K value for faster processing
- Use Neo4j for large document sets

🔍 **For Better Recommendations**:
- Add your research interests and reading history
- Adjust diversity slider for exploration vs. similarity
- Use the discovery engine for unexpected connections

### Example Workflows

#### Research Literature Review
1. Upload 10-20 papers in your field
2. Ask: "What are the main themes in these papers?"
3. Use Temporal Analysis to see topic evolution
4. Get recommendations for related papers

#### Finding Research Gaps
1. Upload papers in your research area
2. Use Discovery Engine to find cross-disciplinary connections
3. Analyze citation patterns with Advanced Metrics
4. Explore emerging topics in Temporal Analysis

#### Tracking Research Impact
1. Upload your published papers
2. Use Citation Explorer to see citation networks
3. Check H-Index Timeline for author impact
4. Analyze disruption index for paper influence

## 🐛 Troubleshooting

### Common Issues

**Ollama Connection Failed**
```bash
ollama list
ollama serve
# Verify: http://localhost:11434
```

**Neo4j Connection Issues**
- Verify URI format: `neo4j://host:port`
- Check credentials in Neo4j Browser first
- Falls back to in-memory NetworkX if unavailable

**Performance Issues**
- Enable caching in Research Assistant tab
- Reduce Top-K value for faster queries
- Use Neo4j for large graphs (faster than NetworkX)

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Neo4j** for graph database technology
- **Meta AI** for FAISS vector search
- **PyTorch Geometric** for GNN framework
- **spaCy** for NLP capabilities
- **Gradio** for the excellent UI framework

---

**Made with dedication for the AI and Research community** 🚀