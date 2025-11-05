# Research Compass# Research Compass# 🧭 Research Compass



Advanced AI-powered research exploration platform with Graph Neural Networks



## Overview**Advanced AI-powered research exploration platform with Graph Neural Networks**> **Advanced AI-powered research exploration platform with Graph Neural Networks**



Research Compass combines knowledge graphs, vector embeddings, and Graph Neural Networks to provide intelligent paper recommendations, citation analysis, and research insights.



## Prerequisites[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)



- Python 3.11+[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

- Neo4j Aura account or Neo4j Desktop

- LLM provider (Ollama, OpenRouter, OpenAI, or LM Studio)[![GraphRAG](https://img.shields.io/badge/type-GraphRAG-green.svg)]()[![GraphRAG](https://img.shields.io/badge/type-GraphRAG-green.svg)]()



## Installation



```bashResearch Compass is a cutting-edge research platform that combines knowledge graphs, vector embeddings, and Graph Neural Networks (GNNs) to provide intelligent paper recommendations, citation analysis, and research insights.**Research Compass** is a cutting-edge research platform that combines knowledge graphs, vector embeddings, and Graph Neural Networks (GNNs) to provide intelligent paper recommendations, citation analysis, and research insights.

# Clone repository

git clone https://github.com/Apc0015/Research-Compass.git

cd "Research Compass"

## Quick Start## 🚀 Quick Start

# Create conda environment

conda create -n research_compass python=3.11 -y

conda activate research_compass

### Prerequisites```bash

# Install dependencies

pip install -r requirements.txt- Python 3.11+ (recommended: conda environment)# 1. Clone and setup

python -m spacy download en_core_web_sm

- Neo4j Aura account (free tier available) or Neo4j Desktopgit clone <repository-url>

# Configure environment

cp .env.example .env- LLM provider (Ollama, OpenRouter, OpenAI, or LM Studio)cd "Research Compass"

# Edit .env with your credentials

python -m venv .venv

# Launch application

python launcher.py### Installationsource .venv/bin/activate  # Windows: .venv\Scripts\activate

```



## Configuration

```bash# 2. Install dependencies

Edit `.env` file:

# Clone repositorypip install -r requirements.txt

```bash

# Neo4j Cloud Configurationgit clone <repository-url>python -m spacy download en_core_web_sm

NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io

NEO4J_USERNAME=neo4jcd "Research Compass"

NEO4J_PASSWORD=your_password

# 3. Setup LLM (Ollama recommended)

# LLM Configuration

LLM_PROVIDER=openrouter# Create conda environment (recommended)curl -fsSL https://ollama.ai/install.sh | sh

LLM_MODEL=your_model

OPENROUTER_API_KEY=your_keyconda create -n research_compass python=3.11 -yollama pull llama3.2



# Application Settingsconda activate research_compassollama serve

GRADIO_PORT=7860

```



## Key Features# Install dependencies# 4. Configure



- Interactive graph visualizationpip install -r requirements.txtcp .env.example .env

- GNN model training and predictions

- Document processing (PDF, DOCX, TXT, Markdown)python -m spacy download en_core_web_sm# Edit .env with your settings

- Citation network analysis

- Research assistant with natural language queries

- Temporal analysis and trend detection

# Configure environment# 5. Launch

## Technology Stack

cp .env.example .envpython launcher.py

- Python 3.11+

- Neo4j / NetworkX# Edit .env with your settings```

- PyTorch Geometric

- Gradio

- FAISS

- spaCy# Launch applicationAccess at **http://localhost:7860**



## Usagepython launcher.py



```bash```## ⚙️ Configuration

# Activate environment

conda activate research_compass



# Start applicationAccess at: http://localhost:7860Create `.env` file in project root:

python launcher.py

```



Access at http://localhost:7860## Configuration```bash



## Troubleshooting# LLM Configuration



**Graph Visualization Empty**: Upload documents first with "Build Knowledge Graph" enabledCreate `.env` file in project root:LLM_PROVIDER=ollama



**Neo4j Connection Failed**: Verify URI format and credentials in .env fileLLM_MODEL=llama3.2



**Python Version**: Use Python 3.11 (NOT 3.13)```bashLLM_TEMPERATURE=0.3



## License# Neo4j ConfigurationLLM_MAX_TOKENS=1000



MIT LicenseNEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io



## AcknowledgmentsNEO4J_USERNAME=neo4j# Local LLM URLs



Built with Neo4j, PyTorch Geometric, spaCy, and GradioNEO4J_PASSWORD=your_passwordOLLAMA_BASE_URL=http://localhost:11434


LMSTUDIO_BASE_URL=http://localhost:1234

# LLM Configuration

LLM_PROVIDER=ollama# Cloud LLM API Keys (optional)

LLM_MODEL=llama3.2OPENROUTER_API_KEY=your_key_here

OLLAMA_BASE_URL=http://localhost:11434OPENAI_API_KEY=your_key_here



# Cloud LLM (optional)# Neo4j Configuration (optional)

OPENROUTER_API_KEY=your_keyNEO4J_URI=neo4j://127.0.0.1:7687

OPENAI_API_KEY=your_keyNEO4J_USER=neo4j

NEO4J_PASSWORD=your_password

# Application Settings

GRADIO_PORT=7860# Embedding Model

```EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2



## Key Features# Cache Settings

CACHE_DIR=data/cache

- **Graph & GNN Dashboard**: Interactive visualization, model training, predictionsMAX_CACHE_ITEMS=1000

- **Document Processing**: PDF, DOCX, TXT, Markdown support with automated graph constructionDEFAULT_CACHE_TTL=3600

- **GNN-Powered Analysis**: Link prediction, node classification, temporal analysis

- **Visualization**: Interactive citation networks, collaboration graphs, attention maps# Server Settings

- **Research Assistant**: Natural language queries with graph-aware responsesGRADIO_PORT=7860

```

## Technology Stack

## 🎯 Key Features

| Component | Technology |

|-----------|-----------|### 🕸️ **NEW: Graph & GNN Dashboard** (Tab 2)

| Language | Python 3.11+ |- **Interactive Graph Visualization**: Explore your knowledge graph with clickable, draggable nodes

| Graph Database | Neo4j 5.0+ / NetworkX |- **GNN Model Training**: Train 4 types of GNN models (GAT, Transformer, Hetero, GCN) directly from UI

| Vector Search | FAISS |- **Live Graph Statistics**: Real-time node/edge counts, type breakdowns, GNN model status

| NLP/NER | spaCy 3.7+ |- **GNN Predictions**: Link prediction, node classification, and similarity search

| Embeddings | Sentence Transformers |- **Graph Export**: Export your knowledge graph as JSON or CSV

| Web UI | Gradio 4.0+ |

| GNN Framework | PyTorch Geometric |### 🧠 **GNN-Powered Core**

| Visualization | Pyvis, Plotly, NetworkX |

---

## Project Structure- **Graph Neural Networks**: Advanced GNN models for node classification, link prediction

- **Temporal Analysis**: Research evolution tracking and trend prediction

## Project Structure- **Neural Recommendations**: GNN-based personalized paper suggestions

- **Graph Search**: Semantic + structural search with graph context

```

Research Compass/### 📚 **Document Processing**

├── launcher.py           # Main application launcher- **Multi-Format Support**: PDF, DOCX, TXT, Markdown processing

├── requirements.txt      # Python dependencies- **Knowledge Graph Construction**: Automated entity and relationship extraction

├── config/              # Configuration files- **Batch Processing**: Handle multiple files simultaneously

├── src/graphrag/        # Core system- **Web URL Import**: Direct processing from arXiv and other sources

│   ├── core/           # System core

│   ├── analytics/      # Analytics modules### 🎨 **Visualization & Analytics**

│   ├── ml/             # Machine learning & GNN- **Interactive Networks**: Clickable citation and collaboration graphs

│   ├── ui/             # User interface- **Attention Visualization**: See how GNN models make decisions

│   └── visualization/  # Graph visualization- **Citation Analysis**: Disruption index, sleeping beauty detection

├── data/               # Data storage- **Temporal Charts**: Research trends over time

├── models/             # Trained models

└── tests/              # Unit tests
```

---

## Usage

### Starting the Application### 🤖 **User Experience**

```- **Streaming Responses**: Real-time word-by-word answer display

- **Intelligent Caching**: 10-100x performance optimization

## Usage- **Settings Panel**: User-friendly configuration with connection testing

- **Multi-Provider LLM**: Ollama, LM Studio, OpenRouter, OpenAI

```bash

# Start application## 🏗️ Technology Stack

conda activate research_compass

python launcher.py| Component | Technology |

|-----------|-----------|

# Custom port| **Language** | Python 3.11+ |

python launcher.py --port 8080| **Graph Database** | Neo4j 5.0+ / NetworkX |

| **Vector Search** | FAISS |

# Public sharing| **NLP/NER** | spaCy 3.7+ |

# Public sharing
python launcher.py --share
```

---

## Troubleshooting

### Common Issues and Solutions| **Embeddings** | Sentence Transformers |

```| **Web UI** | Gradio 4.0+ |

| **GNN Framework** | PyTorch Geometric |

## Troubleshooting| **Visualization** | Pyvis, Plotly, NetworkX |



### Neo4j Connection Issues## 📂 Project Structure

- **Cloud**: Verify URI format `neo4j+s://xxxxx.databases.neo4j.io` and password

- **Local**: Start Neo4j Desktop and verify `neo4j://localhost:7687````

Research Compass/

### Graph Visualization Empty├── launcher.py                    # Main application launcher

Upload documents first with "Build Knowledge Graph" enabled├── requirements.txt               # Python dependencies

├── config/                      # Configuration files

### Python Version│   ├── academic_config.yaml

Use Python 3.11 (NOT 3.13) for compatibility│   └── settings.py

├── src/graphrag/                 # Core GNN system

### LLM Connection│   ├── core/                     # System core

**Ollama**: `ollama serve` then `ollama pull llama3.2`│   ├── analytics/                 # Analytics modules

**OpenRouter/OpenAI**: Verify API key and credits│   ├── ml/                        # Machine learning

│   ├── visualization/             # Graph visualization

## License│   └── ui/                        # User interface

├── data/                         # Data storage

MIT License - see [LICENSE](LICENSE) file for details.│   ├── docs/                     # Research papers

│   ├── indices/                   # Search indices

## Acknowledgments│   └── cache/                     # Application cache

└── output/                        # Generated outputs

- Neo4j for graph database technology```

- PyTorch Geometric for GNN framework  

- spaCy for NLP capabilities## 🔧 How to Use Research Compass

- Gradio for UI framework

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

### Step 4: Explore Your Graph (NEW: Graph & GNN Dashboard)

1. **Go to Graph & GNN Dashboard tab** (Tab 2)
2. **View Statistics**:
   - Click "🔄 Refresh Statistics"
   - See node counts, edge counts, type breakdowns
   - Check GNN model status
3. **Visualize Your Graph**:
   - Go to "🎨 Visualize Graph" sub-tab
   - Set max nodes (start with 50-100)
   - Click "🎨 Generate Visualization"
   - **Interact**: Click nodes, drag to rearrange, zoom in/out
4. **Train GNN Models** (Optional - requires PyTorch Geometric):
   - Go to "🤖 Train GNN Models" sub-tab
   - Select model type: GAT (recommended), Transformer, Hetero, or GCN
   - Select task: Link Prediction (finds missing citations)
   - Set epochs: 50 (default)
   - Click "🚀 Start Training"
   - Wait 2-5 minutes
5. **Get GNN Predictions** (After training):
   - Go to "🔮 GNN Predictions" sub-tab
   - Enter paper title or node ID
   - Select prediction type
   - Get top-K predictions with scores

**Node Colors in Visualization:**
- 🔵 Blue = Papers
- 🟢 Green = Authors
- 🟡 Yellow = Topics
- 🟣 Purple = Venues

### Step 5: Ask Questions (Research Assistant)

1. **Go to Research Assistant tab**
2. **Enter your question**: "What are the main innovations in transformer architecture?"
3. **Enable options**:
   - ✅ Use Knowledge Graph (better context)
   - ✅ Use GNN Reasoning (advanced analysis)
   - ✅ Stream Response (real-time display)
   - ✅ Use Cache (faster repeated queries)
4. **Adjust Top-K sources** (default: 5)
5. **Click "Ask Question"**

### Step 6: Explore Advanced Features

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

### Step 7: Manage Performance

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

**GNN Training Issues**
```bash
# If you see "PyTorch Geometric not installed":
pip install torch torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# For GPU support (CUDA 11.8):
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**Graph Visualization Issues**
- If graph appears empty: Upload documents first (Tab 1)
- If too slow: Reduce max nodes to 50-100
- If nodes overlap: Try different layout (circular, spring)

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