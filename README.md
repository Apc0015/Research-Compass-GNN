# Research Compass

> **Advanced AI-powered research exploration platform combining knowledge graphs, vector embeddings, and Graph Neural Networks**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GraphRAG](https://img.shields.io/badge/type-GraphRAG-green.svg)]()

**Research Compass** is a comprehensive knowledge management system designed for academic research exploration. It combines graph databases, vector embeddings, and Graph Neural Networks (GNNs) to provide intelligent paper recommendations, citation analysis, and research insights through a unified, easy-to-use interface.

---

## Table of Contents

- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

### Core Capabilities

- **Multi-Format Document Processing** - PDF, DOCX, TXT, Markdown support with batch upload
- **Knowledge Graph Construction** - Automated entity and relationship extraction using spaCy
- **Hybrid Search** - Combines vector similarity (FAISS) with graph traversal (Neo4j/NetworkX)
- **LLM-Powered Q&A** - Context-aware answer generation with source attribution
- **Advanced Analytics** - Temporal trends, citation metrics, impact analysis
- **Graph Neural Networks** - Node classification, link prediction, embedding generation

### Advanced Features

#### Temporal Analysis
- **Topic Evolution Tracking** - See how research topics develop over time
- **Citation Velocity Analysis** - Understand citation accumulation patterns
- **H-Index Timeline** - Track researcher impact evolution
- **Emerging Topics Detection** - Identify rapidly growing research areas

#### Personalized Recommendations
- **GNN-Powered Suggestions** - Intelligent paper and author recommendations
- **User Profile Management** - Create profiles based on interests and reading history
- **Diversity Controls** - Balance between similar and exploratory recommendations
- **Multi-Strategy Recommendations** - Hybrid, collaborative, and content-based approaches

#### Citation Analysis
- **Citation Explorer** - Interactive network visualization with forward/backward chains
- **Disruption Index** - Measure how papers disrupt existing research
- **Sleeping Beauty Detection** - Find delayed-recognition papers
- **Citation Cascade Analysis** - Track multi-generation influence patterns

#### Discovery Engine
- **Cross-Disciplinary Connections** - Find unexpected links between research areas
- **Similar Paper Detection** - Multiple similarity metrics (embedding, graph, citation)
- **Community Detection** - Identify research communities and clusters
- **Interdisciplinary Analysis** - Bridge discovery across fields

### Enhanced User Experience

- **Batch Upload** - Process multiple files simultaneously with progress tracking
- **Web URL Processing** - Direct import from arXiv and other sources (no download needed)
- **Streaming Responses** - Real-time word-by-word answer display
- **Intelligent Caching** - Multi-level cache for 10-100x performance boost
- **Settings Management** - User-friendly configuration panel with connection testing

### Multi-Provider LLM Support

| Provider | Type | Cost | Models |
|----------|------|------|--------|
| **Ollama** | Local | Free | Llama, Mistral, DeepSeek, etc. |
| **LM Studio** | Local | Free | Any GGUF model |
| **OpenRouter** | Cloud | Paid | 100+ models (Claude, GPT, Gemini) |
| **OpenAI** | Cloud | Paid | GPT-4o, GPT-4o-mini, GPT-3.5-turbo |

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Neo4j Database** (optional - falls back to in-memory graph)
- **LLM Provider** (choose one):
  - Ollama (local, free) - **Recommended for beginners**
  - LM Studio (local, free)
  - OpenRouter API key (cloud, paid)
  - OpenAI API key (cloud, paid)

### 5-Minute Setup

```bash
# 1. Clone and navigate
git clone <repository-url>
cd "Research Compass"

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Setup LLM (Ollama example - recommended)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2
ollama serve

# 6. Create configuration file
cp .env.example .env
# Edit .env with your settings

# 7. Launch the application
python launcher.py
```

Access at **http://localhost:7860**

---

## Installation

### Step 1: Setup LLM Provider

#### Option A: Ollama (Recommended for Beginners)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2

# Start Ollama (usually auto-starts)
ollama serve
```

#### Option B: LM Studio

1. Download from https://lmstudio.ai
2. Load a model from the catalog
3. Start server from the Server tab
4. Note the port (default: 1234)

#### Option C: Cloud (OpenRouter/OpenAI)

1. Get API key from your chosen provider
2. Add to `.env` file (see Configuration section)

### Step 2: Setup Neo4j (Optional)

#### Option A: Neo4j Desktop (Recommended)
1. Download from https://neo4j.com/download/
2. Create a new database
3. Start the database
4. Note your credentials

#### Option B: Neo4j AuraDB (Cloud)
1. Sign up at https://neo4j.com/cloud/aura/
2. Create free instance
3. Save connection URI and credentials

#### Option C: Skip Neo4j
The app will automatically use an in-memory NetworkX graph if Neo4j is unavailable.

### Step 3: Install Python Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model for entity extraction
python -m spacy download en_core_web_sm
```

### Step 4: Optional Web URL Processing

```bash
# For fetching papers from URLs
pip install requests beautifulsoup4 lxml
```

---

## Configuration

### Create .env File

Create a `.env` file in the project root:

```bash
# LLM Configuration
LLM_PROVIDER=ollama                    # ollama, lmstudio, openrouter, openai
LLM_MODEL=llama3.2                     # Your model name
LLM_TEMPERATURE=0.3                    # 0.0-1.0 (lower = more focused)
LLM_MAX_TOKENS=1000                    # Response length limit

# Local LLM URLs
OLLAMA_BASE_URL=http://localhost:11434
LMSTUDIO_BASE_URL=http://localhost:1234

# Cloud LLM API Keys (if using)
OPENROUTER_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Neo4j Configuration (optional)
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Embedding Model
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2  # Fast and effective

# Cache Settings
CACHE_DIR=data/cache
MAX_CACHE_ITEMS=1000
DEFAULT_CACHE_TTL=3600                 # 1 hour

# Server Settings
GRADIO_PORT=7860
```

### Configuration Tips

**LLM Temperature Settings:**
- `0.0-0.3`: Focused, deterministic (best for facts)
- `0.4-0.7`: Balanced (best for general use)
- `0.8-1.0`: Creative (best for brainstorming)

**Max Tokens:**
- `100-500`: Short answers
- `500-1500`: Detailed responses
- `1500-4000`: Comprehensive analysis

**Embedding Models:**
- `all-MiniLM-L6-v2`: Fast, good quality (384-dim)
- `all-mpnet-base-v2`: Slower, better quality (768-dim)
- `paraphrase-multilingual-MiniLM-L12-v2`: Multilingual support

---

## Usage Guide

### Launch the Application

```bash
# Production mode (default port 7860)
python launcher.py

# Development mode with auto-reload
python launcher.py --dev

# Custom port
python launcher.py --port 8080

# Create public share link
python launcher.py --share

# Combined options
python launcher.py --dev --port 8080 --share

# View help
python launcher.py --help
```

### Step 1: Configure Settings

1. Go to **Settings** tab
2. **LLM Configuration**:
   - Select your provider
   - Enter base URL (local) or API key (cloud)
   - Click **Refresh Available Models**
   - Select a model
   - Click **Test Connection**
   - Click **Save Configuration**

3. **Neo4j Configuration** (optional):
   - Enter URI, username, password
   - Click **Test Connection**
   - Should show: "Connected to Neo4j [version]"
   - Click **Save Configuration**

### Step 2: Upload Documents

#### Single/Multiple Files
1. Go to **Upload & Process** tab
2. Click **Upload Files** - select PDF, DOCX, TXT, or MD files
3. Enable **Build Knowledge Graph**
4. Enable **Extract Metadata** for academic papers
5. Click **Process All**

#### Web URLs
1. In **Web URLs** field, enter URLs (one per line):
   ```
   https://arxiv.org/abs/1706.03762
   https://arxiv.org/abs/1810.04805
   ```
2. Click **Process URLs**

### Step 3: Ask Questions

1. Go to **Research Assistant** tab
2. Enter your question: *"What are the main innovations in transformer architecture?"*
3. Enable **Use Knowledge Graph** for better context
4. Enable **Use GNN Reasoning** for advanced analysis
5. Enable **Stream Response** for real-time display
6. Enable **Use Cache** for faster repeated queries
7. Adjust **Top-K** sources (default: 5)
8. Click **Ask Question**

### Step 4: Explore Features

#### Temporal Analysis
1. Go to **Temporal Analysis** tab
2. **Topic Evolution**: Enter topic name, select time granularity
3. **Citation Velocity**: Enter paper title to see citation growth
4. **H-Index Timeline**: Enter author name to track impact
5. **Emerging Topics**: Set year and acceleration threshold

#### Personalized Recommendations
1. Go to **Recommendations** tab
2. Enter your research interests (comma-separated)
3. Add papers you've read/liked
4. Adjust diversity slider (0 = similar, 1 = exploratory)
5. Get tailored paper and author suggestions

#### Citation Explorer
1. Go to **Citation Explorer** tab
2. Enter paper title
3. Interactive network appears - click nodes to expand
4. Trace idea propagation between papers
5. Export visualization as HTML

#### Discovery Engine
1. Go to **Discovery** tab
2. Enter paper title
3. Find similar papers across disciplines
4. Explore cross-disciplinary connections
5. Toggle exploration mode for surprise papers

#### Advanced Metrics
1. Go to **Advanced Metrics** tab
2. **Disruption Index**: Measure research impact
3. **Sleeping Beauty**: Find delayed-recognition papers
4. **Citation Cascades**: Track multi-generation influence
5. **Citation Patterns**: Analyze diversity and concentration

#### Cache Management
1. Go to **Cache Management** tab
2. View statistics (hit rate, memory usage)
3. Clear expired entries or entire cache
4. Monitor performance improvements

---

## Technology Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.11+ | Primary development language |
| **Graph Database** | Neo4j 5.0+ / NetworkX | Knowledge graph storage |
| **Vector Search** | FAISS | Semantic similarity search |
| **NLP/NER** | spaCy 3.7+ | Entity extraction |
| **Embeddings** | Sentence Transformers | Text vectorization |
| **Web UI** | Gradio 4.0+ | User interface |
| **Visualization** | Pyvis, Plotly, NetworkX | Graph rendering |
| **Document Processing** | PyPDF2, python-docx, pdfplumber | Multi-format support |

### Machine Learning

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Graph Neural Networks** | PyTorch Geometric | Node classification, link prediction |
| **Traditional ML** | scikit-learn | Clustering, classification |
| **Community Detection** | python-louvain | Research community identification |

### LLM Integration

| Provider | Technology | Models |
|----------|-----------|--------|
| **Ollama** | Local inference | Llama, Mistral, DeepSeek |
| **LM Studio** | Local inference | Any GGUF model |
| **OpenRouter** | Cloud API | 100+ models |
| **OpenAI** | Cloud API | GPT series |

### Advanced Features

- **Caching**: Multi-level (memory + disk) with TTL expiration
- **Web Scraping**: requests, BeautifulSoup4, lxml
- **Configuration**: python-dotenv
- **Progress Tracking**: tqdm
- **Date Handling**: python-dateutil

---

## Project Structure

```
Research Compass/
├── launcher.py                     # Main unified launcher
├── requirements.txt                # Python dependencies
├── .env                           # Configuration (create this)
├── README.md                      # This file
├── PROJECT_DETAILED_REPORT.md     # Comprehensive technical documentation
│
├── config/
│   └── settings.py                # Configuration management
│
├── src/graphrag/
│   ├── core/                      # Core functionality
│   │   ├── academic_rag_system.py       # Main system orchestrator
│   │   ├── document_processor.py        # Document loading & chunking
│   │   ├── entity_extractor.py          # NER & relationship extraction
│   │   ├── graph_manager.py             # Graph database operations
│   │   ├── academic_graph_manager.py    # Academic schema manager
│   │   ├── vector_search.py             # FAISS similarity search
│   │   ├── llm_manager.py               # Multi-provider LLM manager
│   │   ├── llm_providers.py             # LLM provider implementations
│   │   ├── cache_manager.py             # Intelligent caching
│   │   ├── web_fetcher.py               # Web content fetching
│   │   ├── metadata_extractor.py        # Academic metadata extraction
│   │   └── container.py                 # Dependency injection
│   │
│   ├── analytics/                 # Analytics modules
│   │   ├── unified_recommendation_engine.py  # All recommendation algorithms
│   │   ├── temporal_analytics.py        # Topic evolution, citation velocity
│   │   ├── advanced_citation_metrics.py # Disruption, sleeping beauty
│   │   ├── discovery_engine.py          # Cross-disciplinary discovery
│   │   ├── citation_network.py          # Citation graph analysis
│   │   ├── collaboration_network.py     # Coauthor analysis
│   │   ├── graph_analytics.py           # PageRank, centrality, communities
│   │   ├── impact_metrics.py            # H-index, impact metrics
│   │   ├── interdisciplinary_analysis.py  # Field bridging
│   │   └── relationship_analytics.py    # Relationship patterns
│   │
│   ├── ml/                        # Machine learning
│   │   ├── gnn_manager.py              # GNN orchestrator
│   │   ├── gnn_explainer.py            # GNN interpretability
│   │   └── link_predictor.py           # Link prediction
│   │
│   ├── visualization/             # Visualization
│   │   ├── enhanced_viz.py             # Graph visualization
│   │   └── citation_explorer.py        # Citation network explorer
│   │
│   ├── ui/                        # User interface
│   │   ├── unified_launcher.py         # Unified UI (all features)
│   │   └── launcher.py                 # Basic launcher (legacy)
│   │
│   ├── indexing/                  # Advanced indexing
│   │   └── llama_index_integration.py
│   │
│   ├── query/                     # Advanced querying
│   │   └── advanced_query.py
│   │
│   └── utils/                     # Utilities
│       └── helpers.py
│
├── data/                          # Data storage
│   ├── documents/                 # Uploaded documents
│   ├── indices/                   # FAISS vector indices
│   └── cache/                     # Cached data
│
├── output/                        # Generated outputs
│   ├── visualizations/            # HTML graphs
│   └── reports/                   # Analysis reports
│
├── tests/                         # Test suite
│   ├── test_academic_schema.py
│   ├── test_integration.py
│   ├── test_metadata_extractor.py
│   └── test_smoke_integration.py
│
├── scripts/                       # Utility scripts
│   └── validate_installation.py
│
├── tools/                         # Development tools
│   ├── check_llm_endpoint.py      # LLM diagnostics
│   └── run_feature_tests.py       # Feature validation
│
└── lib/                           # Vendored JavaScript libraries
    ├── bindings/                  # JavaScript utilities
    ├── tom-select/                # Enhanced dropdowns
    └── vis-9.1.2/                 # Network visualization
```

---

## Documentation

- **[PROJECT_DETAILED_REPORT.md](PROJECT_DETAILED_REPORT.md)** - Complete technical architecture, API reference, development guide, performance analysis, security assessment, and testing information

---

## Troubleshooting

### Common Issues

#### Ollama Connection Failed
```bash
# Check if Ollama is running
ollama list

# Start Ollama
ollama serve

# Verify base URL in Settings tab
http://localhost:11434
```

#### LM Studio Connection Failed
```bash
# In LM Studio:
# 1. Ensure a model is loaded (not just downloaded)
# 2. Check Server tab shows "Running"
# 3. Verify port (default: 1234)
# 4. Test in Settings tab
```

#### Neo4j Connection Failed
```bash
# Check Neo4j is running
# Verify credentials are correct
# Try connecting with Neo4j Browser first
# Check URI format: neo4j://host:port
```

#### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Download spaCy model again
python -m spacy download en_core_web_sm
```

#### Port Already in Use
```bash
# Use a different port
python launcher.py --port 8080
```

#### Low Performance
- Enable caching in Research Assistant tab
- Reduce Top-K value (fewer sources to process)
- Use smaller embedding model
- Check if Neo4j is running (faster than NetworkX for large graphs)

### Python 3.13 Compatibility

The application includes automatic compatibility fixes for Python 3.13. If you encounter issues:

```bash
# Verify Python version
python --version  # Should be 3.11+

# The fix is automatic, but you can verify imports work
python -c "from src.graphrag.core.academic_rag_system import AcademicRAGSystem"
```

### Getting Help

1. Check this README
2. Review [PROJECT_DETAILED_REPORT.md](PROJECT_DETAILED_REPORT.md)
3. Run validation: `python scripts/validate_installation.py`
4. Test features: `python tools/run_feature_tests.py`
5. Check LLM connection: `python tools/check_llm_endpoint.py`

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install pytest black flake8 mypy

# Run linters
black src/
flake8 src/
mypy src/

# Run tests
pytest tests/ -v

# Run with coverage
pytest --cov=src/graphrag tests/
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Neo4j** for graph database technology
- **Meta AI** for FAISS vector search
- **Anthropic, OpenAI, Meta** for language models
- **spaCy** for NLP capabilities
- **Gradio** for the excellent UI framework
- **PyTorch Geometric** for GNN framework

---

## Project Status

**Status**: Active Development
**Version**: 2.0 (Consolidated)
**Last Updated**: October 2025

### What's New (v2.0)

- **Unified Architecture**: Consolidated 4 UI launchers into 1, and 3 recommendation engines into 1
- **Enhanced UX**: Batch upload, web URL processing, streaming responses, intelligent caching
- **Settings Panel**: User-friendly configuration with connection testing
- **Improved Performance**: 10-100x speedup with intelligent caching
- **Better Documentation**: Clean, comprehensive guides

---

**Made with dedication for the AI and Research community**
