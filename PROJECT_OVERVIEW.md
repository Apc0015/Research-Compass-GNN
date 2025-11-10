# Research Compass - Complete Project Overview

## Project Summary
**Research Compass** is an advanced AI-powered research exploration platform that combines:
- Graph Neural Networks (GNN) for intelligent analysis
- Knowledge graphs using Neo4j/NetworkX for entity relationships
- Gradio web UI for interactive research exploration
- Multi-provider LLM support (Ollama, OpenRouter, OpenAI, LM Studio)
- Comprehensive document processing and analysis
- Research assistant with streaming responses
- Citation and collaboration network analysis

**Tech Stack**: Python 3.11+, PyTorch, PyTorch Geometric, Neo4j, Gradio, LlamaIndex, Faiss

---

## Directory Structure

```
/home/user/Research-Compass/
├── launcher.py                      # Main entry point (3222 lines)
├── requirements.txt                 # All dependencies
├── setup.sh                         # Setup script
├── .env.example                     # Environment template
│
├── config/                          # Configuration management
│   ├── config_manager.py            # Unified configuration system
│   ├── settings.py                  # Settings (backward compatibility)
│   └── academic_config.yaml         # YAML configuration
│
├── src/graphrag/                    # Main package
│   ├── __init__.py
│   ├── core/                        # Core RAG system (22 modules)
│   ├── indexing/                    # Document indexing
│   ├── ml/                          # Machine learning & GNN (14 modules)
│   ├── query/                       # Query engines
│   ├── analytics/                   # Analytics & recommendations
│   ├── visualization/               # Visualization components
│   ├── ui/                          # Gradio UI
│   └── utils/                       # Utilities
│
├── data/                            # Data storage
│   ├── docs/                        # Documents (17 sample PDFs)
│   └── indices/                     # Index files
│
└── documentation/
    ├── README.md                    # Main documentation
    ├── GNN_FIXES_APPLIED.md
    ├── GNN_PHASE2_FIXES.md
    ├── GNN_PHASE3_ENHANCEMENTS.md
    ├── CONTRIBUTING.md
    └── CHANGELOG.md
```

---

## Main Entry Point: launcher.py

**Purpose**: Unified launcher for the entire Research Compass platform

**Key Features**:
- 📤 Multiple file upload & web URL processing
- 🔍 Research Assistant with GNN reasoning
- 💬 Streaming responses with intelligent caching
- 📊 Temporal analysis & trend detection
- 💡 Personalized recommendations
- 🕸️ Interactive citation explorer
- 🔬 Discovery engine for cross-disciplinary research
- 📈 Advanced citation metrics
- ⚙️ Settings management with connection testing

**Modes**:
- `python launcher.py` - Production mode
- `python launcher.py --dev` - Development mode with auto-reload
- `python launcher.py --port 8080` - Custom port
- `python launcher.py --share` - Public URL sharing

**Key Functions**:
- `setup_dev_mode()` - Setup auto-reload monitoring (uses watchdog)
- `print_banner()` - Display startup banner
- `initialize_system()` - Initialize AcademicRAGSystem with unified config
- `main()` - Parse arguments and launch unified UI

---

## Configuration System (config/)

### config_manager.py
Unified configuration management consolidating:
- Environment variables
- YAML files (academic_config.yaml)
- Default values
- Type checking & validation

**Configuration Sections** (Dataclasses):
1. **DatabaseConfig** - Neo4j connection & pooling
2. **LLMConfig** - LLM provider settings (Ollama, OpenAI, OpenRouter, LM Studio)
3. **EmbeddingConfig** - Embedding model (all-MiniLM-L6-v2 default)
4. **ProcessingConfig** - Document chunking, file extensions, metadata extraction
5. **AcademicConfig** - GNN, indexing, recommendations, citation analysis
6. **UIConfig** - Gradio port, visualization settings
7. **CacheConfig** - Cache directory & TTL
8. **PathsConfig** - All directory paths
9. **SystemConfig** - Environment, logging, performance tuning
10. **Custom** - Additional configuration sections

---

## Core Module: src/graphrag/core/ (22 modules)

### Core RAG System
- **academic_rag_system.py** - Top-level orchestrator wiring all components
- **container.py** - Dependency injection container

### Graph Management
- **graph_manager.py** - Neo4j database operations for entities/relationships
- **academic_graph_manager.py** - Academic-specific graph operations
- **academic_schema.py** - Graph schema definitions for academic papers

### Document Processing
- **document_processor.py** - PDF/text/doc extraction and processing
- **entity_extractor.py** - Named entity recognition from documents
- **metadata_extractor.py** - Title, authors, dates, citations extraction
- **reference_parser.py** - Parse and extract bibliographic references

### Relationships
- **relationship_extractor.py** - Extract relationships from text
- **relationship_inference.py** - Infer implicit relationships
- **relationship_manager.py** - Manage relationship lifecycle

### Vector & Search
- **vector_search.py** - FAISS-based vector similarity search
- **gnn_enhanced_query.py** - GNN-powered query enhancement

### GNN Pipeline
- **gnn_core_system.py** - Core GNN system orchestration
- **gnn_data_pipeline.py** - Data preparation for GNN training

### LLM & Caching
- **llm_manager.py** - Unified LLM provider management
- **llm_providers.py** - Implementations for Ollama, OpenAI, OpenRouter, LM Studio
- **cache_manager.py** - Intelligent response caching
- **health_checker.py** - System health status monitoring
- **web_fetcher.py** - Download papers from web

---

## Machine Learning: src/graphrag/ml/ (14 modules)

### GNN Models
- **gnn_manager.py** - Orchestrate all ML models (training, prediction, lifecycle)
- **advanced_gnn_models.py** - State-of-the-art GNN architectures
- **temporal_gnn.py** - Temporal Graph Neural Networks

### Core ML
- **node_classifier.py** - Classify paper nodes (PaperClassifier)
- **link_predictor.py** - Predict citation links (CitationPredictor)
- **embeddings_generator.py** - Generate graph embeddings
- **graph_converter.py** - Convert Neo4j to PyTorch Geometric

### GNN Utilities
- **gnn_batch_inference.py** - Batch inference on graphs
- **gnn_utils.py** - Helper functions and utilities
- **gnn_export.py** - Export trained models
- **gnn_interpretation.py** - Explain GNN predictions
- **gnn_visualization.py** - Visualize learned representations

---

## Analytics & Recommendations: src/graphrag/analytics/ (12 modules)

### Recommendations
- **unified_recommendation_engine.py** - Comprehensive recommendation system
  - Hybrid recommendations (content + citation + GNN)
  - Personalized user profiles
  - Collaborative filtering
  - Author & research direction suggestions

- **neural_recommendation_engine.py** - GNN-based neural recommendations

### Network Analysis
- **graph_analytics.py** - General graph metrics and analysis
- **citation_network.py** - Citation network analysis
- **collaboration_network.py** - Author collaboration networks
- **relationship_analytics.py** - Relationship pattern analysis
- **interdisciplinary_analysis.py** - Cross-disciplinary connections

### Metrics & Discovery
- **advanced_citation_metrics.py** - Advanced citation analysis
- **impact_metrics.py** - Research impact measurement
- **temporal_analytics.py** - Trend detection and temporal patterns
- **discovery_engine.py** - Serendipitous cross-disciplinary discovery

---

## Query & Indexing: src/graphrag/query/ & src/graphrag/indexing/

### Query Engines
- **gnn_search_engine.py** - Graph neural network-based search
- **advanced_query.py** - Advanced query building & execution
- **temporal_query.py** - Time-aware queries
- **query_builder.py** - Query construction utilities

### Indexing
- **advanced_indexer.py** - LlamaIndex + FAISS indexing
- **chunking_strategies.py** - Academic paper chunking
- **retrieval_strategies.py** - Multiple retrieval approaches
- **query_engine.py** - Query execution

---

## Visualization: src/graphrag/visualization/ (4 modules)

- **citation_explorer.py** - Interactive citation chain exploration
- **enhanced_viz.py** - Enhanced visualization components
- **gnn_explainer.py** - Explain GNN predictions visually
- **__init__.py**

---

## UI Components: src/graphrag/ui/ (3 modules)

### unified_launcher.py (Main UI - 3222 lines)
Comprehensive Gradio-based interface with **9 major tabs**:

#### 1. **📤 Upload & Process**
   - Multiple file upload (PDF, DOCX, TXT, MD)
   - Web URL input for remote papers
   - Extract metadata option
   - Build knowledge graph option
   - Process all documents
   - Detailed JSON results

#### 2. **🕸️ Graph & GNN Dashboard**
   - **Sub-tabs**:
     - 📊 Graph Statistics - Node/edge counts, metrics
     - 🎨 Visualize Graph - Interactive network visualization
     - 🤖 Train GNN Models - Model training interface
     - 🔮 GNN Predictions - View model predictions
     - 💾 Export Graph - Export graph data

#### 3. **🔍 Research Assistant**
   - Conversational research queries
   - GNN-powered reasoning
   - Streaming responses
   - Multi-turn conversations
   - Caching for performance

#### 4. **📊 Temporal Analysis**
   - **Sub-tabs**:
     - Topic Evolution - Track topic trends over time
     - Citation Velocity - Citation growth rates
     - H-Index Timeline - Author h-index evolution

#### 5. **💡 Recommendations**
   - Personalized paper recommendations
   - Hybrid recommendation algorithm
   - Collaborative filtering
   - Research direction suggestions

#### 6. **🕸️ Citation Explorer**
   - Citation network visualization
   - Citation chain tracing
   - Research lineage exploration
   - Interactive citation browsing

#### 7. **🔬 Discovery**
   - **Sub-tabs**:
     - Similar Papers - Find similar research
     - Cross-Disciplinary - Find related fields

#### 8. **💾 Cache Management**
   - View cache statistics
   - Clear cache
   - Cache performance metrics
   - TTL management

#### 9. **⚙️ Settings**
   - **Sub-tabs**:
     - 🤖 LLM Model - Provider selection, model selection, API keys
     - 🔢 Embedding Model - Embedding configuration
     - 💾 Cache Settings - Cache behavior configuration
     - 🗄️ Database Connection - Neo4j/database settings
     - Connection testing for all services

### graph_gnn_dashboard.py
Dashboard class for graph visualization and GNN management:
- `get_graph_statistics()` - Comprehensive graph metrics
- Graph statistics display
- Node/edge type analysis
- GNN model monitoring

### __init__.py
UI module initialization

---

## Feature Highlights

### 1. **Graph Neural Networks (GNN)**
   - Node classification for paper categorization
   - Link prediction for citation discovery
   - Temporal GNN for evolution analysis
   - Batch inference capabilities
   - Model export and interpretation
   - Advanced GNN architectures

### 2. **Knowledge Graph Management**
   - Neo4j or NetworkX backend (fallback)
   - Entity extraction (papers, authors, concepts)
   - Relationship inference and management
   - Metadata enrichment
   - Graph statistics and analysis

### 3. **Document Processing**
   - Multiple format support (PDF, DOCX, TXT, MD)
   - Web URL fetching
   - Smart chunking strategies
   - Metadata extraction
   - Reference parsing
   - Entity recognition

### 4. **Intelligent Search & Retrieval**
   - FAISS vector search
   - LlamaIndex integration (optional)
   - GNN-enhanced queries
   - Temporal queries
   - Multi-strategy retrieval

### 5. **Recommendation Engine**
   - Hybrid recommendations
   - GNN-powered personalization
   - Collaborative filtering
   - Author recommendations
   - Research direction suggestions

### 6. **Analytics & Insights**
   - Citation network analysis
   - Collaboration network analysis
   - Research impact metrics
   - Temporal trend analysis
   - Cross-disciplinary discovery
   - Community detection

### 7. **LLM Integration**
   - Multi-provider support:
     - Ollama (local)
     - LM Studio (local)
     - OpenRouter (cloud)
     - OpenAI (cloud)
   - Streaming responses
   - Intelligent caching
   - Connection testing

### 8. **Visualization**
   - Interactive citation networks (Pyvis, Plotly)
   - Graph visualization
   - Temporal charts
   - GNN explanation visuals
   - Citation chain exploration

---

## Configuration Files

### academic_config.yaml
Default configuration in YAML format:
```yaml
database:
  uri: "neo4j://127.0.0.1:7687"
  user: "neo4j"
  password: ""
  
llm:
  provider: "ollama"
  model: "llama3.2"
  temperature: 0.3
  max_tokens: 1000
  
embedding:
  model_name: "all-MiniLM-L6-v2"
  provider: "huggingface"
  dimension: 384

processing:
  chunk_size: 500
  chunk_overlap: 50
  max_file_size: 52428800  # 50MB
  allowed_extensions: ['.pdf', '.txt', '.md', '.docx', '.doc']

ui:
  port: 7860
  host: "0.0.0.0"
  share: false
  height: "800px"
  width: "100%"
```

### .env.example
Environment variables template for sensitive data

---

## Dependencies Summary

### Core Framework
- **gradio** >= 4.0.0 - Web UI
- **neo4j** >= 5.0.0 - Graph database
- **networkx** >= 3.0 - In-memory graph fallback

### Machine Learning & GNN
- **torch** >= 2.0.0 - Deep learning
- **torch-geometric** >= 2.3.0 - Graph neural networks
- **scikit-learn** >= 1.3.0 - ML utilities

### NLP & Text Processing
- **sentence-transformers** >= 2.2.0 - Embeddings
- **spacy** >= 3.7.0 - NER
- **nltk** >= 3.8.0 - Text processing
- **transformers** >= 4.30.0 - Pretrained models

### Document Processing
- **PyPDF2** >= 3.0.0, **pdfplumber** >= 0.7.6 - PDF handling
- **python-docx** >= 1.0.0 - Word documents
- **beautifulsoup4** >= 4.12.0 - HTML parsing

### Vector Search & Indexing
- **faiss-cpu** >= 1.7.0 - Vector similarity
- **chromadb** >= 0.4.0 - Vector database
- **llama-index** >= 0.9.0 - Advanced indexing
- **langchain** >= 0.1.0 - LLM chains

### Visualization
- **pyvis** >= 0.3.0 - Network graphs
- **plotly** >= 5.18.0 - Interactive charts
- **matplotlib** >= 3.7.0 - Static plots
- **kaleido** >= 0.2.1 - Image export

### LLM Integration
- **openai** >= 1.0.0 - OpenAI API
- **httpx** >= 0.24.0 - HTTP client
- **requests** >= 2.31.0 - Web requests

### Utilities
- **pandas** >= 2.0.0 - Data manipulation
- **numpy** >= 1.24.0 - Numerical computing
- **scipy** >= 1.11.0 - Scientific computing
- **pyyaml** >= 6.0.0 - YAML parsing
- **python-dotenv** >= 1.0.0 - Environment variables
- **tqdm** >= 4.65.0 - Progress bars

---

## Data Directory Structure

```
data/
├── docs/                    # Document storage (17 sample PDFs)
│   ├── NCT06157684_Prot_000.pdf
│   ├── NCT06155240_Prot_SAP_000.pdf
│   └── ... (15 more clinical trial documents)
│
└── indices/                 # Index files
    ├── chunks.pkl          # Processed document chunks
    └── .gitkeep
```

---

## Testing & Quality

**Current Status**: No test files found in repository
- Recommendation: Add pytest-based tests under `tests/` directory
- Include unit tests, integration tests, and end-to-end tests

---

## Frontend & Assets

**Note**: No custom CSS/JavaScript files found. UI entirely built with Gradio framework which:
- Handles all UI styling with built-in themes
- Uses Gradio's native components
- Built-in responsiveness and accessibility
- Uses `theme=gr.themes.Soft()` for styling

---

## Environment & Setup

### Installation
```bash
git clone <repository-url>
cd "Research Compass"
conda create -n research_compass python=3.11 -y
conda activate research_compass
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Configuration
```bash
cp .env.example .env
# Edit .env with:
# - NEO4J credentials
# - LLM provider details
# - API keys if using cloud providers
```

### Running
```bash
# Production
python launcher.py

# Development (with auto-reload)
python launcher.py --dev

# Custom port
python launcher.py --port 8080

# Public sharing
python launcher.py --share
```

---

## Recent Enhancements (GNN Phases)

### Phase 1: GNN Fixes Applied
- Core GNN system stabilization
- Entity extraction improvements
- Graph creation optimization

### Phase 2: Critical UX & Performance Improvements
- Performance optimization
- UX enhancements
- Bug fixes

### Phase 3: Production & Deployment Features
- Production-ready deployments
- Advanced deployment features
- Enhanced monitoring

---

## Key Statistics

- **Total Python Files**: 71
- **Total Modules**: 
  - Core: 22
  - ML/GNN: 14
  - Analytics: 12
  - Query/Indexing: 7
  - Visualization: 4
  - UI: 3
- **UI Tabs**: 9 main tabs + sub-tabs
- **Configuration Sections**: 10
- **Lines of Code (UI alone)**: 3,222
- **Sample Documents**: 17 PDFs
- **License**: MIT

