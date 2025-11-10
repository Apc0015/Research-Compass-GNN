# Research Compass - Architecture & Data Flow

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RESEARCH COMPASS                                 │
│                      Unified AI Research Platform                        │
└─────────────────────────────────────────────────────────────────────────┘

                           launcher.py
                        (Entry Point)
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
    Config Mgr         System Init            UI Launch
    ┌───────┐          ┌──────────┐         ┌─────────┐
    │Config │          │Academic  │         │Gradio   │
    │Manager│          │RAGSystem │         │UI       │
    └───────┘          └──────────┘         └─────────┘
        │                    │                    │
        │                    │              ┌─────┴─────┐
        │                    │              │  9 Tabs   │
        │              ┌─────┴──────┐       │  + UI     │
        │              │            │       │  Modules  │
        │         ┌────┴──┐    ┌────┴──┐   └───────────┘
        │         │        │    │       │
    Database  Graph Mgr  Doc Proc  LLM Mgr
    (Neo4j)   (Core)    (Core)    (Core)
        │         │        │       │
        │         │        │   ┌───┴──────┐
        │         │        │   │Multi-Prov│
        │         │        │   │Ollama etc│
        │         │        │   └──────────┘
        │    ┌────┴────┐   │
        │    │IndexMgr │   │
        │    └────┬────┘   │
        │         │        │
        │    ┌────┴──────┐ │
        │    │Vector     │ │
        │    │Search     │ │
        │    └───────────┘ │
        │                  │
    ┌───┴──────────────────┴─────────────────┐
    │         Analytics Layer                 │
    │  ┌────────────────────────────────────┐ │
    │  │ GNN Manager (Training & Inference) │ │
    │  │ - Node Classifier                  │ │
    │  │ - Link Predictor                   │ │
    │  │ - Temporal GNN                     │ │
    │  └────────────────────────────────────┘ │
    │  ┌────────────────────────────────────┐ │
    │  │ Recommendation Engine               │ │
    │  │ - Hybrid Recommendations            │ │
    │  │ - Collaborative Filtering           │ │
    │  └────────────────────────────────────┘ │
    │  ┌────────────────────────────────────┐ │
    │  │ Network Analytics                   │ │
    │  │ - Citation Network Analysis         │ │
    │  │ - Collaboration Networks            │ │
    │  │ - Temporal Analysis                 │ │
    │  │ - Discovery Engine                  │ │
    │  └────────────────────────────────────┘ │
    └────────────────────────────────────────┘
        │         │        │        │
    Visualization Layer
    ├─ Citation Explorer
    ├─ Graph Visualizer
    ├─ GNN Explainer
    └─ Enhanced Charts
```

---

## Data Flow Diagram

### 1. Document Upload & Processing Flow

```
User Upload
    ↓
┌─────────────────────────┐
│ Document Processor      │
│ - PDF extraction        │
│ - Text parsing          │
│ - Web fetching          │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Entity Extractor (NER)  │
│ - Extract papers        │
│ - Extract authors       │
│ - Extract concepts      │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Metadata Extractor      │
│ - Title, authors        │
│ - Publication date      │
│ - Abstract              │
│ - Keywords              │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Relationship Extractor  │
│ - Citations             │
│ - Authorship            │
│ - Co-authorship         │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Graph Manager           │
│ Creates nodes & edges   │
│ in Neo4j/NetworkX       │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Vector Indexer          │
│ - Embed documents       │
│ - Store in FAISS        │
└──────────┬──────────────┘
           ↓
Indexed & Ready for Query
```

### 2. Query & Search Flow

```
User Query
    ↓
┌─────────────────────────┐
│ Query Input             │
│ - Text query            │
│ - Parameters            │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ GNN Search Engine       │
│ - Encode query          │
│ - Graph neural encoder  │
└──────────┬──────────────┘
           ↓
       ┌───┴────┐
       ↓        ↓
   Graph Db  Vector DB
   (Neo4j)   (FAISS)
       │        │
       └───┬────┘
           ↓
┌─────────────────────────┐
│ Results Fusion          │
│ - Combine results       │
│ - Rank & rerank         │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ LLM Response Gen        │
│ - Stream generation     │
│ - Use cache if hits     │
└──────────┬──────────────┘
           ↓
Display Results to User
```

### 3. GNN Training Flow

```
Indexed Graph Data
    ↓
┌─────────────────────────┐
│ Graph Converter         │
│ Neo4j → PyTorch Geom    │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ GNN Data Pipeline       │
│ - Create train/val/test │
│ - Node features         │
│ - Edge features         │
└──────────┬──────────────┘
           ↓
       ┌───┴────────────┐
       ↓                ↓
   Node Classifier  Link Predictor
   Train:           Train:
   ├─ Input:        ├─ Input:
   │  Node features │  Graph structure
   │  Node labels   │  Edge labels
   ├─ Model:        ├─ Model:
   │  PaperClassif  │  CitationPredict
   └─ Output:       └─ Output:
      Node pred        Link pred
           ↓                ↓
           └────┬──────────┘
                ↓
        ┌──────────────────────┐
        │ Model Evaluation     │
        │ - Accuracy metrics   │
        │ - Validation scores  │
        └────┬─────────────────┘
             ↓
       ┌──────────────┐
       │ Save Models  │
       │ Export       │
       └──────────────┘
```

### 4. Recommendation Flow

```
User Profile
    ↓
┌─────────────────────────┐
│ Unified Recommendation  │
│ Engine                  │
└──────────┬──────────────┘
           ↓
    ┌──────┼──────┐
    ↓      ↓      ↓
Content  Citation GNN Signal
    │      │      │
    └──────┼──────┘
           ↓
┌─────────────────────────┐
│ Hybrid Score Combine    │
│ - Weight signals        │
│ - Compute score         │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Collaborative Filter    │
│ (User-user similarity)  │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Rank & Diversify        │
│ - Top-k selection       │
│ - Add diversity         │
└──────────┬──────────────┘
           ↓
Display Recommendations
```

---

## Module Interdependencies

### Core System
```
launcher.py
    ↓
AcademicRAGSystem (core/academic_rag_system.py)
    ├─ GraphManager (core/graph_manager.py)
    ├─ AcademicGraphManager (core/academic_graph_manager.py)
    ├─ DocumentProcessor (core/document_processor.py)
    ├─ LLMManager (core/llm_manager.py)
    ├─ AdvancedIndexer (indexing/advanced_indexer.py)
    ├─ GNNManager (ml/gnn_manager.py)
    ├─ RecommendationEngine (analytics/unified_recommendation_engine.py)
    └─ Analytics Modules (analytics/*)
```

### Document Processing Pipeline
```
DocumentProcessor
    ├─ WebFetcher (fetch remote docs)
    ├─ EntityExtractor (NER - spaCy)
    ├─ MetadataExtractor
    ├─ ReferenceParser
    └─ RelationshipExtractor
        ↓
GraphManager (stores in Neo4j/NetworkX)
    ↓
AdvancedIndexer
    ├─ Chunking Strategies
    ├─ FAISS Vector Store
    └─ (Optional) LlamaIndex
        ↓
VectorSearch (FAISS)
```

### Query & Search Pipeline
```
User Query
    ↓
GNNSearchEngine / AdvancedQuery
    ├─ Query Encoding (GraphQueryEncoder)
    ├─ Neo4j/NetworkX Graph Query
    ├─ FAISS Vector Search
    └─ Result Fusion
        ↓
LLMManager
    ├─ LLM Provider (Ollama/OpenAI/etc)
    ├─ Streaming Response
    └─ Cache Manager (caching results)
```

### ML Pipeline
```
Graph Data
    ↓
GraphConverter (Neo4j → PyTorch Geometric)
    ↓
GNNManager
    ├─ NodeClassifier
    │  ├─ PaperClassifier
    │  ├─ Training
    │  └─ Inference
    ├─ LinkPredictor
    │  ├─ CitationPredictor
    │  ├─ Training
    │  └─ Inference
    ├─ EmbeddingsGenerator
    ├─ TemporalGNN
    └─ GNNBatchInference
```

### Analytics Pipeline
```
Graph + GNN Models
    ↓
    ├─ UnifiedRecommendationEngine
    ├─ NeuralRecommendationEngine
    ├─ CitationNetworkAnalysis
    ├─ CollaborationNetworkAnalysis
    ├─ RelationshipAnalytics
    ├─ ImpactMetrics
    ├─ TemporalAnalytics
    ├─ InterdisciplinaryAnalysis
    ├─ DiscoveryEngine
    └─ AdvancedCitationMetrics
```

---

## Configuration Flow

```
launcher.py
    ↓
config_manager.py (get_config_manager())
    ↓
Loads in order:
1. academic_config.yaml (base config)
2. Environment variables (override)
3. Command-line args (override)
    ↓
Config object with:
├─ DatabaseConfig
├─ LLMConfig
├─ EmbeddingConfig
├─ ProcessingConfig
├─ AcademicConfig
├─ UIConfig
├─ CacheConfig
├─ PathsConfig
├─ SystemConfig
└─ Custom
    ↓
Passed to:
├─ AcademicRAGSystem
├─ UnifiedUI
└─ All Managers
```

---

## UI Hierarchy

```
Research Compass (Main App)
│
├─ Tab 1: Upload & Process
│  ├─ File upload input
│  ├─ URL input
│  ├─ Metadata extraction toggle
│  ├─ Graph building toggle
│  ├─ Status panel
│  └─ Results JSON
│
├─ Tab 2: Graph & GNN Dashboard
│  ├─ SubTab: Graph Statistics
│  │  ├─ Node counts by type
│  │  ├─ Edge counts by type
│  │  ├─ Graph metrics
│  │  └─ GNN status
│  ├─ SubTab: Visualize Graph
│  │  └─ Interactive network visualization
│  ├─ SubTab: Train GNN Models
│  │  ├─ Model selection
│  │  ├─ Training parameters
│  │  ├─ Progress tracking
│  │  └─ Results
│  ├─ SubTab: GNN Predictions
│  │  ├─ Node predictions
│  │  ├─ Link predictions
│  │  └─ Confidence scores
│  └─ SubTab: Export Graph
│     ├─ Export formats
│     └─ Download options
│
├─ Tab 3: Research Assistant
│  ├─ Query input
│  ├─ Streaming response display
│  ├─ Conversation history
│  └─ Reference citations
│
├─ Tab 4: Temporal Analysis
│  ├─ SubTab: Topic Evolution
│  ├─ SubTab: Citation Velocity
│  └─ SubTab: H-Index Timeline
│
├─ Tab 5: Recommendations
│  ├─ User profile selection
│  ├─ Recommendation parameters
│  ├─ Results list
│  └─ Reasoning explanation
│
├─ Tab 6: Citation Explorer
│  ├─ Paper selection
│  ├─ Citation chain visualization
│  ├─ Network exploration
│  └─ Statistics
│
├─ Tab 7: Discovery
│  ├─ SubTab: Similar Papers
│  └─ SubTab: Cross-Disciplinary
│
├─ Tab 8: Cache Management
│  ├─ Cache stats
│  ├─ Clear cache button
│  └─ Performance metrics
│
└─ Tab 9: Settings
   ├─ SubTab: LLM Model
   │  ├─ Provider selection
   │  ├─ Model selection
   │  ├─ API key input
   │  └─ Connection test button
   ├─ SubTab: Embedding Model
   │  ├─ Model selection
   │  └─ Settings
   ├─ SubTab: Cache Settings
   │  ├─ Enable/disable cache
   │  └─ TTL configuration
   └─ SubTab: Database Connection
      ├─ Neo4j URI
      ├─ Credentials
      └─ Connection test button
```

---

## Component Communication

```
UI Layer (unified_launcher.py)
    ↕
Core System Layer (AcademicRAGSystem)
    ├─ GraphManager
    ├─ DocumentProcessor
    ├─ LLMManager
    ├─ AdvancedIndexer
    ├─ GNNManager
    └─ Analytics Engines
    ↕
Data Layer
    ├─ Neo4j (Graph DB)
    ├─ FAISS (Vector DB)
    ├─ File System (Docs, Models)
    └─ Cache (Redis/In-memory)
```

---

## Key Execution Paths

### Path 1: Upload & Index Document
1. User uploads file via UI
2. DocumentProcessor extracts text
3. EntityExtractor identifies entities (NER)
4. MetadataExtractor captures metadata
5. ReferenceParser extracts citations
6. RelationshipExtractor finds relationships
7. GraphManager creates nodes/edges in Neo4j
8. AdvancedIndexer embeds and stores in FAISS
9. UI shows success

### Path 2: Query with Results
1. User enters query
2. Query parsed and encoded by GNNSearchEngine
3. Parallel:
   - GraphDB query on Neo4j
   - Vector search on FAISS
4. Results fused and ranked
5. Top-K sent to LLMManager
6. LLM generates response with streaming
7. Response cached
8. Streamed to UI

### Path 3: Train GNN Model
1. User clicks "Train GNN"
2. GraphConverter transforms Neo4j to PyG
3. GNNDataPipeline creates train/val/test
4. GNNManager trains NodeClassifier
5. GNNManager trains LinkPredictor
6. Evaluation metrics computed
7. Models saved to disk
8. UI shows training progress and results

### Path 4: Get Recommendations
1. User requests recommendations
2. UnifiedRecommendationEngine analyzes user
3. Hybrid algorithm combines:
   - Content-based features
   - Citation patterns
   - GNN embeddings
4. Collaborative filtering applied
5. Results ranked and diversified
6. UI displays top-K recommendations

---

## Performance Optimization Strategies

1. **Caching Layer**
   - Query responses cached by cache_manager
   - TTL-based expiration
   - LRU eviction

2. **Batch Processing**
   - GNNBatchInference for GPU efficiency
   - Document batch processing
   - Chunking and indexing batches

3. **Lazy Loading**
   - Components loaded on-demand
   - Optional modules gracefully fallback
   - Container DI prevents unnecessary loads

4. **Connection Pooling**
   - Neo4j connection pool (100 connections)
   - Reusable sessions
   - Timeout management

5. **Vector Indexing**
   - FAISS for efficient similarity search
   - Dimension reduction (384-dim embeddings)
   - CPU-friendly (faiss-cpu)

6. **Streaming Responses**
   - LLM responses streamed word-by-word
   - Better UX for long responses
   - Reduced memory footprint

