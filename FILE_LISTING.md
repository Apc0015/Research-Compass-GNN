# Research Compass - Complete File Listing

## All 71 Python Files

### Root Level (1 file)
1. `/home/user/Research-Compass/launcher.py` - Main entry point & unified launcher (3222 lines)

### Configuration (2 files)
2. `/home/user/Research-Compass/config/config_manager.py` - Unified configuration system
3. `/home/user/Research-Compass/config/settings.py` - Settings module (backward compatibility)

### Core Module: src/graphrag/core/ (22 files)

#### Core RAG System
4. `/home/user/Research-Compass/src/graphrag/core/__init__.py`
5. `/home/user/Research-Compass/src/graphrag/core/academic_rag_system.py` - Top-level orchestrator
6. `/home/user/Research-Compass/src/graphrag/core/container.py` - Dependency injection

#### Graph Management
7. `/home/user/Research-Compass/src/graphrag/core/graph_manager.py` - Neo4j operations
8. `/home/user/Research-Compass/src/graphrag/core/academic_graph_manager.py` - Academic graph ops
9. `/home/user/Research-Compass/src/graphrag/core/academic_schema.py` - Graph schema

#### Document Processing
10. `/home/user/Research-Compass/src/graphrag/core/document_processor.py` - Text extraction
11. `/home/user/Research-Compass/src/graphrag/core/entity_extractor.py` - NER & extraction
12. `/home/user/Research-Compass/src/graphrag/core/metadata_extractor.py` - Metadata capture
13. `/home/user/Research-Compass/src/graphrag/core/reference_parser.py` - Citation parsing

#### Relationships
14. `/home/user/Research-Compass/src/graphrag/core/relationship_extractor.py` - Extract relationships
15. `/home/user/Research-Compass/src/graphrag/core/relationship_inference.py` - Infer relationships
16. `/home/user/Research-Compass/src/graphrag/core/relationship_manager.py` - Manage relationships

#### Vector & Search
17. `/home/user/Research-Compass/src/graphrag/core/vector_search.py` - FAISS search
18. `/home/user/Research-Compass/src/graphrag/core/gnn_enhanced_query.py` - GNN queries

#### GNN Pipeline
19. `/home/user/Research-Compass/src/graphrag/core/gnn_core_system.py` - GNN orchestration
20. `/home/user/Research-Compass/src/graphrag/core/gnn_data_pipeline.py` - GNN data prep

#### LLM & Utilities
21. `/home/user/Research-Compass/src/graphrag/core/llm_manager.py` - LLM management
22. `/home/user/Research-Compass/src/graphrag/core/llm_providers.py` - LLM providers
23. `/home/user/Research-Compass/src/graphrag/core/cache_manager.py` - Response caching
24. `/home/user/Research-Compass/src/graphrag/core/health_checker.py` - System health
25. `/home/user/Research-Compass/src/graphrag/core/web_fetcher.py` - Web document fetching

### ML Module: src/graphrag/ml/ (14 files)

#### Core ML
26. `/home/user/Research-Compass/src/graphrag/ml/__init__.py`
27. `/home/user/Research-Compass/src/graphrag/ml/gnn_manager.py` - ML orchestration
28. `/home/user/Research-Compass/src/graphrag/ml/advanced_gnn_models.py` - GNN architectures
29. `/home/user/Research-Compass/src/graphrag/ml/temporal_gnn.py` - Temporal GNNs

#### Models
30. `/home/user/Research-Compass/src/graphrag/ml/node_classifier.py` - Paper classification
31. `/home/user/Research-Compass/src/graphrag/ml/link_predictor.py` - Citation prediction
32. `/home/user/Research-Compass/src/graphrag/ml/embeddings_generator.py` - Graph embeddings
33. `/home/user/Research-Compass/src/graphrag/ml/graph_converter.py` - Neo4j to PyG conversion

#### Utilities
34. `/home/user/Research-Compass/src/graphrag/ml/gnn_batch_inference.py` - Batch inference
35. `/home/user/Research-Compass/src/graphrag/ml/gnn_utils.py` - Helper functions
36. `/home/user/Research-Compass/src/graphrag/ml/gnn_export.py` - Model export
37. `/home/user/Research-Compass/src/graphrag/ml/gnn_interpretation.py` - Model interpretation
38. `/home/user/Research-Compass/src/graphrag/ml/gnn_visualization.py` - Visualization

### Analytics Module: src/graphrag/analytics/ (12 files)

#### Recommendations
39. `/home/user/Research-Compass/src/graphrag/analytics/__init__.py`
40. `/home/user/Research-Compass/src/graphrag/analytics/unified_recommendation_engine.py` - Main recommendation system
41. `/home/user/Research-Compass/src/graphrag/analytics/neural_recommendation_engine.py` - GNN recommendations

#### Network Analysis
42. `/home/user/Research-Compass/src/graphrag/analytics/graph_analytics.py` - Graph metrics
43. `/home/user/Research-Compass/src/graphrag/analytics/citation_network.py` - Citation analysis
44. `/home/user/Research-Compass/src/graphrag/analytics/collaboration_network.py` - Collaboration analysis
45. `/home/user/Research-Compass/src/graphrag/analytics/relationship_analytics.py` - Relationship patterns
46. `/home/user/Research-Compass/src/graphrag/analytics/interdisciplinary_analysis.py` - Cross-disciplinary

#### Metrics & Discovery
47. `/home/user/Research-Compass/src/graphrag/analytics/advanced_citation_metrics.py` - Citation metrics
48. `/home/user/Research-Compass/src/graphrag/analytics/impact_metrics.py` - Impact measurement
49. `/home/user/Research-Compass/src/graphrag/analytics/temporal_analytics.py` - Temporal trends
50. `/home/user/Research-Compass/src/graphrag/analytics/discovery_engine.py` - Discovery system

### Query Module: src/graphrag/query/ (4 files)

51. `/home/user/Research-Compass/src/graphrag/query/__init__.py`
52. `/home/user/Research-Compass/src/graphrag/query/gnn_search_engine.py` - GNN-based search
53. `/home/user/Research-Compass/src/graphrag/query/advanced_query.py` - Advanced queries
54. `/home/user/Research-Compass/src/graphrag/query/temporal_query.py` - Temporal queries
55. `/home/user/Research-Compass/src/graphrag/query/query_builder.py` - Query building

### Indexing Module: src/graphrag/indexing/ (3 files)

56. `/home/user/Research-Compass/src/graphrag/indexing/__init__.py`
57. `/home/user/Research-Compass/src/graphrag/indexing/advanced_indexer.py` - LlamaIndex + FAISS
58. `/home/user/Research-Compass/src/graphrag/indexing/chunking_strategies.py` - Document chunking
59. `/home/user/Research-Compass/src/graphrag/indexing/retrieval_strategies.py` - Retrieval methods
60. `/home/user/Research-Compass/src/graphrag/indexing/query_engine.py` - Query execution

### Visualization Module: src/graphrag/visualization/ (3 files)

61. `/home/user/Research-Compass/src/graphrag/visualization/__init__.py`
62. `/home/user/Research-Compass/src/graphrag/visualization/citation_explorer.py` - Citation visualization
63. `/home/user/Research-Compass/src/graphrag/visualization/enhanced_viz.py` - Enhanced components
64. `/home/user/Research-Compass/src/graphrag/visualization/gnn_explainer.py` - GNN explanation

### UI Module: src/graphrag/ui/ (3 files)

65. `/home/user/Research-Compass/src/graphrag/ui/__init__.py`
66. `/home/user/Research-Compass/src/graphrag/ui/unified_launcher.py` - Main Gradio UI (3222 lines)
67. `/home/user/Research-Compass/src/graphrag/ui/graph_gnn_dashboard.py` - Graph dashboard

### Utils Module: src/graphrag/utils/ (2 files)

68. `/home/user/Research-Compass/src/graphrag/utils/__init__.py`

### Evaluation Module: src/graphrag/evaluation/ (2 files)

69. `/home/user/Research-Compass/src/graphrag/evaluation/__init__.py`
70. `/home/user/Research-Compass/src/graphrag/evaluation/gnn_evaluator.py` - Model evaluation

### Package Init (1 file)

71. `/home/user/Research-Compass/src/graphrag/__init__.py` - Main package init

---

## Configuration Files

- `/home/user/Research-Compass/config/academic_config.yaml` - Default YAML configuration
- `/home/user/Research-Compass/.env.example` - Environment variables template

---

## Data Files

### Documents
- `/home/user/Research-Compass/data/sample_paper.pdf` - Sample document
- `/home/user/Research-Compass/data/docs/NCT06157684_Prot_000.pdf`
- `/home/user/Research-Compass/data/docs/NCT06155240_Prot_SAP_000.pdf`
- `/home/user/Research-Compass/data/docs/NCT06152666_Prot_000.pdf`
- `/home/user/Research-Compass/data/docs/NCT06156735_Prot_SAP_000.pdf`
- `/home/user/Research-Compass/data/docs/NCT06152679_Prot_000.pdf`
- `/home/user/Research-Compass/data/docs/NCT06159946_Prot_000.pdf`
- `/home/user/Research-Compass/data/docs/NCT06155955_Prot_SAP_000.pdf`
- `/home/user/Research-Compass/data/docs/NCT06154122_Prot_000.pdf`
- `/home/user/Research-Compass/data/docs/NCT06149832_Prot_000.pdf`
- `/home/user/Research-Compass/data/docs/NCT06153992_ICF_001.pdf`
- `/home/user/Research-Compass/data/docs/NCT06151600_Prot_000.pdf`
- `/home/user/Research-Compass/data/docs/NCT06155006_Prot_SAP_000.pdf`
- `/home/user/Research-Compass/data/docs/NCT06145295_Prot_SAP_000.pdf`
- `/home/user/Research-Compass/data/docs/NCT06147739_Prot_SAP_000.pdf`
- `/home/user/Research-Compass/data/docs/NCT06152952_Prot_000.pdf`
- `/home/user/Research-Compass/data/docs/NCT06150378_Prot_SAP_000.pdf`
- `/home/user/Research-Compass/data/docs/NCT06152367_Prot_000.pdf`

### Indices
- `/home/user/Research-Compass/data/indices/chunks.pkl` - Processed chunks
- `/home/user/Research-Compass/data/indices/.gitkeep`

---

## Documentation Files

- `/home/user/Research-Compass/README.md` - Project overview
- `/home/user/Research-Compass/CHANGELOG.md` - Version history
- `/home/user/Research-Compass/CONTRIBUTING.md` - Contributing guidelines
- `/home/user/Research-Compass/GNN_FIXES_APPLIED.md` - GNN Phase 1 fixes
- `/home/user/Research-Compass/GNN_ISSUES_ANALYSIS.md` - Issues analysis
- `/home/user/Research-Compass/GNN_PHASE2_FIXES.md` - GNN Phase 2 improvements
- `/home/user/Research-Compass/GNN_PHASE3_ENHANCEMENTS.md` - GNN Phase 3 features

---

## Support Files

- `/home/user/Research-Compass/requirements.txt` - Python dependencies
- `/home/user/Research-Compass/setup.sh` - Setup script
- `/home/user/Research-Compass/.gitignore` - Git ignore rules
- `/home/user/Research-Compass/LICENSE` - MIT License

---

## Directory Summary

Total directories: 9 main + 2 data subdirectories
```
src/graphrag/
├── core/          22 Python files
├── ml/            14 Python files  
├── analytics/     12 Python files
├── query/         4 Python files
├── indexing/      4 Python files (query_engine + 3 others)
├── visualization/ 3 Python files
├── ui/            3 Python files
├── utils/         1 Python file
├── evaluation/    1 Python file
└── __init__.py    1 Python file

config/           2 Python files
root/             1 Python file (launcher.py)

Total: 71 Python files
```

