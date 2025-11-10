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
        """
        # Research Compass - Architecture (Consolidated)

        This document consolidates the project architecture and the detailed architecture report into a single, de-duplicated reference. It is intended to be the canonical architecture overview for contributors, operators, and maintainers.

        Version: 1.0  •  Last updated: 2025-11-06

        ---

        ## Overview

        Research Compass is an AI-first research platform combining Graph Neural Networks (GNN) for relational reasoning with Large Language Models (LLM) for semantic understanding. It ingests academic documents, constructs a knowledge graph, provides retrieval and recommendation services, and supports GNN training and analytics.

        High-level layers:
        - UI (Gradio web app)
        - Application & Orchestration (AcademicRAGSystem)
        - Core Services (LLMManager, GNNManager, VectorSearch, GraphManager)
        - Data Layer (Neo4j/NetworkX, FAISS/Pinecone/Chroma, local file storage)

        ---

        ## Key Flows

        1) Document Ingestion
        - Documents → DocumentProcessor → Entity/Metadata/Relationship extractors → GraphManager (Neo4j/NetworkX) → AdvancedIndexer → Vector DB (FAISS/Pinecone)

        2) Query & Retrieval
        - User Query → GNNSearchEngine + Vector Search → Result Fusion → LLMManager (generate/explain) → UI

        3) GNN Training
        - Graph → GraphConverter → PyG Data → GNNManager (train/evaluate) → models/

        4) Recommendations
        - UnifiedRecommendationEngine combines GNN signals, semantic LLM scores, and collaborative filters to produce ranked, diversified recommendations.

        ---

        ## Components and Where to Find Them

        - Entry & UI: `launcher.py`, `src/graphrag/ui/unified_launcher.py`
        - Configuration: `config/academic_config.yaml`, `config/config_manager.py`
        - Core orchestration: `src/graphrag/core/academic_rag_system.py`
        - LLM providers & manager: `src/graphrag/core/llm_providers.py`, `src/graphrag/core/llm_manager.py`
        - Vector search: `src/graphrag/core/unified_vector_search.py`, `src/graphrag/core/pinecone_provider.py`
        - Graph manager: `src/graphrag/core/graph_manager.py`, `src/graphrag/core/academic_graph_manager.py`
        - Document processing: `src/graphrag/core/document_processor.py`, `src/graphrag/core/entity_extractor.py`
        - GNN & ML: `src/graphrag/ml/` (gnn_manager.py, advanced_gnn_models.py, graph_converter.py)
        - Analytics: `src/graphrag/analytics/` (unified_recommendation_engine.py, citation_network.py, temporal_analytics.py)

        ---

        ## Configuration & Runtime

        - Load order: defaults → `config/academic_config.yaml` → environment variables (`.env`) → CLI args
        - You can override LLM provider, models, vector DB backend (FAISS/Pinecone), Neo4j URI, and other runtime settings in `.env` or via the config manager API.
        - Note: some long-lived resources (Neo4j drivers, external LLM client sessions, vector DB connections) are initialized at system start and may require a restart unless the codepath supports hot-reload. The UI includes an "Apply .env" action that attempts a best-effort runtime reconfiguration for managers that expose update methods.

        ---

        ## GNN & LLM Integration Patterns

        - Sequential: GNN retrieves candidates, LLM ranks/explains.
        - Parallel: GNN and LLM produce independent scores then merge.
        - Iterative refinement: LLM refines query → GNN expands graph → LLM synthesizes final answer.

        ---

        ## Operational Guidance

        - Development: FAISS (local) + Ollama/LM Studio (local) provides low-cost local workflow.
        - Small-scale production: FAISS with regular backups.
        - Large-scale production: Pinecone (cloud) for vector DB, managed Neo4j or enterprise graph store.
        - Recommended resources for medium workloads: 8 vCPUs, 32GB RAM, optional GPU for GNN training.

        ---

        ## Notable Files & Commands

        - Launch locally: `python launcher.py`
        - Install deps: `pip install -r requirements.txt`
        - Download spaCy model: `python -m spacy download en_core_web_sm`

        ---

        If you want, I can now:
        - Remove the old `ARCHITECTURE_REPORT.md` file (I can delete it now if you want the single canonical file).
        - Produce a compact one-page diagram (SVG/PNG) from this doc.
        - Create a short CHANGELOG section in the file to track future architecture edits.

        """
- Modular, production-ready architecture with defensive design patterns
