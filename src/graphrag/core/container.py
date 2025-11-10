"""Simple dependency injection container for Research Compass.

This small container supports registering singletons and factories and
resolving dependencies by key or type. It's intentionally lightweight to
avoid adding external DI dependencies.

Usage:
    c = Container()
    c.register_singleton('config', {...})
    c.register_factory('graph_manager', lambda: GraphManager(...))
    graph = c.resolve('graph_manager')
"""
from __future__ import annotations

from typing import Any, Callable, Dict
import threading


class Container:
    def __init__(self) -> None:
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._lock = threading.RLock()

    def register_singleton(self, name: str, instance: Any) -> None:
        with self._lock:
            self._singletons[name] = instance

    def register_factory(self, name: str, factory: Callable[[], Any]) -> None:
        with self._lock:
            self._factories[name] = factory

    def resolve(self, name: str) -> Any:
        with self._lock:
            if name in self._singletons:
                return self._singletons[name]
            if name in self._factories:
                inst = self._factories[name]()
                # cache factories as singletons for subsequent resolves
                self._singletons[name] = inst
                return inst
            raise KeyError(f"Dependency '{name}' not found in container")


def build_default_container(config: Dict[str, Any] | None = None) -> Container:
    """Create a container pre-populated with common factories used across the app.

    The factories use lazy imports to avoid heavy imports at module import time.
    """
    import os
    from ...config.config_manager import get_config_manager, get_config
    
    # Use unified configuration manager if no config provided
    if config is None:
        config_manager = get_config_manager()
        unified_config = config_manager.config
        cfg = unified_config.to_dict()
    else:
        cfg = config
        # For backward compatibility, create a minimal config object
        unified_config = get_config()
        # Update with provided config
        for key, value in config.items():
            if hasattr(unified_config, key):
                setattr(unified_config, key, value)

    c = Container()

    # Register unified config
    c.register_singleton('config', unified_config)
    c.register_singleton('config_dict', cfg)

    # Neo4j connection details from unified config
    neo4j_uri = unified_config.database.uri
    neo4j_user = unified_config.database.user
    neo4j_password = unified_config.database.password

    # GraphManager factory
    def _graph_factory():
        from .graph_manager import GraphManager
        return GraphManager(neo4j_uri, neo4j_user, neo4j_password)

    c.register_factory('graph_manager', _graph_factory)

    # DocumentProcessor factory with unified config
    def _docproc_factory():
        from .document_processor import DocumentProcessor
        processor = DocumentProcessor()
        # Apply configuration to processor
        processor.chunk_size = unified_config.processing.chunk_size
        processor.chunk_overlap = unified_config.processing.chunk_overlap
        processor.top_k_chunks = unified_config.processing.top_k_chunks
        processor.max_graph_depth = unified_config.processing.max_graph_depth
        processor.allowed_extensions = unified_config.processing.allowed_extensions
        processor.metadata_extraction = unified_config.processing.metadata_extraction
        return processor

    c.register_factory('document_processor', _docproc_factory)

    # CacheManager factory with unified config
    def _cache_factory():
        from .cache_manager import CacheManager
        from pathlib import Path
        cache_dir = Path(unified_config.cache.cache_dir)
        max_items = unified_config.cache.max_items
        ttl = unified_config.cache.default_ttl
        return CacheManager(cache_dir, max_items, ttl)
    
    c.register_factory('cache_manager', _cache_factory)

    # AcademicGraphManager factory (depends on graph_manager)
    def _academic_graph_factory():
        from .academic_graph_manager import AcademicGraphManager
        graph = c.resolve('graph_manager')
        return AcademicGraphManager(graph)

    c.register_factory('academic_graph_manager', _academic_graph_factory)
    
    # VectorSearch factory with unified config
    def _vector_search_factory():
        from .vector_search import VectorSearch
        return VectorSearch(
            model_name=unified_config.embedding.model_name,
            provider=unified_config.embedding.provider,
            base_url=unified_config.embedding.base_url
        )
    
    c.register_factory('vector_search', _vector_search_factory)
    
    # LLMManager factory with unified config
    def _llm_manager_factory():
        from .llm_manager import LLMManager
        llm_config = unified_config.llm
        
        # Handle provider-specific settings
        if llm_config.provider == 'ollama':
            base_url = llm_config.base_url
        elif llm_config.provider == 'lmstudio':
            base_url = llm_config.base_url
        else:
            base_url = None
        
        # Handle API keys
        if llm_config.provider == 'openrouter':
            api_key = llm_config.api_key
        elif llm_config.provider == 'openai':
            api_key = llm_config.api_key
        else:
            api_key = None
        
        return LLMManager(
            provider=llm_config.provider,
            model=llm_config.model,
            base_url=base_url,
            api_key=api_key,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens
        )
    
    c.register_factory('llm_manager', _llm_manager_factory)
    
    # GNNManager factory (with lazy loading for faster startup)
    def _gnn_manager_factory():
        from ..ml.lazy_gnn_manager import LazyGNNManager
        # Lazy loading: models loaded only when first accessed (faster startup)
        return LazyGNNManager(neo4j_uri, neo4j_user, neo4j_password, auto_initialize=False)

    c.register_factory('gnn_manager', _gnn_manager_factory)
    
    # TemporalGraphAnalytics factory
    def _temporal_analytics_factory():
        from ..analytics.temporal_analytics import TemporalGraphAnalytics
        graph = c.resolve('graph_manager')
        return TemporalGraphAnalytics(graph, neo4j_uri, neo4j_user, neo4j_password)
    
    c.register_factory('temporal_analytics', _temporal_analytics_factory)
    
    # GNNExplainer factory
    def _gnn_explainer_factory():
        from ..ml.gnn_interpretation import GNNExplainer
        gnn = c.resolve('gnn_manager')
        return GNNExplainer(gnn)
    
    c.register_factory('gnn_explainer', _gnn_explainer_factory)
    
    # GNNEnhancedQueryEngine factory
    def _gnn_query_engine_factory():
        from .gnn_enhanced_query import GNNEnhancedQueryEngine
        vector_search = c.resolve('vector_search')
        graph = c.resolve('graph_manager')
        gnn = c.resolve('gnn_manager')
        return GNNEnhancedQueryEngine(vector_search, graph, gnn)
    
    c.register_factory('gnn_query_engine', _gnn_query_engine_factory)
    
    # UnifiedRecommendationEngine factory (consolidates all recommendation functionality)
    def _recommendation_engine_factory():
        from ..analytics.unified_recommendation_engine import UnifiedRecommendationEngine
        graph = c.resolve('graph_manager')
        gnn = c.resolve('gnn_manager')
        vector_search = c.resolve('vector_search')
        # UnifiedRecommendationEngine combines PersonalizedRecommendationEngine,
        # PaperRecommendationEngine, and CollaborativeRecommender functionality
        return UnifiedRecommendationEngine(graph, embedder=None, gnn_manager=gnn, vector_search=vector_search)

    c.register_factory('recommendation_engine', _recommendation_engine_factory)
    
    # SerendipitousDiscoveryEngine factory
    def _discovery_engine_factory():
        from ..analytics.discovery_engine import SerendipitousDiscoveryEngine
        graph = c.resolve('graph_manager')
        gnn = c.resolve('gnn_manager')
        return SerendipitousDiscoveryEngine(graph, gnn)
    
    c.register_factory('discovery_engine', _discovery_engine_factory)
    
    # InteractiveCitationExplorer factory
    def _citation_explorer_factory():
        from ..visualization.citation_explorer import InteractiveCitationExplorer
        graph = c.resolve('graph_manager')
        return InteractiveCitationExplorer(graph)
    
    c.register_factory('citation_explorer', _citation_explorer_factory)
    
    # AdvancedCitationMetrics factory
    def _citation_metrics_factory():
        from ..analytics.advanced_citation_metrics import AdvancedCitationMetrics
        graph = c.resolve('graph_manager')
        return AdvancedCitationMetrics(graph, neo4j_uri, neo4j_user, neo4j_password)
    
    c.register_factory('citation_metrics', _citation_metrics_factory)
    
    # InterdisciplinaryAnalyzer factory
    def _interdisciplinary_factory():
        from ..analytics.interdisciplinary_analysis import InterdisciplinaryAnalyzer
        graph = c.resolve('graph_manager')
        return InterdisciplinaryAnalyzer(graph, neo4j_uri, neo4j_user, neo4j_password)
    
    c.register_factory('interdisciplinary_analyzer', _interdisciplinary_factory)
    
    # CollaborativeRecommender factory (now provided by UnifiedRecommendationEngine)
    # For backwards compatibility, we return the same unified engine instance
    def _collaborative_recommender_factory():
        # The UnifiedRecommendationEngine includes all collaborative filtering functionality
        return c.resolve('recommendation_engine')

    c.register_factory('collaborative_recommender', _collaborative_recommender_factory)

    return c


__all__ = ['Container', 'build_default_container']
