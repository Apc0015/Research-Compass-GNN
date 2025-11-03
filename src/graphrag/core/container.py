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
    cfg = config or {}
    c = Container()

    # Register config
    c.register_singleton('config', cfg)
    
    # Neo4j connection details
    neo = cfg.get('neo4j', {})
    neo4j_uri = neo.get('uri', cfg.get('NEO4J_URI', 'bolt://127.0.0.1:7687'))
    neo4j_user = neo.get('user', cfg.get('NEO4J_USER', 'neo4j'))
    neo4j_password = neo.get('password', cfg.get('NEO4J_PASSWORD', 'password'))

    # GraphManager factory
    def _graph_factory():
        from .graph_manager import GraphManager
        return GraphManager(neo4j_uri, neo4j_user, neo4j_password)

    c.register_factory('graph_manager', _graph_factory)

    # DocumentProcessor factory
    def _docproc_factory():
        from .document_processor import DocumentProcessor
        return DocumentProcessor()

    c.register_factory('document_processor', _docproc_factory)

    # CacheManager factory (NEW)
    def _cache_factory():
        from .cache_manager import CacheManager
        from pathlib import Path
        cache_dir = Path(cfg.get('cache_dir', 'data/cache'))
        max_items = cfg.get('max_cache_items', 1000)
        ttl = cfg.get('default_cache_ttl', 3600)
        return CacheManager(cache_dir, max_items, ttl)
    
    c.register_factory('cache_manager', _cache_factory)

    # AcademicGraphManager factory (depends on graph_manager)
    def _academic_graph_factory():
        from .academic_graph_manager import AcademicGraphManager
        graph = c.resolve('graph_manager')
        return AcademicGraphManager(graph)

    c.register_factory('academic_graph_manager', _academic_graph_factory)
    
    # VectorSearch factory
    def _vector_search_factory():
        from .vector_search import VectorSearch
        import os
        model_name = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
        provider = os.getenv('EMBEDDING_PROVIDER', 'huggingface')
        base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        return VectorSearch(model_name, provider, base_url)
    
    c.register_factory('vector_search', _vector_search_factory)
    
    # LLMManager factory (NEW - for dynamic configuration updates)
    def _llm_manager_factory():
        from .llm_manager import LLMManager
        import os
        provider = os.getenv('LLM_PROVIDER', 'ollama')
        model = os.getenv('LLM_MODEL', 'llama3.2')
        base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434') if provider == 'ollama' else os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234')
        api_key = os.getenv('OPENROUTER_API_KEY') if provider == 'openrouter' else os.getenv('OPENAI_API_KEY')
        temperature = float(os.getenv('LLM_TEMPERATURE', '0.3'))
        max_tokens = int(os.getenv('LLM_MAX_TOKENS', '1000'))
        
        return LLMManager(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    c.register_factory('llm_manager', _llm_manager_factory)
    
    # GNNManager factory
    def _gnn_manager_factory():
        from ..ml.gnn_manager import GNNManager
        return GNNManager(neo4j_uri, neo4j_user, neo4j_password)
    
    c.register_factory('gnn_manager', _gnn_manager_factory)
    
    # TemporalGraphAnalytics factory
    def _temporal_analytics_factory():
        from ..analytics.temporal_analytics import TemporalGraphAnalytics
        graph = c.resolve('graph_manager')
        return TemporalGraphAnalytics(graph, neo4j_uri, neo4j_user, neo4j_password)
    
    c.register_factory('temporal_analytics', _temporal_analytics_factory)
    
    # GNNExplainer factory
    def _gnn_explainer_factory():
        from ..ml.gnn_explainer import GNNExplainer
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
