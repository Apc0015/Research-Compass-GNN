"""Unified AcademicRAGSystem that wires together ingestion, graph, indexing and ML.

This is a high-level orchestrator that composes the various components added by
Prompts 1-6. Implementations prefer existing modules when available and fall
back to conservative placeholders otherwise.
"""
from __future__ import annotations

from typing import List, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from .graph_manager import GraphManager
from .academic_graph_manager import AcademicGraphManager
from .document_processor import DocumentProcessor
from .container import Container, build_default_container

# Optional components
try:
    from src.graphrag.indexing.advanced_indexer import AdvancedDocumentIndexer
except Exception:
    AdvancedDocumentIndexer = None

try:
    from src.graphrag.ml.gnn_manager import GNNManager
except Exception:
    GNNManager = None

try:
    from src.graphrag.analytics.relationship_analytics import RelationshipAnalytics
except Exception:
    RelationshipAnalytics = None

try:
    from src.graphrag.analytics.impact_metrics import ImpactMetricsCalculator
except Exception:
    ImpactMetricsCalculator = None

try:
    from src.graphrag.analytics.unified_recommendation_engine import UnifiedRecommendationEngine
except Exception:
    UnifiedRecommendationEngine = None

# Legacy alias for backwards compatibility
PaperRecommendationEngine = UnifiedRecommendationEngine


class AcademicRAGSystem:
    """Top-level system integrating academic ingestion, indexing, graph and ML.

    Provides convenient workflows for processing papers, querying, analysis and
    recommendations. Methods are defensive and will run with minimal
    functionality if optional components are not present.
    """

    def __init__(self, container: Optional[Container] = None, config: Optional[Dict] = None):
        """Initialize the system using dependencies from a container.

        If no container is provided, a default container is built. This allows
        easier testing and looser coupling between components.
        """
        self.config = config or {}
        self.container = container or build_default_container(self.config)

        # Resolve core components via container
        try:
            self.graph = self.container.resolve('graph_manager')
            self.academic = self.container.resolve('academic_graph_manager')
            self.doc_processor = self.container.resolve('document_processor')
        except KeyError as e:
            logger.exception("Missing core dependency in container: %s", e)
            raise RuntimeError(f"Failed to initialize AcademicRAGSystem due to missing dependency: {e}")

        # Optional subsystems (resolve if registered)
        try:
            self.indexer = self.container.resolve('advanced_indexer')
        except Exception:
            self.indexer = AdvancedDocumentIndexer() if AdvancedDocumentIndexer is not None else None

        # Defensive instantiation for GNNManager: try container first then fallbacks
        self.gnn_manager = None
        try:
            self.gnn_manager = self.container.resolve('gnn_manager')
        except Exception:
            if GNNManager is not None:
                try:
                    self.gnn_manager = GNNManager(self.graph)
                except TypeError:
                    try:
                        self.gnn_manager = GNNManager()
                    except Exception:
                        logger.exception("Failed to instantiate GNNManager; disabling GNNManager")
                        self.gnn_manager = None
                except Exception:
                    logger.exception("Unexpected error instantiating GNNManager; disabling GNNManager")
                    self.gnn_manager = None

        try:
            self.relationship_analytics = self.container.resolve('relationship_analytics')
        except Exception:
            self.relationship_analytics = RelationshipAnalytics(self.graph) if RelationshipAnalytics is not None else None

        try:
            self._impact_metrics = self.container.resolve('impact_metrics')
        except Exception:
            self._impact_metrics = ImpactMetricsCalculator(self.graph) if ImpactMetricsCalculator is not None else None

        try:
            self._recommendation_engine = self.container.resolve('recommendation_engine')
        except Exception:
            self._recommendation_engine = PaperRecommendationEngine(self.graph) if PaperRecommendationEngine is not None else None

    def process_academic_paper(self, pdf_path: str) -> Dict:
        """End-to-end processing: extract metadata, create nodes, index, and update ML.

        Returns the created PaperNode info and additional artifacts.
        """
        # 1. Extract and store metadata + nodes
        result = self.doc_processor.process_academic_paper(pdf_path, academic_graph_manager=self.academic, extract_metadata=True)

        created = result.get('created', {})

        # 2. Index the paper
        if self.indexer is not None:
            try:
                paper_meta = result.get('metadata', {})
                full_text = self.doc_processor.load_pdf(Path(pdf_path))
                self.indexer.index_academic_paper(paper_meta, full_text)
            except Exception:
                logger.exception("Indexing failed for %s", pdf_path)

        # 3. Update GNN embeddings (if available)
        if self.gnn_manager is not None:
            try:
                # Trigger incremental embedding update / retrain as configured
                if getattr(self.gnn_manager, 'update_embeddings', None):
                    self.gnn_manager.update_embeddings()
            except Exception:
                logger.exception("GNN update failed")

        return {'processed': pdf_path, 'created': created}

    def query_academic(self, query: str, retrieval_strategy: str = "hybrid", include_recommendations: bool = False, include_citations: bool = True) -> Dict:
        """Run a query against index and graph; return answer, sources, and optional recommendations.
        """
        response = {'query': query, 'answers': [], 'sources': [], 'recommendations': []}

        # Use indexer / vector search first
        try:
            if self.indexer and getattr(self.indexer, 'vector_search', None):
                vs = self.indexer.vector_search
                hits = vs.search(query, top_k=5)
                response['sources'] = hits
        except Exception:
            logger.exception("Vector search failed for query")

        # Optional recommendations
        if include_recommendations and self.recommendation_engine is not None:
            try:
                recs = self.recommendation_engine.recommend_papers_for_user([], [], top_k=5)
                response['recommendations'] = recs
            except Exception:
                logger.exception("Recommendation engine failed")

        # Placeholder for LLM answer synthesis
        response['answers'] = [f"Top sources: {len(response['sources'])}"]

        return response

    def analyze_research_topic(self, topic: str) -> Dict:
        """Return a combined analysis for a given topic: citations, trends, seminal papers.

        This function uses available analytics modules and falls back to
        conservative summaries.
        """
        analysis = {'topic': topic, 'seminal_papers': [], 'trends': [], 'metrics': {}}
        try:
            if self.relationship_analytics is not None:
                analysis['metrics'] = self.relationship_analytics.analyze_topic_evolution(topic)
            if self.impact_metrics is not None:
                # placeholder: compute topic momentum if implemented
                analysis['metrics']['topic_momentum'] = self.impact_metrics.calculate_topic_momentum(topic)
        except Exception:
            logger.exception("Failed to analyze research topic %s", topic)

        return analysis

    def get_research_recommendations(self, papers_read: List[str], interests: List[str]) -> List[Dict]:
        if self.recommendation_engine is None:
            return []
        try:
            return self.recommendation_engine.recommend_papers_for_user(papers_read, interests, top_k=self.config.get('recommendations', {}).get('top_k', 10))
        except Exception:
            logger.exception("Failed to get research recommendations")
            return []

    def predict_paper_impact(self, paper_id: str) -> Dict:
        out = {'paper_id': paper_id, 'predicted_citations': None, 'confidence': 0.0}
        try:
            if self.impact_metrics is not None:
                out['pagerank'] = self.impact_metrics.calculate_paper_pagerank(paper_id)
            if self.gnn_manager is not None and getattr(self.gnn_manager, 'predict_impact', None):
                out['predicted_citations'] = self.gnn_manager.predict_impact(paper_id)
        except Exception:
            logger.exception("Failed to predict impact for %s", paper_id)
        return out

    # Advanced GNN-powered analytics components
    @property
    def temporal_analytics(self):
        """Temporal graph analytics for research evolution tracking."""
        if not hasattr(self, '_temporal_analytics'):
            self._temporal_analytics = self.container.resolve('temporal_analytics')
        return self._temporal_analytics
    
    @property
    def gnn_explainer(self):
        """GNN explainability for transparent AI decisions."""
        if not hasattr(self, '_gnn_explainer'):
            self._gnn_explainer = self.container.resolve('gnn_explainer')
        return self._gnn_explainer
    
    @property
    def gnn_query_engine(self):
        """GNN-enhanced query engine with multi-source reasoning."""
        if not hasattr(self, '_gnn_query_engine'):
            self._gnn_query_engine = self.container.resolve('gnn_query_engine')
        return self._gnn_query_engine
    
    @property
    def recommendation_engine(self):
        """Personalized recommendation engine."""
        if not hasattr(self, '_recommendation_engine'):
            self._recommendation_engine = self.container.resolve('recommendation_engine')
        return self._recommendation_engine
    
    @property
    def discovery_engine(self):
        """Serendipitous discovery engine for cross-disciplinary exploration."""
        if not hasattr(self, '_discovery_engine'):
            self._discovery_engine = self.container.resolve('discovery_engine')
        return self._discovery_engine
    
    @property
    def citation_explorer(self):
        """Interactive citation network explorer."""
        if not hasattr(self, '_citation_explorer'):
            self._citation_explorer = self.container.resolve('citation_explorer')
        return self._citation_explorer
    
    @property
    def citation_metrics(self):
        """Advanced citation metrics analyzer."""
        if not hasattr(self, '_citation_metrics'):
            self._citation_metrics = self.container.resolve('citation_metrics')
        return self._citation_metrics
    
    @property
    def interdisciplinary_analyzer(self):
        """Interdisciplinary research analyzer."""
        if not hasattr(self, '_interdisciplinary_analyzer'):
            self._interdisciplinary_analyzer = self.container.resolve('interdisciplinary_analyzer')
        return self._interdisciplinary_analyzer
    
    @property
    def collaborative_recommender(self):
        """Collaborative filtering recommendation engine."""
        if not hasattr(self, '_collaborative_recommender'):
            self._collaborative_recommender = self.container.resolve('collaborative_recommender')
        return self._collaborative_recommender
