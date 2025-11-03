"""
Core GraphRAG functionality
"""

from .document_processor import DocumentProcessor, DocumentChunk
from .entity_extractor import EntityExtractor, Entity, Relationship
from .graph_manager import GraphManager
from .vector_search import VectorSearch
from .llm_manager import LLMManager
from .academic_schema import (
    PaperNode,
    AuthorNode,
    TopicNode,
    MethodNode,
    DatasetNode,
    InstitutionNode,
    VenueNode,
)
from .academic_graph_manager import AcademicGraphManager
from .relationship_manager import RelationshipManager
from .relationship_extractor import RelationshipExtractor
from .relationship_inference import RelationshipInferenceEngine

__all__ = [
    'DocumentProcessor',
    'DocumentChunk',
    'EntityExtractor',
    'Entity',
    'Relationship',
    'GraphManager',
    'VectorSearch',
    'LLMManager',
    # VersionManager removed (module deleted)
    # Academic schema exports
    'PaperNode',
    'AuthorNode',
    'TopicNode',
    'MethodNode',
    'DatasetNode',
    'InstitutionNode',
    'VenueNode',
    'AcademicGraphManager',
    'RelationshipManager',
    'RelationshipExtractor',
    'RelationshipInferenceEngine',
]
