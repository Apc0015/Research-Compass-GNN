"""
Machine Learning module for GraphRAG
Includes GNN models for node classification, link prediction, and embeddings generation
"""

from .graph_converter import Neo4jToTorchGeometric
from .node_classifier import PaperClassifier, train_node_classifier
from .link_predictor import CitationPredictor, train_link_predictor
from .embeddings_generator import GraphEmbedder
from .gnn_manager import GNNManager

__all__ = [
    'Neo4jToTorchGeometric',
    'PaperClassifier',
    'CitationPredictor',
    'GraphEmbedder',
    'GNNManager',
    'train_node_classifier',
    'train_link_predictor'
]
