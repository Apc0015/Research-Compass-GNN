"""
Evaluation module for GraphRAG system.

Provides comprehensive evaluation frameworks for GNN models,
recommendation systems, and graph analytics.
"""

from .gnn_evaluator import GNNEvaluator, GNNEvaluationMetrics

__all__ = [
    "GNNEvaluator",
    "GNNEvaluationMetrics"
]