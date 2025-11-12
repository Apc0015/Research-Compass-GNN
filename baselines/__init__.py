"""
Baseline models for comparison with GNNs
"""

from .traditional_ml import (
    MLPBaseline,
    RandomBaseline,
    TraditionalMLBaseline,
    train_mlp_baseline,
    evaluate_all_baselines
)

from .graph_baselines import (
    LabelPropagation,
    SimpleNode2Vec,
    evaluate_graph_baselines
)

__all__ = [
    'MLPBaseline',
    'RandomBaseline',
    'TraditionalMLBaseline',
    'train_mlp_baseline',
    'evaluate_all_baselines',
    'LabelPropagation',
    'SimpleNode2Vec',
    'evaluate_graph_baselines',
]
