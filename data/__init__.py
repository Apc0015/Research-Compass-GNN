"""
Dataset utilities and loaders
"""

from .dataset_utils import (
    create_synthetic_citation_network,
    load_citation_dataset,
    get_dataset_statistics,
    print_dataset_info,
    create_link_prediction_split
)

from .heterogeneous_graph_builder import (
    HeterogeneousGraphBuilder,
    convert_to_heterogeneous
)

__all__ = [
    'create_synthetic_citation_network',
    'load_citation_dataset',
    'get_dataset_statistics',
    'print_dataset_info',
    'create_link_prediction_split',
    'HeterogeneousGraphBuilder',
    'convert_to_heterogeneous',
]
