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

__all__ = [
    'create_synthetic_citation_network',
    'load_citation_dataset',
    'get_dataset_statistics',
    'print_dataset_info',
    'create_link_prediction_split',
]
