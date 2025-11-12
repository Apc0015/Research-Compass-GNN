"""
Configuration module for Research Compass GNN
"""

from .settings import (
    load_config,
    get_config,
    get_model_config,
    get_training_config,
    get_paths_config,
    Settings
)

__all__ = [
    'load_config',
    'get_config',
    'get_model_config',
    'get_training_config',
    'get_paths_config',
    'Settings',
]
