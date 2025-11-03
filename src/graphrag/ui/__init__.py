"""
UI components for Research Compass.

This module provides the unified launcher interface.
"""

# Import from unified launcher
from .unified_launcher import create_unified_ui, launch_unified_ui

# Export unified interface
__all__ = [
    'create_unified_ui',
    'launch_unified_ui'
]
