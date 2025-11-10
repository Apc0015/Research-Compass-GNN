"""
UI Components Module for Research Compass.

This package contains modular UI components extracted from the monolithic
unified_launcher.py as part of Phase 3 architecture improvements.

Modules:
- connection_utils: Connection testing and model detection utilities
"""

from .connection_utils import (
    test_ollama_connection,
    test_lmstudio_connection,
    test_openrouter_connection,
    test_openai_connection,
    detect_ollama_models,
    detect_lmstudio_models,
    detect_openrouter_models,
    detect_openai_models,
    test_neo4j_connection,
)

__all__ = [
    "test_ollama_connection",
    "test_lmstudio_connection",
    "test_openrouter_connection",
    "test_openai_connection",
    "detect_ollama_models",
    "detect_lmstudio_models",
    "detect_openrouter_models",
    "detect_openai_models",
    "test_neo4j_connection",
]
