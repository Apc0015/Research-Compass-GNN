"""
GraphRAG Configuration Settings - Unified Configuration System

This module now uses the unified configuration manager to provide
backward compatibility while consolidating all configuration sources.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Import the unified configuration manager
from .config_manager import get_config, get_config_manager

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Get the unified configuration instance
_config = get_config()

# Neo4j Configuration (backward compatibility)
NEO4J_URI = _config.database.uri
NEO4J_USER = _config.database.user
NEO4J_PASSWORD = _config.database.password

# LLM Configuration - Unified Multi-Provider Support
LLM_PROVIDER = _config.llm.provider
LLM_MODEL = _config.llm.model
LLM_TEMPERATURE = _config.llm.temperature
LLM_MAX_TOKENS = _config.llm.max_tokens
LLM_TIMEOUT = _config.llm.timeout

# Local LLM Providers (Ollama, LM Studio)
OLLAMA_BASE_URL = _config.llm.base_url
LMSTUDIO_BASE_URL = _config.llm.base_url  # Default to same base URL

# Cloud LLM Providers (OpenRouter, OpenAI)
OPENROUTER_API_KEY = _config.llm.api_key
OPENAI_API_KEY = _config.llm.api_key

# Legacy support for backwards compatibility (DEPRECATED - use LLM_PROVIDER instead)
# These variables are maintained for old code but should not be used in new code
_USE_OLLAMA = os.getenv("USE_OLLAMA", "").lower() == "true"
_USE_OPENAI = os.getenv("USE_OPENAI", "").lower() == "true"
# Auto-migrate legacy settings to new unified system
if _USE_OLLAMA and not os.getenv("LLM_PROVIDER"):
    LLM_PROVIDER = "ollama"
    LLM_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
elif _USE_OPENAI and not os.getenv("LLM_PROVIDER"):
    LLM_PROVIDER = "openai"
    LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Embedding Model
EMBEDDING_MODEL_NAME = _config.embedding.model_name

# Chunking Parameters
CHUNK_SIZE = _config.processing.chunk_size
CHUNK_OVERLAP = _config.processing.chunk_overlap

# Retrieval Parameters
TOP_K_CHUNKS = _config.processing.top_k_chunks
MAX_GRAPH_DEPTH = _config.processing.max_graph_depth

# Data Directories
DATA_DIR = Path(_config.paths.data_dir)
DOCUMENTS_DIR = Path(_config.paths.documents_dir)
INDEX_DIR = Path(_config.paths.indices_dir)
CACHE_DIR = Path(_config.paths.cache_dir)
OUTPUT_DIR = Path(_config.paths.output_dir)
VISUALIZATION_DIR = Path(_config.paths.visualization_dir)
REPORTS_DIR = Path(_config.paths.reports_dir)
EXPORTS_DIR = Path(_config.paths.exports_dir)

# Allowed file extensions
ALLOWED_EXTENSIONS = _config.processing.allowed_extensions

# Create directories (using unified config manager)
try:
    config_manager = get_config_manager()
    config_manager._create_directories()
except Exception:
    # Fallback to original directory creation
    for directory in [DOCUMENTS_DIR, INDEX_DIR, CACHE_DIR, VISUALIZATION_DIR, REPORTS_DIR, EXPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Gradio Configuration
GRADIO_PORT = _config.ui.port
GRADIO_HOST = _config.ui.host
GRADIO_SHARE = _config.ui.share

# Visualization Configuration
VIZ_HEIGHT = _config.ui.height
VIZ_WIDTH = _config.ui.width
VIZ_BG_COLOR = _config.ui.bg_color
VIZ_FONT_COLOR = _config.ui.font_color

# Additional configuration for backward compatibility
def get_settings() -> Dict[str, Any]:
    """Get all settings as a dictionary (backward compatibility)."""
    return _config.to_dict()

def get_setting(key: str, default: Any = None) -> Any:
    """Get a specific setting by key (backward compatibility)."""
    return _config.get(key, default)
