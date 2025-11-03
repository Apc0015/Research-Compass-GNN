"""
GraphRAG Configuration Settings
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# LLM Configuration - Unified Multi-Provider Support
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # ollama, lmstudio, openrouter, openai
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30.0"))

# Local LLM Providers (Ollama, LM Studio)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234")

# Cloud LLM Providers (OpenRouter, OpenAI)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

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
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Chunking Parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval Parameters
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "5"))
MAX_GRAPH_DEPTH = int(os.getenv("MAX_GRAPH_DEPTH", "3"))

# Data Directories
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
INDEX_DIR = DATA_DIR / "indices"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
REPORTS_DIR = OUTPUT_DIR / "reports"
EXPORTS_DIR = OUTPUT_DIR / "exports"

# Allowed file extensions
ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.md', '.docx', '.doc']

# Create directories
for directory in [DOCUMENTS_DIR, INDEX_DIR, CACHE_DIR, VISUALIZATION_DIR, REPORTS_DIR, EXPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Gradio Configuration
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_HOST = os.getenv("GRADIO_HOST", "0.0.0.0")
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "False").lower() == "true"

# Visualization Configuration
VIZ_HEIGHT = "800px"
VIZ_WIDTH = "100%"
VIZ_BG_COLOR = "#1a1a1a"
VIZ_FONT_COLOR = "white"
