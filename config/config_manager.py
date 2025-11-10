"""
Unified Configuration Manager for Research Compass

This module provides a centralized configuration system that consolidates
settings from multiple sources (environment variables, YAML files, defaults)
with validation and type checking.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    uri: str = "neo4j://127.0.0.1:7687"
    user: str = "neo4j"
    password: str = ""
    type: str = "neo4j"  # neo4j, memory, postgresql, etc.
    connection_pool_size: int = 100
    connection_timeout: float = 30.0
    max_connection_lifetime: float = 3600.0

    # Pinecone configuration
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "gcp-starter"  # or 'us-east-1-aws', etc.
    pinecone_index_name: str = "research-compass"
    pinecone_dimension: int = 384
    pinecone_metric: str = "cosine"  # cosine, euclidean, dotproduct
    pinecone_use_local: bool = False  # True for Pinecone Lite (local mode)


@dataclass
class LLMConfig:
    """LLM provider configuration settings."""
    provider: str = "ollama"  # ollama, lmstudio, openrouter, openai
    model: str = "llama3.2"
    temperature: float = 0.3
    max_tokens: int = 1000
    timeout: float = 30.0
    max_retries: int = 2
    
    # Provider-specific settings
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    
    # Legacy support
    use_ollama: bool = True
    use_openai: bool = False
    ollama_model: str = "deepseek-r1:1.5b"
    openai_model: str = "gpt-4o-mini"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration settings."""
    model_name: str = "all-MiniLM-L6-v2"
    provider: str = "huggingface"  # huggingface, ollama
    base_url: str = "http://localhost:11434"
    dimension: int = 384
    batch_size: int = 32


@dataclass
class VectorDatabaseConfig:
    """Vector database configuration settings."""
    provider: str = "faiss"  # faiss, pinecone, chroma
    use_pinecone: bool = False  # Shortcut for Pinecone
    use_faiss: bool = True  # Shortcut for FAISS (default)


@dataclass
class ProcessingConfig:
    """Document processing configuration settings."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_chunks: int = 5
    max_graph_depth: int = 3
    use_llama_index: bool = True
    chunk_strategy: str = "hybrid"
    
    # File processing
    allowed_extensions: List[str] = field(default_factory=lambda: ['.pdf', '.txt', '.md', '.docx', '.doc'])
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    
    # Metadata extraction
    metadata_extraction: Dict[str, Any] = field(default_factory=lambda: {
        'use_llm_fallback': True,
        'confidence_threshold': 0.7
    })


@dataclass
class AcademicConfig:
    """Academic-specific configuration settings."""
    gnn: Dict[str, Any] = field(default_factory=lambda: {
        'train_on_startup': False,
        'model_checkpoint_dir': "models/gnn",
        'node_classifier': {
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout': 0.5
        }
    })
    
    indexing: Dict[str, Any] = field(default_factory=lambda: {
        'use_llama_index': True,
        'chunk_strategy': "hybrid",
        'chunk_size': 512,
        'chunk_overlap': 50
    })
    
    recommendations: Dict[str, Any] = field(default_factory=lambda: {
        'top_k': 20,
        'min_confidence': 0.6,
        'diversity_weight': 0.3
    })
    
    citation_analysis: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': 3,
        'min_citations': 5
    })


@dataclass
class UIConfig:
    """User interface configuration settings."""
    port: int = 7860
    host: str = "0.0.0.0"
    share: bool = False
    server_name: str = "127.0.0.1"
    preferred_port: int = 7860
    
    # Visualization
    height: str = "800px"
    width: str = "100%"
    bg_color: str = "#1a1a1a"
    font_color: str = "white"
    
    # Visualization defaults
    max_nodes: int = 200
    layout: str = "force_directed"
    node_sizing: str = "degree"


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    enabled: bool = True
    cache_dir: str = "data/cache"
    max_items: int = 1000
    default_ttl: int = 3600
    cleanup_interval: int = 300


@dataclass
class PathsConfig:
    """File system paths configuration."""
    base_dir: str = ""
    data_dir: str = "data"
    documents_dir: str = "data/documents"
    indices_dir: str = "data/indices"
    cache_dir: str = "data/cache"
    output_dir: str = "output"
    visualization_dir: str = "output/visualizations"
    reports_dir: str = "output/reports"
    exports_dir: str = "output/exports"
    models_dir: str = "models"
    gnn_models_dir: str = "models/gnn"
    chroma_dir: str = "./data/chroma"


@dataclass
class SystemConfig:
    """System-wide configuration settings."""
    environment: str = "development"
    log_level: str = "INFO"
    debug: bool = False
    auto_reload: bool = False
    
    # Performance
    max_workers: int = 4
    request_timeout: float = 60.0
    
    # Version management
    version_snapshot_enabled: bool = True
    max_versions: int = 50
    
    # Temporal queries
    temporal_index_enabled: bool = True
    default_time_window_hours: int = 24


@dataclass
class Config:
    """Main configuration class that consolidates all settings."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_db: VectorDatabaseConfig = field(default_factory=VectorDatabaseConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    academic: AcademicConfig = field(default_factory=AcademicConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    # Additional configuration sections
    custom: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Unified configuration manager that handles multiple configuration sources."""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to the YAML configuration file
        """
        self.config_file = Path(config_file) if config_file else None
        self._config: Optional[Config] = None
        
        # Load configuration from multiple sources
        self._load_config()
        
        # Create directories
        self._create_directories()
    
    def _load_config(self):
        """Load configuration from multiple sources with proper precedence."""
        # Start with defaults
        config_dict = {}
        
        # Load from YAML file if it exists
        if self.config_file and self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        config_dict.update(yaml_config)
                        logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Could not load configuration from {self.config_file}: {e}")
        
        # Override with environment variables
        env_config = self._load_from_environment()
        config_dict.update(env_config)
        
        # Apply legacy migration
        config_dict = self._migrate_legacy_settings(config_dict)
        
        # Validate configuration
        self._validate_config(config_dict)
        
        # Create configuration object
        self._config = self._create_config_from_dict(config_dict)
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Database configuration
        if os.getenv("NEO4J_URI"):
            env_config.setdefault("database", {})["uri"] = os.getenv("NEO4J_URI")
        if os.getenv("NEO4J_USER"):
            env_config.setdefault("database", {})["user"] = os.getenv("NEO4J_USER")
        if os.getenv("NEO4J_PASSWORD"):
            env_config.setdefault("database", {})["password"] = os.getenv("NEO4J_PASSWORD")

        # Pinecone configuration
        if os.getenv("PINECONE_API_KEY"):
            env_config.setdefault("database", {})["pinecone_api_key"] = os.getenv("PINECONE_API_KEY")
        if os.getenv("PINECONE_ENVIRONMENT"):
            env_config.setdefault("database", {})["pinecone_environment"] = os.getenv("PINECONE_ENVIRONMENT")
        if os.getenv("PINECONE_INDEX_NAME"):
            env_config.setdefault("database", {})["pinecone_index_name"] = os.getenv("PINECONE_INDEX_NAME")
        if os.getenv("PINECONE_USE_LOCAL"):
            env_config.setdefault("database", {})["pinecone_use_local"] = os.getenv("PINECONE_USE_LOCAL").lower() == "true"
        
        # LLM configuration
        if os.getenv("LLM_PROVIDER"):
            env_config.setdefault("llm", {})["provider"] = os.getenv("LLM_PROVIDER")
        if os.getenv("LLM_MODEL"):
            env_config.setdefault("llm", {})["model"] = os.getenv("LLM_MODEL")
        if os.getenv("LLM_TEMPERATURE"):
            env_config.setdefault("llm", {})["temperature"] = float(os.getenv("LLM_TEMPERATURE"))
        if os.getenv("LLM_MAX_TOKENS"):
            env_config.setdefault("llm", {})["max_tokens"] = int(os.getenv("LLM_MAX_TOKENS"))
        if os.getenv("LLM_TIMEOUT"):
            env_config.setdefault("llm", {})["timeout"] = float(os.getenv("LLM_TIMEOUT"))
        
        # Embedding configuration
        if os.getenv("EMBEDDING_MODEL_NAME"):
            env_config.setdefault("embedding", {})["model_name"] = os.getenv("EMBEDDING_MODEL_NAME")
        if os.getenv("EMBEDDING_PROVIDER"):
            env_config.setdefault("embedding", {})["provider"] = os.getenv("EMBEDDING_PROVIDER")
        
        # Processing configuration
        if os.getenv("CHUNK_SIZE"):
            env_config.setdefault("processing", {})["chunk_size"] = int(os.getenv("CHUNK_SIZE"))
        if os.getenv("CHUNK_OVERLAP"):
            env_config.setdefault("processing", {})["chunk_overlap"] = int(os.getenv("CHUNK_OVERLAP"))
        if os.getenv("TOP_K_CHUNKS"):
            env_config.setdefault("processing", {})["top_k_chunks"] = int(os.getenv("TOP_K_CHUNKS"))
        if os.getenv("MAX_GRAPH_DEPTH"):
            env_config.setdefault("processing", {})["max_graph_depth"] = int(os.getenv("MAX_GRAPH_DEPTH"))
        
        # UI configuration
        if os.getenv("GRADIO_PORT"):
            env_config.setdefault("ui", {})["port"] = int(os.getenv("GRADIO_PORT"))
        if os.getenv("GRADIO_HOST"):
            env_config.setdefault("ui", {})["host"] = os.getenv("GRADIO_HOST")
        if os.getenv("GRADIO_SHARE"):
            env_config.setdefault("ui", {})["share"] = os.getenv("GRADIO_SHARE").lower() == "true"
        
        # System configuration
        if os.getenv("ENV"):
            env_config.setdefault("system", {})["environment"] = os.getenv("ENV")
        if os.getenv("LOG_LEVEL"):
            env_config.setdefault("system", {})["log_level"] = os.getenv("LOG_LEVEL")
        
        return env_config
    
    def _migrate_legacy_settings(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate legacy settings to the new unified format."""
        # Legacy LLM provider settings
        if os.getenv("USE_OLLAMA", "").lower() == "true" and not config_dict.get("llm", {}).get("provider"):
            config_dict.setdefault("llm", {})["provider"] = "ollama"
            config_dict.setdefault("llm", {})["model"] = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
        
        if os.getenv("USE_OPENAI", "").lower() == "true" and not config_dict.get("llm", {}).get("provider"):
            config_dict.setdefault("llm", {})["provider"] = "openai"
            config_dict.setdefault("llm", {})["model"] = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Legacy database settings
        if os.getenv("GNN_URI") and not config_dict.get("database", {}).get("uri"):
            config_dict.setdefault("database", {})["uri"] = os.getenv("GNN_URI")
        
        if os.getenv("GNN_USER") and not config_dict.get("database", {}).get("user"):
            config_dict.setdefault("database", {})["user"] = os.getenv("GNN_USER")
        
        if os.getenv("GNN_PASSWORD") and not config_dict.get("database", {}).get("password"):
            config_dict.setdefault("database", {})["password"] = os.getenv("GNN_PASSWORD")
        
        return config_dict
    
    def _validate_config(self, config_dict: Dict[str, Any]):
        """Validate configuration values."""
        errors = []
        
        # Validate LLM provider
        llm_config = config_dict.get("llm", {})
        if "provider" in llm_config:
            valid_providers = ["ollama", "lmstudio", "openrouter", "openai"]
            if llm_config["provider"] not in valid_providers:
                errors.append(f"Invalid LLM provider: {llm_config['provider']}. Must be one of: {valid_providers}")
        
        # Validate embedding provider
        embedding_config = config_dict.get("embedding", {})
        if "provider" in embedding_config:
            valid_providers = ["huggingface", "ollama"]
            if embedding_config["provider"] not in valid_providers:
                errors.append(f"Invalid embedding provider: {embedding_config['provider']}. Must be one of: {valid_providers}")
        
        # Validate temperature range
        if "temperature" in llm_config:
            temp = llm_config["temperature"]
            if not (0.0 <= temp <= 1.0):
                errors.append(f"LLM temperature must be between 0.0 and 1.0, got: {temp}")
        
        # Validate chunk size
        if "chunk_size" in config_dict.get("processing", {}):
            chunk_size = config_dict["processing"]["chunk_size"]
            if chunk_size <= 0:
                errors.append(f"Chunk size must be positive, got: {chunk_size}")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
    
    def _create_config_from_dict(self, config_dict: Dict[str, Any]) -> Config:
        """Create Config object from dictionary."""
        config = Config()
        
        # Update each section
        if "database" in config_dict:
            for key, value in config_dict["database"].items():
                if hasattr(config.database, key):
                    setattr(config.database, key, value)
        
        if "llm" in config_dict:
            for key, value in config_dict["llm"].items():
                if hasattr(config.llm, key):
                    setattr(config.llm, key, value)
        
        if "embedding" in config_dict:
            for key, value in config_dict["embedding"].items():
                if hasattr(config.embedding, key):
                    setattr(config.embedding, key, value)

        if "vector_db" in config_dict:
            for key, value in config_dict["vector_db"].items():
                if hasattr(config.vector_db, key):
                    setattr(config.vector_db, key, value)

        if "processing" in config_dict:
            for key, value in config_dict["processing"].items():
                if hasattr(config.processing, key):
                    setattr(config.processing, key, value)
        
        if "academic" in config_dict:
            for key, value in config_dict["academic"].items():
                if hasattr(config.academic, key):
                    setattr(config.academic, key, value)
        
        if "ui" in config_dict:
            for key, value in config_dict["ui"].items():
                if hasattr(config.ui, key):
                    setattr(config.ui, key, value)
        
        if "cache" in config_dict:
            for key, value in config_dict["cache"].items():
                if hasattr(config.cache, key):
                    setattr(config.cache, key, value)
        
        if "paths" in config_dict:
            for key, value in config_dict["paths"].items():
                if hasattr(config.paths, key):
                    setattr(config.paths, key, value)
        
        if "system" in config_dict:
            for key, value in config_dict["system"].items():
                if hasattr(config.system, key):
                    setattr(config.system, key, value)
        
        # Store custom configuration
        if "custom" in config_dict:
            config.custom = config_dict["custom"]
        
        return config
    
    def _create_directories(self):
        """Create necessary directories based on configuration."""
        directories = [
            self._config.paths.data_dir,
            self._config.paths.documents_dir,
            self._config.paths.indices_dir,
            self._config.paths.cache_dir,
            self._config.paths.output_dir,
            self._config.paths.visualization_dir,
            self._config.paths.reports_dir,
            self._config.paths.exports_dir,
            self._config.paths.models_dir,
            self._config.paths.gnn_models_dir,
            self._config.paths.chroma_dir,
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create directory {directory}: {e}")
    
    @property
    def config(self) -> Config:
        """Get the loaded configuration."""
        if self._config is None:
            self._load_config()
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot notation key.
        
        Args:
            key: Configuration key in dot notation (e.g., "llm.provider")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config
        
        try:
            for k in keys:
                value = getattr(value, k)
            return value
        except AttributeError:
            return default
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value by dot notation key.
        
        Args:
            key: Configuration key in dot notation (e.g., "llm.provider")
            value: Value to set
        """
        keys = key.split(".")
        config_obj = self.config
        
        # Navigate to the parent object
        for k in keys[:-1]:
            config_obj = getattr(config_obj, k)
        
        # Set the value
        setattr(config_obj, keys[-1], value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        return dataclass_to_dict(self.config)
    
    def save_to_file(self, file_path: Union[str, Path]):
        """
        Save current configuration to a YAML file.
        
        Args:
            file_path: Path to save the configuration file
        """
        file_path = Path(file_path)
        config_dict = self.to_dict()
        
        try:
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Could not save configuration to {file_path}: {e}")
            raise
    
    def reload(self):
        """Reload configuration from sources."""
        self._load_config()


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: Optional[Union[str, Path]] = None) -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager


def get_config(config_file: Optional[Union[str, Path]] = None) -> Config:
    """
    Get the global configuration.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Config instance
    """
    return get_config_manager(config_file).config


@lru_cache(maxsize=1)
def get_settings() -> Dict[str, Any]:
    """
    Get configuration as dictionary (for backward compatibility).
    
    Returns:
        Configuration dictionary
    """
    return get_config().to_dict()